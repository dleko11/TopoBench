"""End-to-end, feature-free tests for the v2 structural recovery runner."""

import hashlib
import json
import sys
from collections import defaultdict
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import scripts.structural_coverage.run_recovery_diagnostic as runner
from scripts.structural_coverage.recovery_io import load_recovery_bundle
from scripts.structural_coverage.run_recovery_diagnostic import (
    RecoveryGraphAdapter,
    RecoveryRunInputs,
    main,
    preflight_recovery_inputs,
    run_recovery_experiment,
)


def _digest(value):
    return hashlib.sha256(
        np.asarray(value, dtype=np.int64).tobytes()
    ).hexdigest()


def _toy_inputs(*, families=None):
    """A triangle spread across three singleton clusters plus an isolate."""
    directed_edges = [
        (0, 1),
        (1, 0),
        (0, 2),
        (2, 0),
        (1, 2),
        (2, 1),
    ]
    edges = np.asarray(directed_edges, dtype=np.int64).T.copy()
    adapter = RecoveryGraphAdapter(
        np.asarray([0, 1, 2, 3, 4], dtype=np.int64),
        np.asarray([0, 2, 4, 6, 6], dtype=np.int64),
        np.asarray([1, 2, 0, 2, 0, 1], dtype=np.int64),
        np.arange(4, dtype=np.int64),
    )
    topology = hashlib.sha256()
    topology.update(np.asarray([4, 6], dtype=np.int64).tobytes())
    topology.update(edges.tobytes())
    graph_hash = topology.hexdigest()
    order_hash = _digest(edges)
    assignment_hash = (
        _digest(adapter.perm_to_global) + ":" + _digest(adapter.partptr)
    )
    sidecar = {
        "topology_sha256": graph_hash,
        "edge_order_sha256": order_hash,
        "csr_order_sha256": adapter.csr_order_sha256(),
        "partition_assignment_sha256": assignment_hash,
        "num_nodes": 4,
        "num_parts": 4,
        "num_input_edges": 6,
        "basis_order_policy": "native_local_order",
    }
    train_mask = np.ones(4, dtype=bool)
    provenance = {
        "edge_order_sha256": order_hash,
        "dataset": "toy",
        "num_nodes": 4,
        "num_parts": 4,
        "num_edges_directed": 6,
        "max_cell_length": 9,
        "partition_assignment_sha256": assignment_hash,
        "train_mask_sha256": hashlib.sha256(train_mask.tobytes()).hexdigest(),
        "active_cluster_count": 4,
    }
    selected_families = families or ["hypergraph", "simplicial", "cellular"]
    return RecoveryRunInputs(
        dataset="toy",
        edge_index=edges,
        adapters={4: adapter},
        partition_sidecars={4: sidecar},
        provenance_manifests={4: provenance},
        train_masks_perm={4: train_mask},
        configuration_matrix=[
            {"K": 4, "q": q, "families": selected_families} for q in (2, 4)
        ],
        seeds=[0, 1, 2],
        epochs=2,
        sample_every=1,
        loader_config="toy-fixed",
        split_config={"type": "fixed", "seed": 0},
        partition_source="regenerated",
    )


def _run_toy(tmp_path, *, families=None, name="bundle"):
    inputs = _toy_inputs(families=families)
    run_recovery_experiment(
        inputs,
        output_dir=tmp_path / name,
        run_mode="smoke",
        implementation_commit="f" * 40,
        max_wall_seconds=30,
    )
    return load_recovery_bundle(tmp_path / name, publication_only=False)


def _seed_coverage_row(K, q, seed, coverage):
    return {
        "K": K,
        "q": q,
        "seed": seed,
        "epoch": 1,
        "family": "hypergraph",
        "measurement": "support_available",
        "coverage": coverage,
    }


def test_aggregation_keeps_partition_configurations_separate():
    rows = [
        _seed_coverage_row(32, 16, 0, 0.0),
        _seed_coverage_row(32, 16, 1, 0.0),
        _seed_coverage_row(64, 32, 0, 1.0),
        _seed_coverage_row(64, 32, 1, 1.0),
    ]

    summaries = runner._aggregate_observations(
        rows,
        expected_seeds=[0, 1],
        reference_ids={"hypergraph": set(range(4))},
        histograms_by_K={
            32: {"hypergraph": {1: 4}},
            64: {"hypergraph": {1: 4}},
        },
        dataset="toy",
    )

    assert [
        (row["K"], row["q"], row["n_repetitions"], row["coverage_mean"])
        for row in summaries
    ] == [(32, 16, 2, 0.0), (64, 32, 2, 1.0)]


def test_aggregation_rejects_duplicate_seed_as_extra_repetition():
    rows = [_seed_coverage_row(4, 2, 0, 0.0)] * 2

    with pytest.raises(ValueError, match="duplicate.*seed"):
        runner._aggregate_observations(
            rows,
            expected_seeds=[0],
            reference_ids={"hypergraph": set(range(4))},
            histograms_by_K={4: {"hypergraph": {1: 4}}},
            dataset="toy",
        )


def test_aggregation_rejects_missing_seed_instead_of_narrower_band():
    rows = [_seed_coverage_row(4, 2, 0, 0.0)]

    with pytest.raises(ValueError, match="missing.*seed"):
        runner._aggregate_observations(
            rows,
            expected_seeds=[0, 1],
            reference_ids={"hypergraph": set(range(4))},
            histograms_by_K={4: {"hypergraph": {1: 4}}},
            dataset="toy",
        )


def test_single_seed_smoke_bundle_does_not_invent_uncertainty(tmp_path):
    inputs = replace(_toy_inputs(), seeds=[0])
    run_recovery_experiment(
        inputs,
        output_dir=tmp_path / "single-seed",
        run_mode="smoke",
        implementation_commit="f" * 40,
        max_wall_seconds=30,
    )
    bundle = load_recovery_bundle(
        tmp_path / "single-seed", publication_only=False
    )

    assert {row["n_repetitions"] for row in bundle["summary"]} == {1}
    assert all(row["coverage_sample_sd"] is None for row in bundle["summary"])


def test_toy_run_writes_and_reloads_independently_known_counts(tmp_path):
    bundle = _run_toy(tmp_path)
    assert bundle["manifest"]["completed"] is True
    assert bundle["manifest"]["run_mode"] == "smoke"
    assert bundle["manifest"]["partition_source"] == "regenerated"
    assert bundle["manifest"]["seeds"] == [0, 1, 2]
    assert bundle["manifest"]["epochs"] == 2
    assert len(bundle["observations"]) == 2 * 3 * 3 * 4

    by_key = defaultdict(set)
    for row in bundle["observations"]:
        key = (row["q"], row["family"], row["measurement"], row["epoch"])
        by_key[key].add((row["recovered_count"], row["reference_count"]))
    for epoch in range(3):
        assert by_key[2, "hypergraph", "support_available", epoch] == {
            (0 if epoch == 0 else 1, 4)
        }
        assert by_key[4, "hypergraph", "support_available", epoch] == {
            (0 if epoch == 0 else 4, 4)
        }
        for family in ("simplicial", "cellular"):
            assert by_key[2, family, "support_available", epoch] == {(0, 1)}
            assert by_key[4, family, "support_available", epoch] == {
                (0 if epoch == 0 else 1, 1)
            }
        assert by_key[2, "cellular", "actual_basis_recovery", epoch] == {
            (0, 1)
        }
        assert by_key[4, "cellular", "actual_basis_recovery", epoch] == {
            (0 if epoch == 0 else 1, 1)
        }


def test_family_iteration_order_does_not_change_counts(tmp_path):
    forward = _run_toy(tmp_path, name="forward")
    reverse = _run_toy(
        tmp_path,
        families=["cellular", "simplicial", "hypergraph"],
        name="reverse",
    )

    def order(row):
        return tuple(str(row[key]) for key in sorted(row))

    for key in (
        "reference_summary",
        "span_histogram",
        "theory",
        "observations",
        "summary",
    ):
        assert sorted(forward[key], key=order) == sorted(
            reverse[key], key=order
        )


def test_each_seeded_batch_is_built_once_for_all_families(
    tmp_path, monkeypatch
):
    inputs = _toy_inputs()
    adapter = inputs.adapters[4]
    calls = []
    original = adapter.batch_for_clusters

    def counted(parts):
        calls.append(tuple(parts))
        return original(parts)

    monkeypatch.setattr(adapter, "batch_for_clusters", counted)
    run_recovery_experiment(
        inputs,
        output_dir=tmp_path / "once",
        run_mode="smoke",
        implementation_commit="f" * 40,
        max_wall_seconds=30,
    )
    # 3 seeds x 2 epochs x (2 batches for q=2 + 1 batch for q=4).
    # One extra full-batch construction is allowed for the q=K order audit.
    assert len(calls) == 18 or len(calls) == 19


def test_one_local_graph_is_reused_by_hypergraph_and_cellular(
    tmp_path, monkeypatch
):
    calls = []
    original = runner.RecoveryBatch.local_graph

    def counted(batch):
        calls.append(batch.global_nodes)
        return original(batch)

    monkeypatch.setattr(runner.RecoveryBatch, "local_graph", counted)
    _run_toy(tmp_path)
    # 18 sampled batches, one native q=K audit, and one identity-order audit.
    assert len(calls) == 20


def test_dirty_native_source_changes_patch_digest_with_unchanged_status(
    tmp_path, monkeypatch
):
    native_file = (
        tmp_path
        / "topobench"
        / "transforms"
        / "liftings"
        / "graph2cell"
        / "cycle_lifting.py"
    )
    native_file.parent.mkdir(parents=True)
    native_file.write_text("version = 1\n", encoding="utf-8")

    def fake_git(args, **_kwargs):
        if args[:3] == ["git", "rev-parse", "HEAD"]:
            return SimpleNamespace(stdout="f" * 40 + "\n")
        assert args[:3] == ["git", "status", "--porcelain"]
        return SimpleNamespace(
            stdout=" M topobench/transforms/liftings/graph2cell/cycle_lifting.py\n"
        )

    monkeypatch.setattr(runner.subprocess, "run", fake_git)
    first = runner._code_provenance(root=tmp_path)
    native_file.write_text("version = 2\n", encoding="utf-8")
    second = runner._code_provenance(root=tmp_path)
    assert first[0] == second[0] == "f" * 40
    assert first[1] != second[1]


def test_one_seeded_schedule_is_shared_across_families(tmp_path, monkeypatch):
    original = runner.generate_epoch_schedules
    requested = []

    def counted(K, q, epochs, seed):
        requested.append((K, q, epochs, seed))
        return original(K, q, epochs, seed)

    monkeypatch.setattr(runner, "generate_epoch_schedules", counted)
    _run_toy(tmp_path)
    assert sorted(requested) == sorted(
        (4, q, 2, seed) for q in (2, 4) for seed in (0, 1, 2)
    )


def test_reference_snapshot_is_explicit_bootstrap_then_must_match(tmp_path):
    inputs = _toy_inputs()
    candidate = preflight_recovery_inputs(inputs)["reference_fingerprints"]
    assert candidate["cellular"]["count"] == 1
    assert len(candidate["cellular"]["sha256"]) == 64
    with pytest.raises(ValueError, match="snapshot is required"):
        run_recovery_experiment(
            inputs,
            output_dir=tmp_path / "unfrozen",
            run_mode="smoke",
            pinned_config={"dataset": "cora_full"},
        )
    bad = {name: dict(value) for name, value in candidate.items()}
    bad["cellular"]["count"] += 1
    with pytest.raises(ValueError, match="reference count or hash drift"):
        preflight_recovery_inputs(inputs, reference_snapshot=bad)
    assert not (tmp_path / "unfrozen").exists()


def test_cellular_smoke_accepts_full_preflight_snapshot_subset(tmp_path):
    full_snapshot = preflight_recovery_inputs(_toy_inputs())[
        "reference_fingerprints"
    ]
    bundle = run_recovery_experiment(
        _toy_inputs(families=["cellular"]),
        output_dir=tmp_path / "cellular-pilot",
        run_mode="smoke",
        implementation_commit="f" * 40,
        reference_snapshot=full_snapshot,
        max_wall_seconds=30,
    )
    assert bundle["manifest"]["run_mode"] == "smoke"
    assert set(bundle["manifest"]["reference_hashes"]) == {"cellular"}
    with pytest.raises(ValueError, match="smoke bundle"):
        load_recovery_bundle(tmp_path / "cellular-pilot")


def test_cellular_smoke_requires_matching_saved_cellular_reference(tmp_path):
    full_snapshot = preflight_recovery_inputs(_toy_inputs())[
        "reference_fingerprints"
    ]
    missing = {
        family: value
        for family, value in full_snapshot.items()
        if family != "cellular"
    }
    with pytest.raises(ValueError, match="snapshot: cellular"):
        run_recovery_experiment(
            _toy_inputs(families=["cellular"]),
            output_dir=tmp_path / "missing-cellular",
            run_mode="smoke",
            implementation_commit="f" * 40,
            reference_snapshot=missing,
            max_wall_seconds=30,
        )
    assert not (tmp_path / "missing-cellular").exists()

    mismatched = {name: dict(value) for name, value in full_snapshot.items()}
    mismatched["cellular"]["sha256"] = "0" * 64
    with pytest.raises(
        ValueError, match="cellular reference count or hash drift"
    ):
        run_recovery_experiment(
            _toy_inputs(families=["cellular"]),
            output_dir=tmp_path / "mismatched-cellular",
            run_mode="smoke",
            implementation_commit="f" * 40,
            reference_snapshot=mismatched,
            max_wall_seconds=30,
        )
    assert not (tmp_path / "mismatched-cellular").exists()


def test_full_preflight_rejects_a_subset_reference_snapshot():
    full_snapshot = preflight_recovery_inputs(_toy_inputs())[
        "reference_fingerprints"
    ]
    with pytest.raises(ValueError, match="reference families differ"):
        preflight_recovery_inputs(
            _toy_inputs(),
            reference_snapshot={"cellular": full_snapshot["cellular"]},
        )


def test_mismatched_saved_sidecar_fails_before_output(tmp_path):
    inputs = _toy_inputs()
    incorrect = dict(inputs.partition_sidecars[4])
    incorrect["edge_order_sha256"] = "0" * 64
    inputs = replace(inputs, partition_sidecars={4: incorrect})
    with pytest.raises(ValueError, match="sidecar edge order"):
        run_recovery_experiment(
            inputs,
            output_dir=tmp_path / "invalid",
            run_mode="smoke",
            implementation_commit="f" * 40,
            max_wall_seconds=30,
        )
    assert not (tmp_path / "invalid").exists()


def test_bounded_smoke_timeout_writes_no_partial_bundle(tmp_path):
    output = tmp_path / "expired"
    with pytest.raises(TimeoutError, match="checked between batches"):
        run_recovery_experiment(
            _toy_inputs(),
            output_dir=output,
            run_mode="smoke",
            implementation_commit="f" * 40,
            max_wall_seconds=1e-12,
        )
    assert not output.exists()


def test_q_one_recovers_only_span_one_structures(tmp_path):
    inputs = replace(
        _toy_inputs(),
        configuration_matrix=[
            {
                "K": 4,
                "q": 1,
                "families": ["hypergraph", "simplicial", "cellular"],
            }
        ],
        seeds=[0],
        epochs=1,
    )
    bundle = run_recovery_experiment(
        inputs,
        output_dir=tmp_path / "q1",
        run_mode="smoke",
        implementation_commit="f" * 40,
        max_wall_seconds=30,
    )
    actual = {
        (row["family"], row["measurement"]): (
            row["recovered_count"],
            row["observable_count"],
            row["reference_count"],
        )
        for row in bundle["observations"]
        if row["epoch"] == 1
    }
    assert actual == {
        ("hypergraph", "support_available"): (1, 1, 4),
        ("simplicial", "support_available"): (0, 0, 1),
        ("cellular", "support_available"): (0, 0, 1),
        ("cellular", "actual_basis_recovery"): (0, 0, 1),
    }


def test_toy_counting_path_never_initializes_model_or_cuda(
    tmp_path, monkeypatch
):
    def forbidden(*_args, **_kwargs):
        raise AssertionError("model/CUDA/W&B path is forbidden")

    monkeypatch.setattr(torch.nn.Module, "__init__", forbidden)
    monkeypatch.setattr(torch.Tensor, "cuda", forbidden)
    monkeypatch.setattr(torch.cuda, "set_device", forbidden)
    monkeypatch.setitem(sys.modules, "wandb", None)
    _run_toy(tmp_path)


def test_pinned_train_mask_matches_topobench_first_random_fold():
    n = 19_793
    node_order = np.arange(n - 1, -1, -1, dtype=np.int64)
    adapter = RecoveryGraphAdapter(
        np.linspace(0, n, 33, dtype=np.int64),
        np.zeros(n + 1, dtype=np.int64),
        np.empty(0, dtype=np.int64),
        node_order,
    )
    original_mask = np.zeros(n, dtype=bool)
    permutation = np.random.RandomState(42).permutation(n)
    original_mask[permutation[: int(n * 0.7)]] = True
    permuted_mask = original_mask[node_order]
    assert runner.preflight_pinned_training_split(
        adapter, permuted_mask, runner.load_recovery_config()
    ) == int(n * 0.7)

    changed = permuted_mask.copy()
    train_node = int(np.flatnonzero(changed)[0])
    other_node = int(np.flatnonzero(~changed)[0])
    changed[train_node] = False
    changed[other_node] = True
    with pytest.raises(ValueError, match="random split"):
        runner.preflight_pinned_training_split(
            adapter, changed, runner.load_recovery_config()
        )


def test_preflight_identity_basis_check_uses_a_real_identity_order_batch(
    monkeypatch,
):
    inputs = _toy_inputs()
    original = runner.local_basis_ids_in_global_coordinates

    def reject_identity_batch(batch, *, max_length=9):
        if batch.global_nodes == tuple(range(4)):
            return set()
        return original(batch, max_length=max_length)

    monkeypatch.setattr(
        runner, "local_basis_ids_in_global_coordinates", reject_identity_batch
    )
    with pytest.raises(ValueError, match="same-order full-graph"):
        preflight_recovery_inputs(inputs)


def test_chorded_permuted_full_batch_has_support_three_actual_two(tmp_path):
    node_order = (1, 0, 2, 3, 4)
    local_edges = [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 4),
        (4, 0),
        (0, 2),
        (1, 3),
    ]
    directed = [
        edge
        for u, v in local_edges
        for edge in (
            (node_order[u], node_order[v]),
            (node_order[v], node_order[u]),
        )
    ]
    edges = np.asarray(directed, dtype=np.int64).T.copy()
    inverse = {node: local for local, node in enumerate(node_order)}
    adjacency = defaultdict(list)
    for source, target in directed:
        adjacency[inverse[source]].append(inverse[target])
    indptr = [0]
    indices = []
    for local in range(5):
        indices.extend(adjacency[local])
        indptr.append(len(indices))
    adapter = RecoveryGraphAdapter(
        np.arange(6, dtype=np.int64),
        np.asarray(indptr, dtype=np.int64),
        np.asarray(indices, dtype=np.int64),
        np.asarray(node_order, dtype=np.int64),
    )
    topology = hashlib.sha256()
    topology.update(np.asarray([5, len(directed)], dtype=np.int64).tobytes())
    topology.update(edges.tobytes())
    graph_hash = topology.hexdigest()
    edge_hash = _digest(edges)
    assignment_hash = (
        _digest(adapter.perm_to_global) + ":" + _digest(adapter.partptr)
    )
    sidecar = {
        "topology_sha256": graph_hash,
        "edge_order_sha256": edge_hash,
        "csr_order_sha256": adapter.csr_order_sha256(),
        "partition_assignment_sha256": assignment_hash,
        "num_nodes": 5,
        "num_parts": 5,
        "num_input_edges": len(directed),
        "basis_order_policy": "native_local_order",
    }
    train_mask = np.ones(5, dtype=bool)
    provenance = {
        "edge_order_sha256": edge_hash,
        "dataset": "toy",
        "num_nodes": 5,
        "num_parts": 5,
        "num_edges_directed": len(directed),
        "max_cell_length": 9,
        "partition_assignment_sha256": assignment_hash,
        "train_mask_sha256": hashlib.sha256(train_mask.tobytes()).hexdigest(),
        "active_cluster_count": 5,
    }
    inputs = RecoveryRunInputs(
        dataset="toy",
        edge_index=edges,
        adapters={5: adapter},
        partition_sidecars={5: sidecar},
        provenance_manifests={5: provenance},
        train_masks_perm={5: train_mask},
        configuration_matrix=[{"K": 5, "q": 5, "families": ["cellular"]}],
        seeds=[0],
        epochs=1,
        sample_every=1,
        loader_config="toy-chorded",
        split_config={"type": "fixed", "seed": 0},
        partition_source="regenerated",
    )
    bundle = run_recovery_experiment(
        inputs,
        output_dir=tmp_path / "chorded",
        run_mode="smoke",
        implementation_commit="f" * 40,
        max_wall_seconds=30,
    )
    counts = {
        row["measurement"]: (row["recovered_count"], row["reference_count"])
        for row in bundle["observations"]
        if row["epoch"] == 1
    }
    assert counts == {
        "support_available": (3, 3),
        "actual_basis_recovery": (2, 3),
    }
    assert (
        bundle["checks"]["native_full_batch_order"]["5"]["same_cycle_basis"]
        is False
    )
    assert (
        bundle["checks"]["same_order_full_graph_basis_identity_verified"]
        is True
    )


def test_smoke_bundle_cannot_be_loaded_as_publication(tmp_path):
    _run_toy(tmp_path)
    with pytest.raises(ValueError, match="smoke bundle"):
        load_recovery_bundle(tmp_path / "bundle")


def test_runner_rejects_existing_output_without_resume(tmp_path):
    output = tmp_path / "existing"
    output.mkdir()
    with pytest.raises(FileExistsError, match="already exists"):
        run_recovery_experiment(
            _toy_inputs(),
            output_dir=output,
            run_mode="smoke",
            implementation_commit="f" * 40,
            max_wall_seconds=30,
        )


def test_cli_requires_real_input_paths_and_does_not_substitute_cora(
    tmp_path, capsys
):
    output = tmp_path / "no-input-bundle"
    with pytest.raises(SystemExit) as exc:
        main(["--preflight-only", "--output-dir", str(output)])
    assert exc.value.code == 2
    assert "--processed-edge-index" in capsys.readouterr().err
    assert not output.exists()


def test_cli_help_describes_modes_and_explicit_inputs(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    assert exc.value.code == 0
    help_text = capsys.readouterr().out
    for token in (
        "--preflight-only",
        "--smoke",
        "--processed-edge-index",
        "--partition-k32-dir",
        "--partition-k32-sidecar",
        "--output-dir",
        "cooperative",
    ):
        assert token in help_text


def test_default_smoke_selection_preserves_pinned_primary_schedule():
    config = runner.load_recovery_config()
    schedule, q, epochs = runner._select_smoke_schedule(
        config, K=32, q=4, epochs=2
    )
    assert schedule == config["primary"]
    assert (q, epochs) == (4, 2)
    assert config["seeds"][0] == 0


@pytest.mark.parametrize("q", [2, 16])
def test_smoke_selection_accepts_pinned_k32_q_values(q):
    config = runner.load_recovery_config()
    schedule, selected_q, epochs = runner._select_smoke_schedule(
        config, K=32, q=q, epochs=3
    )
    assert schedule["K"] == 32
    assert schedule["families"] == ["hypergraph", "simplicial", "cellular"]
    assert (selected_q, epochs) == (q, 3)


def test_smoke_selection_accepts_pinned_k64_cellular_schedule():
    config = runner.load_recovery_config()
    schedule, q, epochs = runner._select_smoke_schedule(
        config, K=64, q=8, epochs=5
    )
    assert schedule == config["additional"][0]
    assert (q, epochs) == (8, 5)


@pytest.mark.parametrize(
    ("K", "q", "epochs", "message"),
    [
        (16, 4, 2, "pinned"),
        (32, 3, 2, "pinned"),
        (64, 4, 2, "pinned"),
        (32, 4, 0, "epochs"),
        (32, 4, 201, "epochs"),
    ],
)
def test_smoke_selection_rejects_unpinned_values(K, q, epochs, message):
    config = runner.load_recovery_config()
    with pytest.raises(ValueError, match=message):
        runner._select_smoke_schedule(config, K=K, q=q, epochs=epochs)


def test_cli_smoke_k64_requires_only_k64_partition_paths(tmp_path, capsys):
    with pytest.raises(SystemExit) as exc:
        main(
            [
                "--smoke",
                "--smoke-k",
                "64",
                "--smoke-q",
                "8",
                "--output-dir",
                str(tmp_path / "smoke-k64"),
            ]
        )
    assert exc.value.code == 2
    error = capsys.readouterr().err.split("error:", 1)[-1]
    assert "--partition-k64-dir" in error
    assert "--partition-k32-dir" not in error


def _sparse_cora_full_cli_inputs(tmp_path):
    """Write sparse, synthetic topology with the pinned node/partition counts."""
    node_count = 19_793
    edge_index = np.asarray(
        [[0, 1, 0, 2, 1, 2], [1, 0, 2, 0, 2, 1]], dtype=np.int64
    )
    edge_path = tmp_path / "processed_edge_index.npy"
    np.save(edge_path, edge_index)
    training_mask = np.zeros(node_count, dtype=bool)
    selected = np.random.RandomState(42).permutation(node_count)
    training_mask[selected[: int(node_count * 0.7)]] = True
    graph_digest = hashlib.sha256()
    graph_digest.update(
        np.asarray([node_count, edge_index.shape[1]], dtype=np.int64).tobytes()
    )
    graph_digest.update(edge_index.tobytes())
    indptr = np.full(node_count + 1, 6, dtype=np.int64)
    indptr[:4] = [0, 2, 4, 6]
    indices = np.asarray([1, 2, 0, 2, 0, 1], dtype=np.int64)
    permutation = np.arange(node_count, dtype=np.int64)
    partitions = {}
    for K in (32, 64):
        partptr = np.asarray(
            [node_count * part // K for part in range(K + 1)],
            dtype=np.int64,
        )
        adapter = RecoveryGraphAdapter(partptr, indptr, indices, permutation)
        directory = tmp_path / f"k{K}"
        native = directory / "native_csr"
        native.mkdir(parents=True)
        for name, array in (
            ("partptr", partptr),
            ("indptr", indptr),
            ("indices", indices),
            ("perm_to_global", permutation),
        ):
            np.save(native / f"{name}.npy", array)
        mask_path = directory / "train_mask_perm.npy"
        np.save(mask_path, training_mask)
        assignment = _digest(permutation) + ":" + _digest(partptr)
        sidecar = {
            "topology_sha256": graph_digest.hexdigest(),
            "edge_order_sha256": _digest(edge_index),
            "csr_order_sha256": adapter.csr_order_sha256(),
            "partition_assignment_sha256": assignment,
            "num_nodes": node_count,
            "num_parts": K,
            "num_input_edges": edge_index.shape[1],
            "basis_order_policy": "native_local_order",
        }
        provenance = {
            "edge_order_sha256": _digest(edge_index),
            "dataset": "cora_full",
            "num_nodes": node_count,
            "num_parts": K,
            "num_edges_directed": edge_index.shape[1],
            "max_cell_length": 9,
            "partition_assignment_sha256": assignment,
            "train_mask_sha256": hashlib.sha256(
                training_mask.tobytes()
            ).hexdigest(),
            "active_cluster_count": K,
        }
        sidecar_path = directory / "metadata.json"
        provenance_path = directory / "provenance.json"
        sidecar_path.write_text(json.dumps(sidecar), encoding="utf-8")
        provenance_path.write_text(json.dumps(provenance), encoding="utf-8")
        partitions[K] = [
            f"--partition-k{K}-dir",
            str(native),
            f"--partition-k{K}-sidecar",
            str(sidecar_path),
            f"--partition-k{K}-provenance",
            str(provenance_path),
            f"--partition-k{K}-train-mask",
            str(mask_path),
        ]

    model_settings = {
        "GCN": (32, 8),
        "EDHNN": (32, 4),
        "UniGNN": (32, 4),
        "CWN": (32, 4),
        "TopoTune": (64, 8),
        "SCN": (32, 16),
        "SCCNN": (32, 8),
    }
    manuscript = [r"\label{tab:optuna_hparams}"]
    for model, (K, q) in model_settings.items():
        manuscript.extend(
            [
                rf"\multicolumn{{9}}{{c}}{{\textbf{{{model}}}}}",
                r"\multirow{2}{*}{Cora Full} & F & x",
                f" & P & a & b & c & d & ${K}$ & ${q}$ & x",
            ]
        )
    manuscript.append(r"\end{longtable}")
    manuscript_path = tmp_path / "main.tex"
    manuscript_path.write_text("\n".join(manuscript), encoding="utf-8")
    common = [
        "--processed-edge-index",
        str(edge_path),
        "--manuscript",
        str(manuscript_path),
        "--partition-source",
        "regenerated",
    ]
    return common, partitions


def test_cli_preflight_saved_snapshot_handoff_to_k64_cellular_pilot(tmp_path):
    common, partitions = _sparse_cora_full_cli_inputs(tmp_path)
    preflight_dir = tmp_path / "preflight"
    assert (
        main(
            [
                "--preflight-only",
                *common,
                *partitions[32],
                *partitions[64],
                "--output-dir",
                str(preflight_dir),
            ]
        )
        == 0
    )
    snapshot_path = preflight_dir / "reference_snapshot.json"
    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    assert set(snapshot) == {"hypergraph", "simplicial", "cellular"}

    pilot_dir = tmp_path / "k64-cellular-pilot"
    assert (
        main(
            [
                "--smoke",
                "--smoke-k",
                "64",
                "--smoke-q",
                "8",
                "--smoke-epochs",
                "1",
                "--max-wall-seconds",
                "300",
                *common,
                *partitions[64],
                "--reference-snapshot",
                str(snapshot_path),
                "--output-dir",
                str(pilot_dir),
            ]
        )
        == 0
    )
    bundle = load_recovery_bundle(pilot_dir, publication_only=False)
    assert bundle["manifest"]["configuration_matrix"] == [
        {"K": 64, "q": 8, "families": ["cellular"]}
    ]
    assert bundle["manifest"]["reference_hashes"] == {
        "cellular": snapshot["cellular"]["sha256"]
    }
    with pytest.raises(ValueError, match="smoke bundle"):
        load_recovery_bundle(pilot_dir, publication_only=True)


@pytest.mark.parametrize(
    "option,value",
    [
        ("--smoke-k", "32"),
        ("--smoke-q", "4"),
        ("--smoke-epochs", "2"),
    ],
)
def test_cli_rejects_diagnostic_overrides_without_smoke(
    tmp_path, capsys, option, value
):
    with pytest.raises(SystemExit) as exc:
        main(["--output-dir", str(tmp_path / "not-smoke"), option, value])
    assert exc.value.code == 2
    assert "requires --smoke" in capsys.readouterr().err
