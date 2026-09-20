"""Tests for the support-only, multi-dataset recovery counter."""

import pytest

from scripts.structural_coverage.recovery_core import (
    ReferenceStructure,
    generate_epoch_schedules,
    support_available_ids,
)
from scripts.structural_coverage.support_recovery_multidataset import (
    analyze_recovery,
    build_signature_groups,
    count_support_by_epoch,
    labels_from_partition,
    validate_ordered_graph,
)


def test_support_counter_matches_explicit_batch_node_oracle():
    references = [
        ReferenceStructure(("one", 0), frozenset({0})),
        ReferenceStructure(("two", 0), frozenset({0, 1})),
        ReferenceStructure(("two", 1), frozenset({0, 1})),
        ReferenceStructure(("three", 0), frozenset({0, 1, 2})),
    ]
    labels = [0, 1, 2, 3]
    schedules = generate_epoch_schedules(4, 2, 6, 7)
    groups, histogram = build_signature_groups(references, labels, q=2)

    observed = count_support_by_epoch(
        groups, K=4, q=2, schedules=schedules
    )
    seen = set()
    oracle = [0]
    for epoch_groups in schedules:
        for batch in epoch_groups:
            seen.update(support_available_ids(references, batch))
        oracle.append(len(seen))

    assert observed == oracle
    assert histogram == {1: 1, 2: 2, 3: 1}
    assert observed[1] >= 1
    assert observed[-1] <= 3  # The span-three reference is unobservable.


def test_support_counter_rejects_invalid_node_labels():
    references = [ReferenceStructure(("one", 0), frozenset({4}))]
    with pytest.raises(ValueError, match="node outside partition"):
        build_signature_groups(references, [0, 1, 2, 3], q=2)


def test_support_counter_does_not_mutate_group_state_between_seeds():
    references = [ReferenceStructure(("two", 0), frozenset({0, 1}))]
    groups, _ = build_signature_groups(references, [0, 1, 2, 3], q=2)
    schedules = generate_epoch_schedules(4, 2, 4, 3)

    first = count_support_by_epoch(groups, K=4, q=2, schedules=schedules)
    second = count_support_by_epoch(groups, K=4, q=2, schedules=schedules)

    assert first == second
    assert first[0] == 0


def test_partition_labels_preserve_original_node_coordinates():
    labels = labels_from_partition(
        partptr=[0, 2, 4],
        perm_to_global=[2, 0, 3, 1],
        train_mask_perm=[True, False, True, False],
    )
    assert labels.tolist() == [0, 1, 0, 1]


def test_partition_rejects_inactive_cluster_and_duplicate_node():
    with pytest.raises(ValueError, match="inactive cluster"):
        labels_from_partition(
            partptr=[0, 2, 4],
            perm_to_global=[2, 0, 3, 1],
            train_mask_perm=[True, False, False, False],
        )
    with pytest.raises(ValueError, match="permutation"):
        labels_from_partition(
            partptr=[0, 2, 4],
            perm_to_global=[2, 0, 3, 3],
            train_mask_perm=[True, False, True, False],
        )


def test_graph_manifest_rejects_ordered_edge_mismatch():
    import hashlib
    import numpy as np

    edge_index = np.asarray([[0, 1], [1, 0]], dtype=np.int64)
    manifest = {
        "num_nodes": 2,
        "num_edges_directed": 2,
        "edge_order_sha256": hashlib.sha256(edge_index.tobytes()).hexdigest(),
    }
    validate_ordered_graph(edge_index, num_nodes=2, manifest=manifest)
    with pytest.raises(ValueError, match="ordered edge hash"):
        validate_ordered_graph(edge_index[:, ::-1], num_nodes=2, manifest=manifest)


def test_analysis_uses_fixed_reference_denominator_and_exact_theory():
    references = {
        "cellular": [
            ReferenceStructure(("a",), frozenset({0})),
            ReferenceStructure(("b",), frozenset({0, 1})),
            ReferenceStructure(("c",), frozenset({0, 1, 2})),
        ]
    }
    result = analyze_recovery(
        references,
        labels=[0, 1, 2, 3],
        K=4,
        q=2,
        seeds=[0, 1],
        epochs=3,
    )
    family = result["families"]["cellular"]

    assert family["reference_count"] == 3
    assert family["observable_count"] == 2
    assert family["span_histogram"] == {1: 1, 2: 1, 3: 1}
    assert family["expected_coverage"][0] == 0
    assert family["expected_coverage"][1] == pytest.approx(4 / 9)
    assert len(family["counts_by_seed"]) == 2
    assert all(len(counts) == 4 for counts in family["counts_by_seed"].values())


def test_q_sweep_matches_independent_seeded_recovery_and_endpoints():
    from scripts.structural_coverage.support_recovery_multidataset import analyze_q_sweep

    references = {
        "cellular": [
            ReferenceStructure(("one",), frozenset({0})),
            ReferenceStructure(("pair",), frozenset({0, 1})),
            ReferenceStructure(("wide",), frozenset({0, 1, 2})),
        ]
    }
    labels = [0, 1, 2, 3]
    sweep = analyze_q_sweep(
        references, labels=labels, K=4, q_values=[1, 2, 4],
        seeds=[0, 1], epochs=3,
    )

    for q in (1, 2, 4):
        independent = analyze_recovery(
            references, labels=labels, K=4, q=q, seeds=[0, 1], epochs=3,
        )
        assert sweep[q]["families"]["cellular"] == independent["families"]["cellular"]
    assert sweep[1]["families"]["cellular"]["counts_by_seed"][0] == [0, 1, 1, 1]
    assert sweep[4]["families"]["cellular"]["counts_by_seed"][0] == [0, 3, 3, 3]


@pytest.mark.parametrize("q_values", [[1, 3, 4], [1, 2, 2], [2, 1]])
def test_q_sweep_rejects_invalid_grid(q_values):
    from scripts.structural_coverage.support_recovery_multidataset import analyze_q_sweep

    references = {"cellular": [ReferenceStructure(("one",), frozenset({0}))]}
    with pytest.raises(ValueError, match="q grid"):
        analyze_q_sweep(
            references, labels=[0, 1, 2, 3], K=4,
            q_values=q_values, seeds=[0], epochs=3,
        )


def test_q_grids_are_valid_and_include_selected_training_size():
    from scripts.structural_coverage.run_support_recovery_multidataset import (
        DATASETS,
        Q_GRIDS,
    )

    for dataset, grid in Q_GRIDS.items():
        K = DATASETS[dataset]["K"]
        assert grid[0] == 1
        assert grid[-1] == K
        assert DATASETS[dataset]["q"] in grid
        assert grid == sorted(set(grid))
        assert all(K % q == 0 for q in grid)


def test_q_export_and_plot_match_seed_level_epoch_endpoint(tmp_path):
    import csv
    import matplotlib.pyplot as plt

    from scripts.structural_coverage.run_support_recovery_multidataset import (
        FAMILIES,
        export_q_sweep,
        make_q_plot,
    )
    from scripts.structural_coverage.support_recovery_multidataset import analyze_q_sweep

    references = {
        family: [
            ReferenceStructure((family, "one"), frozenset({0})),
            ReferenceStructure((family, "pair"), frozenset({0, 1})),
            ReferenceStructure((family, "wide"), frozenset({0, 1, 2})),
        ]
        for family in FAMILIES
    }
    sweep = analyze_q_sweep(
        references, labels=[0, 1, 2, 3], K=4,
        q_values=[1, 2, 4], seeds=[0, 1], epochs=3,
    )
    export_q_sweep(
        sweep, dataset="cora_full", output_dir=tmp_path,
        input_manifest={"dataset": "cora_full", "partition_source": "toy"},
    )
    with (tmp_path / "q_recovery_source_data.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 9
    q2_cell = next(row for row in rows if row["family"] == "cellular"
                   and int(row["q"]) == 2)
    source = sweep[2]["families"]["cellular"]
    assert int(q2_cell["reference_count"]) == 3
    assert int(q2_cell["q_observable_count"]) == 2
    assert float(q2_cell["empirical_mean"]) == source["coverage_mean"][-1]
    assert float(q2_cell["empirical_sample_sd"]) == source["coverage_sample_sd"][-1]
    assert [int(q2_cell[f"seed_{seed}_count"]) for seed in (0, 1)] == [
        source["counts_by_seed"][seed][-1] for seed in (0, 1)
    ]

    figure = make_q_plot(sweep, dataset="cora_full")
    try:
        ax = figure.axes[0]
        assert ax.get_xlabel() == "Clusters per mini-batch, $q$"
        assert list(ax.lines[1].get_xdata()) == [1, 2, 4]
        assert list(ax.lines[1].get_ydata()) == [
            sweep[q]["families"]["cellular"]["coverage_mean"][-1]
            for q in (1, 2, 4)
        ]
    finally:
        plt.close(figure)


def test_q_plot_uses_dataset_specific_percentage_axes():
    import matplotlib.pyplot as plt

    from scripts.structural_coverage.run_support_recovery_multidataset import (
        FAMILIES,
        make_q_plot,
    )
    from scripts.structural_coverage.support_recovery_multidataset import analyze_q_sweep

    references = {
        family: [ReferenceStructure((family, 0), frozenset({0}))]
        for family in FAMILIES
    }
    sweep = analyze_q_sweep(
        references, labels=[0, 1, 2, 3], K=4,
        q_values=[1, 2, 4], seeds=[0, 1], epochs=3,
    )

    amazon = make_q_plot(sweep, dataset="amazon_ratings")
    cora = make_q_plot(sweep, dataset="cora_full")
    questions = make_q_plot(sweep, dataset="questions")
    try:
        assert amazon.axes[0].get_ylim() == pytest.approx((0.75, 1.025))
        assert list(amazon.axes[0].get_yticks()) == pytest.approx(
            [0.75, 0.80, 0.85, 0.90, 0.95, 1.0]
        )
        assert cora.axes[0].get_ylim() == pytest.approx((0.50, 1.025))
        assert list(cora.axes[0].get_yticks()) == pytest.approx(
            [0.50, 0.60, 0.70, 0.80, 0.90, 1.0]
        )
        assert questions.axes[0].get_ylim() == pytest.approx((-0.015, 1.025))
    finally:
        plt.close(amazon)
        plt.close(cora)
        plt.close(questions)


def test_q_runner_writes_all_three_family_curves_from_one_graph(tmp_path, monkeypatch):
    import csv
    import numpy as np

    from scripts.structural_coverage import run_support_recovery_multidataset as runner

    edges = np.asarray([
        [0, 1, 1, 2, 2, 0],
        [1, 0, 2, 1, 0, 2],
    ], dtype=np.int64)
    monkeypatch.setitem(runner.DATASETS, "cora_full", {
        "K": 4, "q": 2, "nodes": 4, "counts": (4, 1, 1),
    })
    monkeypatch.setitem(runner.Q_GRIDS, "cora_full", [1, 2, 4])
    monkeypatch.setattr(runner, "load_graph", lambda *args, **kwargs: (edges, 4))
    monkeypatch.setattr(
        runner, "load_saved_partition",
        lambda *args, **kwargs: (
            np.asarray([0, 1, 2, 3]), {"dataset": "cora_full"}
        ),
    )

    output = runner.run_q_sweep_one(
        "cora_full", source_root=tmp_path, partition_root=tmp_path,
        coauthor_root=tmp_path, cora_edge_index=tmp_path / "unused.npy",
        output_root=tmp_path, epochs=3, seeds=[0, 1],
    )

    assert output["q_values"] == [1, 2, 4]
    assert output["reference_counts"] == {
        "hypergraph": 4, "cellular": 1, "simplicial": 1,
    }
    with (tmp_path / "cora_full" / "q_recovery_source_data.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 9
    assert (tmp_path / "cora_full" / "q_recovery.png").exists()
