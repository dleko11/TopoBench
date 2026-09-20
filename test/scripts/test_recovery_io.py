"""Configuration contract for the Cora Full structural-recovery study."""

import csv
import json
from copy import deepcopy
from pathlib import Path

import pytest

from scripts.structural_coverage import recovery_io
from scripts.structural_coverage.recovery_io import (
    CANONICAL_FAMILY_DEFINITIONS,
    load_recovery_bundle,
    load_recovery_config,
    validate_graph_identity,
    validate_recovery_bundle,
    validate_recovery_config,
    write_recovery_bundle,
)


@pytest.fixture
def config():
    return load_recovery_config()


def test_primary_configuration_is_cora_full_not_cora(config):
    assert config["schema_version"] == 2
    assert config["dataset"] == "cora_full"
    assert config["expected_num_nodes"] == 19_793
    assert config["primary"] == {
        "K": 32,
        "q_values": [2, 4, 8, 16],
        "families": ["hypergraph", "simplicial", "cellular"],
    }
    assert config["cycle_max_length"] == 9
    assert config["seeds"] == list(range(10))
    assert config["epochs"] == 200
    assert config["entropy_normalization"] == "all_reference"


def test_additional_configuration_matches_selected_cell_topotune_setting(
    config,
):
    assert config["additional"] == [
        {"K": 64, "q_values": [8], "families": ["cellular"]}
    ]


def test_configuration_records_final_training_split_policy(config):
    assert config["cluster_policy"] == "training_active_require_all"
    assert config["split_seed"] == 0
    assert config["split_type"] == "random"
    assert config["train_prop"] == 0.7
    assert config["learning_setting"] == "transductive"
    assert config["hypergraph_identity"] == "center_indexed"
    assert config["sample_every"] == 5


def test_valid_configuration_passes_validation(config):
    validate_recovery_config(config)


def test_cora_full_metadata_rejects_cora_graph(config):
    with pytest.raises(ValueError, match="node count"):
        validate_graph_identity(config, dataset="cora_full", num_nodes=2_708)


def test_graph_identity_rejects_wrong_dataset(config):
    with pytest.raises(ValueError, match="dataset"):
        validate_graph_identity(config, dataset="cora", num_nodes=19_793)


def test_graph_identity_accepts_cora_full(config):
    validate_graph_identity(config, dataset="cora_full", num_nodes=19_793)


@pytest.mark.parametrize("bad_q", [0, 33, -1])
def test_invalid_primary_q_is_rejected(config, bad_q):
    bad_config = deepcopy(config)
    bad_config["primary"]["q_values"] = [bad_q]
    with pytest.raises(ValueError, match="q"):
        validate_recovery_config(bad_config)


def test_nondivisible_k_q_is_rejected(config):
    bad_config = deepcopy(config)
    bad_config["primary"]["q_values"] = [3]
    with pytest.raises(ValueError, match="divisible"):
        validate_recovery_config(bad_config)


def test_duplicate_seeds_are_rejected(config):
    bad_config = deepcopy(config)
    bad_config["seeds"] = [0, 0]
    with pytest.raises(ValueError, match="seed"):
        validate_recovery_config(bad_config)


def test_empty_seeds_are_rejected(config):
    bad_config = deepcopy(config)
    bad_config["seeds"] = []
    with pytest.raises(ValueError, match="seed"):
        validate_recovery_config(bad_config)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("K", 64),
        ("q_values", [2, 4]),
        ("families", ["hypergraph"]),
    ],
)
def test_primary_schedule_drift_is_rejected(config, field, value):
    bad_config = deepcopy(config)
    bad_config["primary"][field] = value
    with pytest.raises(ValueError, match="primary"):
        validate_recovery_config(bad_config)


@pytest.mark.parametrize(
    "additional",
    [[], [{"K": 32, "q_values": [8], "families": ["cellular"]}]],
)
def test_additional_schedule_drift_is_rejected(config, additional):
    bad_config = deepcopy(config)
    bad_config["additional"] = additional
    with pytest.raises(ValueError, match="additional"):
        validate_recovery_config(bad_config)


def test_shortened_reshuffling_seed_list_is_rejected(config):
    bad_config = deepcopy(config)
    bad_config["seeds"] = [0]
    with pytest.raises(ValueError, match="seeds"):
        validate_recovery_config(bad_config)


@pytest.mark.parametrize("field", ["epochs", "sample_every"])
def test_shortened_experiment_duration_is_rejected(config, field):
    bad_config = deepcopy(config)
    bad_config[field] = 1
    with pytest.raises(ValueError, match=field):
        validate_recovery_config(bad_config)


def test_unknown_top_level_protocol_key_is_rejected(config):
    bad_config = deepcopy(config)
    bad_config["cycle_backend"] = "all_simple_cycles"
    with pytest.raises(ValueError, match="unknown"):
        validate_recovery_config(bad_config)


@pytest.mark.parametrize("schedule_name", ["primary", "additional"])
def test_unknown_schedule_protocol_key_is_rejected(config, schedule_name):
    bad_config = deepcopy(config)
    schedule = (
        bad_config["primary"]
        if schedule_name == "primary"
        else bad_config["additional"][0]
    )
    schedule["cycle_backend"] = "all_simple_cycles"
    with pytest.raises(ValueError, match="unknown"):
        validate_recovery_config(bad_config)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("split_seed", False),
        ("split_seed", 0.0),
        ("schema_version", 2.0),
        ("expected_num_nodes", 19_793.0),
        ("cycle_max_length", 9.0),
    ],
)
def test_scalar_type_mismatch_is_rejected(config, field, value):
    bad_config = deepcopy(config)
    bad_config[field] = value
    with pytest.raises(ValueError, match=field):
        validate_recovery_config(bad_config)


def test_malformed_family_is_rejected_as_configuration_error(config):
    bad_config = deepcopy(config)
    bad_config["primary"]["families"] = [["hypergraph"]]
    with pytest.raises(ValueError, match="families"):
        validate_recovery_config(bad_config)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("cycle_max_length", 10),
        ("hypergraph_identity", "membership_set"),
        ("entropy_normalization", "observable_reference"),
        ("cluster_policy", "training_active_filter"),
        ("split_type", "fixed"),
        ("train_prop", 0.5),
        ("learning_setting", "inductive"),
    ],
)
def test_protocol_changing_values_are_rejected(config, field, value):
    bad_config = deepcopy(config)
    bad_config[field] = value
    with pytest.raises(ValueError, match=field):
        validate_recovery_config(bad_config)


@pytest.fixture
def tiny_bundle():
    """Independent K=4, q=2 counts, not derived by the recovery helpers."""
    graph_hash = "a" * 64
    order_hash = "b" * 64
    partition_hash = "c" * 64 + ":" + "e" * 64
    reference_hash = "d" * 64
    manifest = {
        "schema_version": 2,
        "dataset": "toy",
        "loader_config": "toy-fixed",
        "num_nodes": 4,
        "canonical_edge_count": 3,
        "graph_hash": graph_hash,
        "graph_order_hash": order_hash,
        "partition_hashes": {"4": partition_hash},
        "split_config": {"type": "fixed", "seed": 0},
        "active_cluster_counts": {"4": 4},
        "family_definitions": {
            "cellular": CANONICAL_FAMILY_DEFINITIONS["cellular"]
        },
        "reference_hashes": {"cellular": reference_hash},
        "cycle_max_length": 9,
        "hypergraph_identity": "center_indexed",
        "basis_order_policy": "native_local_order",
        "configuration_matrix": [{"K": 4, "q": 2, "families": ["cellular"]}],
        "seeds": [0, 1],
        "epochs": 2,
        "sample_every": 1,
        "entropy_normalization": "all_reference",
        "implementation_commit": "f" * 40,
        "local_patch_hash_if_dirty": None,
        "package_versions": {"networkx": "3.4"},
        "partition_source": "saved_experiment",
        "run_mode": "smoke",
        "completed": False,
    }
    metadata = {
        "4": {
            "topology_sha256": graph_hash,
            "edge_order_sha256": order_hash,
            "csr_order_sha256": "e" * 64 + ":" + "f" * 64,
            "partition_assignment_sha256": partition_hash,
            "num_nodes": 4,
            "num_parts": 4,
            "num_input_edges": 6,
            "basis_order_policy": "native_local_order",
        }
    }
    observations = []
    for seed, counts in ((0, (0, 1, 2)), (1, (0, 2, 2))):
        for epoch, count in enumerate(counts):
            observations.append(
                {
                    "dataset": "toy",
                    "K": 4,
                    "q": 2,
                    "seed": seed,
                    "epoch": epoch,
                    "family": "cellular",
                    "measurement": "support_available",
                    "recovered_count": count,
                    "reference_count": 3,
                    "observable_count": 2,
                    "coverage": count / 3,
                }
            )
    for seed, counts in ((0, (0, 0, 1)), (1, (0, 1, 1))):
        for epoch, count in enumerate(counts):
            observations.append(
                {
                    "dataset": "toy",
                    "K": 4,
                    "q": 2,
                    "seed": seed,
                    "epoch": epoch,
                    "family": "cellular",
                    "measurement": "actual_basis_recovery",
                    "recovered_count": count,
                    "reference_count": 3,
                    "observable_count": 2,
                    "coverage": count / 3,
                }
            )
    return {
        "manifest": manifest,
        "partition_metadata": metadata,
        "reference_summary": [
            {
                "dataset": "toy",
                "family": "cellular",
                "reference_count": 3,
                "reference_hash": reference_hash,
            }
        ],
        "span_histogram": [
            {
                "dataset": "toy",
                "K": 4,
                "family": "cellular",
                "span": span,
                "count": 1,
                "reference_hash": reference_hash,
            }
            for span in (1, 2, 3)
        ],
        "theory": [
            {
                "dataset": "toy",
                "K": 4,
                "q": 2,
                "epoch": epoch,
                "family": "cellular",
                "reference_count": 3,
                "observable_count": 2,
                "expected_coverage": expected,
                "entropy_nats_per_reference": entropy,
            }
            for epoch, expected, entropy in (
                (0, 0.0, 0.0),
                (1, 4 / 9, 0.21217138943160427),
                (2, 14 / 27, 0.2289871921991078),
            )
        ],
        "observations": observations,
        "summary": [
            {
                "dataset": "toy",
                "K": 4,
                "q": 2,
                "epoch": epoch,
                "family": "cellular",
                "measurement": "support_available",
                "n_repetitions": 2,
                "coverage_mean": mean,
                "coverage_sample_sd": sd,
                "reference_count": 3,
                "observable_fraction": 2 / 3,
            }
            for epoch, mean, sd in (
                (0, 0.0, 0.0),
                (1, 0.5, 0.23570226039551584),
                (2, 2 / 3, 0.0),
            )
        ]
        + [
            {
                "dataset": "toy",
                "K": 4,
                "q": 2,
                "epoch": epoch,
                "family": "cellular",
                "measurement": "actual_basis_recovery",
                "n_repetitions": 2,
                "coverage_mean": mean,
                "coverage_sample_sd": sd,
                "reference_count": 3,
                "observable_fraction": 2 / 3,
            }
            for epoch, mean, sd in (
                (0, 0.0, 0.0),
                (1, 1 / 6, 0.23570226039551584),
                (2, 1 / 3, 0.0),
            )
        ],
        "checks": {
            "passed": True,
            "graph_partition_consistency_verified": True,
            "cellular_set_inclusion_verified": True,
            "independent_oracle": "K4_q2_three_spans",
        },
    }


def test_tiny_bundle_round_trip_preserves_integer_counts_and_hashes(
    tmp_path, tiny_bundle
):
    destination = tmp_path / "new-bundle"
    write_recovery_bundle(destination, **tiny_bundle)
    loaded = load_recovery_bundle(destination, publication_only=False)
    assert loaded["manifest"]["completed"] is True
    assert loaded["manifest"]["reference_hashes"] == {"cellular": "d" * 64}
    assert [row["recovered_count"] for row in loaded["observations"]] == [
        0,
        1,
        2,
        0,
        2,
        2,
        0,
        0,
        1,
        0,
        1,
        1,
    ]
    assert all(
        type(row["recovered_count"]) is int for row in loaded["observations"]
    )
    assert loaded["partition_metadata"]["4"]["csr_order_sha256"] == (
        "e" * 64 + ":" + "f" * 64
    )
    assert (destination / "checks.json").exists()


@pytest.mark.parametrize(
    ("field", "bad_value", "message"),
    [
        ("recovered_count", 4, "exceeds"),
        ("recovered_count", 1.5, "integer"),
        ("reference_count", 4, "denominator"),
        ("coverage", 0.9, "coverage"),
    ],
)
def test_bundle_rejects_invalid_observation_counts(
    tmp_path, tiny_bundle, field, bad_value, message
):
    tiny_bundle["observations"][2][field] = bad_value
    with pytest.raises(ValueError, match=message):
        write_recovery_bundle(tmp_path / "invalid", **tiny_bundle)


def test_bundle_rejects_decreasing_cumulative_count(tmp_path, tiny_bundle):
    tiny_bundle["observations"][2]["recovered_count"] = 0
    tiny_bundle["observations"][2]["coverage"] = 0.0
    with pytest.raises(ValueError, match="decreasing"):
        write_recovery_bundle(tmp_path / "invalid", **tiny_bundle)


def test_bundle_rejects_duplicate_observation_key(tmp_path, tiny_bundle):
    tiny_bundle["observations"].append(
        deepcopy(tiny_bundle["observations"][0])
    )
    with pytest.raises(ValueError, match="duplicate"):
        write_recovery_bundle(tmp_path / "invalid", **tiny_bundle)


@pytest.mark.parametrize("missing_index", [2, 3])
def test_bundle_rejects_missing_epoch_or_seed(
    tmp_path, tiny_bundle, missing_index
):
    tiny_bundle["observations"].pop(missing_index)
    with pytest.raises(ValueError, match="missing"):
        write_recovery_bundle(tmp_path / "invalid", **tiny_bundle)


def test_sample_every_does_not_drop_simulated_epoch_rows(
    tmp_path, tiny_bundle
):
    tiny_bundle["manifest"]["sample_every"] = 2
    for name in ("observations", "theory", "summary"):
        tiny_bundle[name] = [
            row for row in tiny_bundle[name] if row["epoch"] != 1
        ]
    with pytest.raises(ValueError, match="missing"):
        write_recovery_bundle(tmp_path / "invalid", **tiny_bundle)


def test_cellular_bundle_requires_both_measurements(tmp_path, tiny_bundle):
    for name in ("observations", "summary"):
        tiny_bundle[name] = [
            row
            for row in tiny_bundle[name]
            if row["measurement"] != "actual_basis_recovery"
        ]
    with pytest.raises(ValueError, match="measurement"):
        write_recovery_bundle(tmp_path / "invalid", **tiny_bundle)


def test_cellular_actual_count_cannot_exceed_available_support(
    tmp_path, tiny_bundle
):
    support = tiny_bundle["observations"][1]
    actual = tiny_bundle["observations"][7]
    assert support["seed"] == actual["seed"] == 0
    assert support["epoch"] == actual["epoch"] == 1
    actual["recovered_count"] = 2
    actual["coverage"] = 2 / 3
    tiny_bundle["observations"][8]["recovered_count"] = 2
    tiny_bundle["observations"][8]["coverage"] = 2 / 3
    with pytest.raises(ValueError, match="actual basis.*support"):
        write_recovery_bundle(tmp_path / "invalid", **tiny_bundle)


@pytest.mark.parametrize(
    ("target", "field", "message"),
    [
        ("partition_metadata", "topology_sha256", "graph"),
        ("partition_metadata", "edge_order_sha256", "order"),
        ("partition_metadata", "partition_assignment_sha256", "partition"),
        ("partition_metadata", "basis_order_policy", "order-policy"),
        ("reference_summary", "reference_hash", "reference"),
    ],
)
def test_bundle_rejects_inconsistent_provenance(
    tmp_path, tiny_bundle, target, field, message
):
    record = (
        tiny_bundle[target]["4"]
        if target == "partition_metadata"
        else tiny_bundle[target][0]
    )
    record[field] = "wrong"
    with pytest.raises(ValueError, match=message):
        write_recovery_bundle(tmp_path / "invalid", **tiny_bundle)


def test_matching_but_malformed_partition_hashes_are_rejected(
    tmp_path, tiny_bundle
):
    tiny_bundle["manifest"]["partition_hashes"]["4"] = "not-a-hash"
    tiny_bundle["partition_metadata"]["4"]["partition_assignment_sha256"] = (
        "not-a-hash"
    )
    with pytest.raises(ValueError, match="partition.*SHA-256"):
        write_recovery_bundle(tmp_path / "invalid", **tiny_bundle)


def test_bundle_rejects_observable_entropy_label(tmp_path, tiny_bundle):
    tiny_bundle["manifest"]["entropy_normalization"] = "observable_reference"
    with pytest.raises(ValueError, match="entropy_normalization"):
        write_recovery_bundle(tmp_path / "invalid", **tiny_bundle)


@pytest.mark.parametrize(
    ("epoch", "field", "bad_value"),
    [
        (0, "expected_coverage", 0.1),
        (1, "expected_coverage", 1 / 2),
        (2, "entropy_nats_per_reference", 0.0),
    ],
)
def test_theory_rows_must_match_histogram_equations(
    tmp_path, tiny_bundle, epoch, field, bad_value
):
    tiny_bundle["theory"][epoch][field] = bad_value
    with pytest.raises(ValueError, match=field):
        write_recovery_bundle(tmp_path / "invalid", **tiny_bundle)


@pytest.mark.parametrize(
    ("field", "bad_value"),
    [
        ("coverage_mean", 0.6),
        ("coverage_sample_sd", 0.1),
        ("coverage_sample_sd", -0.1),
    ],
)
def test_summary_must_recompute_seed_mean_and_sample_sd(
    tmp_path, tiny_bundle, field, bad_value
):
    tiny_bundle["summary"][1][field] = bad_value
    with pytest.raises(ValueError, match=field):
        write_recovery_bundle(tmp_path / "invalid", **tiny_bundle)


@pytest.mark.parametrize(
    "field",
    [
        "passed",
        "graph_partition_consistency_verified",
        "cellular_set_inclusion_verified",
    ],
)
def test_failed_attestation_prevents_completion(tmp_path, tiny_bundle, field):
    tiny_bundle["checks"][field] = False
    with pytest.raises(ValueError, match=field):
        write_recovery_bundle(tmp_path / "invalid", **tiny_bundle)


def test_old_partition_provenance_attestation_is_not_sufficient(
    tmp_path, tiny_bundle
):
    tiny_bundle["checks"]["graph_partition_provenance_verified"] = True
    tiny_bundle["checks"].pop("graph_partition_consistency_verified")
    with pytest.raises(
        ValueError, match="graph_partition_consistency_verified"
    ):
        write_recovery_bundle(tmp_path / "invalid", **tiny_bundle)


def test_publication_mode_rejects_toy_or_shortened_protocol(
    tmp_path, tiny_bundle
):
    tiny_bundle["manifest"]["run_mode"] = "publication"
    with pytest.raises(ValueError, match="publication.*config"):
        write_recovery_bundle(tmp_path / "invalid", **tiny_bundle)


def test_publication_bundle_rejects_simple_cycles_called_fixed_basis(
    tiny_bundle, monkeypatch
):
    """Keep the semantics check independent of the large Cora Full data files."""
    manifest = tiny_bundle["manifest"]
    manifest["run_mode"] = "publication"
    manifest["completed"] = True
    manifest["split_config"] = {
        "seed": 0,
        "type": "fixed",
        "learning_setting": "transductive",
        "train_prop": 0.7,
    }
    pinned = {
        "dataset": "toy",
        "expected_num_nodes": 4,
        "primary": {"K": 4, "q_values": [2], "families": ["cellular"]},
        "additional": [],
        "seeds": [0, 1],
        "epochs": 2,
        "sample_every": 1,
        "cycle_max_length": 9,
        "hypergraph_identity": "center_indexed",
        "entropy_normalization": "all_reference",
        "split_seed": 0,
        "split_type": "fixed",
        "learning_setting": "transductive",
        "train_prop": 0.7,
    }
    monkeypatch.setattr(recovery_io, "load_recovery_config", lambda: pinned)
    assert validate_recovery_bundle(tiny_bundle)["manifest"]["completed"]
    manifest["family_definitions"]["cellular"] = "all_simple_cycles_len_le_9"

    with pytest.raises(ValueError, match="family definition"):
        validate_recovery_bundle(tiny_bundle)


def test_smoke_bundle_also_rejects_simple_cycles_called_fixed_basis(
    tiny_bundle,
):
    tiny_bundle["manifest"]["family_definitions"]["cellular"] = (
        "all_simple_cycles_len_le_9"
    )
    with pytest.raises(ValueError, match="family definition"):
        validate_recovery_bundle(
            tiny_bundle, require_complete=False, publication_only=False
        )


def test_bundle_rejects_unknown_reference_family_with_clear_error(tiny_bundle):
    tiny_bundle["manifest"]["configuration_matrix"][0]["families"] = [
        "all_simple_cycles"
    ]
    with pytest.raises(ValueError, match="families are invalid"):
        validate_recovery_bundle(
            tiny_bundle, require_complete=False, publication_only=False
        )


def test_publication_loader_rejects_completed_smoke_bundle(
    tmp_path, tiny_bundle
):
    destination = tmp_path / "smoke"
    write_recovery_bundle(destination, **tiny_bundle)
    with pytest.raises(ValueError, match="smoke"):
        load_recovery_bundle(destination)


def test_manifest_rejects_missing_implementation_provenance(
    tmp_path, tiny_bundle
):
    tiny_bundle["manifest"]["implementation_commit"] = "unknown"
    with pytest.raises(ValueError, match="implementation_commit"):
        write_recovery_bundle(tmp_path / "invalid", **tiny_bundle)


def test_incomplete_bundle_cannot_be_loaded_for_publication(
    tmp_path, tiny_bundle
):
    destination = tmp_path / "bundle"
    write_recovery_bundle(destination, **tiny_bundle)
    path = destination / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["completed"] = False
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="incomplete"):
        load_recovery_bundle(destination)


def test_complete_bundle_is_never_overwritten(tmp_path, tiny_bundle):
    destination = tmp_path / "bundle"
    write_recovery_bundle(destination, **tiny_bundle)
    with pytest.raises(FileExistsError, match="exists"):
        write_recovery_bundle(destination, **tiny_bundle)


def test_corrupted_csv_is_rejected_on_reload(tmp_path, tiny_bundle):
    destination = tmp_path / "bundle"
    write_recovery_bundle(destination, **tiny_bundle)
    path = destination / "observations.csv"
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    rows[1]["coverage"] = "0.9"
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(ValueError, match="coverage"):
        load_recovery_bundle(destination, publication_only=False)


@pytest.mark.parametrize(
    "field",
    ["partition_metadata", "partition_hashes", "active_cluster_counts"],
)
def test_validation_rejects_extra_or_traversal_partition_key(
    tiny_bundle, field
):
    dangerous = "4/../../../outside"
    if field == "partition_metadata":
        tiny_bundle[field][dangerous] = deepcopy(tiny_bundle[field]["4"])
    else:
        values = tiny_bundle["manifest"][field]
        values[dangerous] = values["4"]
    with pytest.raises(ValueError, match="partition.*keys"):
        validate_recovery_bundle(
            tiny_bundle, require_complete=False, publication_only=False
        )


def test_writer_rejects_extra_sidecar_before_creating_directories(
    tmp_path, tiny_bundle
):
    outside = tmp_path / "outside"
    outside.mkdir()
    sentinel = outside / "metadata.json"
    sentinel.write_text("do not replace", encoding="utf-8")
    tiny_bundle["partition_metadata"]["4/../../../outside"] = deepcopy(
        tiny_bundle["partition_metadata"]["4"]
    )
    destination = tmp_path / "bundle"
    with pytest.raises(ValueError, match="partition.*keys"):
        write_recovery_bundle(destination, **tiny_bundle)
    assert not destination.exists()
    assert sentinel.read_text(encoding="utf-8") == "do not replace"


def test_loader_rejects_manifest_keys_before_opening_any_partition_path(
    tmp_path, tiny_bundle, monkeypatch
):
    destination = tmp_path / "bundle"
    write_recovery_bundle(destination, **tiny_bundle)
    outside = tmp_path / "outside"
    outside.mkdir()
    sentinel = outside / "metadata.json"
    sentinel.write_text("do not read", encoding="utf-8")
    manifest_path = destination / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["partition_hashes"]["4/../../../outside"] = manifest[
        "partition_hashes"
    ]["4"]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    opened_after_manifest = []
    original_open = Path.open

    def tracked_open(path, *args, **kwargs):
        if path != manifest_path:
            opened_after_manifest.append(path)
        return original_open(path, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(Path, "open", tracked_open)
        with pytest.raises(ValueError, match="partition.*keys"):
            load_recovery_bundle(destination, publication_only=False)
    assert opened_after_manifest == []
    assert sentinel.read_text(encoding="utf-8") == "do not read"
