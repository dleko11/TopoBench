"""Validate recovery configuration and auditable numerical result bundles."""

import csv
import json
import math
import re
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

DEFAULT_CONFIG = (
    Path(__file__).parent / "configs" / "corafull_recovery_v2.json"
)
FAMILIES = {"hypergraph", "simplicial", "cellular"}
CANONICAL_FAMILY_DEFINITIONS = {
    "hypergraph": "one indexed closed one-hop neighbourhood per centre",
    "simplicial": "all graph triangles as filled two-cells",
    "cellular": "full-graph cycle basis filtered to length at most nine",
}
SCHEDULE_FIELDS = {"K", "q_values", "families"}
CONFIG_FIELDS = {
    "schema_version",
    "dataset",
    "expected_num_nodes",
    "primary",
    "additional",
    "seeds",
    "epochs",
    "cycle_max_length",
    "hypergraph_identity",
    "entropy_normalization",
    "cluster_policy",
    "split_seed",
    "split_type",
    "learning_setting",
    "train_prop",
    "sample_every",
}


def _require_value(config: dict[str, Any], field: str, expected: Any) -> None:
    actual = config.get(field)
    if type(actual) is not type(expected) or actual != expected:
        raise ValueError(f"{field} must be {expected!r}")


def _validate_schedule(schedule: Any, name: str) -> None:
    if not isinstance(schedule, dict):
        raise ValueError(f"{name} must be an object")
    unknown_fields = set(schedule) - SCHEDULE_FIELDS
    if unknown_fields:
        raise ValueError(f"{name} has unknown keys: {sorted(unknown_fields)}")
    k = schedule.get("K")
    if type(k) is not int or k < 1:
        raise ValueError(f"{name}.K must be a positive integer")
    q_values = schedule.get("q_values")
    if not isinstance(q_values, list) or not q_values:
        raise ValueError(f"{name}.q_values must be a nonempty list")
    if len(q_values) != len(set(map(str, q_values))):
        raise ValueError(
            f"{name}.q_values must not contain duplicate q values"
        )
    for q in q_values:
        if type(q) is not int or q < 1 or q > k:
            raise ValueError(f"{name}: q must be an integer between 1 and K")
        if k % q:
            raise ValueError(f"{name}: K must be divisible by q")
    families = schedule.get("families")
    if (
        not isinstance(families, list)
        or not families
        or any(type(family) is not str for family in families)
        or len(families) != len(set(families))
        or any(family not in FAMILIES for family in families)
    ):
        raise ValueError(
            f"{name}.families must list distinct supported domains"
        )


def validate_recovery_config(config: dict[str, Any]) -> None:
    """Reject configuration drift that would change the measured experiment."""
    if not isinstance(config, dict):
        raise ValueError("configuration must be an object")
    unknown_fields = set(config) - CONFIG_FIELDS
    if unknown_fields:
        raise ValueError(
            f"unknown configuration keys: {sorted(unknown_fields)}"
        )
    for field, expected in (
        ("schema_version", 2),
        ("dataset", "cora_full"),
        ("expected_num_nodes", 19_793),
        ("cycle_max_length", 9),
        ("hypergraph_identity", "center_indexed"),
        ("entropy_normalization", "all_reference"),
        ("cluster_policy", "training_active_require_all"),
        ("split_seed", 0),
        ("split_type", "random"),
        ("learning_setting", "transductive"),
        ("train_prop", 0.7),
    ):
        _require_value(config, field, expected)
    _validate_schedule(config.get("primary"), "primary")
    additional = config.get("additional")
    if not isinstance(additional, list):
        raise ValueError("additional must be a list")
    for index, schedule in enumerate(additional):
        _validate_schedule(schedule, f"additional[{index}]")
    seeds = config.get("seeds")
    if (
        not isinstance(seeds, list)
        or not seeds
        or any(type(seed) is not int or seed < 0 for seed in seeds)
        or len(seeds) != len(set(seeds))
    ):
        raise ValueError("seeds must be distinct nonnegative integers")
    for field in ("epochs", "sample_every"):
        value = config.get(field)
        if type(value) is not int or value < 1:
            raise ValueError(f"{field} must be a positive integer")
    _require_value(
        config,
        "primary",
        {
            "K": 32,
            "q_values": [2, 4, 8, 16],
            "families": ["hypergraph", "simplicial", "cellular"],
        },
    )
    _require_value(
        config,
        "additional",
        [{"K": 64, "q_values": [8], "families": ["cellular"]}],
    )
    for field, expected in (
        ("seeds", list(range(10))),
        ("epochs", 200),
        ("sample_every", 5),
    ):
        _require_value(config, field, expected)


def load_recovery_config(path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    """Read a recovery configuration and validate it before use."""
    with Path(path).open(encoding="utf-8") as config_file:
        config = json.load(config_file)
    validate_recovery_config(config)
    return config


def validate_graph_identity(
    config: dict[str, Any], *, dataset: str, num_nodes: int
) -> None:
    """Guard against accidentally substituting Cora for Cora Full."""
    validate_recovery_config(config)
    if dataset != config["dataset"]:
        raise ValueError(
            f"dataset must be {config['dataset']!r}, got {dataset!r}"
        )
    if num_nodes != config["expected_num_nodes"]:
        raise ValueError(
            f"node count must be {config['expected_num_nodes']}, got {num_nodes}"
        )


CSV_COLUMNS = {
    "reference_summary": (
        "dataset",
        "family",
        "reference_count",
        "reference_hash",
    ),
    "span_histogram": (
        "dataset",
        "K",
        "family",
        "span",
        "count",
        "reference_hash",
    ),
    "theory": (
        "dataset",
        "K",
        "q",
        "epoch",
        "family",
        "reference_count",
        "observable_count",
        "expected_coverage",
        "entropy_nats_per_reference",
    ),
    "observations": (
        "dataset",
        "K",
        "q",
        "seed",
        "epoch",
        "family",
        "measurement",
        "recovered_count",
        "reference_count",
        "observable_count",
        "coverage",
    ),
    "summary": (
        "dataset",
        "K",
        "q",
        "epoch",
        "family",
        "measurement",
        "n_repetitions",
        "coverage_mean",
        "coverage_sample_sd",
        "reference_count",
        "observable_fraction",
    ),
}
INTEGER_COLUMNS = {
    "K",
    "q",
    "seed",
    "epoch",
    "span",
    "count",
    "reference_count",
    "observable_count",
    "recovered_count",
    "n_repetitions",
}
FLOAT_COLUMNS = {
    "coverage",
    "expected_coverage",
    "entropy_nats_per_reference",
    "coverage_mean",
    "coverage_sample_sd",
    "observable_fraction",
}
MANIFEST_FIELDS = {
    "schema_version",
    "dataset",
    "loader_config",
    "num_nodes",
    "canonical_edge_count",
    "graph_hash",
    "graph_order_hash",
    "partition_hashes",
    "split_config",
    "active_cluster_counts",
    "family_definitions",
    "reference_hashes",
    "cycle_max_length",
    "hypergraph_identity",
    "basis_order_policy",
    "configuration_matrix",
    "seeds",
    "epochs",
    "entropy_normalization",
    "implementation_commit",
    "local_patch_hash_if_dirty",
    "package_versions",
    "partition_source",
    "run_mode",
    "completed",
}
SIDECAR_FIELDS = {
    "topology_sha256",
    "edge_order_sha256",
    "csr_order_sha256",
    "partition_assignment_sha256",
    "num_nodes",
    "num_parts",
    "num_input_edges",
    "basis_order_policy",
}
HASH_PATTERN = re.compile(r"[0-9a-f]{64}\Z")


def _integer(value: Any, label: str) -> int:
    if type(value) is int:
        result = value
    elif isinstance(value, str) and re.fullmatch(r"(?:0|[1-9][0-9]*)", value):
        result = int(value)
    else:
        raise ValueError(f"{label} must be an integer")
    if result < 0:
        raise ValueError(f"{label} must be a nonnegative integer")
    return result


def _fraction(value: Any, label: str) -> float | None:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        raise ValueError(f"{label} must be numeric")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be numeric") from exc
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _sha256(value: Any, label: str) -> None:
    if not isinstance(value, str) or not HASH_PATTERN.fullmatch(value):
        raise ValueError(f"{label} must be a SHA-256 hex digest")


def _sha256_pair(value: Any, label: str) -> None:
    if not isinstance(value, str) or len(parts := value.split(":")) != 2:
        raise ValueError(f"{label} must contain two SHA-256 hex digests")
    for digest in parts:
        _sha256(digest, label)


def _check_close(actual: Any, expected: float | None, label: str) -> None:
    number = _fraction(actual, label)
    if expected is None:
        if number is not None:
            raise ValueError(f"{label} must be missing for an empty reference")
    elif number is None or not math.isclose(
        number, expected, rel_tol=1e-12, abs_tol=1e-12
    ):
        raise ValueError(f"{label} differs from integer counts")


def _validated_rows(
    name: str, rows: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    if not isinstance(rows, list):
        raise ValueError(f"{name} must be a list of rows")
    fields = set(CSV_COLUMNS[name])
    converted = []
    for row in rows:
        if not isinstance(row, dict) or set(row) != fields:
            raise ValueError(
                f"{name} row must contain exactly {sorted(fields)}"
            )
        converted_row = dict(row)
        for key in fields & INTEGER_COLUMNS:
            converted_row[key] = _integer(row[key], key)
        for key in fields & FLOAT_COLUMNS:
            converted_row[key] = _fraction(row[key], key)
        converted.append(converted_row)
    return converted


def _expected_epochs(manifest: dict[str, Any]) -> list[int]:
    maximum = _integer(manifest["epochs"], "epochs")
    if maximum < 1:
        raise ValueError("epochs must be positive")
    if _integer(manifest.get("sample_every", 1), "sample_every") < 1:
        raise ValueError("epochs and sample_every must be positive")
    return list(range(maximum + 1))


def _partition_labels_from_manifest(manifest: dict[str, Any]) -> set[str]:
    """Validate K-derived path labels before opening partition sidecars."""
    matrix = manifest.get("configuration_matrix")
    if not isinstance(matrix, list) or not matrix:
        raise ValueError("partition keys require a configuration matrix")
    labels = set()
    for schedule in matrix:
        if not isinstance(schedule, dict):
            raise ValueError("partition keys require object schedules")
        K = schedule.get("K")
        if type(K) is not int or K < 1:
            raise ValueError("partition keys require positive integer K")
        labels.add(str(K))
    for field in ("partition_hashes", "active_cluster_counts"):
        values = manifest.get(field)
        if not isinstance(values, dict) or set(values) != labels:
            raise ValueError(
                f"partition keys in {field} differ from configured K"
            )
    return labels


def _validate_publication_manifest(manifest: dict[str, Any]) -> None:
    """Separate the full pinned protocol from explicitly labelled smoke runs."""
    if manifest["run_mode"] == "smoke":
        return
    if manifest["run_mode"] != "publication":
        raise ValueError("run_mode must be publication or smoke")
    pinned = load_recovery_config()
    matrix = [
        {"K": schedule["K"], "q": q, "families": schedule["families"]}
        for schedule in [pinned["primary"], *pinned["additional"]]
        for q in schedule["q_values"]
    ]
    expected = {
        "dataset": pinned["dataset"],
        "num_nodes": pinned["expected_num_nodes"],
        "configuration_matrix": matrix,
        "seeds": pinned["seeds"],
        "epochs": pinned["epochs"],
        "sample_every": pinned["sample_every"],
        "cycle_max_length": pinned["cycle_max_length"],
        "hypergraph_identity": pinned["hypergraph_identity"],
        "entropy_normalization": pinned["entropy_normalization"],
    }
    if any(manifest.get(field) != value for field, value in expected.items()):
        raise ValueError("publication bundle differs from pinned config")
    split = manifest["split_config"]
    if not isinstance(split, dict) or any(
        split.get(key) != pinned[field]
        for key, field in (
            ("seed", "split_seed"),
            ("type", "split_type"),
            ("learning_setting", "learning_setting"),
            ("train_prop", "train_prop"),
        )
    ):
        raise ValueError("publication split differs from pinned config")


def validate_recovery_bundle(
    bundle: dict[str, Any],
    *,
    require_complete: bool = True,
    publication_only: bool = True,
) -> dict[str, Any]:
    """Check provenance and every numerical row before plotting or release."""
    expected = {"manifest", "partition_metadata", "checks", *CSV_COLUMNS}
    if set(bundle) != expected:
        raise ValueError("bundle artifact set is incomplete or unexpected")
    manifest = bundle["manifest"]
    if not isinstance(manifest, dict) or not set(manifest) >= MANIFEST_FIELDS:
        raise ValueError("manifest is missing required provenance fields")
    if manifest["schema_version"] != 2:
        raise ValueError("schema_version must be 2")
    if manifest["entropy_normalization"] != "all_reference":
        raise ValueError("entropy_normalization must be all_reference")
    if type(manifest["completed"]) is not bool:
        raise ValueError("completed must be boolean")
    if require_complete and not manifest["completed"]:
        raise ValueError("incomplete bundle cannot be plotted")
    _validate_publication_manifest(manifest)
    if publication_only and manifest["run_mode"] == "smoke":
        raise ValueError("smoke bundle cannot be used for publication plots")
    if manifest["partition_source"] not in {"saved_experiment", "regenerated"}:
        raise ValueError("partition_source is invalid")
    dataset = manifest["dataset"]
    nodes = _integer(manifest["num_nodes"], "num_nodes")
    if (
        nodes < 1
        or not isinstance(manifest["loader_config"], str)
        or not manifest["loader_config"]
    ):
        raise ValueError("loader_config and num_nodes must be meaningful")
    _integer(manifest["canonical_edge_count"], "canonical_edge_count")
    if not re.fullmatch(
        r"[0-9a-f]{40}", str(manifest["implementation_commit"])
    ):
        raise ValueError("implementation_commit must be a full Git SHA")
    patch_hash = manifest["local_patch_hash_if_dirty"]
    if patch_hash is not None:
        _sha256(patch_hash, "local_patch_hash_if_dirty")
    packages = manifest["package_versions"]
    if (
        not isinstance(packages, dict)
        or not packages
        or any(
            not isinstance(name, str)
            or not isinstance(version, str)
            or not version
            for name, version in packages.items()
        )
    ):
        raise ValueError("package_versions must identify installed packages")
    for field in ("graph_hash", "graph_order_hash"):
        _sha256(manifest[field], field)
    if (
        not isinstance(manifest["basis_order_policy"], str)
        or not manifest["basis_order_policy"]
    ):
        raise ValueError("basis_order_policy must be named")
    if manifest["hypergraph_identity"] != "center_indexed":
        raise ValueError("hypergraph_identity must be center_indexed")
    if _integer(manifest["cycle_max_length"], "cycle_max_length") != 9:
        raise ValueError("cycle_max_length must be 9")
    seeds = manifest["seeds"]
    if (
        not isinstance(seeds, list)
        or not seeds
        or any(type(seed) is not int or seed < 0 for seed in seeds)
        or len(seeds) != len(set(seeds))
    ):
        raise ValueError("seeds must be distinct nonnegative integers")
    epochs = _expected_epochs(manifest)
    schedules = manifest["configuration_matrix"]
    if not isinstance(schedules, list) or not schedules:
        raise ValueError("configuration_matrix must be nonempty")
    configs: dict[tuple[int, int, str], None] = {}
    for schedule in schedules:
        if not isinstance(schedule, dict) or set(schedule) != {
            "K",
            "q",
            "families",
        }:
            raise ValueError("configuration_matrix row is malformed")
        K = _integer(schedule["K"], "K")
        q = _integer(schedule["q"], "q")
        if K < 1 or q < 1 or q > K or K % q:
            raise ValueError("configuration_matrix has invalid K,q")
        families = schedule["families"]
        if (
            not isinstance(families, list)
            or not families
            or len(families) != len(set(families))
            or set(families) - FAMILIES
        ):
            raise ValueError("configuration_matrix families are invalid")
        for family in families:
            key = (K, q, family)
            if key in configs:
                raise ValueError("duplicate configuration_matrix entry")
            configs[key] = None
    for field in ("partition_hashes", "active_cluster_counts"):
        if not isinstance(manifest[field], dict):
            raise ValueError(f"{field} must be a mapping")
    for field in ("reference_hashes", "family_definitions"):
        if not isinstance(manifest[field], dict):
            raise ValueError(f"{field} must be a mapping")
    labels = _partition_labels_from_manifest(manifest)
    if (
        not isinstance(bundle["partition_metadata"], dict)
        or set(bundle["partition_metadata"]) != labels
    ):
        raise ValueError("partition metadata keys differ from configured K")
    input_edge_counts = set()
    for K in {K for K, _, _ in configs}:
        label = str(K)
        sidecar = bundle["partition_metadata"].get(label)
        if not isinstance(sidecar, dict) or not set(sidecar) >= SIDECAR_FIELDS:
            raise ValueError(f"missing verified partition sidecar for K={K}")
        if sidecar["topology_sha256"] != manifest["graph_hash"]:
            raise ValueError("partition graph hash disagrees with manifest")
        if sidecar["edge_order_sha256"] != manifest["graph_order_hash"]:
            raise ValueError(
                "partition graph order hash disagrees with manifest"
            )
        if sidecar["partition_assignment_sha256"] != manifest[
            "partition_hashes"
        ].get(label):
            raise ValueError(
                "partition assignment hash disagrees with manifest"
            )
        _sha256_pair(manifest["partition_hashes"][label], "partition hash")
        _sha256_pair(sidecar["partition_assignment_sha256"], "partition hash")
        if sidecar["basis_order_policy"] != manifest["basis_order_policy"]:
            raise ValueError("basis order-policy disagrees with manifest")
        _sha256_pair(sidecar["csr_order_sha256"], "csr order hash")
        if (
            _integer(sidecar["num_nodes"], "num_nodes") != nodes
            or _integer(sidecar["num_parts"], "num_parts") != K
        ):
            raise ValueError("partition sidecar graph counts disagree")
        input_edge_counts.add(
            _integer(sidecar["num_input_edges"], "num_input_edges")
        )
        if (
            _integer(
                manifest["active_cluster_counts"].get(label),
                "active_cluster_counts",
            )
            != K
        ):
            raise ValueError("active_cluster_counts must equal K")
    if len(input_edge_counts) != 1:
        raise ValueError("processed graph edge count differs across K")
    required_families = {family for _, _, family in configs}
    if (
        set(manifest["family_definitions"]) != required_families
        or set(manifest["reference_hashes"]) != required_families
        or any(
            not isinstance(description, str) or not description
            for description in manifest["family_definitions"].values()
        )
    ):
        raise ValueError(
            "family definitions and reference hashes are incomplete"
        )
    if manifest["family_definitions"] != {
        family: CANONICAL_FAMILY_DEFINITIONS[family]
        for family in required_families
    }:
        raise ValueError("family definition differs from the v2 protocol")
    checks = bundle["checks"]
    if not isinstance(checks, dict):
        raise ValueError("checks must contain passed attestations")
    required_attestations = ["passed", "graph_partition_consistency_verified"]
    if "cellular" in required_families:
        required_attestations.append("cellular_set_inclusion_verified")
    for field in required_attestations:
        if checks.get(field) is not True:
            raise ValueError(f"checks {field} attestation must be true")
    rows = {name: _validated_rows(name, bundle[name]) for name in CSV_COLUMNS}
    references: dict[str, int] = {}
    for row in rows["reference_summary"]:
        family = row["family"]
        if row["dataset"] != dataset or family in references:
            raise ValueError("duplicate or inconsistent reference summary")
        if row["reference_hash"] != manifest["reference_hashes"].get(family):
            raise ValueError("reference hash disagrees with manifest")
        _sha256(row["reference_hash"], "reference hash")
        references[family] = row["reference_count"]
    if set(references) != {family for _, _, family in configs}:
        raise ValueError("missing reference family")
    histograms: dict[tuple[int, str], dict[int, int]] = defaultdict(dict)
    for row in rows["span_histogram"]:
        key = (row["K"], row["family"])
        if row["dataset"] != dataset or key not in {
            (K, family) for K, _, family in configs
        }:
            raise ValueError("span histogram has unexpected configuration")
        if row["reference_hash"] != manifest["reference_hashes"].get(
            row["family"]
        ):
            raise ValueError("span histogram reference hash disagrees")
        span = row["span"]
        if span < 1 or span > row["K"] or span in histograms[key]:
            raise ValueError("span histogram has invalid or duplicate span")
        histograms[key][span] = row["count"]
    for K, family in {(K, family) for K, _, family in configs}:
        if sum(histograms[K, family].values()) != references[family]:
            raise ValueError("span histogram denominator disagrees")
    theory_keys: set[tuple[int, int, int, str]] = set()
    from scripts.structural_coverage.recovery_core import (
        expected_coverage,
        recovery_entropy,
    )

    for row in rows["theory"]:
        K, q, family = row["K"], row["q"], row["family"]
        key = (K, q, row["epoch"], family)
        if row["dataset"] != dataset or (K, q, family) not in configs:
            raise ValueError("theory has unexpected configuration")
        if key in theory_keys:
            raise ValueError("duplicate theory row")
        theory_keys.add(key)
        observable = sum(
            count for span, count in histograms[K, family].items() if span <= q
        )
        if (
            row["reference_count"] != references[family]
            or row["observable_count"] != observable
        ):
            raise ValueError("theory denominator disagrees")
        _check_close(
            row["expected_coverage"],
            expected_coverage(histograms[K, family], K, q, row["epoch"]),
            "expected_coverage",
        )
        _check_close(
            row["entropy_nats_per_reference"],
            recovery_entropy(histograms[K, family], K, q, row["epoch"]),
            "entropy_nats_per_reference",
        )
    expected_theory = {
        (K, q, epoch, family) for K, q, family in configs for epoch in epochs
    }
    if theory_keys != expected_theory:
        raise ValueError("missing theory epoch rows")
    observations = rows["observations"]
    observed: dict[tuple[int, int, int, str, str], dict[int, int]] = (
        defaultdict(dict)
    )
    for row in observations:
        K, q, family = row["K"], row["q"], row["family"]
        if row["dataset"] != dataset or (K, q, family) not in configs:
            raise ValueError("observation has unexpected configuration")
        if row["seed"] not in seeds:
            raise ValueError("observation has unexpected seed")
        key = (K, q, row["seed"], family, row["measurement"])
        epoch = row["epoch"]
        if epoch in observed[key]:
            raise ValueError("duplicate observation key")
        if row["reference_count"] != references[family]:
            raise ValueError("observation denominator changes")
        observable = sum(
            count for span, count in histograms[K, family].items() if span <= q
        )
        if row["observable_count"] != observable:
            raise ValueError("observation observable count changes")
        recovered = row["recovered_count"]
        if recovered > row["reference_count"]:
            raise ValueError("recovered_count exceeds reference_count")
        if recovered > observable:
            raise ValueError("recovered_count exceeds observable_count")
        _check_close(
            row["coverage"],
            recovered / references[family] if references[family] else None,
            "coverage",
        )
        observed[key][epoch] = recovered
    measurement_types: dict[tuple[int, int, str], set[str]] = defaultdict(set)
    for K, q, _seed, family, measurement in observed:
        measurement_types[K, q, family].add(measurement)
    for K, q, family in configs:
        measures = measurement_types[K, q, family]
        required = {"support_available"}
        if family == "cellular":
            required.add("actual_basis_recovery")
        if measures != required:
            raise ValueError(
                f"missing or unexpected measurement for {family}: {measures}"
            )
        for seed in seeds:
            for measurement in measures:
                counts = observed.get((K, q, seed, family, measurement), {})
                if set(counts) != set(epochs):
                    raise ValueError("missing observation epoch rows or seed")
                if counts[0] != 0:
                    raise ValueError(
                        "T=0 cumulative recovered_count must be 0"
                    )
                if any(
                    counts[later] < counts[earlier]
                    for earlier, later in zip(epochs, epochs[1:], strict=False)
                ):
                    raise ValueError("decreasing cumulative recovered_count")
            if family == "cellular":
                support = observed[K, q, seed, family, "support_available"]
                actual = observed[K, q, seed, family, "actual_basis_recovery"]
                if any(actual[epoch] > support[epoch] for epoch in epochs):
                    raise ValueError("actual basis count exceeds support")
    summary_keys: set[tuple[int, int, int, str, str]] = set()
    for row in rows["summary"]:
        K, q, family = row["K"], row["q"], row["family"]
        key = (K, q, row["epoch"], family, row["measurement"])
        if key in summary_keys:
            raise ValueError("duplicate summary row")
        summary_keys.add(key)
        if (
            row["dataset"] != dataset
            or (K, q, family) not in configs
            or row["measurement"] not in measurement_types[K, q, family]
        ):
            raise ValueError("summary has unexpected configuration")
        if row["n_repetitions"] != len(seeds):
            raise ValueError("summary repetition count disagrees with seeds")
        if row["reference_count"] != references[family]:
            raise ValueError("summary denominator disagrees")
        observable = sum(
            count for span, count in histograms[K, family].items() if span <= q
        )
        _check_close(
            row["observable_fraction"],
            observable / references[family] if references[family] else None,
            "observable_fraction",
        )
        counts = [
            observed[K, q, seed, family, row["measurement"]][row["epoch"]]
            for seed in seeds
        ]
        if references[family]:
            values = [count / references[family] for count in counts]
            expected_mean = statistics.mean(values)
            expected_sd = statistics.stdev(values) if len(values) > 1 else None
        else:
            expected_mean = expected_sd = None
        _check_close(row["coverage_mean"], expected_mean, "coverage_mean")
        _check_close(
            row["coverage_sample_sd"], expected_sd, "coverage_sample_sd"
        )
    expected_summary = {
        (K, q, epoch, family, measurement)
        for (K, q, family), measures in measurement_types.items()
        for measurement in measures
        for epoch in epochs
    }
    if summary_keys != expected_summary:
        raise ValueError("missing summary epoch rows")
    return {**bundle, **rows}


def _write_json(path: Path, value: Any) -> None:
    with path.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")


def write_recovery_bundle(
    output_dir: str | Path,
    *,
    manifest: dict[str, Any],
    partition_metadata: dict[str, dict[str, Any]],
    reference_summary: list[dict[str, Any]],
    span_histogram: list[dict[str, Any]],
    theory: list[dict[str, Any]],
    observations: list[dict[str, Any]],
    summary: list[dict[str, Any]],
    checks: dict[str, Any],
) -> None:
    """Write a fresh bundle, marking complete only after disk validation."""
    directory = Path(output_dir)
    if directory.exists():
        raise FileExistsError(f"result directory already exists: {directory}")
    pending = dict(manifest)
    pending["completed"] = False
    bundle = validate_recovery_bundle(
        {
            "manifest": pending,
            "partition_metadata": partition_metadata,
            "reference_summary": reference_summary,
            "span_histogram": span_histogram,
            "theory": theory,
            "observations": observations,
            "summary": summary,
            "checks": checks,
        },
        require_complete=False,
        publication_only=False,
    )
    directory.mkdir(parents=True, exist_ok=False)
    _write_json(directory / "manifest.json", pending)
    for K, metadata in partition_metadata.items():
        child = directory / "partitions" / f"k{K}"
        child.mkdir(parents=True, exist_ok=False)
        _write_json(child / "metadata.json", metadata)
    for name, fields in CSV_COLUMNS.items():
        with (directory / f"{name}.csv").open(
            "w", encoding="utf-8", newline=""
        ) as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(bundle[name])
    _write_json(directory / "checks.json", checks)
    load_recovery_bundle(
        directory, require_complete=False, publication_only=False
    )
    complete = dict(pending)
    complete["completed"] = True
    _write_json(directory / "manifest.json", complete)


def load_recovery_bundle(
    output_dir: str | Path,
    *,
    require_complete: bool = True,
    publication_only: bool = True,
) -> dict[str, Any]:
    """Load a bundle and reject incomplete or contradictory numerical data."""
    directory = Path(output_dir)
    with (directory / "manifest.json").open(encoding="utf-8") as stream:
        manifest = json.load(stream)
    if require_complete and not manifest.get("completed"):
        raise ValueError("incomplete bundle cannot be plotted")
    labels = _partition_labels_from_manifest(manifest)
    metadata = {}
    for K in sorted(labels, key=int):
        path = directory / "partitions" / f"k{K}" / "metadata.json"
        with path.open(encoding="utf-8") as stream:
            metadata[K] = json.load(stream)
    rows = {}
    for name, fields in CSV_COLUMNS.items():
        with (directory / f"{name}.csv").open(
            encoding="utf-8", newline=""
        ) as stream:
            reader = csv.DictReader(stream)
            if tuple(reader.fieldnames or ()) != fields:
                raise ValueError(f"{name} CSV columns differ from schema")
            rows[name] = list(reader)
    with (directory / "checks.json").open(encoding="utf-8") as stream:
        checks = json.load(stream)
    return validate_recovery_bundle(
        {
            "manifest": manifest,
            "partition_metadata": metadata,
            "checks": checks,
            **rows,
        },
        require_complete=require_complete,
        publication_only=publication_only,
    )
