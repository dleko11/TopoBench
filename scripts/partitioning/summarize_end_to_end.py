#!/usr/bin/env python3
"""Summarize end-to-end runtime and run-wide memory peaks across seeds.

The non-overlapping time components are preprocessing (dataset loading through
the start of fitting), training (the complete fit call, including validation),
and final evaluation (the end of fitting through the final test protocol).
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

DEFAULT_INPUT_DIR = Path("outputs/phase_tracking")
DEFAULT_OUTPUT_DIR = DEFAULT_INPUT_DIR / "end_to_end"
COMPONENTS = (
    "preprocessing_sec",
    "training_sec",
    "final_evaluation_sec",
    "total_wall_clock_sec",
)
RESOURCES = (
    "cpu_process_peak_gib",
    "cpu_process_tree_peak_gib",
    "gpu_peak_reserved_gib",
    "gpu_peak_allocated_gib",
)
MANUSCRIPT_RESOURCES = (
    "cpu_process_peak_gib",
    "gpu_peak_allocated_gib",
)
GIB_TO_GB = 1024**3 / 1000**3
MEASURES = COMPONENTS + RESOURCES
TEST_PHASES = (
    "test_inference_batched",
    "test_inference_full_graph",
    "test_inference_ensemble",
    "test_best_rerun",
)
IDENTITY = ("dataset", "model", "mode", "seed")
DATASET_ORDER = ("cora_full", "amazon_ratings", "questions")
MODEL_ORDER = ("gcn", "edgnn", "unignn", "cwn", "topotune", "scn", "sccnn")
DATASET_NAMES = {
    "amazon_ratings": "Amazon Ratings",
    "questions": "Questions",
    "cora_full": "Cora Full",
}
MODEL_NAMES = {
    "gcn": "GCN",
    "edgnn": "EDHNN",
    "unignn": "UniGNN",
    "cwn": "CWN",
    "topotune": "Cell TopoTune",
    "scn": "SCN",
    "sccnn": "SCCNN",
}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runs",
        type=Path,
        default=DEFAULT_INPUT_DIR / "runs.csv",
    )
    parser.add_argument(
        "--phase-metrics",
        type=Path,
        default=DEFAULT_INPUT_DIR / "phase_metrics.csv",
    )
    parser.add_argument(
        "--fresh-partition-phase-metrics",
        type=Path,
        help=(
            "Optional W&B phase export from fresh partition-build runs. "
            "When provided, partitioned preprocessing time and CPU peaks are "
            "replaced by the matching fresh dataset/K/seed measurements."
        ),
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser


def _require_columns(
    frame: pd.DataFrame,
    required: set[str],
    source: Path,
) -> None:
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{source} is missing required columns: {missing}")


def _single_phase(group: pd.DataFrame, phase: str) -> pd.Series:
    selected = group[group["phase"] == phase]
    if len(selected) != 1:
        key = tuple(group.iloc[0][column] for column in IDENTITY)
        raise ValueError(
            f"Expected one {phase!r} row for {key}, found {len(selected)}."
        )
    row = selected.iloc[0]
    if not bool(row["events_complete"]):
        raise ValueError(f"Incomplete {phase!r} markers for {key}.")
    return row


def _evaluation_end(group: pd.DataFrame) -> tuple[float, str]:
    selected = group[group["phase"].isin(TEST_PHASES)].dropna(
        subset=["last_end_timestamp"]
    )
    if selected.empty:
        key = tuple(group.iloc[0][column] for column in IDENTITY)
        raise ValueError(f"No completed final test phase found for {key}.")
    row = selected.loc[selected["last_end_timestamp"].idxmax()]
    if not bool(row["events_complete"]):
        key = tuple(group.iloc[0][column] for column in IDENTITY)
        raise ValueError(f"Incomplete final test markers for {key}.")
    return float(row["last_end_timestamp"]), str(row["phase"])


def _metadata_row(runs: pd.DataFrame, key: tuple[Any, ...]) -> pd.Series:
    mask = pd.Series(True, index=runs.index)
    for column, value in zip(IDENTITY, key, strict=True):
        mask &= runs[column] == value
    selected = runs[mask]
    if len(selected) != 1:
        raise ValueError(f"Expected one run row for {key}, found {len(selected)}.")
    return selected.iloc[0]


def _run_components(runs: pd.DataFrame, phases: pd.DataFrame) -> pd.DataFrame:
    records = []
    for key, group in phases.groupby(list(IDENTITY), sort=False):
        metadata = _metadata_row(runs, key)
        dataset_load = _single_phase(group, "dataset_load")
        fit = _single_phase(group, "fit")
        evaluation_end, evaluation_phase = _evaluation_end(group)

        start = float(dataset_load["first_start_timestamp"])
        fit_start = float(fit["first_start_timestamp"])
        fit_end = float(fit["last_end_timestamp"])
        if not start <= fit_start <= fit_end <= evaluation_end:
            raise ValueError(f"Non-monotonic phase boundaries for {key}.")

        preprocessing = fit_start - start
        training = fit_end - fit_start
        final_evaluation = evaluation_end - fit_end
        total = evaluation_end - start
        if not np.isclose(
            total,
            preprocessing + training + final_evaluation,
            rtol=0,
            atol=1e-6,
        ):
            raise ValueError(f"Component accounting failed for {key}.")

        record = {
            column: value
            for column, value in zip(IDENTITY, key, strict=True)
        }
        for column in (
            "run_id",
            "run_name",
            "run_state",
            "model_params_total",
            "final_epoch",
            "final_global_step",
            "gpu",
            "cpu_count",
            "host_memory_total_mb",
            "git_commit",
            "num_parts",
            "stream_q",
            "stream_q_val",
            "stream_q_test",
            "cache_val",
            "val_shuffle",
            "test_protocols",
            "ensemble_runs",
            "force_reload_preprocessing",
            "partition_cache_namespace",
        ):
            record[column] = metadata.get(column)
        record.update(
            {
                "final_test_phase": evaluation_phase,
                "preprocessing_sec": preprocessing,
                "training_sec": training,
                "final_evaluation_sec": final_evaluation,
                "total_wall_clock_sec": total,
            }
        )
        records.append(record)
    return pd.DataFrame.from_records(records)


def _run_resource_peaks(phases: pd.DataFrame) -> pd.DataFrame:
    """Return run-wide process and CUDA peaks across all tracked phases."""
    sources = {
        "cpu_process_peak_gib": "rss_peak_max_mb",
        "cpu_process_tree_peak_gib": "tree_rss_peak_max_mb",
        "gpu_peak_reserved_gib": "cuda_peak_reserved_max_mb",
        "gpu_peak_allocated_gib": "cuda_peak_allocated_max_mb",
    }
    records = []
    groups = phases.groupby([*IDENTITY, "run_id"], sort=False)
    for key, group in groups:
        record = {
            column: value
            for column, value in zip(
                (*IDENTITY, "run_id"), key, strict=True
            )
        }
        for output, source in sources.items():
            values = pd.to_numeric(group[source], errors="coerce").dropna()
            record[output] = values.max() / 1024 if not values.empty else np.nan
        records.append(record)
    return pd.DataFrame.from_records(records)


def _fresh_partition_preprocessing(
    phases: pd.DataFrame,
    source: Path,
) -> pd.DataFrame:
    required = {
        "dataset",
        "mode",
        "model",
        "seed",
        "num_parts",
        "run_id",
        "phase",
        "events_complete",
        "first_start_timestamp",
        "last_end_timestamp",
        "duration_total_sec",
        "rss_peak_max_mb",
        "tree_rss_peak_max_mb",
    }
    _require_columns(phases, required, source)
    phases = phases.copy()
    phases["events_complete"] = (
        phases["events_complete"].astype(str).str.lower() == "true"
    )
    phases = phases[
        (phases["mode"] == "partitioning") & (phases["model"] == "gcn")
    ]
    if phases.empty:
        raise ValueError(f"No fresh partitioning GCN runs found in {source}.")

    records = []
    group_columns = ["dataset", "num_parts", "seed", "run_id"]
    for key, group in phases.groupby(group_columns, sort=False):
        dataset_load = _single_phase(group, "dataset_load")
        trainer_init = _single_phase(group, "trainer_init")
        partition_build = _single_phase(group, "partition_build")
        start = float(dataset_load["first_start_timestamp"])
        end = float(trainer_init["last_end_timestamp"])
        if start > end:
            raise ValueError(f"Non-monotonic fresh preprocessing phases for {key}.")

        contained = group[
            (group["first_start_timestamp"] >= start)
            & (group["last_end_timestamp"] <= end)
        ]
        if contained.empty or not contained["events_complete"].all():
            raise ValueError(f"Incomplete fresh preprocessing phases for {key}.")

        dataset, num_parts, seed, run_id = key
        records.append(
            {
                "dataset": dataset,
                "num_parts": int(num_parts),
                "seed": int(seed),
                "fresh_preprocessing_sec": end - start,
                "fresh_partition_build_sec": float(
                    partition_build["duration_total_sec"]
                ),
                "fresh_cpu_process_peak_gib": (
                    pd.to_numeric(
                        contained["rss_peak_max_mb"], errors="coerce"
                    ).max()
                    / 1024
                ),
                "fresh_cpu_process_tree_peak_gib": (
                    pd.to_numeric(
                        contained["tree_rss_peak_max_mb"], errors="coerce"
                    ).max()
                    / 1024
                ),
                "fresh_preprocessing_run_id": run_id,
            }
        )

    fresh = pd.DataFrame.from_records(records)
    key_columns = ["dataset", "num_parts", "seed"]
    duplicates = fresh.duplicated(key_columns, keep=False)
    if duplicates.any():
        duplicate_keys = fresh.loc[duplicates, key_columns].to_dict("records")
        raise ValueError(
            "Duplicate fresh preprocessing measurements found: "
            f"{duplicate_keys[:5]}"
        )
    return fresh


def _apply_fresh_partition_preprocessing(
    run_components: pd.DataFrame,
    fresh: pd.DataFrame,
) -> pd.DataFrame:
    corrected = run_components.copy()
    corrected["num_parts"] = pd.to_numeric(
        corrected["num_parts"], errors="coerce"
    )
    corrected["seed"] = pd.to_numeric(corrected["seed"], errors="raise").astype(
        int
    )
    corrected["cached_preprocessing_sec"] = corrected["preprocessing_sec"]
    corrected["cached_cpu_process_peak_gib"] = corrected[
        "cpu_process_peak_gib"
    ]
    corrected["cached_cpu_process_tree_peak_gib"] = corrected[
        "cpu_process_tree_peak_gib"
    ]

    partitioned = corrected["mode"] == "partitioning"
    partitioned_rows = corrected.loc[partitioned].copy()
    partitioned_rows["__source_index"] = partitioned_rows.index
    merged = partitioned_rows.merge(
        fresh,
        on=["dataset", "num_parts", "seed"],
        how="left",
        validate="many_to_one",
    )
    missing = merged["fresh_preprocessing_sec"].isna()
    if missing.any():
        keys = merged.loc[missing, ["dataset", "num_parts", "seed"]]
        raise ValueError(
            "Missing fresh preprocessing measurements for: "
            f"{keys.drop_duplicates().to_dict('records')[:5]}"
        )

    merged["preprocessing_sec"] = merged["fresh_preprocessing_sec"]
    merged["total_wall_clock_sec"] = (
        merged["preprocessing_sec"]
        + merged["training_sec"]
        + merged["final_evaluation_sec"]
    )
    merged["cpu_process_peak_gib"] = merged[
        ["cpu_process_peak_gib", "fresh_cpu_process_peak_gib"]
    ].max(axis=1)
    merged["cpu_process_tree_peak_gib"] = merged[
        ["cpu_process_tree_peak_gib", "fresh_cpu_process_tree_peak_gib"]
    ].max(axis=1)

    fresh_columns = [
        "fresh_preprocessing_sec",
        "fresh_partition_build_sec",
        "fresh_cpu_process_peak_gib",
        "fresh_cpu_process_tree_peak_gib",
        "fresh_preprocessing_run_id",
    ]
    for column in fresh_columns:
        corrected[column] = None if column.endswith("run_id") else np.nan
    merged = merged.set_index("__source_index")
    source_indices = corrected.index[partitioned]
    corrected.loc[source_indices, corrected.columns] = merged.loc[
        source_indices, corrected.columns
    ].to_numpy()
    return corrected


def _display_unique(values: pd.Series) -> str:
    unique = sorted(str(value) for value in values.dropna().unique())
    return "|".join(unique)


def _stats(values: pd.Series) -> dict[str, float | int]:
    array = pd.to_numeric(values, errors="coerce").dropna().to_numpy()
    if array.size == 0:
        return {
            "n": 0,
            "mean": np.nan,
            "std": np.nan,
            "median": np.nan,
            "q1": np.nan,
            "q3": np.nan,
        }
    return {
        "n": int(array.size),
        "mean": float(np.mean(array)),
        "std": float(np.std(array, ddof=1)) if array.size > 1 else np.nan,
        "median": float(np.median(array)),
        "q1": float(np.quantile(array, 0.25)),
        "q3": float(np.quantile(array, 0.75)),
    }


def _component_summary(run_components: pd.DataFrame) -> pd.DataFrame:
    records = []
    groups = run_components.groupby(["dataset", "model", "mode"], sort=False)
    for (dataset, model, mode), group in groups:
        metadata = {
            "dataset": dataset,
            "model": model,
            "mode": mode,
            "seeds": _display_unique(group["seed"]),
            "num_parts": _display_unique(group["num_parts"]),
            "q": _display_unique(group["stream_q"]),
            "q_val": _display_unique(group["stream_q_val"]),
            "q_test": _display_unique(group["stream_q_test"]),
            "cache_val": _display_unique(group["cache_val"]),
            "val_shuffle": _display_unique(group["val_shuffle"]),
            "test_protocols": _display_unique(group["test_protocols"]),
            "gpu": _display_unique(group["gpu"]),
            "cpu_count": _display_unique(group["cpu_count"]),
            "git_commit": _display_unique(group["git_commit"]),
        }
        for component in COMPONENTS:
            record = dict(metadata)
            record["component"] = component.removesuffix("_sec")
            record["unit"] = "s"
            record.update(_stats(group[component]))
            records.append(record)
    return pd.DataFrame.from_records(records)


def _resource_summary(run_components: pd.DataFrame) -> pd.DataFrame:
    records = []
    groups = run_components.groupby(["dataset", "model", "mode"], sort=False)
    for (dataset, model, mode), group in groups:
        for resource in RESOURCES:
            record = {
                "dataset": dataset,
                "model": model,
                "mode": mode,
                "resource": resource,
                "unit": "GiB",
                "seeds": _display_unique(group["seed"]),
            }
            record.update(_stats(group[resource]))
            records.append(record)
    return pd.DataFrame.from_records(records)


def _paired_comparison(run_components: pd.DataFrame) -> pd.DataFrame:
    join = ["dataset", "model", "seed"]
    metadata = [
        "model_params_total",
        "gpu",
        "cpu_count",
        "git_commit",
        "num_parts",
        "stream_q",
        "cache_val",
        "val_shuffle",
        "force_reload_preprocessing",
        "partition_cache_namespace",
    ]
    full = run_components[run_components["mode"] == "full"]
    partitioning = run_components[run_components["mode"] == "partitioning"]
    full = full[join + metadata + list(MEASURES)].add_prefix("full_")
    full = full.rename(columns={f"full_{column}": column for column in join})
    partitioning = partitioning[
        join + metadata + list(MEASURES)
    ].add_prefix("partitioning_")
    partitioning = partitioning.rename(
        columns={f"partitioning_{column}": column for column in join}
    )
    paired = full.merge(
        partitioning,
        on=join,
        how="inner",
        validate="one_to_one",
    )
    if paired.empty:
        return paired

    paired["same_model_params"] = (
        paired["full_model_params_total"]
        == paired["partitioning_model_params_total"]
    )
    paired["same_gpu"] = paired["full_gpu"] == paired["partitioning_gpu"]
    paired["same_cpu_count"] = (
        paired["full_cpu_count"] == paired["partitioning_cpu_count"]
    )
    paired["same_git_commit"] = (
        paired["full_git_commit"] == paired["partitioning_git_commit"]
    )
    for measure in MEASURES:
        paired[f"{measure}_ratio"] = (
            paired[f"partitioning_{measure}"] / paired[f"full_{measure}"]
        )
    return paired


def _paired_summary(paired: pd.DataFrame) -> pd.DataFrame:
    if paired.empty:
        return paired
    records = []
    for (dataset, model), group in paired.groupby(
        ["dataset", "model"], sort=False
    ):
        for measure in MEASURES:
            full_stats = _stats(group[f"full_{measure}"])
            partitioning_stats = _stats(group[f"partitioning_{measure}"])
            ratio_stats = _stats(group[f"{measure}_ratio"])
            records.append(
                {
                    "dataset": dataset,
                    "model": model,
                    "measure": measure,
                    "unit": "s" if measure in COMPONENTS else "GiB",
                    "n_pairs": full_stats["n"],
                    "full_mean": full_stats["mean"],
                    "full_std": full_stats["std"],
                    "partitioning_mean": partitioning_stats["mean"],
                    "partitioning_std": partitioning_stats["std"],
                    "paired_ratio_mean": ratio_stats["mean"],
                    "paired_ratio_std": ratio_stats["std"],
                    "same_model_params": bool(group["same_model_params"].all()),
                    "same_gpu": bool(group["same_gpu"].all()),
                    "same_cpu_count": bool(group["same_cpu_count"].all()),
                    "same_git_commit": bool(group["same_git_commit"].all()),
                    "num_parts": _display_unique(
                        group["partitioning_num_parts"]
                    ),
                    "q": _display_unique(group["partitioning_stream_q"]),
                    "cache_val": _display_unique(
                        group["partitioning_cache_val"]
                    ),
                    "val_shuffle": _display_unique(
                        group["partitioning_val_shuffle"]
                    ),
                    "full_force_reload_preprocessing": _display_unique(
                        group["full_force_reload_preprocessing"]
                    ),
                    "partitioning_partition_cache_namespace": (
                        _display_unique(
                            group["partitioning_partition_cache_namespace"]
                        )
                    ),
                }
            )
    return pd.DataFrame.from_records(records)


def _compact_comparison(paired_summary: pd.DataFrame) -> pd.DataFrame:
    records = []
    groups = paired_summary.groupby(["dataset", "model"], sort=False)
    for (dataset, model), group in groups:
        first = group.iloc[0]
        record = {
            "dataset": dataset,
            "model": model,
            "num_parts": first["num_parts"],
            "q": first["q"],
            "n_pairs": int(first["n_pairs"]),
            "same_model_params": bool(first["same_model_params"]),
            "same_gpu": bool(first["same_gpu"]),
            "same_cpu_count": bool(first["same_cpu_count"]),
            "same_git_commit": bool(first["same_git_commit"]),
            "cache_val": first["cache_val"],
            "val_shuffle": first["val_shuffle"],
            "full_force_reload_preprocessing": first[
                "full_force_reload_preprocessing"
            ],
            "partition_cache_namespace": first[
                "partitioning_partition_cache_namespace"
            ],
        }
        indexed = group.set_index("measure")
        for measure in (*COMPONENTS, *MANUSCRIPT_RESOURCES):
            if measure not in indexed.index:
                continue
            row = indexed.loc[measure]
            for column in (
                "full_mean",
                "full_std",
                "partitioning_mean",
                "partitioning_std",
                "paired_ratio_mean",
                "paired_ratio_std",
            ):
                record[f"{measure}_{column}"] = row[column]
        records.append(record)
    compact = pd.DataFrame.from_records(records)
    dataset_order = {value: index for index, value in enumerate(DATASET_ORDER)}
    model_order = {value: index for index, value in enumerate(MODEL_ORDER)}
    compact["__dataset_order"] = compact["dataset"].map(dataset_order)
    compact["__model_order"] = compact["model"].map(model_order)
    return (
        compact.sort_values(["__dataset_order", "__model_order"])
        .drop(columns=["__dataset_order", "__model_order"])
        .reset_index(drop=True)
    )


def _format_mean_std(
    mean: Any,
    std: Any,
    decimals: int,
    scale: float = 1.0,
) -> str:
    if pd.isna(mean):
        return "--"
    mean = float(mean) * scale
    if pd.isna(std):
        return f"{mean:.{decimals}f}"
    std = float(std) * scale
    return f"{mean:.{decimals}f} $\\pm$ {std:.{decimals}f}"


def _int_cell(value: Any) -> str:
    if pd.isna(value) or value == "":
        return "--"
    return str(int(float(value)))


def _latex_rows(
    compact: pd.DataFrame,
    measures: tuple[str, ...],
    decimals: int,
    header: str,
    scale: float = 1.0,
) -> str:
    rows = [f"% {header}"]
    for row in compact.itertuples(index=False):
        cells = [
            DATASET_NAMES.get(row.dataset, str(row.dataset)),
            MODEL_NAMES.get(row.model, str(row.model)),
            _int_cell(row.num_parts),
            _int_cell(row.q),
            str(int(row.n_pairs)),
        ]
        values = row._asdict()
        cells.extend(
            _format_mean_std(
                values[f"{measure}_{mode}_mean"],
                values[f"{measure}_{mode}_std"],
                decimals,
                scale,
            )
            for measure in measures
            for mode in ("full", "partitioning")
        )
        rows.append(" & ".join(cells) + r" \\")
    return "\n".join(rows) + "\n"


def main() -> None:
    args = _build_parser().parse_args()
    runs = pd.read_csv(args.runs)
    phases = pd.read_csv(args.phase_metrics)
    _require_columns(
        runs,
        {
            *IDENTITY,
            "run_id",
            "model_params_total",
            "num_parts",
            "stream_q",
            "cache_val",
            "val_shuffle",
        },
        args.runs,
    )
    _require_columns(
        phases,
        {
            *IDENTITY,
            "phase",
            "events_complete",
            "first_start_timestamp",
            "last_end_timestamp",
            "rss_peak_max_mb",
            "tree_rss_peak_max_mb",
            "cuda_peak_reserved_max_mb",
            "cuda_peak_allocated_max_mb",
        },
        args.phase_metrics,
    )
    phases["events_complete"] = (
        phases["events_complete"].astype(str).str.lower() == "true"
    )

    run_components = _run_components(runs, phases)
    resource_peaks = _run_resource_peaks(phases)
    run_components = run_components.merge(
        resource_peaks,
        on=[*IDENTITY, "run_id"],
        how="left",
        validate="one_to_one",
    )
    if args.fresh_partition_phase_metrics is not None:
        fresh_phases = pd.read_csv(args.fresh_partition_phase_metrics)
        fresh_preprocessing = _fresh_partition_preprocessing(
            fresh_phases,
            args.fresh_partition_phase_metrics,
        )
        run_components = _apply_fresh_partition_preprocessing(
            run_components,
            fresh_preprocessing,
        )
    component_summary = _component_summary(run_components)
    resource_summary = _resource_summary(run_components)
    paired = _paired_comparison(run_components)
    paired_summary = _paired_summary(paired)
    compact = _compact_comparison(paired_summary)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "run_components.csv": run_components,
        "component_summary.csv": component_summary,
        "resource_summary.csv": resource_summary,
        "paired_runs.csv": paired,
        "paired_summary.csv": paired_summary,
        "compact_comparison.csv": compact,
    }
    for filename, frame in outputs.items():
        path = args.output_dir / filename
        frame.to_csv(path, index=False)
        print(path)

    timing_rows = args.output_dir / "manuscript_timing_rows.tex"
    timing_rows.write_text(
        _latex_rows(
            compact,
            COMPONENTS,
            decimals=1,
            header=(
                "Dataset & Model & K & q & n & Full preprocessing & "
                "Partitioned preprocessing & Full training & "
                "Partitioned training & Full final evaluation & "
                "Partitioned final evaluation & Full total & "
                "Partitioned total"
            ),
        ),
        encoding="utf-8",
    )
    print(timing_rows)
    memory_rows = args.output_dir / "manuscript_memory_rows.tex"
    memory_rows.write_text(
        _latex_rows(
            compact,
            MANUSCRIPT_RESOURCES,
            decimals=2,
            header=(
                "Dataset & Model & K & q & n & Full peak CPU & "
                "Partitioned peak CPU & Full peak GPU allocated & "
                "Partitioned peak GPU allocated"
            ),
            scale=GIB_TO_GB,
        ),
        encoding="utf-8",
    )
    print(memory_rows)

    print(f"Runs summarized: {len(run_components)}")
    print(f"Paired runs: {len(paired)}")
    if not paired.empty:
        print(
            "Hardware-matched pairs: "
            f"{int((paired['same_gpu'] & paired['same_cpu_count']).sum())}"
            f" / {len(paired)}"
        )


if __name__ == "__main__":
    main()
