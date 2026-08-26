#!/usr/bin/env python3
"""Summarize extracted W&B phase metrics across paired seeds."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

DEFAULT_INPUT_DIR = Path("outputs/phase_tracking")
DEFAULT_OUTPUT_DIR = DEFAULT_INPUT_DIR / "analysis"

DATASET_ORDER = ("cora_full", "amazon_ratings", "questions")
MODEL_ORDER = ("gcn", "edgnn", "unignn", "cwn", "topotune", "scn", "sccnn")
PHASE_ORDER = (
    "dataset_load",
    "full_graph_preprocessing",
    "partition_build",
    "datamodule_init",
    "model_init",
    "callbacks_init",
    "trainer_init",
    "val_cache_build",
    "fit",
    "train_epoch",
    "validation_epoch",
    "test_epoch",
    "checkpoint_load",
    "val_best_rerun",
    "test_best_rerun",
    "test_inference_batched",
    "test_inference_full_graph",
    "test_inference_ensemble",
)


@dataclass(frozen=True)
class MetricSpec:
    name: str
    source: str
    family: str
    unit: str
    scale: float = 1.0


METRICS = (
    MetricSpec(
        "time_total_sec",
        "duration_total_sec",
        "time",
        "s",
    ),
    MetricSpec(
        "time_per_occurrence_sec",
        "duration_mean_sec",
        "time",
        "s",
    ),
    MetricSpec(
        "cuda_peak_allocated_gib",
        "cuda_peak_allocated_max_mb",
        "memory",
        "GiB",
        1 / 1024,
    ),
    MetricSpec(
        "cuda_peak_reserved_gib",
        "cuda_peak_reserved_max_mb",
        "memory",
        "GiB",
        1 / 1024,
    ),
    MetricSpec(
        "driver_rss_boundary_gib",
        "rss_boundary_max_mb",
        "memory",
        "GiB",
        1 / 1024,
    ),
    MetricSpec(
        "tree_rss_boundary_gib",
        "tree_rss_boundary_max_mb",
        "memory",
        "GiB",
        1 / 1024,
    ),
)

REPEATED_PHASES = {"train_epoch", "validation_epoch", "test_epoch"}
CUDA_MAIN_PHASES = {
    "train_epoch",
    "validation_epoch",
    "val_best_rerun",
    "test_inference_batched",
}
RSS_APPENDIX_PHASES = {
    "dataset_load",
    "full_graph_preprocessing",
    "train_epoch",
    "validation_epoch",
    "val_best_rerun",
    "test_inference_batched",
}
TIME_TOTAL_MAIN_PHASES = {
    "dataset_load",
    "full_graph_preprocessing",
    "datamodule_init",
    "model_init",
    "callbacks_init",
    "trainer_init",
    "fit",
    "checkpoint_load",
    "val_best_rerun",
    "test_inference_batched",
}

SUMMARY_GROUPS = (
    "dataset",
    "model",
    "mode",
    "phase",
    "metric",
    "family",
    "unit",
    "source_field",
)

COMPARISON_GROUPS = (
    "dataset",
    "model",
    "phase",
    "metric",
    "family",
    "unit",
    "source_field",
)


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
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )
    return parser


def _require_columns(
    frame: pd.DataFrame,
    required: set[str],
    source: Path,
) -> None:
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{source} is missing required columns: {missing}")


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() == "true"


def _stats(values: pd.Series | np.ndarray) -> dict[str, float | int]:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return {
            "n": 0,
            "median": np.nan,
            "q1": np.nan,
            "q3": np.nan,
            "mean": np.nan,
            "std": np.nan,
        }
    return {
        "n": int(array.size),
        "median": float(np.median(array)),
        "q1": float(np.quantile(array, 0.25)),
        "q3": float(np.quantile(array, 0.75)),
        "mean": float(np.mean(array)),
        "std": float(np.std(array, ddof=1)) if array.size > 1 else np.nan,
    }


def _unique_numbers(values: pd.Series) -> tuple[int | float, ...]:
    unique = []
    for value in values.dropna().unique():
        numeric = float(value)
        unique.append(int(numeric) if numeric.is_integer() else numeric)
    return tuple(sorted(unique))


def _unique_text(values: pd.Series) -> tuple[str, ...]:
    return tuple(sorted(str(value) for value in values.dropna().unique()))


def _display_values(values: tuple[Any, ...]) -> str:
    return "|".join(str(value) for value in values)


def _metadata_lookups(
    runs: pd.DataFrame,
) -> tuple[
    dict[tuple[str, str, str], tuple[int | float, ...]],
    dict[tuple[str, str, str], tuple[str, ...]],
    dict[tuple[str, str, str], tuple[int | float, ...]],
]:
    params = {}
    gpus = {}
    cpus = {}
    for key, group in runs.groupby(["dataset", "model", "mode"], sort=False):
        params[key] = _unique_numbers(group["model_params_total"])
        gpus[key] = _unique_text(group["gpu"])
        cpus[key] = _unique_numbers(group["cpu_count"])
    return params, gpus, cpus


def _metric_values(phases: pd.DataFrame) -> pd.DataFrame:
    identity = [
        "dataset",
        "model",
        "mode",
        "seed",
        "phase",
        "events_complete",
    ]
    values = []
    for spec in METRICS:
        metric = phases[identity + [spec.source]].copy()
        metric = metric.rename(columns={spec.source: "value"})
        metric["value"] = pd.to_numeric(metric["value"], errors="coerce")
        metric["value"] *= spec.scale
        metric["metric"] = spec.name
        metric["family"] = spec.family
        metric["unit"] = spec.unit
        metric["source_field"] = spec.source
        values.append(metric)
    result = pd.concat(values, ignore_index=True)
    return result.dropna(subset=["value"])


def _summary(values: pd.DataFrame) -> pd.DataFrame:
    records = []
    for key, group in values.groupby(list(SUMMARY_GROUPS), sort=False):
        stats = _stats(group["value"])
        record = dict(zip(SUMMARY_GROUPS, key, strict=True))
        record.update(
            {
                "n_seeds": stats["n"],
                "median": stats["median"],
                "q1": stats["q1"],
                "q3": stats["q3"],
                "mean": stats["mean"],
                "std": stats["std"],
                "incomplete_phase_rows": int(
                    (~group["events_complete"]).sum()
                ),
            }
        )
        records.append(record)
    return pd.DataFrame.from_records(records)


def _recommended_for_main(metric: str, phase: str) -> bool:
    if metric == "time_total_sec":
        return phase in TIME_TOTAL_MAIN_PHASES
    if metric == "time_per_occurrence_sec":
        return phase in REPEATED_PHASES
    if metric in {"cuda_peak_allocated_gib", "cuda_peak_reserved_gib"}:
        return phase in CUDA_MAIN_PHASES
    return False


def _measurement_note(metric: str, phase: str) -> str:
    if metric == "time_total_sec" and phase in REPEATED_PHASES:
        return "Total repeated-phase time overlaps the fit duration."
    if metric == "time_per_occurrence_sec":
        return "Mean occurrence duration within each run, summarized across seeds."
    if metric.startswith("cuda_peak_") and phase == "fit":
        return (
            "Fit-wide CUDA peak is invalidated by nested phase counter resets."
        )
    if metric == "cuda_peak_reserved_gib":
        return (
            "Allocator reservation includes live allocation and cached blocks."
        )
    if metric == "driver_rss_boundary_gib":
        return "Driver-process RSS sampled only at phase boundaries."
    if metric == "tree_rss_boundary_gib":
        return "Aggregate process-tree RSS at boundaries may double-count shared pages."
    return ""


def _paired_values(values: pd.DataFrame) -> pd.DataFrame:
    join_columns = [
        "dataset",
        "model",
        "seed",
        "phase",
        "metric",
        "family",
        "unit",
        "source_field",
    ]
    full = values[values["mode"] == "full"].rename(
        columns={
            "value": "full_value",
            "events_complete": "full_events_complete",
        }
    )
    partitioning = values[values["mode"] == "partitioning"].rename(
        columns={
            "value": "partitioning_value",
            "events_complete": "partitioning_events_complete",
        }
    )
    return full[join_columns + ["full_value", "full_events_complete"]].merge(
        partitioning[
            join_columns
            + ["partitioning_value", "partitioning_events_complete"]
        ],
        on=join_columns,
        how="inner",
        validate="one_to_one",
    )


def _comparison(
    paired: pd.DataFrame,
    runs: pd.DataFrame,
) -> pd.DataFrame:
    params, gpus, cpus = _metadata_lookups(runs)
    records = []
    for key, group in paired.groupby(list(COMPARISON_GROUPS), sort=False):
        record = dict(zip(COMPARISON_GROUPS, key, strict=True))
        full_stats = _stats(group["full_value"])
        partitioning_stats = _stats(group["partitioning_value"])
        delta = group["partitioning_value"] - group["full_value"]
        delta_stats = _stats(delta)
        nonzero = group["full_value"] != 0
        ratio = (
            group.loc[nonzero, "partitioning_value"]
            / group.loc[nonzero, "full_value"]
        )
        ratio_stats = _stats(ratio)
        reduction_stats = _stats((1 - ratio) * 100)

        dataset = record["dataset"]
        model = record["model"]
        full_key = (dataset, model, "full")
        partitioning_key = (dataset, model, "partitioning")
        full_params = params.get(full_key, ())
        partitioning_params = params.get(partitioning_key, ())
        full_gpus = gpus.get(full_key, ())
        partitioning_gpus = gpus.get(partitioning_key, ())
        full_cpus = cpus.get(full_key, ())
        partitioning_cpus = cpus.get(partitioning_key, ())

        record.update(
            {
                "n_pairs": full_stats["n"],
                "full_median": full_stats["median"],
                "full_q1": full_stats["q1"],
                "full_q3": full_stats["q3"],
                "full_mean": full_stats["mean"],
                "full_std": full_stats["std"],
                "partitioning_median": partitioning_stats["median"],
                "partitioning_q1": partitioning_stats["q1"],
                "partitioning_q3": partitioning_stats["q3"],
                "partitioning_mean": partitioning_stats["mean"],
                "partitioning_std": partitioning_stats["std"],
                "paired_delta_median": delta_stats["median"],
                "paired_delta_mean": delta_stats["mean"],
                "paired_delta_std": delta_stats["std"],
                "n_ratio_pairs": ratio_stats["n"],
                "paired_ratio_median": ratio_stats["median"],
                "paired_ratio_q1": ratio_stats["q1"],
                "paired_ratio_q3": ratio_stats["q3"],
                "paired_ratio_mean": ratio_stats["mean"],
                "paired_ratio_std": ratio_stats["std"],
                "paired_reduction_pct_median": reduction_stats["median"],
                "paired_reduction_pct_mean": reduction_stats["mean"],
                "paired_reduction_pct_std": reduction_stats["std"],
                "incomplete_pairs": int(
                    (
                        ~group["full_events_complete"]
                        | ~group["partitioning_events_complete"]
                    ).sum()
                ),
                "full_model_params": _display_values(full_params),
                "partitioning_model_params": _display_values(
                    partitioning_params
                ),
                "same_model_params": full_params == partitioning_params,
                "full_gpu": _display_values(full_gpus),
                "partitioning_gpu": _display_values(partitioning_gpus),
                "same_gpu": full_gpus == partitioning_gpus,
                "full_cpu_count": _display_values(full_cpus),
                "partitioning_cpu_count": _display_values(partitioning_cpus),
                "same_cpu_count": full_cpus == partitioning_cpus,
                "recommended_for_main": _recommended_for_main(
                    record["metric"], record["phase"]
                ),
                "measurement_note": _measurement_note(
                    record["metric"], record["phase"]
                ),
            }
        )
        records.append(record)
    return pd.DataFrame.from_records(records)


def _mean_std(mean: float, std: float) -> str:
    if not np.isfinite(mean):
        return ""
    if not np.isfinite(std):
        return f"{mean:.3f}"
    return f"{mean:.3f} ± {std:.3f}"


def _appendix_table(comparison: pd.DataFrame) -> pd.DataFrame:
    table = comparison.copy()
    table["full_mean_std"] = [
        _mean_std(mean, std)
        for mean, std in zip(
            table["full_mean"], table["full_std"], strict=True
        )
    ]
    table["partitioning_mean_std"] = [
        _mean_std(mean, std)
        for mean, std in zip(
            table["partitioning_mean"],
            table["partitioning_std"],
            strict=True,
        )
    ]
    table["paired_reduction_pct_mean_std"] = [
        _mean_std(mean, std)
        for mean, std in zip(
            table["paired_reduction_pct_mean"],
            table["paired_reduction_pct_std"],
            strict=True,
        )
    ]
    columns = [
        "dataset",
        "model",
        "phase",
        "metric",
        "unit",
        "n_pairs",
        "full_mean_std",
        "partitioning_mean_std",
        "paired_reduction_pct_mean_std",
        "same_model_params",
        "incomplete_pairs",
        "measurement_note",
    ]
    return table[columns]


def _sort(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    ordering = {
        "dataset": DATASET_ORDER,
        "model": MODEL_ORDER,
        "phase": PHASE_ORDER,
        "metric": tuple(spec.name for spec in METRICS),
        "mode": ("full", "partitioning"),
    }
    sort_columns = []
    temporary_columns = []
    for column, categories in ordering.items():
        if column in result.columns:
            temporary = f"__{column}_order"
            order = {value: index for index, value in enumerate(categories)}
            result[temporary] = result[column].map(order).fillna(len(order))
            sort_columns.append(temporary)
            temporary_columns.append(temporary)
    result = result.sort_values(sort_columns).reset_index(drop=True)
    return result.drop(columns=temporary_columns)


def main() -> None:
    args = _build_parser().parse_args()
    runs = pd.read_csv(args.runs)
    phases = pd.read_csv(args.phase_metrics)

    _require_columns(
        runs,
        {
            "dataset",
            "model",
            "mode",
            "seed",
            "model_params_total",
            "gpu",
            "cpu_count",
        },
        args.runs,
    )
    _require_columns(
        phases,
        {
            "dataset",
            "model",
            "mode",
            "seed",
            "phase",
            "events_complete",
            *(spec.source for spec in METRICS),
        },
        args.phase_metrics,
    )

    duplicate_key = ["dataset", "model", "mode", "seed", "phase"]
    duplicates = phases.duplicated(duplicate_key, keep=False)
    if duplicates.any():
        examples = (
            phases.loc[duplicates, duplicate_key].head().to_dict("records")
        )
        raise ValueError(f"Duplicate run-phase rows found: {examples}")

    phases["events_complete"] = phases["events_complete"].map(_as_bool)
    values = _metric_values(phases)
    summary = _sort(_summary(values))
    paired = _paired_values(values)
    comparison = _sort(_comparison(paired, runs))
    plot_medians = comparison[comparison["recommended_for_main"]].copy()
    appendix = _appendix_table(comparison)
    appendix_time = appendix[
        (appendix["metric"] == "time_total_sec")
        | (
            (appendix["metric"] == "time_per_occurrence_sec")
            & appendix["phase"].isin(REPEATED_PHASES)
        )
    ]
    appendix_memory = appendix[
        (
            appendix["metric"].isin(
                {"cuda_peak_allocated_gib", "cuda_peak_reserved_gib"}
            )
            & appendix["phase"].isin(CUDA_MAIN_PHASES)
        )
        | (
            appendix["metric"].isin(
                {"driver_rss_boundary_gib", "tree_rss_boundary_gib"}
            )
            & appendix["phase"].isin(RSS_APPENDIX_PHASES)
        )
    ]

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_dir / "phase_summary.csv", index=False)
    comparison.to_csv(output_dir / "phase_comparison.csv", index=False)
    plot_medians.to_csv(output_dir / "plot_medians.csv", index=False)
    appendix_time.to_csv(output_dir / "appendix_time.csv", index=False)
    appendix_memory.to_csv(output_dir / "appendix_memory.csv", index=False)

    print(f"Runs read: {len(runs)}")
    print(f"Run-phase rows read: {len(phases)}")
    print(f"Summary rows: {len(summary)}")
    print(f"Paired comparison rows: {len(comparison)}")
    print(f"Plot-median rows: {len(plot_medians)}")
    print(
        "Comparison seed counts: "
        f"{sorted(comparison['n_pairs'].unique().tolist())}"
    )
    mismatched_models = comparison.loc[
        ~comparison["same_model_params"], ["dataset", "model"]
    ].drop_duplicates()
    print(
        f"Parameter-mismatched dataset-model pairs: {len(mismatched_models)}"
    )
    print(
        "Comparisons containing incomplete phase markers: "
        f"{int((comparison['incomplete_pairs'] > 0).sum())}"
    )
    for filename in (
        "phase_summary.csv",
        "phase_comparison.csv",
        "plot_medians.csv",
        "appendix_time.csv",
        "appendix_memory.csv",
    ):
        print(output_dir / filename)


if __name__ == "__main__":
    main()
