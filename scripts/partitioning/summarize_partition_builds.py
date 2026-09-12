#!/usr/bin/env python3
"""Summarize fresh graph-partition construction measurements."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_INPUT = Path("outputs/partition_build_benchmark/phase_metrics.csv")
DEFAULT_OUTPUT_DIR = Path("outputs/partition_build_benchmark/summary")
MIB_TO_GB = 1024**2 / 1000**3


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase-metrics", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--namespace-prefix",
        help="Only include cache namespaces beginning with this value.",
    )
    parser.add_argument(
        "--expected-repetitions",
        default="0,1,2,3,4",
        help="Comma-separated repetition identifiers stored in the seed field.",
    )
    return parser


def _as_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() == "true"


def _require_columns(frame: pd.DataFrame, columns: set[str], path: Path) -> None:
    missing = sorted(columns - set(frame.columns))
    if missing:
        raise ValueError(f"{path} is missing columns: {missing}")


def _sample_std(values: pd.Series) -> float:
    return float(values.std(ddof=1)) if len(values) > 1 else np.nan


def _summarize(runs: pd.DataFrame) -> pd.DataFrame:
    records = []
    for (dataset, num_parts), group in runs.groupby(
        ["dataset", "num_parts"], sort=True
    ):
        records.append(
            {
                "dataset": dataset,
                "num_parts": int(num_parts),
                "n": len(group),
                "repetitions": ",".join(
                    str(value) for value in sorted(group["repetition"])
                ),
                "partition_time_mean_sec": float(
                    group["partition_time_sec"].mean()
                ),
                "partition_time_std_sec": _sample_std(
                    group["partition_time_sec"]
                ),
                "partition_time_median_sec": float(
                    group["partition_time_sec"].median()
                ),
                "peak_cpu_memory_mean_gb": float(
                    group["peak_cpu_memory_gb"].mean()
                ),
                "peak_cpu_memory_std_gb": _sample_std(
                    group["peak_cpu_memory_gb"]
                ),
                "peak_cpu_memory_median_gb": float(
                    group["peak_cpu_memory_gb"].median()
                ),
            }
        )
    return pd.DataFrame.from_records(records)


def main() -> None:
    args = _build_parser().parse_args()
    expected_repetitions = {
        int(value.strip())
        for value in args.expected_repetitions.split(",")
        if value.strip()
    }
    if not expected_repetitions:
        raise ValueError("--expected-repetitions must not be empty.")

    phases = pd.read_csv(args.phase_metrics)
    required = {
        "dataset",
        "mode",
        "model",
        "seed",
        "num_parts",
        "partition_cache_namespace",
        "run_id",
        "run_name",
        "phase",
        "events_complete",
        "duration_total_sec",
        "rss_peak_max_mb",
    }
    _require_columns(phases, required, args.phase_metrics)

    selected = phases[
        (phases["mode"] == "partitioning")
        & (phases["model"] == "gcn")
        & (phases["phase"] == "partition_build")
    ].copy()
    if args.namespace_prefix:
        selected = selected[
            selected["partition_cache_namespace"]
            .fillna("")
            .str.startswith(args.namespace_prefix)
        ]
    if selected.empty:
        raise ValueError("No matching partition_build phase rows found.")
    if not selected["events_complete"].map(_as_bool).all():
        raise ValueError("At least one partition_build phase is incomplete.")

    for column in ("seed", "num_parts", "duration_total_sec", "rss_peak_max_mb"):
        selected[column] = pd.to_numeric(selected[column], errors="raise")
    if selected[["duration_total_sec", "rss_peak_max_mb"]].isna().any().any():
        raise ValueError("Partition time or peak CPU memory is missing.")

    selected["repetition"] = selected["seed"].astype(int)
    selected["num_parts"] = selected["num_parts"].astype(int)
    duplicate_key = ["dataset", "num_parts", "repetition"]
    duplicates = selected.duplicated(duplicate_key, keep=False)
    if duplicates.any():
        examples = selected.loc[duplicates, duplicate_key].to_dict("records")
        raise ValueError(f"Duplicate partition builds found: {examples[:5]}")

    for key, group in selected.groupby(["dataset", "num_parts"], sort=True):
        repetitions = set(group["repetition"])
        if repetitions != expected_repetitions:
            raise ValueError(
                f"Unexpected repetitions for {key}: {sorted(repetitions)}; "
                f"expected {sorted(expected_repetitions)}."
            )

    runs = selected[
        [
            "dataset",
            "num_parts",
            "repetition",
            "duration_total_sec",
            "rss_peak_max_mb",
            "partition_cache_namespace",
            "run_id",
            "run_name",
        ]
    ].rename(columns={"duration_total_sec": "partition_time_sec"})
    runs["peak_cpu_memory_gb"] = runs.pop("rss_peak_max_mb") * MIB_TO_GB
    runs = runs.sort_values(
        ["dataset", "num_parts", "repetition"]
    ).reset_index(drop=True)
    summary = _summarize(runs)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    runs_path = args.output_dir / "partition_build_runs.csv"
    summary_path = args.output_dir / "partition_build_summary.csv"
    runs.to_csv(runs_path, index=False)
    summary.to_csv(summary_path, index=False)

    print(
        "| Dataset | K | n | Partition time (s), mean +/- std | "
        "Peak CPU memory (GB), mean +/- std |"
    )
    print("|---|---:|---:|---:|---:|")
    for row in summary.itertuples(index=False):
        print(
            f"| {row.dataset} | {row.num_parts} | {row.n} | "
            f"{row.partition_time_mean_sec:.2f} +/- "
            f"{row.partition_time_std_sec:.2f} | "
            f"{row.peak_cpu_memory_mean_gb:.3f} +/- "
            f"{row.peak_cpu_memory_std_gb:.3f} |"
        )
    print(runs_path)
    print(summary_path)


if __name__ == "__main__":
    main()
