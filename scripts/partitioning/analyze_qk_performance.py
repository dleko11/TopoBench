#!/usr/bin/env python3
"""Export K-q performance cells and heatmaps from W&B partitioning runs.

Run with ``python -m scripts.partitioning.analyze_qk_performance --project
questions:PROJECT --project amazon_ratings:PROJECT``. CSVs retain anchor runs
outside the plotted grid. Only finished runs with finite test metrics enter
cell means; the latest successful retry is retained for each seed pair.
"""

from __future__ import annotations

import argparse
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import wandb

from scripts.partitioning.extract_phase_tracking import (
    DEFAULT_ENTITY,
    RUN_FIELDS,
    ProjectSpec,
    _run_metadata,
    _write_csv,
)
from scripts.partitioning.summarize_statistical_efficiency import (
    DATASET_METRICS,
    DATASET_NAMES,
    MODEL_NAMES,
)

CELL_FIELDS = (
    "dataset",
    "project",
    "model",
    "num_parts",
    "stream_q",
    "stream_q_val",
    "stream_q_test",
    "split_type",
    "learning_setting",
    "metric",
    "protocol",
    "ensemble_passes",
)
SEED_FIELDS = ("seed", "data_seed")
EXTRA_FIELDS = (
    "split_type",
    "learning_setting",
    "metric",
    "protocol",
    "ensemble_passes",
    "test_metric",
    "validation_metric",
    "usable",
)
SUMMARY_FIELDS = CELL_FIELDS + (
    "n",
    "test_mean",
    "test_std",
    "runtime_mean_sec",
    "run_ids",
)
FIGURE_FIELDS = (
    "dataset",
    "project",
    "model",
    "split_type",
    "learning_setting",
    "metric",
    "protocol",
    "ensemble_passes",
)


def extract_run(spec: ProjectSpec, run: Any) -> list[dict[str, Any]]:
    metric = DATASET_METRICS[spec.dataset]
    metadata = _run_metadata(spec, run)
    split = run.config.get("dataset", {}).get("split_params", {})
    rows = []
    for protocol in ("batched", "ensemble"):
        value = run.summary.get(f"test_inference/{protocol}/{metric}")
        finite = isinstance(value, (int, float)) and math.isfinite(value)
        rows.append(
            {
                **metadata,
                "split_type": split.get("split_type"),
                "learning_setting": split.get("learning_setting"),
                "metric": metric,
                "protocol": protocol,
                "ensemble_passes": metadata["ensemble_runs"]
                if protocol == "ensemble"
                else 1,
                "test_metric": float(value) if finite else None,
                "validation_metric": run.summary.get(
                    f"val_best_rerun/{metric}"
                ),
                "usable": run.state == "finished" and finite,
            }
        )
    return rows


def aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    latest = {}
    for row in sorted(
        rows, key=lambda row: (row["created_at"], row["run_id"])
    ):
        if not row["usable"]:
            continue
        required = CELL_FIELDS + SEED_FIELDS
        if any(row.get(key) is None for key in required):
            raise ValueError(
                f"Run {row['run_id']} is missing grid identity fields."
            )
        latest[tuple(row[key] for key in required)] = row
    groups = defaultdict(list)
    for row in latest.values():
        groups[tuple(row[key] for key in CELL_FIELDS)].append(row)
    result = []
    for key, group in sorted(groups.items()):
        values = [row["test_metric"] for row in group]
        runtimes = [
            row["runtime_sec"]
            for row in group
            if row["runtime_sec"] is not None
        ]
        result.append(
            {
                **dict(zip(CELL_FIELDS, key, strict=True)),
                "n": len(values),
                "test_mean": statistics.mean(values),
                "test_std": statistics.stdev(values)
                if len(values) > 1
                else None,
                "runtime_mean_sec": statistics.mean(runtimes)
                if runtimes
                else None,
                "run_ids": ",".join(row["run_id"] for row in group),
            }
        )
    return result


def plot_cells(
    rows: list[dict[str, Any]], output: Path, ks: list[int], qs: list[int]
) -> list[Path]:
    """One panel per model/protocol; color limits shared within each dataset."""
    groups = defaultdict(list)
    limits = defaultdict(list)
    for row in rows:
        if row["num_parts"] in ks and row["stream_q"] in qs:
            groups[tuple(row[key] for key in FIGURE_FIELDS)].append(row)
            limits[(row["dataset"], row["metric"])].append(
                100 * row["test_mean"]
            )
    paths = []
    plt.rcParams.update(
        {"font.size": 8, "pdf.fonttype": 42, "svg.fonttype": "none"}
    )
    for key, group in sorted(groups.items()):
        identity = dict(zip(FIGURE_FIELDS, key, strict=True))
        values = np.full((len(qs), len(ks)), np.nan)
        counts = np.zeros_like(values, dtype=int)
        for row in group:
            i, j = qs.index(row["stream_q"]), ks.index(row["num_parts"])
            if counts[i, j]:
                raise ValueError(
                    "Conflicting evaluation settings for one K-q cell."
                )
            values[i, j] = 100 * row["test_mean"]
            counts[i, j] = row["n"]
        data_limits = limits[(identity["dataset"], identity["metric"])]
        vmin, vmax = min(data_limits), max(data_limits)
        if vmin == vmax:
            vmin, vmax = max(0, vmin - 0.5), min(100, vmax + 0.5)
        cmap = plt.get_cmap("viridis").copy()
        cmap.set_bad("#eeeeee")
        fig, ax = plt.subplots(figsize=(3.5, 3), layout="constrained")
        mesh = ax.imshow(
            np.ma.masked_invalid(values),
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            origin="lower",
            aspect="auto",
        )
        for i in range(len(qs)):
            for j in range(len(ks)):
                value = values[i, j]
                label = (
                    f"{value:.2f}\nn={counts[i, j]}"
                    if counts[i, j]
                    else "missing"
                )
                color = (
                    "white"
                    if counts[i, j] and (value - vmin) / (vmax - vmin) < 0.5
                    else "black"
                )
                ax.text(
                    j,
                    i,
                    label,
                    ha="center",
                    va="center",
                    color=color,
                    fontsize=7,
                )
        ax.set(
            xticks=range(len(ks)),
            xticklabels=ks,
            yticks=range(len(qs)),
            yticklabels=qs,
            xlabel="Number of clusters K",
            ylabel="Clusters per batch q",
        )
        model = MODEL_NAMES.get(identity["model"], identity["model"])
        dataset = DATASET_NAMES[identity["dataset"]]
        protocol = identity["protocol"]
        if protocol == "ensemble":
            protocol += f" ({identity['ensemble_passes']} passes)"
        ax.set_title(f"{dataset} · {model}\n{protocol}")
        metric_label = "AUROC" if identity["metric"] == "auroc" else "Accuracy"
        fig.colorbar(mesh, ax=ax, label=f"{metric_label} (%)")
        fig.supxlabel(
            "Cell mean; n = distinct (model seed, split seed) pairs",
            fontsize=6,
        )
        name = "_".join(str(identity[field]) for field in FIGURE_FIELDS)
        for suffix in ("pdf", "svg", "png"):
            path = output / f"{name}.{suffix}"
            fig.savefig(path, dpi=300)
            paths.append(path)
        plt.close(fig)
    return paths


def parse_project(value: str) -> ProjectSpec:
    dataset, separator, project = value.partition(":")
    if not separator or not project or dataset not in DATASET_METRICS:
        raise argparse.ArgumentTypeError(
            "Expected DATASET:PROJECT (questions, amazon_ratings, cora_full)."
        )
    return ProjectSpec(dataset, "partitioning", project)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project", action="append", required=True, type=parse_project
    )
    parser.add_argument("--entity", default=DEFAULT_ENTITY)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/qk_performance")
    )
    parser.add_argument("--ks", nargs="+", type=int, default=[32, 64, 128])
    parser.add_argument("--qs", nargs="+", type=int, default=[4, 8, 16])
    args = parser.parse_args()
    if any(value <= 0 for value in args.ks + args.qs) or max(args.qs) > min(
        args.ks
    ):
        parser.error(
            "Grid values must be positive, with q <= K for every cell."
        )
    api = wandb.Api()
    rows = []
    for spec in args.project:
        for run in api.runs(f"{args.entity}/{spec.project}"):
            rows.extend(extract_run(spec, run))
    cells = aggregate(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(args.output_dir / "runs.csv", rows, RUN_FIELDS + EXTRA_FIELDS)
    _write_csv(args.output_dir / "cells.csv", cells, SUMMARY_FIELDS)
    paths = plot_cells(
        cells, args.output_dir, sorted(set(args.ks)), sorted(set(args.qs))
    )
    print(
        f"Exported {len(rows)} run/protocol rows, {len(cells)} cells, {len(paths)} figures to {args.output_dir}"
    )


if __name__ == "__main__":
    main()
