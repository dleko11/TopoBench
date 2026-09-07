#!/usr/bin/env python3
"""Aggregate and plot structural observability span histograms."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import wandb
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter

from scripts.partitioning.structural_observability import (
    CURVE_FIELDS,
    HISTOGRAM_FIELDS,
    THRESHOLD_FIELDS,
    observability_rows,
    write_csv,
)

DATASET_ORDER = (
    "amazon_ratings",
    "questions",
    "cora_full",
    "coauthor_physics",
    "reddit",
)
DATASET_LABELS = {
    "amazon_ratings": "Amazon Ratings",
    "questions": "Questions",
    "cora_full": "Cora Full",
    "coauthor_physics": "Coauthor Physics",
    "reddit": "Reddit",
}
FAMILY_ORDER = ("hypergraph", "cell", "simplicial")
FAMILY_LABELS = {
    "hypergraph": "Hyperedges",
    "cell": "Cycle-basis cells",
    "simplicial": "Clique 2-simplices",
}
COLORS = {
    "hypergraph": "#2878B5",
    "cell": "#D97A2B",
    "simplicial": "#7656A4",
}
MARKERS = {
    "hypergraph": "o",
    "cell": "s",
    "simplicial": "^",
}


def download_histograms(
    *,
    entity: str,
    project: str,
    output_dir: Path,
) -> int:
    """Download the latest histogram artifact for each W&B run config."""
    api = wandb.Api()
    project_path = f"{entity}/{project}"
    artifact_names = set()
    for run in api.runs(project_path):
        if run.job_type != "structure_span_count":
            continue
        dataset = run.config.get("dataset")
        family = run.config.get("family")
        num_parts = run.config.get("num_parts")
        if dataset is None or family is None or num_parts is None:
            continue
        artifact_names.add(
            (
                str(dataset),
                str(family),
                int(num_parts),
                f"structural-observability-{dataset}-{family}-k{num_parts}",
            )
        )
    if not artifact_names:
        raise RuntimeError(
            f"No structural observability artifacts found in {project_path}."
        )

    for dataset, _family, num_parts, artifact_name in sorted(artifact_names):
        artifact = api.artifact(
            f"{project_path}/{artifact_name}:latest",
            type="structural-observability",
        )
        artifact.download(root=str(output_dir / f"{dataset}_k{num_parts}"))
    return len(artifact_names)


def _read_histograms(input_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(input_dir.glob("*_k*/span_histogram_*.csv")):
        with path.open(encoding="utf-8", newline="") as handle:
            rows.extend(csv.DictReader(handle))
    if not rows:
        raise FileNotFoundError(
            f"No span_histogram_*.csv files found under {input_dir}."
        )
    return rows


def aggregate_histograms(
    histogram_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return observability curves and thresholds for every histogram."""
    grouped: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(
        list
    )
    parts_by_dataset: dict[str, set[int]] = defaultdict(set)
    for row in histogram_rows:
        dataset = str(row["dataset"])
        family = str(row["family"])
        num_parts = int(row["num_parts"])
        grouped[(dataset, family, num_parts)].append(row)
        parts_by_dataset[dataset].add(num_parts)

    ambiguous = {
        dataset: sorted(parts)
        for dataset, parts in parts_by_dataset.items()
        if len(parts) > 1
    }
    if ambiguous:
        raise ValueError(
            f"Each dataset must have one K value for this figure: {ambiguous}"
        )

    curves = []
    thresholds = []
    for key in sorted(grouped):
        group_curves, group_threshold = observability_rows(grouped[key])
        curves.extend(group_curves)
        thresholds.append(group_threshold)
    return curves, thresholds


def _configure_matplotlib() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [
                "Arial",
                "Helvetica",
                "DejaVu Sans",
                "sans-serif",
            ],
            "font.size": 7.2,
            "axes.titlesize": 8.2,
            "axes.labelsize": 7.7,
            "xtick.labelsize": 6.7,
            "ytick.labelsize": 6.7,
            "axes.linewidth": 0.7,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def _save_figure(fig: plt.Figure, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=600, bbox_inches="tight")
    plt.close(fig)


def plot_observability(
    curves: list[dict[str, Any]],
    thresholds: list[dict[str, Any]],
    output_stem: Path,
) -> None:
    """Plot observable structure fraction against sampled graph fraction."""
    _configure_matplotlib()
    curve_groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(
        list
    )
    for row in curves:
        curve_groups[(str(row["dataset"]), str(row["family"]))].append(row)
    threshold_map = {
        (str(row["dataset"]), str(row["family"])): row for row in thresholds
    }

    datasets = [
        dataset
        for dataset in DATASET_ORDER
        if any(key[0] == dataset for key in curve_groups)
    ]
    if len(datasets) > 5:
        raise ValueError("The compact layout supports at most five datasets.")

    fig, axes_array = plt.subplots(
        2,
        3,
        figsize=(7.2, 4.25),
        sharex=True,
        sharey=True,
    )
    axes = list(axes_array.flat)
    for index, (ax, dataset) in enumerate(zip(axes, datasets, strict=False)):
        available_families = [
            family
            for family in FAMILY_ORDER
            if (dataset, family) in curve_groups
        ]
        num_parts_values = {
            int(row["num_parts"])
            for family in available_families
            for row in curve_groups[(dataset, family)]
        }
        if len(num_parts_values) != 1:
            raise ValueError(f"Inconsistent K values for {dataset}.")
        num_parts = num_parts_values.pop()

        for family in available_families:
            family_rows = sorted(
                curve_groups[(dataset, family)],
                key=lambda row: int(row["q"]),
            )
            x = [float(row["q_over_k_percent"]) for row in family_rows]
            y = [float(row["observable_fraction"]) for row in family_rows]
            ax.plot(
                x,
                y,
                color=COLORS[family],
                linewidth=1.55,
                solid_capstyle="round",
                zorder=2,
            )
            threshold = threshold_map[(dataset, family)]
            q_95 = int(threshold["q_95"])
            threshold_row = family_rows[q_95 - 1]
            ax.scatter(
                float(threshold_row["q_over_k_percent"]),
                float(threshold_row["observable_fraction"]),
                color=COLORS[family],
                marker=MARKERS[family],
                s=20,
                edgecolor="white",
                linewidth=0.55,
                zorder=3,
            )

        letter = chr(ord("a") + index)
        ax.set_title(
            f"({letter})  {DATASET_LABELS[dataset]}  ($K={num_parts:,}$)",
            loc="left",
            fontweight="semibold",
            pad=6,
        )
        ax.axhline(0.95, color="#9CA3AF", linewidth=0.7, linestyle="--")
        ax.grid(axis="y", color="#E5E7EB", linewidth=0.55)
        ax.set_axisbelow(True)
        ax.set_xscale("log")
        ax.set_xlim(0.009, 110)
        ax.set_ylim(0, 1.025)
        ax.set_yticks((0, 0.25, 0.5, 0.75, 0.95, 1.0))
        ax.yaxis.set_major_formatter(PercentFormatter(1.0))
        ax.tick_params(direction="out", length=3, width=0.7)

    for ax in axes[len(datasets) : -1]:
        ax.set_visible(False)

    legend_ax = axes[-1]
    legend_ax.axis("off")
    handles = [
        Line2D(
            [0],
            [0],
            color=COLORS[family],
            marker=MARKERS[family],
            markeredgecolor="white",
            markeredgewidth=0.55,
            linewidth=1.55,
            label=FAMILY_LABELS[family],
        )
        for family in FAMILY_ORDER
    ]
    handles.append(
        Line2D(
            [0],
            [0],
            color="#9CA3AF",
            linestyle="--",
            linewidth=0.8,
            label="95% observable",
        )
    )
    legend_ax.legend(
        handles=handles,
        loc="center",
        title="Full-graph structures",
        title_fontsize=7.4,
        handlelength=2.4,
        labelspacing=0.9,
    )

    for row in range(2):
        axes[row * 3].set_ylabel("Observable structures")
    for ax in axes[3:5]:
        ax.set_xlabel(r"Sampled partitions, $q/K$ (%)")
    fig.subplots_adjust(
        left=0.08,
        right=0.985,
        top=0.96,
        bottom=0.11,
        wspace=0.2,
        hspace=0.42,
    )
    _save_figure(fig, output_stem)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("outputs/structural_observability"),
    )
    parser.add_argument(
        "--output-stem",
        type=Path,
        default=Path(
            "outputs/structural_observability/structural_observability"
        ),
    )
    parser.add_argument("--wandb-project")
    parser.add_argument("--wandb-entity", default="topobench-scalability")
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    if args.wandb_project:
        count = download_histograms(
            entity=args.wandb_entity,
            project=args.wandb_project,
            output_dir=args.input_dir,
        )
        print(f"Downloaded {count} histogram artifacts.")
    histogram_rows = _read_histograms(args.input_dir)
    curves, thresholds = aggregate_histograms(histogram_rows)
    write_csv(
        histogram_rows,
        args.input_dir / "structure_span_histograms.csv",
        HISTOGRAM_FIELDS,
    )
    write_csv(
        curves,
        args.input_dir / "structural_observability_curves.csv",
        CURVE_FIELDS,
    )
    write_csv(
        thresholds,
        args.input_dir / "structural_observability_thresholds.csv",
        THRESHOLD_FIELDS,
    )
    plot_observability(curves, thresholds, args.output_stem)
    print(args.output_stem)


if __name__ == "__main__":
    main()
