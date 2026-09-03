#!/usr/bin/env python3
"""Summarize and plot resource peaks from a partitioning Q/K pilot."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D

DEFAULT_INPUT_DIR = Path("outputs/cora_qk_pilot/raw")
DEFAULT_OUTPUT_DIR = Path("outputs/cora_qk_pilot/analysis")

MODEL_ORDER = ("gcn", "edgnn", "unignn", "cwn", "topotune", "scn", "sccnn")
MODEL_LABELS = {
    "gcn": "GCN",
    "edgnn": "EDGNN",
    "unignn": "UniGNN",
    "cwn": "CWN",
    "topotune": "Cell TopoTune",
    "scn": "SCN",
    "sccnn": "SCCNN",
}
MODEL_COLORS = {
    "gcn": "#747474",
    "edgnn": "#4D9693",
    "unignn": "#71925B",
    "cwn": "#CE833E",
    "topotune": "#896DA7",
    "scn": "#426F9F",
    "sccnn": "#B45D5D",
}
K_COLORS = {
    32: "#B9CBE0",
    64: "#809FC3",
    128: "#4F76A4",
    256: "#244E7C",
}
K_MARKERS = {32: "o", 64: "s", 128: "^", 256: "D"}
HIGHER_ORDER_MODELS = ("cwn", "topotune", "scn", "sccnn")
MEASURED_PHASES = ("train_epoch", "validation_epoch")
PEAK_METRICS = {
    "cuda_peak_allocated_gib": "cuda_peak_allocated_max_mb",
    "cuda_peak_reserved_gib": "cuda_peak_reserved_max_mb",
    "driver_rss_peak_gib": "rss_peak_max_mb",
    "tree_rss_peak_gib": "tree_rss_peak_max_mb",
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
        "--full-runs",
        type=Path,
        help="runs.csv extracted from the matching full-graph pilot",
    )
    parser.add_argument(
        "--full-phase-metrics",
        type=Path,
        help="phase_metrics.csv extracted from the matching full-graph pilot",
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


def _validate_inputs(
    runs: pd.DataFrame,
    phases: pd.DataFrame,
    runs_path: Path,
    phases_path: Path,
    *,
    require_partition_metadata: bool,
) -> pd.DataFrame:
    run_columns = {
        "run_id",
        "run_name",
        "run_state",
        "model",
        "seed",
    }
    if require_partition_metadata:
        run_columns.update({"num_parts", "stream_q"})
    _require_columns(
        runs,
        run_columns,
        runs_path,
    )
    _require_columns(
        phases,
        {
            "run_id",
            "phase",
            "end_event_count",
            "duration_total_sec",
            *PEAK_METRICS.values(),
        },
        phases_path,
    )

    if not (runs["run_state"] == "finished").all():
        states = runs["run_state"].value_counts().to_dict()
        raise ValueError(f"Expected only finished runs, found: {states}")
    if runs["run_id"].duplicated().any():
        raise ValueError("runs.csv contains duplicate run IDs.")

    measured = phases[phases["phase"].isin(MEASURED_PHASES)].copy()
    expected_rows = len(runs) * len(MEASURED_PHASES)
    if len(measured) != expected_rows:
        raise ValueError(
            f"Expected {expected_rows} train/validation rows, found {len(measured)}."
        )
    if measured.duplicated(["run_id", "phase"]).any():
        raise ValueError("Found duplicate train/validation phase rows.")
    if not (measured["end_event_count"] == 1).all():
        counts = measured["end_event_count"].value_counts().to_dict()
        raise ValueError(
            f"Expected one phase-end event per row, found: {counts}"
        )
    if measured[list(PEAK_METRICS.values())].isna().any().any():
        missing = measured[list(PEAK_METRICS.values())].isna().sum()
        raise ValueError(
            f"Peak metric values are missing:\n{missing[missing > 0]}"
        )

    run_ids = set(runs["run_id"])
    if set(measured["run_id"]) != run_ids:
        raise ValueError("Run IDs in the phase table do not match runs.csv.")
    return measured


def _phase_at_max(group: pd.DataFrame, source: str) -> str:
    return str(group.loc[group[source].idxmax(), "phase"])


def build_resource_table(
    runs: pd.DataFrame,
    measured: pd.DataFrame,
    *,
    pipeline: str,
) -> pd.DataFrame:
    """Return one resource-summary row per pilot run."""
    records = []
    phase_groups = {
        run_id: group for run_id, group in measured.groupby("run_id")
    }
    for run in runs.itertuples(index=False):
        group = phase_groups[run.run_id]
        times = group.set_index("phase")["duration_total_sec"]
        if pipeline == "partitioning":
            num_parts = int(run.num_parts)
            q = int(run.stream_q)
            graph_fraction = q / num_parts
        elif pipeline == "full_graph":
            num_parts = None
            q = None
            graph_fraction = 1.0
        else:
            raise ValueError(f"Unknown pipeline: {pipeline}")
        record = {
            "pipeline": pipeline,
            "model": run.model,
            "model_label": MODEL_LABELS.get(run.model, run.model),
            "seed": int(run.seed),
            "num_parts": num_parts,
            "q": q,
            "graph_fraction": graph_fraction,
            "graph_fraction_pct": 100 * graph_fraction,
            "train_epoch_time_sec": float(times["train_epoch"]),
            "validation_epoch_time_sec": float(times["validation_epoch"]),
        }
        for output, source in PEAK_METRICS.items():
            record[output] = float(group[source].max()) / 1024
            record[f"{output}_phase"] = _phase_at_max(group, source)
        records.append(record)

    table = pd.DataFrame.from_records(records)
    model_rank = {model: index for index, model in enumerate(MODEL_ORDER)}
    table["_model_rank"] = table["model"].map(model_rank)
    if table["_model_rank"].isna().any():
        unknown = sorted(
            table.loc[table["_model_rank"].isna(), "model"].unique()
        )
        raise ValueError(f"Unknown models: {unknown}")

    table = table.sort_values(
        ["_model_rank", "graph_fraction", "num_parts", "seed"]
    ).drop(columns="_model_rank")
    duplicate_columns = ["pipeline", "model", "seed"]
    if pipeline == "partitioning":
        duplicate_columns.extend(["num_parts", "q"])
    if table.duplicated(duplicate_columns).any():
        raise ValueError(f"Found duplicate {pipeline} runs.")
    return table.reset_index(drop=True)


def _validate_full_baseline(
    partitioned: pd.DataFrame,
    full_graph: pd.DataFrame,
) -> None:
    counts = full_graph["model"].value_counts()
    if set(counts.index) != set(MODEL_ORDER) or not (counts == 1).all():
        raise ValueError(
            "Expected exactly one full-graph run for every pilot model, found: "
            f"{counts.to_dict()}"
        )
    partition_seeds = set(partitioned["seed"].unique())
    full_seeds = set(full_graph["seed"].unique())
    if partition_seeds != full_seeds:
        raise ValueError(
            "Partitioned and full-graph seed sets differ: "
            f"{sorted(partition_seeds)} != {sorted(full_seeds)}"
        )


def build_model_summary(table: pd.DataFrame) -> pd.DataFrame:
    """Return worst-case resource peaks and their combinations per model."""
    records = []
    for model in MODEL_ORDER:
        group = table[table["model"] == model]
        if group.empty:
            continue
        gpu_row = group.loc[group["cuda_peak_reserved_gib"].idxmax()]
        cpu_row = group.loc[group["tree_rss_peak_gib"].idxmax()]
        records.append(
            {
                "model": model,
                "model_label": MODEL_LABELS[model],
                "gpu_peak_reserved_gib": gpu_row["cuda_peak_reserved_gib"],
                "gpu_peak_num_parts": int(gpu_row["num_parts"]),
                "gpu_peak_q": int(gpu_row["q"]),
                "gpu_peak_fraction_pct": gpu_row["graph_fraction_pct"],
                "cpu_tree_rss_peak_gib": cpu_row["tree_rss_peak_gib"],
                "cpu_peak_num_parts": int(cpu_row["num_parts"]),
                "cpu_peak_q": int(cpu_row["q"]),
                "cpu_peak_fraction_pct": cpu_row["graph_fraction_pct"],
            }
        )
    return pd.DataFrame.from_records(records)


def _matrix(
    table: pd.DataFrame,
    combinations: list[tuple[int, int]],
    metric: str,
) -> np.ndarray:
    values = table.set_index(["model", "num_parts", "q"])[metric]
    matrix = np.full((len(MODEL_ORDER), len(combinations)), np.nan)
    for row, model in enumerate(MODEL_ORDER):
        for column, (num_parts, q) in enumerate(combinations):
            key = (model, num_parts, q)
            if key in values.index:
                matrix[row, column] = float(values.loc[key])
    if np.isnan(matrix).any():
        raise ValueError(f"The {metric} plotting grid is incomplete.")
    return matrix


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
            "font.size": 7,
            "axes.linewidth": 0.7,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "legend.frameon": False,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )


def _save_figure(fig: plt.Figure, stem: Path) -> None:
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=400, bbox_inches="tight")
    plt.close(fig)


def _annotate_heatmap(
    ax: plt.Axes, matrix: np.ndarray, maximum: float
) -> None:
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value = matrix[row, column]
            color = "white" if value / maximum > 0.58 else "#252525"
            ax.text(
                column,
                row,
                f"{value:.1f}",
                ha="center",
                va="center",
                color=color,
                fontsize=4.6,
            )


def plot_resource_heatmaps(table: pd.DataFrame, output_dir: Path) -> None:
    """Plot aligned GPU and CPU resource heatmaps."""
    combinations = sorted(
        {
            (int(row.num_parts), int(row.q))
            for row in table[["num_parts", "q"]].itertuples(index=False)
        },
        key=lambda item: (item[0], item[1] / item[0]),
    )
    gpu = _matrix(table, combinations, "cuda_peak_reserved_gib")
    cpu = _matrix(table, combinations, "tree_rss_peak_gib")

    gpu_cmap = LinearSegmentedColormap.from_list(
        "gpu_memory", ["#F7F9FC", "#A9C4DF", "#245A8D"]
    )
    cpu_cmap = LinearSegmentedColormap.from_list(
        "cpu_memory", ["#F6FAF8", "#A8CDBE", "#2E715E"]
    )

    fig = plt.figure(figsize=(7.2, 4.7), layout="constrained")
    grid = fig.add_gridspec(
        2, 2, width_ratios=(1, 0.025), hspace=0.16, wspace=0.03
    )
    axes = [fig.add_subplot(grid[index, 0]) for index in range(2)]
    color_axes = [fig.add_subplot(grid[index, 1]) for index in range(2)]
    panels = (
        (gpu, gpu_cmap, "a", "Peak GPU memory", "GiB"),
        (cpu, cpu_cmap, "b", "Peak CPU memory", "GiB"),
    )

    group_sizes: dict[int, int] = {}
    for num_parts, _ in combinations:
        group_sizes[num_parts] = group_sizes.get(num_parts, 0) + 1

    for index, (ax, cax, panel) in enumerate(
        zip(axes, color_axes, panels, strict=True)
    ):
        matrix, cmap, letter, title, colorbar_label = panel
        maximum = float(np.ceil(matrix.max()))
        image = ax.imshow(
            matrix, cmap=cmap, vmin=0, vmax=maximum, aspect="auto"
        )
        _annotate_heatmap(ax, matrix, maximum)
        colorbar = fig.colorbar(image, cax=cax)
        colorbar.set_label(colorbar_label, rotation=90, labelpad=5)
        colorbar.outline.set_linewidth(0.6)
        colorbar.ax.tick_params(labelsize=6, length=2)

        ax.set_yticks(
            range(len(MODEL_ORDER)), [MODEL_LABELS[m] for m in MODEL_ORDER]
        )
        ax.set_title(
            f"{letter}   {title} (GiB)",
            loc="left",
            fontweight="bold",
            pad=25,
        )
        for spine in ax.spines.values():
            spine.set_visible(False)

        offset = 0
        for num_parts, size in group_sizes.items():
            if offset:
                ax.axvline(offset - 0.5, color="white", linewidth=2.0)
            center = offset + (size - 1) / 2
            if index == 0:
                ax.text(
                    center,
                    1.015,
                    f"K = {num_parts}",
                    transform=ax.get_xaxis_transform(),
                    ha="center",
                    va="bottom",
                    fontsize=6.5,
                    fontweight="bold",
                )
            offset += size

    labels = [
        f"q={q}\n{100 * q / num_parts:g}%" for num_parts, q in combinations
    ]
    axes[0].set_xticks(range(len(combinations)), [])
    axes[1].set_xticks(range(len(combinations)), labels)

    output_dir.mkdir(parents=True, exist_ok=True)
    _save_figure(fig, output_dir / "cora_qk_resource_heatmaps")


def plot_fraction_scaling(
    table: pd.DataFrame,
    output_dir: Path,
    full_graph: pd.DataFrame | None = None,
) -> None:
    """Plot median resource scaling and the range across K values."""
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(7.2, 3.1),
        layout="constrained",
        sharex=True,
    )
    panels = (
        ("cuda_peak_reserved_gib", "a", "Peak GPU memory (GiB)"),
        ("tree_rss_peak_gib", "b", "Peak CPU memory (GiB)"),
    )
    for ax, (metric, letter, label) in zip(axes, panels, strict=True):
        for model in MODEL_ORDER:
            group = table[table["model"] == model]
            stats = (
                group.groupby("graph_fraction_pct")[metric]
                .agg(["median", "min", "max"])
                .reset_index()
            )
            x = stats["graph_fraction_pct"].to_numpy()
            color = MODEL_COLORS[model]
            ax.fill_between(
                x,
                stats["min"].to_numpy(),
                stats["max"].to_numpy(),
                color=color,
                alpha=0.11,
                linewidth=0,
            )
            ax.plot(
                x,
                stats["median"],
                color=color,
                marker="o",
                markersize=3.1,
                linewidth=1.35,
                label=MODEL_LABELS[model],
            )
            if full_graph is not None:
                full_value = float(
                    full_graph.loc[full_graph["model"] == model, metric].iloc[
                        0
                    ]
                )
                ax.plot(
                    [x[-1], 100],
                    [stats["median"].iloc[-1], full_value],
                    color=color,
                    linewidth=0.9,
                    linestyle=(0, (2, 2)),
                )
                ax.scatter(
                    100,
                    full_value,
                    color=color,
                    edgecolor="#252525",
                    linewidth=0.45,
                    marker="D",
                    s=18,
                    zorder=3,
                )
        ax.set_title(letter, loc="left", fontweight="bold")
        ax.set_ylabel(label)
        ax.set_xlabel("Graph fraction q/K (%)")
        ticks = [12.5, 25, 50, 75]
        if full_graph is not None:
            ticks.append(100)
        ax.set_xticks(ticks)
        ax.grid(axis="y", color="#DADADA", linewidth=0.55)
        ax.set_axisbelow(True)

    handles = [
        Line2D(
            [0],
            [0],
            color=MODEL_COLORS[model],
            marker="o",
            markersize=3.5,
            linewidth=1.4,
            label=MODEL_LABELS[model],
        )
        for model in MODEL_ORDER
    ]
    if full_graph is not None:
        handles.append(
            Line2D(
                [0],
                [0],
                color="#252525",
                marker="D",
                markerfacecolor="white",
                linestyle="none",
                markersize=4,
                label="Full graph",
            )
        )
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.08),
        ncol=4,
        columnspacing=1.3,
        handlelength=1.6,
    )
    _save_figure(fig, output_dir / "cora_qk_fraction_scaling")


def plot_k_sensitivity(
    table: pd.DataFrame,
    output_dir: Path,
    *,
    metric: str,
    ylabel: str,
    filename: str,
    full_graph: pd.DataFrame | None = None,
) -> None:
    """Plot K-specific scaling curves in one panel per model."""
    fig, axes = plt.subplots(
        2,
        4,
        figsize=(7.2, 4.15),
        sharex=True,
        sharey=True,
        layout="constrained",
    )
    flat_axes = axes.ravel()
    for ax, model in zip(flat_axes, MODEL_ORDER, strict=False):
        group = table[table["model"] == model]
        for num_parts in sorted(group["num_parts"].unique()):
            values = group[group["num_parts"] == num_parts].sort_values(
                "graph_fraction_pct"
            )
            ax.plot(
                values["graph_fraction_pct"],
                values[metric],
                color=K_COLORS[int(num_parts)],
                marker="o",
                markersize=2.8,
                linewidth=1.25,
            )
        if full_graph is not None:
            full_value = float(
                full_graph.loc[full_graph["model"] == model, metric].iloc[0]
            )
            ax.axhline(
                full_value,
                color="#555555",
                linewidth=0.9,
                linestyle=(0, (3, 2)),
            )
        ax.set_title(MODEL_LABELS[model], fontweight="bold", pad=4)
        ax.set_xticks([12.5, 25, 50, 75])
        ax.grid(axis="y", color="#DEDEDE", linewidth=0.5)
        ax.set_axisbelow(True)

    legend_ax = flat_axes[-1]
    legend_ax.axis("off")
    handles = [
        Line2D(
            [0],
            [0],
            color=K_COLORS[num_parts],
            marker="o",
            markersize=3.5,
            linewidth=1.4,
            label=f"K = {num_parts}",
        )
        for num_parts in K_COLORS
    ]
    if full_graph is not None:
        handles.append(
            Line2D(
                [0],
                [0],
                color="#555555",
                linewidth=0.9,
                linestyle=(0, (3, 2)),
                label="Full graph",
            )
        )
    legend_ax.legend(handles=handles, loc="center", title="Partitions")
    fig.supxlabel("Graph fraction q/K (%)")
    fig.supylabel(ylabel)
    _save_figure(fig, output_dir / filename)


def _point_size(fraction_pct: float) -> float:
    return 12 + 0.8 * fraction_pct


def plot_resource_plane(
    table: pd.DataFrame,
    output_dir: Path,
    full_graph: pd.DataFrame | None = None,
) -> None:
    """Plot the joint CPU/GPU resource envelope of higher-order models."""
    selected = table[table["model"].isin(HIGHER_ORDER_MODELS)]
    fig, ax = plt.subplots(figsize=(7.2, 3.65))
    for model in HIGHER_ORDER_MODELS:
        group = selected[selected["model"] == model]
        for num_parts in sorted(group["num_parts"].unique()):
            values = group[group["num_parts"] == num_parts]
            ax.scatter(
                values["tree_rss_peak_gib"],
                values["cuda_peak_reserved_gib"],
                s=[
                    _point_size(value)
                    for value in values["graph_fraction_pct"]
                ],
                marker=K_MARKERS[int(num_parts)],
                color=MODEL_COLORS[model],
                edgecolor="white",
                linewidth=0.45,
                alpha=0.88,
            )
        if full_graph is not None:
            full_row = full_graph[full_graph["model"] == model].iloc[0]
            ax.scatter(
                full_row["tree_rss_peak_gib"],
                full_row["cuda_peak_reserved_gib"],
                s=58,
                marker="*",
                color=MODEL_COLORS[model],
                edgecolor="#252525",
                linewidth=0.55,
                zorder=4,
            )

    ax.set_xlabel("Peak CPU memory (GiB)")
    ax.set_ylabel("Peak GPU memory (GiB)")
    ax.grid(color="#DDDDDD", linewidth=0.5)
    ax.set_axisbelow(True)

    model_handles = [
        Line2D(
            [0],
            [0],
            color=MODEL_COLORS[model],
            marker="o",
            linestyle="none",
            markersize=5,
            label=MODEL_LABELS[model],
        )
        for model in HIGHER_ORDER_MODELS
    ]
    model_legend = ax.legend(
        handles=model_handles,
        title="Model",
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
    )
    ax.add_artist(model_legend)

    k_handles = [
        Line2D(
            [0],
            [0],
            color="#555555",
            marker=marker,
            linestyle="none",
            markersize=4.8,
            label=str(num_parts),
        )
        for num_parts, marker in K_MARKERS.items()
    ]
    if full_graph is not None:
        k_handles.append(
            Line2D(
                [0],
                [0],
                color="#555555",
                marker="*",
                linestyle="none",
                markersize=6,
                label="Full graph",
            )
        )
    k_legend = ax.legend(
        handles=k_handles,
        title="K",
        loc="center left",
        bbox_to_anchor=(1.01, 0.48),
        ncol=2,
        columnspacing=0.8,
    )
    ax.add_artist(k_legend)

    fraction_handles = [
        ax.scatter(
            [],
            [],
            s=_point_size(fraction),
            color="#858585",
            edgecolor="white",
            linewidth=0.45,
            label=f"{fraction:g}%",
        )
        for fraction in (12.5, 50, 75)
    ]
    ax.legend(
        handles=fraction_handles,
        title="q/K",
        loc="lower left",
        bbox_to_anchor=(1.01, 0.0),
        ncol=3,
        columnspacing=0.5,
        handletextpad=0.2,
    )
    fig.subplots_adjust(right=0.76)
    _save_figure(fig, output_dir / "cora_qk_resource_plane")


def main() -> None:
    args = _build_parser().parse_args()
    if (args.full_runs is None) != (args.full_phase_metrics is None):
        raise ValueError(
            "--full-runs and --full-phase-metrics must be provided together."
        )
    _configure_matplotlib()
    runs = pd.read_csv(args.runs)
    phases = pd.read_csv(args.phase_metrics)
    measured = _validate_inputs(
        runs,
        phases,
        args.runs,
        args.phase_metrics,
        require_partition_metadata=True,
    )
    table = build_resource_table(
        runs,
        measured,
        pipeline="partitioning",
    )
    summary = build_model_summary(table)
    full_graph = None
    if args.full_runs is not None and args.full_phase_metrics is not None:
        full_runs = pd.read_csv(args.full_runs)
        full_phases = pd.read_csv(args.full_phase_metrics)
        full_measured = _validate_inputs(
            full_runs,
            full_phases,
            args.full_runs,
            args.full_phase_metrics,
            require_partition_metadata=False,
        )
        full_graph = build_resource_table(
            full_runs,
            full_measured,
            pipeline="full_graph",
        )
        _validate_full_baseline(table, full_graph)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    table_path = args.output_dir / "resource_peaks.csv"
    summary_path = args.output_dir / "model_peak_summary.csv"
    table.to_csv(table_path, index=False)
    summary.to_csv(summary_path, index=False)
    plot_resource_heatmaps(table, args.output_dir)
    plot_fraction_scaling(table, args.output_dir, full_graph)
    plot_k_sensitivity(
        table,
        args.output_dir,
        metric="cuda_peak_reserved_gib",
        ylabel="Peak GPU memory (GiB)",
        filename="cora_qk_gpu_k_sensitivity",
        full_graph=full_graph,
    )
    plot_k_sensitivity(
        table,
        args.output_dir,
        metric="tree_rss_peak_gib",
        ylabel="Peak CPU memory (GiB)",
        filename="cora_qk_cpu_k_sensitivity",
        full_graph=full_graph,
    )
    plot_resource_plane(table, args.output_dir, full_graph)

    if full_graph is not None:
        full_path = args.output_dir / "full_graph_resource_peaks.csv"
        comparison_path = args.output_dir / "resource_comparison.csv"
        full_graph.to_csv(full_path, index=False)
        pd.concat([table, full_graph], ignore_index=True).to_csv(
            comparison_path,
            index=False,
        )
        print(f"Full-graph runs summarized: {len(full_graph)}")
        print(full_path)
        print(comparison_path)

    print(f"Runs summarized: {len(table)}")
    print(f"Models: {table['model'].nunique()}")
    print(
        f"Partition combinations: {table[['num_parts', 'q']].drop_duplicates().shape[0]}"
    )
    print(table_path)
    print(summary_path)
    print(args.output_dir / "cora_qk_resource_heatmaps.svg")
    print(args.output_dir / "cora_qk_fraction_scaling.svg")
    print(args.output_dir / "cora_qk_gpu_k_sensitivity.svg")
    print(args.output_dir / "cora_qk_cpu_k_sensitivity.svg")
    print(args.output_dir / "cora_qk_resource_plane.svg")


if __name__ == "__main__":
    main()
