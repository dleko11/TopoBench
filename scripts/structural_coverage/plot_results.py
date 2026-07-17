"""Plot structural coverage experiment artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": [
            "Arial",
            "Helvetica",
            "DejaVu Sans",
            "sans-serif",
        ],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 9,
        "axes.linewidth": 0.8,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "legend.frameon": False,
        "lines.solid_capstyle": "round",
        "savefig.facecolor": "white",
    }
)

RUN_RE = re.compile(
    r"(?P<dataset>.+?)_(?P<label>.+)_q(?P<q>\d+)_seed(?P<seed>\d+)_"
    r"(?P<ts>\d{8}_\d{6})"
)
MIN_PLOT_EPOCH = 1
FIGURE_WIDTH = 7.0
Q_PALETTE = (
    "#BFD7EA",
    "#91BBD5",
    "#5F99BE",
    "#2F78A8",
    "#185A8D",
    "#123B63",
)
RANK_COLORS = {0: "#A8A8A8", 1: "#315B7D", 2: "#C66A3D"}
NEUTRAL_COLOR = "#666666"
REFERENCE_COLOR = "#B8B8B8"


def save_figure(
    fig: plt.Figure,
    output_dir: Path,
    stem: str,
    *,
    dpi: int = 300,
) -> Path:
    """Save editable vector outputs and a raster preview."""
    output_dir.mkdir(parents=True, exist_ok=True)
    for suffix in ("svg", "pdf"):
        fig.savefig(
            output_dir / f"{stem}.{suffix}",
            bbox_inches="tight",
            pad_inches=0.04,
        )
    preview = output_dir / f"{stem}.png"
    fig.savefig(
        preview,
        dpi=dpi,
        bbox_inches="tight",
        pad_inches=0.04,
    )
    plt.close(fig)
    return preview


def q_colors(runs: list[dict[str, Any]]) -> dict[int, str]:
    """Assign a stable sequential color to each partition batch size."""
    q_values = sorted({int(run["q"]) for run in runs})
    if len(q_values) == 1:
        return {q_values[0]: Q_PALETTE[3]}
    if len(q_values) <= len(Q_PALETTE):
        indices = [
            round(index * (len(Q_PALETTE) - 1) / (len(q_values) - 1))
            for index in range(len(q_values))
        ]
        return {
            q: Q_PALETTE[color_index]
            for q, color_index in zip(q_values, indices, strict=True)
        }
    cmap = plt.get_cmap("Blues")
    return {
        q: cmap(0.35 + 0.55 * index / (len(q_values) - 1))
        for index, q in enumerate(q_values)
    }


def add_panel_label(ax: plt.Axes, label: str) -> None:
    """Add a compact Nature-style panel label."""
    ax.text(
        -0.13,
        1.06,
        label,
        transform=ax.transAxes,
        fontsize=9,
        fontweight="bold",
        ha="left",
        va="bottom",
    )


def read_csv(path: Path) -> list[dict[str, str]]:
    """Read CSV rows as dictionaries."""
    with path.open() as f:
        return list(csv.DictReader(f))


def to_float(row: dict[str, str], key: str) -> float | None:
    """Parse a CSV field as float, returning None for blanks/missing fields."""
    value = row.get(key)
    if value in (None, ""):
        return None
    return float(value)


def is_complete_run(path: Path) -> bool:
    """Return whether a directory contains all core coverage artifacts."""
    if not path.is_dir() or RUN_RE.fullmatch(path.name) is None:
        return False
    required = [
        path / "empirical_coverage.csv",
        path / "theory_curves.csv",
        path / "span_histogram.csv",
        path / "run_metadata.json",
    ]
    return all(file.exists() for file in required)


def complete_runs(results_root: Path) -> list[Path]:
    """Return all complete structural coverage run directories."""
    return sorted(
        path for path in results_root.iterdir() if is_complete_run(path)
    )


def latest_complete_runs(results_root: Path) -> list[Path]:
    """Return the latest complete result directory for each q value."""
    by_q: dict[int, Path] = {}
    for path in complete_runs(results_root):
        match = RUN_RE.fullmatch(path.name)
        assert match is not None
        q = int(match.group("q"))
        if q not in by_q or path.name > by_q[q].name:
            by_q[q] = path
    return [by_q[q] for q in sorted(by_q)]


def load_run(path: Path) -> dict[str, Any]:
    """Load all artifacts needed for plotting from one run directory."""
    metadata = json.loads((path / "run_metadata.json").read_text())
    return {
        "path": path,
        "q": int(metadata["q"]),
        "seed": int(metadata["config"]["seed"]),
        "metadata": metadata,
        "empirical": read_csv(path / "empirical_coverage.csv"),
        "theory": read_csv(path / "theory_curves.csv"),
        "spans": read_csv(path / "span_histogram.csv"),
        "metrics": read_csv(path / "metrics.csv")
        if (path / "metrics.csv").exists()
        else [],
    }


def ensure_output_dir(args: argparse.Namespace) -> Path:
    """Resolve and create the plot output directory."""
    if args.output_dir is not None:
        output_dir = Path(args.output_dir)
    else:
        output_dir = (
            Path(args.results_root) / f"plots_{time.strftime('%Y%m%d_%H%M%S')}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def setup_axes(ax: plt.Axes, *, title: str, ylabel: str) -> None:
    """Apply shared plot styling."""
    ax.set_title(title, loc="left", pad=6, fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.set_xlim(left=MIN_PLOT_EPOCH)
    ax.tick_params(direction="out", length=3, width=0.7)


def coverage_limits(
    runs: list[dict[str, Any]],
    group: str,
) -> tuple[float, float]:
    """Return a data-aware coverage range with room for uncertainty."""
    values = []
    for run in runs:
        values.extend(
            value
            for _, value in epoch_value_rows(
                run["empirical"], f"realized_coverage_{group}"
            )
        )
        values.extend(
            value
            for _, value in epoch_value_rows(
                run["theory"], f"expected_coverage_{group}"
            )
        )
    if not values:
        return 0.0, 1.0
    lower = max(0.0, min(values) - 0.04)
    return lower, 1.005


def epoch_value_rows(
    rows: list[dict[str, str]],
    field: str,
) -> list[tuple[int, float]]:
    """Return epoch/value pairs from the first plotted epoch onward."""
    pairs = []
    for row in rows:
        value = to_float(row, field)
        if value is None:
            continue
        epoch = int(float(row["epoch"]))
        if epoch < MIN_PLOT_EPOCH:
            continue
        pairs.append((epoch, value))
    return pairs


def group_by_q(runs: list[dict[str, Any]]) -> dict[int, list[dict[str, Any]]]:
    """Group loaded runs by q value."""
    grouped: dict[int, list[dict[str, Any]]] = {}
    for run in runs:
        grouped.setdefault(run["q"], []).append(run)
    return {
        q: sorted(q_runs, key=lambda run: run["seed"])
        for q, q_runs in sorted(grouped.items())
    }


def epoch_stats(
    runs: list[dict[str, Any]],
    table_name: str,
    field: str,
) -> tuple[list[int], list[float], list[float], list[int]]:
    """Compute mean and sample standard deviation by epoch."""
    values_by_epoch: dict[int, list[float]] = {}
    for run in runs:
        for row in run[table_name]:
            value = to_float(row, field)
            if value is None:
                continue
            epoch = int(float(row["epoch"]))
            if epoch < MIN_PLOT_EPOCH:
                continue
            values_by_epoch.setdefault(epoch, []).append(value)

    epochs = sorted(values_by_epoch)
    means = [statistics.fmean(values_by_epoch[epoch]) for epoch in epochs]
    stds = [
        statistics.stdev(values_by_epoch[epoch])
        if len(values_by_epoch[epoch]) > 1
        else 0.0
        for epoch in epochs
    ]
    counts = [len(values_by_epoch[epoch]) for epoch in epochs]
    return epochs, means, stds, counts


def entropy_density_stats(
    runs: list[dict[str, Any]],
    group: str = "rank1_2",
) -> tuple[list[int], list[float], list[float], list[int]]:
    """Return appendix entropy in nats per global structure by epoch."""
    values_by_epoch: dict[int, list[float]] = {}
    total_field = f"total_count_{group}"
    entropy_field = f"entropy_nats_{group}"
    for run in runs:
        if not run["empirical"]:
            continue
        total = to_float(run["empirical"][0], total_field)
        if total in (None, 0.0):
            continue
        for row in run["theory"]:
            entropy = to_float(row, entropy_field)
            if entropy is None:
                continue
            epoch = int(float(row["epoch"]))
            if epoch < MIN_PLOT_EPOCH:
                continue
            values_by_epoch.setdefault(epoch, []).append(entropy / total)

    epochs = sorted(values_by_epoch)
    means = [statistics.fmean(values_by_epoch[epoch]) for epoch in epochs]
    stds = [
        statistics.stdev(values_by_epoch[epoch])
        if len(values_by_epoch[epoch]) > 1
        else 0.0
        for epoch in epochs
    ]
    counts = [len(values_by_epoch[epoch]) for epoch in epochs]
    return epochs, means, stds, counts


def plot_coverage_aggregate(
    runs: list[dict[str, Any]],
    output_dir: Path,
    group: str,
    title: str,
    filename: str,
) -> Path:
    """Plot mean empirical coverage with a standard-deviation band."""
    fig, ax = plt.subplots(figsize=(FIGURE_WIDTH, 3.7))
    colors = q_colors(runs)
    for q, q_runs in group_by_q(runs).items():
        theory_epochs, theory_means, _, _ = epoch_stats(
            q_runs,
            "theory",
            f"expected_coverage_{group}",
        )
        empirical_epochs, empirical_means, empirical_stds, _ = epoch_stats(
            q_runs,
            "empirical",
            f"realized_coverage_{group}",
        )
        lower = [
            max(0.0, mean - std)
            for mean, std in zip(empirical_means, empirical_stds, strict=False)
        ]
        upper = [
            min(1.0, mean + std)
            for mean, std in zip(empirical_means, empirical_stds, strict=False)
        ]

        ax.plot(
            theory_epochs,
            theory_means,
            color=colors[q],
            linestyle="--",
            linewidth=1.35,
            alpha=0.9,
            label=f"q={q}, theory",
        )
        ax.plot(
            empirical_epochs,
            empirical_means,
            color=colors[q],
            marker="o",
            markevery=max(1, len(empirical_epochs) // 14),
            markersize=3.2,
            markeredgecolor="white",
            markeredgewidth=0.45,
            linewidth=1.8,
            label=f"q={q}, empirical mean",
        )
        if any(std > 0.0 for std in empirical_stds):
            ax.fill_between(
                empirical_epochs,
                lower,
                upper,
                color=colors[q],
                alpha=0.18,
                linewidth=0,
            )

    setup_axes(ax, title=title, ylabel="Recovered fraction")
    ax.set_ylim(*coverage_limits(runs, group))
    ax.axhline(1.0, color=REFERENCE_COLOR, linewidth=0.75, zorder=0)
    ax.legend(ncol=2, loc="lower right")
    ax.text(
        0.0,
        -0.24,
        "Solid: empirical mean; shading: ±1 s.d.; dashed: theory",
        transform=ax.transAxes,
        color=NEUTRAL_COLOR,
        fontsize=6.5,
        ha="left",
    )
    fig.tight_layout()
    return save_figure(fig, output_dir, Path(filename).stem)


def plot_coverage(
    runs: list[dict[str, Any]],
    output_dir: Path,
    group: str,
    title: str,
    filename: str,
) -> Path:
    """Plot empirical and theoretical coverage for one coverage group."""
    fig, ax = plt.subplots(figsize=(FIGURE_WIDTH, 3.7))
    colors = q_colors(runs)
    for run in runs:
        q = run["q"]
        empirical_pairs = epoch_value_rows(
            run["empirical"],
            f"realized_coverage_{group}",
        )
        theory_pairs = epoch_value_rows(
            run["theory"],
            f"expected_coverage_{group}",
        )
        ax.plot(
            [epoch for epoch, _ in theory_pairs],
            [value for _, value in theory_pairs],
            color=colors[q],
            linestyle="--",
            linewidth=1.35,
            alpha=0.9,
            label=f"q={q}, theory",
        )
        ax.plot(
            [epoch for epoch, _ in empirical_pairs],
            [value for _, value in empirical_pairs],
            color=colors[q],
            marker="o",
            markevery=max(1, len(empirical_pairs) // 14),
            markersize=3.2,
            markeredgecolor="white",
            markeredgewidth=0.45,
            linewidth=1.8,
            label=f"q={q}, empirical",
        )

    setup_axes(ax, title=title, ylabel="Recovered fraction")
    ax.set_ylim(*coverage_limits(runs, group))
    ax.axhline(1.0, color=REFERENCE_COLOR, linewidth=0.75, zorder=0)
    ax.legend(ncol=2, loc="lower right")
    fig.tight_layout()
    return save_figure(fig, output_dir, Path(filename).stem)


def plot_entropy(
    runs: list[dict[str, Any]],
    output_dir: Path,
) -> Path:
    """Plot appendix rank 1+2 entropy in nats per global structure."""
    fig, ax = plt.subplots(figsize=(FIGURE_WIDTH, 3.7))
    colors = q_colors(runs)
    for q, q_runs in group_by_q(runs).items():
        epochs, means, _, _ = entropy_density_stats(q_runs)
        ax.plot(
            epochs,
            means,
            color=colors[q],
            linewidth=1.8,
            label=f"q={q}",
        )

    setup_axes(
        ax,
        title="Cumulative recovery entropy (ranks 1+2)",
        ylabel="Entropy (nats per global structure)",
    )
    ax.set_ylim(bottom=0.0)
    ax.legend(title="Partitions per batch")
    fig.tight_layout()
    return save_figure(fig, output_dir, "entropy_rank1_2_nats")


def plot_span_histogram(
    run: dict[str, Any],
    output_dir: Path,
) -> Path:
    """Plot global structure counts by rank and cluster span."""
    spans = sorted({int(row["span"]) for row in run["spans"]})
    ranks = [0, 1, 2]
    counts = {rank: {span: 0 for span in spans} for rank in ranks}
    for row in run["spans"]:
        counts[int(row["rank"])][int(row["span"])] = int(float(row["count"]))

    fig, ax = plt.subplots(figsize=(FIGURE_WIDTH, 3.7))
    width = 0.24
    offsets = {-1: -width, 0: 0.0, 1: width}
    xs = list(range(len(spans)))
    for offset_key, rank in zip(offsets, ranks, strict=False):
        bars = ax.bar(
            [x + offsets[offset_key] for x in xs],
            [counts[rank][span] for span in spans],
            width=width,
            color=RANK_COLORS[rank],
            label=f"rank {rank}",
        )
        for bar, span in zip(bars, spans, strict=True):
            value = counts[rank][span]
            if value <= 0:
                continue
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value * 1.15,
                f"{value:,}",
                ha="center",
                va="bottom",
                fontsize=6,
                color=NEUTRAL_COLOR,
            )

    ax.set_title(
        "Global structures by cluster span",
        loc="left",
        pad=6,
        fontweight="bold",
    )
    ax.set_xlabel("Cluster span")
    ax.set_ylabel("Structure count")
    ax.set_xticks(xs)
    ax.set_xticklabels([str(span) for span in spans])
    nonzero = [
        value
        for rank_counts in counts.values()
        for value in rank_counts.values()
        if value > 0
    ]
    if nonzero and max(nonzero) / min(nonzero) > 40:
        ax.set_yscale("log")
        ax.set_ylim(0.8, max(nonzero) * 2.2)
    ax.tick_params(direction="out", length=3, width=0.7)
    ax.legend(ncol=3, loc="upper right")
    fig.tight_layout()
    return save_figure(fig, output_dir, "span_histogram_by_rank")


def plot_validation_metrics(
    runs: list[dict[str, Any]],
    output_dir: Path,
    *,
    aggregate_seeds: bool = False,
) -> Path | None:
    """Plot validation accuracy if Lightning metrics are available."""
    fig, ax = plt.subplots(figsize=(FIGURE_WIDTH, 3.7))
    colors = q_colors(runs)
    plotted = False
    plotted_values: list[float] = []
    if aggregate_seeds:
        for q, q_runs in group_by_q(runs).items():
            epochs, means, stds, _ = epoch_stats(
                q_runs,
                "metrics",
                "val/accuracy",
            )
            if not epochs:
                continue
            plotted = True
            plotted_values.extend(means)
            lower = [
                max(0.0, mean - std)
                for mean, std in zip(means, stds, strict=False)
            ]
            upper = [
                min(1.0, mean + std)
                for mean, std in zip(means, stds, strict=False)
            ]
            ax.plot(
                epochs,
                means,
                color=colors[q],
                marker="o",
                markevery=max(1, len(epochs) // 14),
                markersize=3.2,
                markeredgecolor="white",
                markeredgewidth=0.45,
                linewidth=1.8,
                label=f"q={q}",
            )
            if any(std > 0.0 for std in stds):
                ax.fill_between(
                    epochs,
                    lower,
                    upper,
                    color=colors[q],
                    alpha=0.18,
                    linewidth=0,
                )
    else:
        for run in runs:
            pairs = epoch_value_rows(run["metrics"], "val/accuracy")
            if not pairs:
                continue
            plotted = True
            plotted_values.extend(value for _, value in pairs)
            ax.plot(
                [epoch for epoch, _ in pairs],
                [value for _, value in pairs],
                color=colors[run["q"]],
                marker="o",
                markevery=max(1, len(pairs) // 14),
                markersize=3.2,
                linewidth=1.5,
                label=f"q={run['q']}, seed={run['seed']}",
            )

    if not plotted:
        plt.close(fig)
        return None

    setup_axes(ax, title="Validation accuracy", ylabel="Accuracy")
    if plotted_values:
        value_range = max(plotted_values) - min(plotted_values)
        margin = max(0.02, 0.18 * value_range)
        ax.set_ylim(
            max(0.0, min(plotted_values) - margin),
            min(1.0, max(plotted_values) + margin),
        )
    ax.legend(title="Partitions per batch")
    if aggregate_seeds:
        ax.text(
            0.0,
            -0.24,
            "Mean ±1 s.d. across seeds",
            transform=ax.transAxes,
            color=NEUTRAL_COLOR,
            fontsize=6.5,
            ha="left",
        )
    fig.tight_layout()
    return save_figure(fig, output_dir, "validation_accuracy")


def threshold_epoch(
    runs: list[dict[str, Any]],
    table_name: str,
    field: str,
    *,
    threshold: float = 0.95,
) -> int | None:
    """Return the first epoch at which the mean curve reaches a threshold."""
    epochs, means, _, _ = epoch_stats(runs, table_name, field)
    return next(
        (
            epoch
            for epoch, mean in zip(epochs, means, strict=True)
            if mean >= threshold
        ),
        None,
    )


def plot_publication_overview(
    runs: list[dict[str, Any]],
    output_dir: Path,
    *,
    figure_note: str | None = None,
) -> Path:
    """Create the manuscript-style structural reconstruction overview."""
    colors = q_colors(runs)
    grouped = group_by_q(runs)
    fig = plt.figure(figsize=(FIGURE_WIDTH, 5.15))
    grid = fig.add_gridspec(
        2,
        6,
        height_ratios=(1.55, 1.0),
        hspace=0.55,
        wspace=1.05,
    )
    ax_a = fig.add_subplot(grid[0, :])
    ax_b = fig.add_subplot(grid[1, 0:2])
    ax_c = fig.add_subplot(grid[1, 2:4])
    ax_d = fig.add_subplot(grid[1, 4:6])

    # a | Hero evidence: empirical convergence against the theoretical curve.
    for q, q_runs in grouped.items():
        theory_epochs, theory_means, _, _ = epoch_stats(
            q_runs, "theory", "expected_coverage_rank1_2"
        )
        empirical_epochs, empirical_means, empirical_stds, _ = epoch_stats(
            q_runs, "empirical", "realized_coverage_rank1_2"
        )
        ax_a.plot(
            theory_epochs,
            theory_means,
            color=colors[q],
            linestyle="--",
            linewidth=1.25,
            alpha=0.9,
            label=f"q={q}, theory",
        )
        ax_a.plot(
            empirical_epochs,
            empirical_means,
            color=colors[q],
            linewidth=1.9,
            marker="o",
            markevery=max(1, len(empirical_epochs) // 12),
            markersize=3.2,
            markeredgecolor="white",
            markeredgewidth=0.45,
            label=f"q={q}, empirical",
        )
        if any(value > 0.0 for value in empirical_stds):
            ax_a.fill_between(
                empirical_epochs,
                [
                    max(0.0, mean - std)
                    for mean, std in zip(
                        empirical_means, empirical_stds, strict=True
                    )
                ],
                [
                    min(1.0, mean + std)
                    for mean, std in zip(
                        empirical_means, empirical_stds, strict=True
                    )
                ],
                color=colors[q],
                alpha=0.18,
                linewidth=0,
            )
        if len(grouped) == 1 and empirical_epochs:
            ax_a.annotate(
                f"{empirical_means[-1] * 100:.2f}%",
                xy=(empirical_epochs[-1], empirical_means[-1]),
                xytext=(-4, -13),
                textcoords="offset points",
                ha="right",
                va="top",
                fontsize=7,
                color=colors[q],
                fontweight="bold",
            )
    setup_axes(
        ax_a,
        title="Cumulative recovery of rank-1 and rank-2 structures",
        ylabel="Recovered fraction",
    )
    ax_a.set_ylim(*coverage_limits(runs, "rank1_2"))
    ax_a.axhline(1.0, color=REFERENCE_COLOR, linewidth=0.75, zorder=0)
    ax_a.legend(ncol=min(4, 2 * len(grouped)), loc="lower right")
    seed_counts = sorted({len(q_runs) for q_runs in grouped.values()})
    if len(seed_counts) == 1:
        ax_a.text(
            0.0,
            -0.26,
            f"Mean ±1 s.d.; n={seed_counts[0]} seeds",
            transform=ax_a.transAxes,
            color=NEUTRAL_COLOR,
            fontsize=6.5,
            ha="left",
        )
    add_panel_label(ax_a, "a")

    # b | Compact rank-resolved summary that scales to the full q sweep.
    q_values = list(grouped)
    x_positions = list(range(len(q_values)))
    max_epoch = max(
        int(float(row["epoch"])) for run in runs for row in run["empirical"]
    )
    crossing_values: list[int] = []
    for rank, offset in ((1, -0.12), (2, 0.12)):
        for x, q in zip(x_positions, q_values, strict=True):
            empirical = threshold_epoch(
                grouped[q],
                "empirical",
                f"realized_coverage_rank{rank}",
            )
            theory = threshold_epoch(
                grouped[q],
                "theory",
                f"expected_coverage_rank{rank}",
            )
            empirical_y = empirical if empirical is not None else max_epoch
            theory_y = theory if theory is not None else max_epoch
            crossing_values.extend([empirical_y, theory_y])
            ax_b.plot(
                [x + offset, x + offset],
                [empirical_y, theory_y],
                color=REFERENCE_COLOR,
                linewidth=0.8,
                zorder=1,
            )
            ax_b.scatter(
                x + offset,
                empirical_y,
                s=22,
                color=RANK_COLORS[rank],
                edgecolor="white",
                linewidth=0.5,
                zorder=3,
            )
            ax_b.scatter(
                x + offset,
                theory_y,
                s=24,
                facecolor="white",
                edgecolor=RANK_COLORS[rank],
                linewidth=1.0,
                marker="s",
                zorder=2,
            )
            if empirical is None or theory is None:
                ax_b.text(
                    x + offset,
                    max_epoch,
                    ">",
                    ha="center",
                    va="bottom",
                    fontsize=6,
                    color=RANK_COLORS[rank],
                )
    ax_b.set_title(
        "Time to 95% recovery", loc="left", pad=6, fontweight="bold"
    )
    ax_b.set_xlabel("Partitions per batch, q")
    ax_b.set_ylabel("Epoch")
    ax_b.set_xticks(x_positions)
    ax_b.set_xticklabels([str(q) for q in q_values])
    ax_b.set_ylim(0, max(crossing_values + [1]) * 1.22)
    ax_b.tick_params(direction="out", length=3, width=0.7)
    ax_b.legend(
        handles=[
            Line2D(
                [],
                [],
                marker="o",
                linestyle="none",
                color=RANK_COLORS[1],
                label="rank 1",
            ),
            Line2D(
                [],
                [],
                marker="o",
                linestyle="none",
                color=RANK_COLORS[2],
                label="rank 2",
            ),
            Line2D(
                [],
                [],
                marker="o",
                linestyle="none",
                color=NEUTRAL_COLOR,
                label="empirical",
            ),
            Line2D(
                [],
                [],
                marker="s",
                linestyle="none",
                markerfacecolor="white",
                markeredgecolor=NEUTRAL_COLOR,
                color="white",
                label="theory",
            ),
        ],
        ncol=2,
        loc="upper right",
        handletextpad=0.3,
        columnspacing=0.7,
    )
    add_panel_label(ax_b, "b")

    # c | Structural explanation: the global span distribution.
    span_run = runs[-1]
    spans = sorted({int(row["span"]) for row in span_run["spans"]})
    span_counts = {
        rank: {
            int(row["span"]): int(float(row["count"]))
            for row in span_run["spans"]
            if int(row["rank"]) == rank
        }
        for rank in (0, 1, 2)
    }
    all_span_counts = []
    for rank, offset in ((0, -0.12), (1, 0.0), (2, 0.12)):
        rank_spans = [
            span for span in spans if span_counts[rank].get(span, 0) > 0
        ]
        rank_values = [span_counts[rank][span] for span in rank_spans]
        all_span_counts.extend(rank_values)
        ax_c.scatter(
            [span + offset for span in rank_spans],
            rank_values,
            s=25,
            color=RANK_COLORS[rank],
            edgecolor="white",
            linewidth=0.5,
            label=f"rank {rank}",
            zorder=3,
        )
    ax_c.set_title(
        "Global span distribution", loc="left", pad=6, fontweight="bold"
    )
    ax_c.set_xlabel("Cluster span")
    ax_c.set_ylabel("Structure count")
    ax_c.set_xticks(spans)
    if all_span_counts and max(all_span_counts) / min(all_span_counts) > 40:
        ax_c.set_yscale("log")
    ax_c.tick_params(direction="out", length=3, width=0.7)
    ax_c.legend(ncol=1, loc="upper right", handletextpad=0.3)
    add_panel_label(ax_c, "c")

    # d | Consequence of convergence: marginal uncertainty vanishes.
    for q, q_runs in grouped.items():
        epochs, means, _, _ = entropy_density_stats(q_runs)
        ax_d.plot(
            epochs,
            means,
            color=colors[q],
            linewidth=1.7,
            label=f"q={q}",
        )
    setup_axes(
        ax_d,
        title="Recovery uncertainty",
        ylabel="Nats per structure",
    )
    ax_d.set_ylim(bottom=0.0)
    ax_d.legend(loc="upper right")
    add_panel_label(ax_d, "d")

    fig.suptitle(
        "Structural reconstruction",
        x=0.075,
        y=0.995,
        ha="left",
        va="top",
        fontsize=11,
        fontweight="bold",
    )
    if figure_note:
        fig.text(
            0.99,
            0.995,
            figure_note,
            ha="right",
            va="top",
            fontsize=6.5,
            color=NEUTRAL_COLOR,
        )
    fig.subplots_adjust(top=0.91, bottom=0.09, left=0.09, right=0.985)
    return save_figure(fig, output_dir, "structural_reconstruction_overview")


def write_summary(
    runs: list[dict[str, Any]],
    output_dir: Path,
    plot_paths: list[Path],
    *,
    aggregate_seeds: bool,
) -> Path:
    """Write a compact Markdown summary next to the plots."""
    lines = [
        "# Structural Coverage Plot Summary",
        "",
        "Figures are exported as editable SVG, PDF, and PNG preview files.",
        "",
    ]
    if aggregate_seeds:
        lines.append("## Aggregates")
        for q, q_runs in group_by_q(runs).items():
            epochs, means, stds, counts = epoch_stats(
                q_runs,
                "empirical",
                "realized_coverage_rank1_2",
            )
            if epochs:
                lines.append(
                    "- "
                    f"q={q}, seeds={[run['seed'] for run in q_runs]}, "
                    f"last_epoch={epochs[-1]}, "
                    f"rank1+2 empirical_mean={means[-1]:.4f}, "
                    f"empirical_std={stds[-1]:.4f}, "
                    f"n_at_last_epoch={counts[-1]}"
                )
        lines.append("")

    lines.append("## Runs")
    for run in runs:
        last = run["empirical"][-1]
        lines.append(
            "- "
            f"q={run['q']}, seed={run['seed']}, "
            f"epochs={int(float(last['epoch']))}, "
            f"rank1+2 empirical={float(last['realized_coverage_rank1_2']):.4f}, "
            f"source={run['path']}"
        )
    lines.extend(["", "## Plots"])
    lines.extend(f"- {path.name}" for path in plot_paths)
    summary = output_dir / "summary.md"
    summary.write_text("\n".join(lines) + "\n")
    return summary


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Plot Cora structural coverage experiment results."
    )
    parser.add_argument(
        "run_dirs",
        nargs="*",
        help="Specific run directories. Defaults to latest complete run per q.",
    )
    parser.add_argument(
        "--results-root",
        default="scripts/structural_coverage/results",
        help="Directory containing structural coverage result runs.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory to write plots. Defaults to results-root/plots_TIMESTAMP.",
    )
    parser.add_argument(
        "--aggregate-seeds",
        action="store_true",
        help=(
            "Use all complete runs and plot mean empirical curves with "
            "+/-1 standard deviation bands grouped by q."
        ),
    )
    parser.add_argument(
        "--all-runs",
        action="store_true",
        help=(
            "Use all complete runs in results-root. Without this or "
            "--aggregate-seeds, the default is the latest complete run per q."
        ),
    )
    parser.add_argument(
        "--figure-note",
        default=None,
        help=(
            "Optional small note on the publication overview, for example "
            "'Design preview - legacy runs'."
        ),
    )
    return parser.parse_args()


def main() -> None:
    """CLI entrypoint."""
    args = parse_args()
    run_paths = (
        [Path(path) for path in args.run_dirs]
        if args.run_dirs
        else complete_runs(Path(args.results_root))
        if args.aggregate_seeds or args.all_runs
        else latest_complete_runs(Path(args.results_root))
    )
    if not run_paths:
        raise SystemExit("No complete structural coverage runs found.")

    runs = [load_run(path) for path in run_paths]
    runs.sort(key=lambda run: (run["q"], run["seed"], run["path"].name))
    output_dir = ensure_output_dir(args)
    coverage_plotter = (
        plot_coverage_aggregate if args.aggregate_seeds else plot_coverage
    )

    plot_paths = [
        plot_publication_overview(
            runs,
            output_dir,
            figure_note=args.figure_note,
        ),
        coverage_plotter(
            runs,
            output_dir,
            "rank1_2",
            "Cumulative recovery: ranks 1+2",
            "coverage_rank1_2.png",
        ),
        coverage_plotter(
            runs,
            output_dir,
            "rank1",
            "Cumulative recovery: rank 1",
            "coverage_rank1.png",
        ),
        coverage_plotter(
            runs,
            output_dir,
            "rank2",
            "Cumulative recovery: rank 2",
            "coverage_rank2.png",
        ),
        coverage_plotter(
            runs,
            output_dir,
            "all",
            "Cumulative recovery: all ranks",
            "coverage_all_ranks.png",
        ),
        plot_entropy(runs, output_dir),
        plot_span_histogram(runs[-1], output_dir),
    ]
    metrics_plot = plot_validation_metrics(
        runs,
        output_dir,
        aggregate_seeds=args.aggregate_seeds,
    )
    if metrics_plot is not None:
        plot_paths.append(metrics_plot)

    summary = write_summary(
        runs,
        output_dir,
        plot_paths,
        aggregate_seeds=args.aggregate_seeds,
    )
    print(f"Wrote plots to {output_dir}")
    print(f"Wrote summary to {summary}")


if __name__ == "__main__":
    main()
