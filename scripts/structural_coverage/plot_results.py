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

import matplotlib.pyplot as plt

RUN_RE = re.compile(
    r"cora_(?P<label>.+)_q(?P<q>\d+)_seed(?P<seed>\d+)_"
    r"(?P<ts>\d{8}_\d{6})"
)
MIN_PLOT_EPOCH = 1


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
            Path(args.results_root)
            / f"plots_{time.strftime('%Y%m%d_%H%M%S')}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def setup_axes(ax: plt.Axes, *, title: str, ylabel: str) -> None:
    """Apply shared plot styling."""
    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.set_xlim(left=MIN_PLOT_EPOCH)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


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


def plot_coverage_aggregate(
    runs: list[dict[str, Any]],
    output_dir: Path,
    group: str,
    title: str,
    filename: str,
) -> Path:
    """Plot mean empirical coverage with a standard-deviation band."""
    fig, ax = plt.subplots(figsize=(9, 5.2))
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
            linestyle="--",
            linewidth=1.9,
            label=f"q={q} theory mean",
        )
        ax.plot(
            empirical_epochs,
            empirical_means,
            marker="o",
            markersize=3,
            linewidth=1.7,
            label=f"q={q} empirical mean",
        )
        if any(std > 0.0 for std in empirical_stds):
            ax.fill_between(
                empirical_epochs,
                lower,
                upper,
                alpha=0.18,
                linewidth=0,
                label=f"q={q} empirical +/-1 std",
            )

    setup_axes(ax, title=title, ylabel="Coverage")
    ax.set_ylim(-0.02, 1.02)
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    path = output_dir / filename
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_coverage(
    runs: list[dict[str, Any]],
    output_dir: Path,
    group: str,
    title: str,
    filename: str,
) -> Path:
    """Plot empirical and theoretical coverage for one coverage group."""
    fig, ax = plt.subplots(figsize=(9, 5.2))
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
            linestyle="--",
            linewidth=1.8,
            label=f"q={q} theory",
        )
        ax.plot(
            [epoch for epoch, _ in empirical_pairs],
            [value for _, value in empirical_pairs],
            marker="o",
            markersize=3,
            linewidth=1.6,
            label=f"q={q} empirical",
        )

    setup_axes(ax, title=title, ylabel="Coverage")
    ax.set_ylim(-0.02, 1.02)
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    path = output_dir / filename
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_entropy(
    runs: list[dict[str, Any]],
    output_dir: Path,
) -> Path:
    """Plot rank 1+2 entropy in nats."""
    fig, ax = plt.subplots(figsize=(9, 5.2))
    for run in runs:
        q = run["q"]
        entropy_pairs = epoch_value_rows(
            run["theory"],
            "entropy_nats_rank1_2",
        )
        ax.plot(
            [epoch for epoch, _ in entropy_pairs],
            [value for _, value in entropy_pairs],
            linewidth=1.8,
            label=f"q={q}",
        )

    setup_axes(
        ax,
        title="Theoretical Cumulative Recovery Entropy (Rank 1+2)",
        ylabel="Entropy (nats)",
    )
    ax.legend(title="Batch size")
    fig.tight_layout()
    path = output_dir / "entropy_rank1_2_nats.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_span_histogram(
    run: dict[str, Any],
    output_dir: Path,
) -> Path:
    """Plot global structure counts by rank and cluster span."""
    spans = sorted({int(row["span"]) for row in run["spans"]})
    ranks = [0, 1, 2]
    counts = {
        rank: {span: 0 for span in spans}
        for rank in ranks
    }
    for row in run["spans"]:
        counts[int(row["rank"])][int(row["span"])] = int(float(row["count"]))

    fig, ax = plt.subplots(figsize=(8, 5.2))
    width = 0.24
    offsets = {-1: -width, 0: 0.0, 1: width}
    xs = list(range(len(spans)))
    for offset_key, rank in zip(offsets, ranks, strict=False):
        ax.bar(
            [x + offsets[offset_key] for x in xs],
            [counts[rank][span] for span in spans],
            width=width,
            label=f"rank {rank}",
        )

    ax.set_title("Global Simplicial Structures by Cluster Span")
    ax.set_xlabel("Cluster span")
    ax.set_ylabel("Structure count")
    ax.set_xticks(xs)
    ax.set_xticklabels([str(span) for span in spans])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend()
    fig.tight_layout()
    path = output_dir / "span_histogram_by_rank.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_validation_metrics(
    runs: list[dict[str, Any]],
    output_dir: Path,
    *,
    aggregate_seeds: bool = False,
) -> Path | None:
    """Plot validation accuracy if Lightning metrics are available."""
    fig, ax = plt.subplots(figsize=(9, 5.2))
    plotted = False
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
                marker="o",
                markersize=3,
                linewidth=1.6,
                label=f"q={q} mean",
            )
            if any(std > 0.0 for std in stds):
                ax.fill_between(
                    epochs,
                    lower,
                    upper,
                    alpha=0.18,
                    linewidth=0,
                    label=f"q={q} +/-1 std",
                )
    else:
        for run in runs:
            pairs = epoch_value_rows(run["metrics"], "val/accuracy")
            if not pairs:
                continue
            plotted = True
            ax.plot(
                [epoch for epoch, _ in pairs],
                [value for _, value in pairs],
                marker="o",
                markersize=3,
                linewidth=1.6,
                label=f"q={run['q']}",
            )

    if not plotted:
        plt.close(fig)
        return None

    setup_axes(ax, title="Validation Accuracy by q", ylabel="Validation accuracy")
    ax.set_ylim(0.0, 1.0)
    ax.legend(title="Batch size")
    fig.tight_layout()
    path = output_dir / "validation_accuracy.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def write_summary(
    runs: list[dict[str, Any]],
    output_dir: Path,
    plot_paths: list[Path],
    *,
    aggregate_seeds: bool,
) -> Path:
    """Write a compact Markdown summary next to the plots."""
    lines = ["# Structural Coverage Plot Summary", ""]
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
        coverage_plotter(
            runs,
            output_dir,
            "rank1_2",
            "Structural Coverage: Rank 1+2",
            "coverage_rank1_2.png",
        ),
        coverage_plotter(
            runs,
            output_dir,
            "rank1",
            "Structural Coverage: Rank 1 Structures",
            "coverage_rank1.png",
        ),
        coverage_plotter(
            runs,
            output_dir,
            "rank2",
            "Structural Coverage: Rank 2 Structures",
            "coverage_rank2.png",
        ),
        coverage_plotter(
            runs,
            output_dir,
            "all",
            "Structural Coverage: All Ranks",
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
