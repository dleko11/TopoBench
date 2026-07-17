"""Render a legacy-data preview of the grouped sweep figure design.

This script is deliberately separate from the final sweep plotter. It uses
available legacy q=8 runs to evaluate the family-grouped layout without
claiming statistical comparability between those historical experiments.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from scripts.structural_coverage.plot_results import (
    FIGURE_WIDTH,
    Q_PALETTE,
    REFERENCE_COLOR,
    add_panel_label,
    complete_runs,
    epoch_stats,
    load_run,
    save_figure,
)

Q_VALUES = (1, 2, 4, 8, 16, 32)
Q8_COLOR = Q_PALETTE[3]
FAMILY_STYLES = {
    "hypergraph": {
        "color": "#2F78A8",
        "marker": "o",
        "linestyle": "-",
    },
    "simplicial": {
        "color": "#715A9C",
        "marker": "D",
        "linestyle": "-",
    },
    "cell_basis": {
        "color": "#D39A64",
        "marker": "s",
        "linestyle": "--",
    },
    "cell_simple": {
        "color": "#B85C2E",
        "marker": "^",
        "linestyle": "-",
    },
}


@dataclass(frozen=True)
class ProfileView:
    """One lifting profile shown in the grouped design preview."""

    key: str
    heading: str
    structure_label: str
    legend_label: str
    group: str
    runs: list[dict[str, Any]]


def load_profile(
    *,
    key: str,
    heading: str,
    structure_label: str,
    legend_label: str,
    group: str,
    root: str | Path,
) -> ProfileView:
    """Load all complete legacy runs under one profile root."""
    paths = complete_runs(Path(root))
    if not paths:
        raise FileNotFoundError(f"No complete legacy runs found under {root}.")
    runs = [load_run(path) for path in paths]
    runs.sort(key=lambda run: (run["q"], run["seed"], run["path"].name))
    return ProfileView(
        key=key,
        heading=heading,
        structure_label=structure_label,
        legend_label=legend_label,
        group=group,
        runs=runs,
    )


def observable_ceiling(
    run: dict[str, Any],
    *,
    rank: int,
    q_values: tuple[int, ...] = Q_VALUES,
) -> list[float]:
    """Compute the q-observable ceiling from one span histogram."""
    rows = [row for row in run["spans"] if int(row["rank"]) == rank]
    total = sum(int(float(row["count"])) for row in rows)
    if total == 0:
        return [0.0 for _ in q_values]
    return [
        sum(int(float(row["count"])) for row in rows if int(row["span"]) <= q)
        / total
        for q in q_values
    ]


def draw_recovery_axis(
    ax: plt.Axes,
    profile: ProfileView,
    *,
    max_epoch: int,
    show_ylabel: bool,
) -> None:
    """Draw the available q=8 empirical and theoretical recovery curves."""
    theory_epochs, theory_means, _, _ = epoch_stats(
        profile.runs,
        "theory",
        f"expected_coverage_{profile.group}",
    )
    empirical_epochs, empirical_means, empirical_stds, _ = epoch_stats(
        profile.runs,
        "empirical",
        f"realized_coverage_{profile.group}",
    )
    theory = [
        (epoch, mean)
        for epoch, mean in zip(theory_epochs, theory_means, strict=True)
        if epoch <= max_epoch
    ]
    empirical = [
        (epoch, mean, std)
        for epoch, mean, std in zip(
            empirical_epochs,
            empirical_means,
            empirical_stds,
            strict=True,
        )
        if epoch <= max_epoch
    ]

    ax.plot(
        [epoch for epoch, _ in theory],
        [mean for _, mean in theory],
        color=Q8_COLOR,
        linestyle="--",
        linewidth=1.15,
        alpha=0.95,
        zorder=2,
    )
    ax.plot(
        [epoch for epoch, _, _ in empirical],
        [mean for _, mean, _ in empirical],
        color=Q8_COLOR,
        linewidth=1.7,
        marker="o",
        markevery=max(1, len(empirical) // 7),
        markersize=2.8,
        markeredgecolor="white",
        markeredgewidth=0.4,
        zorder=3,
    )
    if len(profile.runs) > 1 and any(std > 0 for _, _, std in empirical):
        ax.fill_between(
            [epoch for epoch, _, _ in empirical],
            [max(0.0, mean - std) for _, mean, std in empirical],
            [min(1.0, mean + std) for _, mean, std in empirical],
            color=Q8_COLOR,
            alpha=0.16,
            linewidth=0,
            zorder=1,
        )

    ax.set_title(profile.heading, loc="left", pad=16, fontweight="bold")
    ax.text(
        0.0,
        1.01,
        profile.structure_label,
        transform=ax.transAxes,
        fontsize=6.5,
        color="#666666",
        ha="left",
        va="bottom",
    )
    ax.text(
        0.98,
        0.05,
        f"n={len(profile.runs)}",
        transform=ax.transAxes,
        fontsize=6.5,
        color="#666666",
        ha="right",
        va="bottom",
    )
    ax.set_xlim(1, max_epoch)
    ax.set_ylim(0.0, 1.02)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Recovered fraction" if show_ylabel else "")
    ax.set_yticks((0.0, 0.5, 1.0))
    if not show_ylabel:
        ax.tick_params(axis="y", labelleft=False)
    ax.axhline(1.0, color=REFERENCE_COLOR, linewidth=0.65, zorder=0)
    ax.tick_params(direction="out", length=2.8, width=0.7)


def draw_observable_ceiling_axis(
    ax: plt.Axes,
    profiles: list[ProfileView],
) -> None:
    """Draw the span-derived observable ceiling across the planned q sweep."""
    for profile in profiles:
        rank = int(profile.group.removeprefix("rank"))
        values = observable_ceiling(profile.runs[0], rank=rank)
        style = FAMILY_STYLES[profile.key]
        marker_face = (
            "white" if profile.key == "cell_basis" else style["color"]
        )
        ax.plot(
            Q_VALUES,
            values,
            color=style["color"],
            linestyle=style["linestyle"],
            linewidth=1.7,
            marker=style["marker"],
            markersize=4.0,
            markerfacecolor=marker_face,
            markeredgecolor=style["color"],
            markeredgewidth=0.9,
            label=profile.legend_label,
        )

    ax.set_title(
        "Observable structural ceiling across q",
        loc="left",
        pad=7,
        fontweight="bold",
    )
    ax.set_xscale("log", base=2)
    ax.set_xlim(0.85, 38)
    ax.set_ylim(0.0, 1.03)
    ax.set_xticks(Q_VALUES)
    ax.set_xticklabels([str(q) for q in Q_VALUES])
    ax.set_xlabel("Partitions per batch, q")
    ax.set_ylabel("Observable fraction")
    ax.axhline(1.0, color=REFERENCE_COLOR, linewidth=0.7, zorder=0)
    ax.tick_params(direction="out", length=3, width=0.7)
    ax.legend(
        ncol=2,
        loc="lower right",
        columnspacing=1.0,
        handlelength=2.0,
    )


def plot_preview(
    profiles: list[ProfileView],
    output_dir: Path,
    *,
    max_epoch: int = 50,
) -> Path:
    """Render panels a and c of the proposed family-grouped sweep figure."""
    fig = plt.figure(figsize=(FIGURE_WIDTH, 5.15))
    grid = fig.add_gridspec(
        2,
        4,
        height_ratios=(1.0, 1.15),
        hspace=0.62,
        wspace=0.22,
    )
    recovery_axes = [fig.add_subplot(grid[0, index]) for index in range(4)]
    ceiling_ax = fig.add_subplot(grid[1, :])

    for index, (ax, profile) in enumerate(
        zip(recovery_axes, profiles, strict=True)
    ):
        draw_recovery_axis(
            ax,
            profile,
            max_epoch=max_epoch,
            show_ylabel=index == 0,
        )

    add_panel_label(recovery_axes[0], "a")
    ceiling_ax.text(
        -0.035,
        1.06,
        "c",
        transform=ceiling_ax.transAxes,
        fontsize=9,
        fontweight="bold",
        ha="left",
        va="bottom",
    )
    draw_observable_ceiling_axis(ceiling_ax, profiles)

    semantic_handles = [
        Line2D(
            [],
            [],
            color=Q8_COLOR,
            linewidth=1.7,
            marker="o",
            markersize=3,
            label="empirical mean, q=8",
        ),
        Line2D(
            [],
            [],
            color=Q8_COLOR,
            linewidth=1.15,
            linestyle="--",
            label="theory, q=8",
        ),
        Patch(
            facecolor=Q8_COLOR,
            alpha=0.16,
            edgecolor="none",
            label="±1 s.d.",
        ),
    ]
    fig.legend(
        handles=semantic_handles,
        ncol=3,
        loc="center",
        bbox_to_anchor=(0.51, 0.505),
        columnspacing=1.2,
        handlelength=2.0,
    )

    cell_left = recovery_axes[2].get_position().x0
    cell_right = recovery_axes[3].get_position().x1
    fig.text(
        (cell_left + cell_right) / 2,
        0.94,
        "Cell complex",
        fontsize=7.5,
        fontweight="bold",
        ha="center",
        va="bottom",
        color=FAMILY_STYLES["cell_simple"]["color"],
    )
    fig.add_artist(
        Line2D(
            [cell_left, cell_right],
            [0.936, 0.936],
            transform=fig.transFigure,
            color=FAMILY_STYLES["cell_simple"]["color"],
            linewidth=1.0,
        )
    )

    fig.suptitle(
        "Structural recovery across lifting families",
        x=0.075,
        y=0.995,
        ha="left",
        va="top",
        fontsize=11,
        fontweight="bold",
    )
    fig.text(
        0.99,
        0.995,
        "Design preview · legacy runs",
        ha="right",
        va="top",
        fontsize=6.5,
        color="#666666",
    )
    fig.subplots_adjust(top=0.84, bottom=0.14, left=0.09, right=0.985)
    fig.text(
        0.09,
        0.025,
        "Legacy span histograms only; no empirical q values are imputed.",
        fontsize=6.5,
        color="#666666",
        ha="left",
        va="bottom",
    )
    return save_figure(fig, output_dir, "grouped_sweep_panels_a_c")


def parse_args() -> argparse.Namespace:
    """Parse preview input roots and export location."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--hypergraph-root",
        default=(
            "scripts/structural_coverage/results/"
            "cora_hypergraph_np64_q8_5seeds"
        ),
    )
    parser.add_argument(
        "--simplicial-root",
        default="scripts/structural_coverage/results/cora_np64_q8_5seeds",
    )
    parser.add_argument(
        "--cell-basis-root",
        default=(
            "scripts/structural_coverage/results/cora_cell_cwn_np64_q8_3seeds"
        ),
    )
    parser.add_argument(
        "--cell-simple-root",
        default=(
            "scripts/structural_coverage/results/"
            "cora_cell_simple_cycles_n8_np64_q8_seed0_50epochs"
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=(
            "scripts/structural_coverage/results/"
            "legacy_grouped_sweep_design_preview"
        ),
    )
    parser.add_argument("--max-epoch", type=int, default=50)
    return parser.parse_args()


def main() -> None:
    """CLI entry point."""
    args = parse_args()
    profiles = [
        load_profile(
            key="hypergraph",
            heading="Hypergraph",
            structure_label="rank-1 hyperedges",
            legend_label="Hyperedges",
            group="rank1",
            root=args.hypergraph_root,
        ),
        load_profile(
            key="simplicial",
            heading="Simplicial complex",
            structure_label="rank-2 triangles",
            legend_label="Triangles",
            group="rank2",
            root=args.simplicial_root,
        ),
        load_profile(
            key="cell_basis",
            heading="Cycle basis",
            structure_label="rank-2 cells · stress test",
            legend_label="Cycle-basis cells",
            group="rank2",
            root=args.cell_basis_root,
        ),
        load_profile(
            key="cell_simple",
            heading="All simple cycles",
            structure_label="rank-2 cells · bounded support",
            legend_label="All simple cycles",
            group="rank2",
            root=args.cell_simple_root,
        ),
    ]
    output_dir = Path(args.output_dir)
    preview = plot_preview(
        profiles,
        output_dir,
        max_epoch=args.max_epoch,
    )
    print(f"Wrote grouped sweep design preview to {preview}")


if __name__ == "__main__":
    main()
