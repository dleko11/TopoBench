"""Render two legacy-data previews for the appendix figure architecture.

The unfinished sweep does not yet provide empirical trajectories at every q.
This preview therefore uses exact span-histogram theory for q=1,...,32,
overlays only the available empirical q=8 runs, and labels that limitation in
the figures. No missing empirical values are simulated or imputed.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.ticker import PercentFormatter

from scripts.structural_coverage.coverage import (
    bernoulli_entropy_bits,
    per_epoch_probability,
)
from scripts.structural_coverage.plot_results import (
    FIGURE_WIDTH,
    epoch_stats,
    save_figure,
)
from scripts.structural_coverage.plot_sweep_design_preview import (
    Q_VALUES,
    ProfileView,
    load_profile,
    observable_ceiling,
)

EMPIRICAL_Q = 8
INK_COLOR = "#26343C"
NEUTRAL_COLOR = "#66757E"
REFERENCE_COLOR = "#AEBAC0"
Q_PALETTE = (
    "#DCE9F5",
    "#B7CEE5",
    "#88ADD0",
    "#4779B2",
    "#3B6F9B",
    "#285A82",
)
FAMILY_STYLES = {
    "hypergraph": {
        "color": "#3B6F9B",
        "marker": "o",
        "linestyle": "-",
    },
    "simplicial": {
        "color": "#7763A5",
        "marker": "D",
        "linestyle": "-",
    },
    "cell_basis": {
        "color": "#D8872E",
        "marker": "s",
        "linestyle": "--",
    },
    "cell_simple": {
        "color": "#C5534B",
        "marker": "^",
        "linestyle": "-",
    },
}

plt.rcParams.update(
    {
        "text.color": INK_COLOR,
        "axes.labelcolor": INK_COLOR,
        "axes.edgecolor": INK_COLOR,
        "xtick.color": INK_COLOR,
        "ytick.color": INK_COLOR,
    }
)


def target_histogram(profile: ProfileView) -> list[tuple[int, int]]:
    """Return ``(span, count)`` rows for the profile's target rank."""
    rank = int(profile.group.removeprefix("rank"))
    return [
        (int(row["span"]), int(float(row["count"])))
        for row in profile.runs[0]["spans"]
        if int(row["rank"]) == rank
    ]


def theory_from_histogram(
    profile: ProfileView,
    *,
    q: int,
    max_epoch: int,
) -> tuple[list[int], list[float], list[float]]:
    """Compute target-rank coverage and entropy from a span histogram."""
    histogram = target_histogram(profile)
    total = sum(count for _, count in histogram)
    k_eff = int(profile.runs[0]["metadata"]["K_eff"])
    epochs = list(range(1, max_epoch + 1))
    coverage: list[float] = []
    entropy: list[float] = []
    for epoch in epochs:
        expected = 0.0
        entropy_nats = 0.0
        for span, count in histogram:
            probability = per_epoch_probability(span, q, k_eff)
            rho = 1.0 - (1.0 - probability) ** epoch
            expected += count * rho
            if probability > 0.0:
                entropy_nats += (
                    count * bernoulli_entropy_bits(rho) * math.log(2.0)
                )
        coverage.append(expected / total if total else 0.0)
        entropy.append(entropy_nats / total if total else 0.0)
    return epochs, coverage, entropy


def style_axis(ax: plt.Axes) -> None:
    """Apply compact shared axis styling."""
    ax.tick_params(direction="out", length=2.8, width=0.7)
    ax.grid(False)


def add_facet_title(ax: plt.Axes, profile: ProfileView) -> None:
    """Add one concise structure title with a family-colour marker."""
    color = FAMILY_STYLES[profile.key]["color"]
    ax.add_patch(
        Rectangle(
            (-0.005, 1.075),
            0.018,
            0.105,
            transform=ax.transAxes,
            facecolor=color,
            edgecolor="none",
            clip_on=False,
        )
    )
    ax.text(
        0.035,
        1.125,
        profile.legend_label,
        transform=ax.transAxes,
        fontsize=8.5,
        fontweight="bold",
        ha="left",
        va="center",
    )


def draw_q_key(
    ax: plt.Axes,
    *,
    entropy_only: bool = False,
) -> None:
    """Draw a compact discrete q key in a dedicated legend axis."""
    ax.set_axis_off()
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.text(0.12, 0.96, r"$q$", fontsize=7.8, fontweight="bold")
    positions = tuple(0.84 - 0.105 * index for index in range(len(Q_VALUES)))
    for y, q, color in zip(positions, Q_VALUES, Q_PALETTE, strict=True):
        ax.add_patch(
            Rectangle(
                (0.12, y),
                0.26,
                0.050,
                facecolor=color,
                edgecolor="none",
            )
        )
        ax.text(0.47, y + 0.025, str(q), fontsize=6.8, va="center")

    if entropy_only:
        ax.plot(
            [0.12, 0.38],
            [0.13, 0.13],
            color=Q_PALETTE[3],
            linewidth=1.5,
        )
        ax.text(0.47, 0.13, "Analytic", fontsize=6.2, va="center")


def draw_recovery_facet(
    ax: plt.Axes,
    profile: ProfileView,
    *,
    max_epoch: int,
    show_ylabel: bool,
    show_xlabel: bool,
) -> None:
    """Draw all-q theory and the available q=8 empirical trajectory."""
    for q, color in zip(Q_VALUES, Q_PALETTE, strict=True):
        epochs, coverage, _ = theory_from_histogram(
            profile,
            q=q,
            max_epoch=max_epoch,
        )
        ax.plot(
            epochs,
            coverage,
            color=color,
            linestyle="--",
            linewidth=1.05,
            alpha=0.95,
            zorder=1,
        )

    empirical_epochs, empirical_means, empirical_stds, _ = epoch_stats(
        profile.runs,
        "empirical",
        f"realized_coverage_{profile.group}",
    )
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
    q8_color = Q_PALETTE[Q_VALUES.index(EMPIRICAL_Q)]
    ax.plot(
        [epoch for epoch, _, _ in empirical],
        [mean for _, mean, _ in empirical],
        color=q8_color,
        linewidth=1.75,
        marker="o",
        markevery=max(1, len(empirical) // 7),
        markersize=2.8,
        markeredgecolor="white",
        markeredgewidth=0.35,
        zorder=3,
    )
    if len(profile.runs) > 1 and any(std > 0 for _, _, std in empirical):
        ax.fill_between(
            [epoch for epoch, _, _ in empirical],
            [max(0.0, mean - std) for _, mean, std in empirical],
            [min(1.0, mean + std) for _, mean, std in empirical],
            color=q8_color,
            alpha=0.14,
            linewidth=0,
            zorder=2,
        )

    add_facet_title(ax, profile)
    ax.text(
        0.03,
        0.04,
        f"n={len(profile.runs)}",
        transform=ax.transAxes,
        fontsize=6.0,
        color=NEUTRAL_COLOR,
        ha="left",
    )
    ax.set_xlim(1, max_epoch)
    ax.set_ylim(0.0, 1.02)
    ax.set_yticks((0.0, 0.5, 1.0))
    ax.set_ylabel(r"$R_{q,T}$" if show_ylabel else "")
    ax.set_xlabel(r"$T$" if show_xlabel else "")
    if not show_ylabel:
        ax.tick_params(axis="y", labelleft=False)
    if not show_xlabel:
        ax.tick_params(axis="x", labelbottom=False)
    ax.axhline(1.0, color=REFERENCE_COLOR, linewidth=0.65, zorder=0)
    style_axis(ax)


def add_curve_key(ax: plt.Axes) -> None:
    """Place the theory/empirical line-style key inside a recovery facet."""
    q8_color = Q_PALETTE[Q_VALUES.index(EMPIRICAL_Q)]
    handles = [
        Line2D(
            [0],
            [0],
            color=INK_COLOR,
            linewidth=1.05,
            linestyle="--",
            label="Expected",
        ),
        Line2D(
            [0],
            [0],
            color=q8_color,
            linewidth=1.75,
            marker="o",
            markersize=2.8,
            label=r"Empirical ($q=8$)",
        ),
    ]
    ax.legend(
        handles=handles,
        loc="lower right",
        bbox_to_anchor=(0.99, 0.02),
        fontsize=5.8,
        handlelength=2.5,
        handletextpad=0.6,
        borderaxespad=0.0,
        labelspacing=0.35,
    )


def draw_observable_structures_panel(
    ax: plt.Axes,
    profiles: list[ProfileView],
) -> None:
    """Draw the asymptotic q-observable fraction for each structure family."""
    for profile in profiles:
        rank = int(profile.group.removeprefix("rank"))
        coverage = observable_ceiling(profile.runs[0], rank=rank)
        style = FAMILY_STYLES[profile.key]
        ax.plot(
            Q_VALUES,
            coverage,
            color=style["color"],
            linestyle=style["linestyle"],
            linewidth=1.55,
            marker=style["marker"],
            markersize=3.8,
            markeredgecolor="white",
            markeredgewidth=0.45,
            label=profile.legend_label,
            zorder=3,
        )
    ax.set_title(
        "Observable structures",
        loc="left",
        fontweight="bold",
    )
    ax.set_xscale("log", base=2)
    ax.set_xlim(0.85, 39)
    ax.set_ylim(0.0, 1.025)
    ax.set_xticks(Q_VALUES)
    ax.set_xticklabels([str(q) for q in Q_VALUES])
    ax.set_yticks((0.0, 0.25, 0.5, 0.75, 1.0))
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    ax.set_xlabel(r"$q$")
    ax.set_ylabel(r"$|S_q^\ast|/|S^\ast|$")
    ax.axhline(1.0, color=REFERENCE_COLOR, linewidth=0.65, zorder=0)
    ax.legend(
        loc="lower right",
        fontsize=5.6,
        handlelength=2.2,
        handletextpad=0.55,
        labelspacing=0.35,
        borderaxespad=0.4,
    )
    style_axis(ax)


def draw_structure_count_panel(
    ax: plt.Axes,
    profiles: list[ProfileView],
) -> None:
    """Draw target-structure counts on a logarithmic horizontal scale."""
    short_labels = [
        "1-hop hyperedges",
        "2-simplices",
        "Cycle-basis 2-cells",
        "Cycle-span 2-cells",
    ]
    totals = [sum(count for _, count in target_histogram(p)) for p in profiles]
    for y, (profile, total) in enumerate(zip(profiles, totals, strict=True)):
        color = FAMILY_STYLES[profile.key]["color"]
        ax.barh(
            y,
            total - 1,
            left=1,
            height=0.56,
            color=color,
            alpha=0.88,
            edgecolor="none",
        )
        ax.text(
            total * 1.10,
            y,
            f"{total:,}",
            fontsize=6.5,
            color=color,
            fontweight="bold",
            ha="left",
            va="center",
        )
    ax.set_title(
        "Global higher-order structures",
        loc="left",
        fontweight="bold",
    )
    ax.set_xscale("log")
    ax.set_xlim(1, max(totals) * 2.1)
    ax.set_xlabel(r"$|S^\ast|$")
    ax.set_yticks(range(len(profiles)))
    ax.set_yticklabels(short_labels)
    ax.invert_yaxis()
    ax.tick_params(axis="y", length=0, pad=3)
    style_axis(ax)


def plot_structural_coverage_preview(
    profiles: list[ProfileView],
    output_dir: Path,
    *,
    max_epoch: int,
) -> Path:
    """Render appendix figure 1: recovery, blind spots, and target scale."""
    fig = plt.figure(figsize=(FIGURE_WIDTH, 7.55))
    outer = fig.add_gridspec(
        2,
        1,
        height_ratios=(2.15, 1.0),
        hspace=0.43,
    )
    hero = outer[0].subgridspec(
        2,
        3,
        width_ratios=(1.0, 1.0, 0.25),
        hspace=0.48,
        wspace=0.26,
    )
    hero_axes = [
        fig.add_subplot(hero[0, 0]),
        fig.add_subplot(hero[0, 1]),
        fig.add_subplot(hero[1, 0]),
        fig.add_subplot(hero[1, 1]),
    ]
    key_ax = fig.add_subplot(hero[:, 2])
    diagnostics = outer[1].subgridspec(1, 2, wspace=0.34)
    ceiling_ax = fig.add_subplot(diagnostics[0, 0])
    count_ax = fig.add_subplot(diagnostics[0, 1])

    for index, (ax, profile) in enumerate(
        zip(hero_axes, profiles, strict=True)
    ):
        draw_recovery_facet(
            ax,
            profile,
            max_epoch=max_epoch,
            show_ylabel=index in (0, 2),
            show_xlabel=index in (2, 3),
        )
    add_curve_key(hero_axes[0])
    draw_q_key(key_ax)
    draw_observable_structures_panel(ceiling_ax, profiles)
    draw_structure_count_panel(count_ax, profiles)

    hero_left = hero_axes[0].get_position().x0
    hero_top = hero_axes[0].get_position().y1
    fig.text(
        hero_left - 0.036,
        hero_top + 0.062,
        "a",
        fontsize=9,
        fontweight="bold",
        ha="left",
    )
    fig.text(
        hero_left,
        hero_top + 0.062,
        "Cumulative structural coverage",
        fontsize=9,
        fontweight="bold",
        ha="left",
    )
    for ax, label in ((ceiling_ax, "b"), (count_ax, "c")):
        ax.text(
            -0.13,
            1.07,
            label,
            transform=ax.transAxes,
            fontsize=9,
            fontweight="bold",
            ha="left",
            va="bottom",
        )

    fig.suptitle(
        "Structural coverage under clustered mini-batch lifting",
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
        fontsize=6.3,
        color=NEUTRAL_COLOR,
    )
    fig.text(
        0.075,
        0.018,
        "Panel a: expected recovery for q=1-32; solid curves and shading show "
        "the empirical q=8 mean ±1 s.d. Panel b shows the asymptotic "
        "q-observable fraction.",
        fontsize=6.0,
        color=NEUTRAL_COLOR,
        ha="left",
    )
    fig.subplots_adjust(top=0.88, bottom=0.10, left=0.09, right=0.985)
    return save_figure(
        fig,
        output_dir,
        "appendix_structural_coverage_design",
    )


def plot_entropy_preview(
    profiles: list[ProfileView],
    output_dir: Path,
    *,
    max_epoch: int,
) -> Path:
    """Render appendix figure 2: four target-rank entropy facets."""
    fig = plt.figure(figsize=(FIGURE_WIDTH, 5.35))
    grid = fig.add_gridspec(
        2,
        3,
        width_ratios=(1.0, 1.0, 0.25),
        hspace=0.50,
        wspace=0.26,
    )
    axes = [
        fig.add_subplot(grid[0, 0]),
        fig.add_subplot(grid[0, 1]),
        fig.add_subplot(grid[1, 0]),
        fig.add_subplot(grid[1, 1]),
    ]
    key_ax = fig.add_subplot(grid[:, 2])

    entropy_by_profile: list[list[tuple[int, list[float], list[float]]]] = []
    all_values: list[float] = []
    for profile in profiles:
        series = []
        for q in Q_VALUES:
            epochs, _, entropy = theory_from_histogram(
                profile,
                q=q,
                max_epoch=max_epoch,
            )
            series.append((q, epochs, entropy))
            all_values.extend(entropy)
        entropy_by_profile.append(series)
    ymax = max(all_values, default=0.0) * 1.08

    for index, (ax, profile, series) in enumerate(
        zip(axes, profiles, entropy_by_profile, strict=True)
    ):
        for (q, epochs, entropy), color in zip(
            series,
            Q_PALETTE,
            strict=True,
        ):
            ax.plot(
                epochs,
                entropy,
                color=color,
                linewidth=1.5,
                label=f"q={q}",
            )
        add_facet_title(ax, profile)
        ax.set_xlim(1, max_epoch)
        ax.set_ylim(0.0, ymax)
        ax.set_ylabel(
            "Entropy per global structure (nats)" if index in (0, 2) else ""
        )
        ax.set_xlabel("Epoch" if index in (2, 3) else "")
        if index not in (0, 2):
            ax.tick_params(axis="y", labelleft=False)
        if index not in (2, 3):
            ax.tick_params(axis="x", labelbottom=False)
        style_axis(ax)
        ax.text(
            -0.13,
            1.08,
            chr(ord("a") + index),
            transform=ax.transAxes,
            fontsize=9,
            fontweight="bold",
            ha="left",
            va="bottom",
        )
    draw_q_key(key_ax, entropy_only=True)

    fig.suptitle(
        "Cumulative recovery entropy",
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
        "Design preview · legacy span histograms",
        ha="right",
        va="top",
        fontsize=6.3,
        color=NEUTRAL_COLOR,
    )
    fig.text(
        0.075,
        0.018,
        "Reported higher-order structures are 1-hop hyperedges, 2-simplices, "
        "cycle-basis 2-cells and cycle-span 2-cells.",
        fontsize=6.0,
        color=NEUTRAL_COLOR,
        ha="left",
    )
    fig.subplots_adjust(top=0.86, bottom=0.11, left=0.09, right=0.985)
    return save_figure(fig, output_dir, "appendix_entropy_design")


def parse_args() -> argparse.Namespace:
    """Parse legacy roots and preview options."""
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
            "legacy_appendix_observable_preview"
        ),
    )
    parser.add_argument("--max-epoch", type=int, default=50)
    return parser.parse_args()


def main() -> None:
    """Load profiles and render both appendix design previews."""
    args = parse_args()
    profiles = [
        load_profile(
            key="hypergraph",
            heading="1-hop neighbourhood hypergraph",
            structure_label="1-hop hyperedges",
            legend_label="1-hop hyperedges",
            group="rank1",
            root=args.hypergraph_root,
        ),
        load_profile(
            key="simplicial",
            heading="Clique complex",
            structure_label="2-simplices",
            legend_label="2-simplices",
            group="rank2",
            root=args.simplicial_root,
        ),
        load_profile(
            key="cell_basis",
            heading="Cycle-basis cell lifting",
            structure_label="cycle-basis 2-cells",
            legend_label="Cycle-basis 2-cells",
            group="rank2",
            root=args.cell_basis_root,
        ),
        load_profile(
            key="cell_simple",
            heading="Cycle-span cell lifting",
            structure_label="cycle-span 2-cells",
            legend_label="Cycle-span 2-cells",
            group="rank2",
            root=args.cell_simple_root,
        ),
    ]
    output_dir = Path(args.output_dir)
    coverage = plot_structural_coverage_preview(
        profiles,
        output_dir,
        max_epoch=args.max_epoch,
    )
    entropy = plot_entropy_preview(
        profiles,
        output_dir,
        max_epoch=args.max_epoch,
    )
    print(f"Wrote structural coverage preview to {coverage}")
    print(f"Wrote entropy preview to {entropy}")


if __name__ == "__main__":
    main()
