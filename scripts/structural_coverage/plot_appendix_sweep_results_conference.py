"""Render compact conference-style alternatives for the Cora sweep figures."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter

from scripts.structural_coverage.plot_appendix_sweep_results_combined import (
    Q_VALUES,
    ProfileData,
    discover_profiles,
)
from scripts.structural_coverage.plot_appendix_sweep_results_split import (
    target_count,
)
from scripts.structural_coverage.plot_results import (
    FIGURE_WIDTH,
    epoch_stats,
    save_figure,
    to_float,
)

INK = "#2F3437"
MUTED = "#6B7280"
GRID = "#E5E7EB"
CONTEXT = "#CDD2D7"
FOCUS_Q = 8
Q_COLORS = (
    "#4C78A8",
    "#F58518",
    "#54A24B",
    "#E45756",
    "#B279A2",
    "#9D755D",
)
PROFILE_COLORS = ("#4C78A8", "#F58518", "#54A24B", "#E45756")


def configure_style() -> None:
    """Apply a familiar compact ML-conference plotting style."""
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 7.5,
            "axes.titlesize": 8.5,
            "axes.labelsize": 8.0,
            "xtick.labelsize": 7.0,
            "ytick.labelsize": 7.0,
            "legend.fontsize": 6.5,
            "axes.edgecolor": INK,
            "axes.labelcolor": INK,
            "text.color": INK,
            "xtick.color": INK,
            "ytick.color": INK,
            "axes.linewidth": 0.75,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": True,
            "legend.framealpha": 1.0,
            "legend.facecolor": "white",
            "legend.edgecolor": GRID,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def style_axis(ax: plt.Axes, *, grid_axis: str = "y") -> None:
    """Apply conventional light gridlines and outward ticks."""
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.55, zorder=0)
    ax.tick_params(direction="out", length=3.0, width=0.7)
    ax.set_axisbelow(True)


def add_panel_title(ax: plt.Axes, letter: str, title: str) -> None:
    """Use an ordinary left-aligned conference subplot title."""
    ax.set_title(
        f"({letter})  {title}",
        loc="left",
        fontsize=8.5,
        fontweight="semibold",
        pad=6,
    )


def coverage_series(
    profile: ProfileData,
    q: int,
    *,
    max_epoch: int,
) -> tuple[list[tuple[int, float]], list[tuple[int, float, float]]]:
    """Return filtered expected and empirical recovery aggregates."""
    runs = profile.runs_by_q[q]
    theory_epochs, theory_means, _, _ = epoch_stats(
        runs,
        "theory",
        f"expected_coverage_{profile.spec.group}",
    )
    empirical_epochs, empirical_means, empirical_stds, _ = epoch_stats(
        runs,
        "empirical",
        f"realized_coverage_{profile.spec.group}",
    )
    theory = [
        (epoch, value)
        for epoch, value in zip(theory_epochs, theory_means, strict=True)
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
    return theory, empirical


def draw_recovery_panel(
    ax: plt.Axes,
    profile: ProfileData,
    *,
    letter: str,
    max_epoch: int,
    show_ylabel: bool,
    show_xlabel: bool,
) -> None:
    """Draw one conventional recovery panel with cellular focus/context."""
    cellular = profile.spec.key in {"cell_basis", "cell_simple_coverage"}

    for q, q_color in zip(Q_VALUES, Q_COLORS, strict=True):
        theory, empirical = coverage_series(
            profile,
            q,
            max_epoch=max_epoch,
        )
        foreground = not cellular or q == FOCUS_Q
        color = q_color if foreground else CONTEXT

        ax.plot(
            [epoch for epoch, _ in theory],
            [value for _, value in theory],
            color=color,
            linestyle="--",
            linewidth=1.0 if foreground else 0.7,
            alpha=0.95 if foreground else 0.72,
            zorder=2 if foreground else 1,
        )
        ax.plot(
            [epoch for epoch, _, _ in empirical],
            [mean for _, mean, _ in empirical],
            color=color,
            linewidth=1.55 if foreground else 0.8,
            marker="o" if foreground else None,
            markevery=max(1, len(empirical) // 7),
            markersize=2.4 if foreground else 0.0,
            markeredgecolor="white",
            markeredgewidth=0.35,
            alpha=1.0 if foreground else 0.78,
            zorder=4 if foreground else 2,
        )
        if foreground and any(std > 0.0 for _, _, std in empirical):
            ax.fill_between(
                [epoch for epoch, _, _ in empirical],
                [max(0.0, mean - std) for _, mean, std in empirical],
                [min(1.0, mean + std) for _, mean, std in empirical],
                color=color,
                alpha=0.10,
                linewidth=0,
                zorder=3,
            )

    add_panel_title(ax, letter, profile.spec.label)
    ax.text(
        0.97,
        0.04,
        f"n={len(profile.runs_by_q[Q_VALUES[0]])}",
        transform=ax.transAxes,
        color=MUTED,
        fontsize=6.2,
        ha="right",
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
    style_axis(ax)


def recovery_legend(fig: plt.Figure) -> None:
    """Add one conventional combined line-style and q legend."""
    handles = [
        Line2D([0], [0], color=INK, linestyle="--", linewidth=1.0),
        Line2D([0], [0], color=INK, linewidth=1.55, marker="o", markersize=3),
    ]
    labels = ["Expected", "Empirical"]
    handles.extend(
        Line2D([0], [0], color=color, linewidth=2.4)
        for color in Q_COLORS
    )
    labels.extend(f"q={q}" for q in Q_VALUES)
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.005),
        ncol=8,
        columnspacing=1.0,
        handlelength=2.0,
        handletextpad=0.45,
        borderpad=0.45,
    )


def plot_recovery(
    profiles: list[ProfileData],
    output_dir: Path,
    *,
    max_epoch: int,
) -> Path:
    """Render a standard two-by-two recovery comparison."""
    fig, axes_array = plt.subplots(
        2,
        2,
        figsize=(FIGURE_WIDTH, 5.0),
        sharex=True,
        sharey=True,
    )
    axes = list(axes_array.flat)
    for index, (ax, profile, letter) in enumerate(
        zip(axes, profiles, "abcd", strict=True)
    ):
        draw_recovery_panel(
            ax,
            profile,
            letter=letter,
            max_epoch=max_epoch,
            show_ylabel=index in (0, 2),
            show_xlabel=index in (2, 3),
        )
    recovery_legend(fig)
    fig.subplots_adjust(
        top=0.96,
        bottom=0.16,
        left=0.09,
        right=0.985,
        hspace=0.38,
        wspace=0.20,
    )
    return save_figure(fig, output_dir, "conference_recovery")


def plot_entropy(
    profiles: list[ProfileData],
    output_dir: Path,
    *,
    max_epoch: int,
) -> Path:
    """Render conventional entropy small multiples."""
    fig, axes_array = plt.subplots(
        2,
        2,
        figsize=(FIGURE_WIDTH, 4.8),
        sharex=True,
        sharey=True,
    )
    axes = list(axes_array.flat)
    series_by_profile: list[
        list[tuple[list[int], list[float], list[float]]]
    ] = []
    upper_values: list[float] = []

    for profile in profiles:
        profile_series = []
        field = f"normalized_entropy_nats_{profile.spec.group}"
        for q in Q_VALUES:
            epochs, means, stds, _ = epoch_stats(
                profile.runs_by_q[q],
                "theory",
                field,
            )
            filtered = [
                (epoch, mean, std)
                for epoch, mean, std in zip(
                    epochs,
                    means,
                    stds,
                    strict=True,
                )
                if epoch <= max_epoch
            ]
            plot_epochs = [epoch for epoch, _, _ in filtered]
            plot_means = [mean for _, mean, _ in filtered]
            plot_stds = [std for _, _, std in filtered]
            profile_series.append((plot_epochs, plot_means, plot_stds))
            upper_values.extend(
                mean + std
                for mean, std in zip(plot_means, plot_stds, strict=True)
            )
        series_by_profile.append(profile_series)

    ymax = max(upper_values, default=0.0) * 1.08
    for index, (ax, profile, profile_series, letter) in enumerate(
        zip(axes, profiles, series_by_profile, "abcd", strict=True)
    ):
        for (epochs, means, stds), color in zip(
            profile_series,
            Q_COLORS,
            strict=True,
        ):
            ax.plot(epochs, means, color=color, linewidth=1.35)
            if any(std > 0.0 for std in stds):
                ax.fill_between(
                    epochs,
                    [
                        max(0.0, mean - std)
                        for mean, std in zip(means, stds, strict=True)
                    ],
                    [
                        mean + std
                        for mean, std in zip(means, stds, strict=True)
                    ],
                    color=color,
                    alpha=0.08,
                    linewidth=0,
                )
        add_panel_title(ax, letter, profile.spec.label)
        ax.set_xlim(1, max_epoch)
        ax.set_ylim(0.0, ymax)
        ax.set_ylabel(r"$H_{q,T}$" if index in (0, 2) else "")
        ax.set_xlabel(r"$T$" if index in (2, 3) else "")
        if index not in (0, 2):
            ax.tick_params(axis="y", labelleft=False)
        if index not in (2, 3):
            ax.tick_params(axis="x", labelbottom=False)
        style_axis(ax)

    handles = [
        Line2D([0], [0], color=color, linewidth=2.2, label=f"q={q}")
        for q, color in zip(Q_VALUES, Q_COLORS, strict=True)
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.005),
        ncol=6,
        columnspacing=1.2,
        handlelength=2.0,
        borderpad=0.45,
    )
    fig.subplots_adjust(
        top=0.96,
        bottom=0.15,
        left=0.09,
        right=0.985,
        hspace=0.38,
        wspace=0.20,
    )
    return save_figure(fig, output_dir, "conference_entropy")


def observable_values(profile: ProfileData) -> list[float]:
    """Return each q-observable fraction for one profile."""
    field = f"observable_ceiling_{profile.spec.group}"
    values = []
    for q in Q_VALUES:
        value = to_float(profile.runs_by_q[q][0]["theory"][0], field)
        if value is None:
            raise ValueError(f"Missing {field} for {profile.spec.label}, q={q}")
        values.append(value)
    return values


def plot_structural_summary(
    profiles: list[ProfileData],
    output_dir: Path,
) -> Path:
    """Render a conventional observable-fraction and structure-count pair."""
    fig, (observable_ax, count_ax) = plt.subplots(
        1,
        2,
        figsize=(FIGURE_WIDTH, 3.0),
        gridspec_kw={"wspace": 0.36},
    )

    for profile, color in zip(profiles, PROFILE_COLORS, strict=True):
        observable_ax.plot(
            Q_VALUES,
            observable_values(profile),
            color=color,
            linewidth=1.55,
            marker=profile.spec.marker,
            markersize=3.5,
            markeredgecolor="white",
            markeredgewidth=0.4,
            label=profile.spec.label,
        )
    add_panel_title(observable_ax, "a", "Observable structures")
    observable_ax.set_xscale("log", base=2)
    observable_ax.set_xlim(0.85, 39)
    observable_ax.set_ylim(0.0, 1.025)
    observable_ax.set_xticks(Q_VALUES)
    observable_ax.set_xticklabels([str(q) for q in Q_VALUES])
    observable_ax.set_yticks((0.0, 0.25, 0.5, 0.75, 1.0))
    observable_ax.yaxis.set_major_formatter(
        PercentFormatter(xmax=1.0, decimals=0)
    )
    observable_ax.set_xlabel(r"$q$")
    observable_ax.legend(loc="lower right", borderpad=0.5)
    style_axis(observable_ax)

    totals = [target_count(profile) for profile in profiles]
    labels = [profile.spec.label for profile in profiles]
    y_positions = list(range(len(profiles)))
    count_ax.barh(
        y_positions,
        totals,
        color=PROFILE_COLORS,
        height=0.55,
        edgecolor="none",
    )
    for y, total in zip(y_positions, totals, strict=True):
        count_ax.text(
            total * 1.08,
            y,
            f"{total:,}",
            fontsize=6.5,
            va="center",
        )
    add_panel_title(count_ax, "b", "Global higher-order structures")
    count_ax.set_xscale("log")
    count_ax.set_xlim(1, max(totals) * 2.25)
    count_ax.set_yticks(y_positions)
    count_ax.set_yticklabels(labels)
    count_ax.invert_yaxis()
    count_ax.set_xlabel(r"$|S^\ast|$")
    style_axis(count_ax, grid_axis="x")

    fig.subplots_adjust(top=0.91, bottom=0.17, left=0.09, right=0.985)
    return save_figure(fig, output_dir, "conference_structural_summary")


def parse_args() -> argparse.Namespace:
    """Parse portable sweep and output locations."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sweep-root",
        default=(
            "scripts/structural_coverage/results_for_plotting/"
            "cora_np64_sweep"
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=(
            "scripts/structural_coverage/results_for_plotting/"
            "cora_np64_sweep/appendix_figures_conference"
        ),
    )
    parser.add_argument("--max-epoch", type=int, default=200)
    return parser.parse_args()


def main() -> None:
    """Render the three conference-style alternatives."""
    args = parse_args()
    configure_style()
    profiles = discover_profiles(Path(args.sweep_root))
    output_dir = Path(args.output_dir)

    summary = plot_structural_summary(profiles, output_dir)
    recovery = plot_recovery(
        profiles,
        output_dir,
        max_epoch=args.max_epoch,
    )
    entropy = plot_entropy(
        profiles,
        output_dir,
        max_epoch=args.max_epoch,
    )

    print(f"Wrote structural summary to {summary}")
    print(f"Wrote recovery comparison to {recovery}")
    print(f"Wrote recovery entropy to {entropy}")


if __name__ == "__main__":
    main()
