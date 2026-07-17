"""Render the three-part appendix figure organization for the real sweep.

The export bundle contains: (1) structural observability and global target-set
sizes, (2) four-family recovery entropy, and (3) cumulative recovery with two
compact local-lifting panels above two full-width cellular-lifting panels. It
also exports symmetric local/cellular pairs and a symmetric four-panel grid.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt

from scripts.structural_coverage.plot_appendix_sweep_results_combined import (
    NEUTRAL_COLOR,
    Q_PALETTE,
    Q_VALUES,
    REFERENCE_COLOR,
    ProfileData,
    add_curve_key,
    add_facet_title,
    discover_profiles,
    draw_q_key,
    draw_recovery_facet,
    style_axis,
    write_source_data,
)
from scripts.structural_coverage.plot_appendix_sweep_results_split import (
    draw_count_panel,
    draw_observable_panel,
)
from scripts.structural_coverage.plot_results import (
    FIGURE_WIDTH,
    epoch_stats,
    save_figure,
)

FOCUS_Q = 8
CONTEXT_COLOR = "#D8E0E3"


def add_aligned_panel_labels(
    fig: plt.Figure,
    axes: list[plt.Axes],
    *,
    y_axes: float,
    labels: str = "abcd",
) -> None:
    """Align panel labels to one axis-relative title baseline."""
    fig.canvas.draw()
    labels = labels[: len(axes)]
    if len(labels) != len(axes):
        raise ValueError("Panel label count must match the number of axes")
    for ax, label in zip(axes, labels, strict=True):
        _, y_figure = fig.transFigure.inverted().transform(
            ax.transAxes.transform((0.0, y_axes))
        )
        fig.text(
            ax.get_position().x0 - 0.035,
            y_figure,
            label,
            fontsize=9,
            fontweight="bold",
            ha="left",
            va="center",
        )


def draw_focused_cell_recovery_facet(
    ax: plt.Axes,
    profile: ProfileData,
    *,
    max_epoch: int,
    show_ylabel: bool = True,
    show_xlabel: bool = True,
) -> None:
    """Emphasize one representative q while retaining the full sweep."""
    focus_color = Q_PALETTE[Q_VALUES.index(FOCUS_Q)]
    focus_endpoint: tuple[int, float] | None = None

    for q in Q_VALUES:
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
            for epoch, value in zip(
                theory_epochs,
                theory_means,
                strict=True,
            )
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
        is_focus = q == FOCUS_Q
        color = focus_color if is_focus else CONTEXT_COLOR

        ax.plot(
            [epoch for epoch, _ in theory],
            [value for _, value in theory],
            color=color,
            linestyle="--",
            linewidth=1.15 if is_focus else 0.70,
            alpha=0.98 if is_focus else 0.90,
            zorder=3 if is_focus else 1,
        )
        ax.plot(
            [epoch for epoch, _, _ in empirical],
            [mean for _, mean, _ in empirical],
            color=color,
            linewidth=1.90 if is_focus else 0.85,
            marker="o" if is_focus else None,
            markevery=max(1, len(empirical) // 8),
            markersize=2.7 if is_focus else 0.0,
            markeredgecolor="white",
            markeredgewidth=0.35,
            alpha=1.0 if is_focus else 0.95,
            zorder=4 if is_focus else 2,
        )
        if is_focus and any(std > 0.0 for _, _, std in empirical):
            ax.fill_between(
                [epoch for epoch, _, _ in empirical],
                [max(0.0, mean - std) for _, mean, std in empirical],
                [min(1.0, mean + std) for _, mean, std in empirical],
                color=focus_color,
                alpha=0.12,
                linewidth=0,
                zorder=2.5,
            )
        if is_focus and empirical:
            focus_endpoint = empirical[-1][0], empirical[-1][1]

    if focus_endpoint is not None:
        ax.annotate(
            rf"$q={FOCUS_Q}$",
            xy=focus_endpoint,
            xytext=(-7, 6),
            textcoords="offset points",
            color=focus_color,
            fontsize=6.4,
            fontweight="bold",
            ha="right",
            va="bottom",
        )

    add_facet_title(ax, profile.spec)
    seed_count = len(profile.runs_by_q[FOCUS_Q])
    ax.text(
        0.03,
        0.04,
        f"n={seed_count}",
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


def add_compact_q_key(ax: plt.Axes) -> None:
    """Place the full q palette as a compact strip inside one panel."""
    x_positions = (0.51, 0.595, 0.68, 0.765, 0.86, 0.96)
    ax.text(
        0.42,
        0.105,
        r"$q$",
        transform=ax.transAxes,
        fontsize=6.8,
        fontweight="bold",
        ha="center",
        va="center",
    )
    for x, q, color in zip(
        x_positions,
        Q_VALUES,
        Q_PALETTE,
        strict=True,
    ):
        ax.plot(
            [x - 0.025, x + 0.025],
            [0.145, 0.145],
            transform=ax.transAxes,
            color=color,
            linewidth=3.8,
            solid_capstyle="butt",
            clip_on=False,
            zorder=6,
        )
        ax.text(
            x,
            0.065,
            str(q),
            transform=ax.transAxes,
            fontsize=5.8,
            ha="center",
            va="center",
        )


def plot_structural_summary(
    profiles: list[ProfileData],
    output_dir: Path,
) -> Path:
    """Render observable fractions and global structure counts."""
    fig, (observable_ax, count_ax) = plt.subplots(
        1,
        2,
        figsize=(FIGURE_WIDTH, 3.15),
        gridspec_kw={"wspace": 0.34},
    )
    draw_observable_panel(observable_ax, profiles)
    draw_count_panel(
        count_ax,
        profiles,
        title="Global higher-order structures",
    )
    # fig.suptitle(
    #     "Structural observability across lifting families",
    #     x=0.075,
    #     y=0.995,
    #     ha="left",
    #     va="top",
    #     fontsize=11,
    #     fontweight="bold",
    # )
    fig.subplots_adjust(top=0.88, bottom=0.16, left=0.09, right=0.985)
    add_aligned_panel_labels(
        fig,
        [observable_ax, count_ax],
        y_axes=1.05,
    )
    return save_figure(fig, output_dir, "appendix_structural_observability")


def plot_recovery_dynamics(
    profiles: list[ProfileData],
    output_dir: Path,
    *,
    max_epoch: int,
) -> Path:
    """Render two compact local and two full-width cellular recovery panels."""
    fig = plt.figure(figsize=(FIGURE_WIDTH, 8.25))
    grid = fig.add_gridspec(
        3,
        3,
        width_ratios=(1.0, 1.0, 0.22),
        height_ratios=(1.0, 1.0, 1.0),
        hspace=0.58,
        wspace=0.26,
    )
    axes = [
        fig.add_subplot(grid[0, 0]),
        fig.add_subplot(grid[0, 1]),
        fig.add_subplot(grid[1, 0:2]),
        fig.add_subplot(grid[2, 0:2]),
    ]
    key_ax = fig.add_subplot(grid[:, 2])

    for index, (ax, profile) in enumerate(zip(axes, profiles, strict=True)):
        if index < 2:
            draw_recovery_facet(
                ax,
                profile,
                max_epoch=max_epoch,
                show_ylabel=index != 1,
                show_xlabel=True,
            )
        else:
            draw_focused_cell_recovery_facet(
                ax,
                profile,
                max_epoch=max_epoch,
            )
    add_curve_key(axes[0])
    draw_q_key(key_ax)

    # fig.suptitle(
    #     "Cumulative structural coverage",
    #     x=0.075,
    #     y=0.995,
    #     ha="left",
    #     va="top",
    #     fontsize=11,
    #     fontweight="bold",
    # )
    fig.subplots_adjust(top=0.94, bottom=0.055, left=0.09, right=0.985)
    add_aligned_panel_labels(fig, axes, y_axes=1.125)
    return save_figure(fig, output_dir, "appendix_recovery_dynamics")


def plot_recovery_pair(
    profiles: list[ProfileData],
    output_dir: Path,
    *,
    max_epoch: int,
    cellular: bool,
) -> Path:
    """Render one symmetric two-panel recovery figure."""
    fig, axes_array = plt.subplots(
        1,
        2,
        figsize=(FIGURE_WIDTH, 3.0),
        gridspec_kw={"wspace": 0.24},
    )
    axes = list(axes_array)
    selected = profiles[2:] if cellular else profiles[:2]

    for index, (ax, profile) in enumerate(
        zip(axes, selected, strict=True)
    ):
        if cellular:
            draw_focused_cell_recovery_facet(
                ax,
                profile,
                max_epoch=max_epoch,
                show_ylabel=index == 0,
                show_xlabel=True,
            )
        else:
            draw_recovery_facet(
                ax,
                profile,
                max_epoch=max_epoch,
                show_ylabel=index == 0,
                show_xlabel=True,
            )

    add_curve_key(axes[0])
    if cellular:
        labels = "cd"
        stem = "appendix_recovery_cellular"
    else:
        add_compact_q_key(axes[1])
        labels = "ab"
        stem = "appendix_recovery_local"

    fig.subplots_adjust(top=0.84, bottom=0.19, left=0.09, right=0.985)
    add_aligned_panel_labels(
        fig,
        axes,
        y_axes=1.125,
        labels=labels,
    )
    return save_figure(fig, output_dir, stem)


def plot_recovery_quadrants(
    profiles: list[ProfileData],
    output_dir: Path,
    *,
    max_epoch: int,
) -> Path:
    """Render all four recovery families in a symmetric two-by-two grid."""
    fig, axes_array = plt.subplots(
        2,
        2,
        figsize=(FIGURE_WIDTH, 5.35),
        gridspec_kw={"hspace": 0.50, "wspace": 0.24},
    )
    axes = list(axes_array.flat)

    for index, (ax, profile) in enumerate(
        zip(axes, profiles, strict=True)
    ):
        if index < 2:
            draw_recovery_facet(
                ax,
                profile,
                max_epoch=max_epoch,
                show_ylabel=index == 0,
                show_xlabel=False,
            )
        else:
            draw_focused_cell_recovery_facet(
                ax,
                profile,
                max_epoch=max_epoch,
                show_ylabel=index == 2,
                show_xlabel=True,
            )

    add_curve_key(axes[0])
    add_compact_q_key(axes[1])
    fig.subplots_adjust(top=0.91, bottom=0.08, left=0.09, right=0.985)
    add_aligned_panel_labels(fig, axes, y_axes=1.125)
    return save_figure(fig, output_dir, "appendix_recovery_quadrants")


def plot_entropy(
    profiles: list[ProfileData],
    output_dir: Path,
    *,
    max_epoch: int,
) -> Path:
    """Render the four-family entropy block with only the shared q key."""
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

    series_by_profile: list[
        list[tuple[int, list[int], list[float], list[float]]]
    ] = []
    all_upper_values: list[float] = []
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
            profile_series.append((q, plot_epochs, plot_means, plot_stds))
            all_upper_values.extend(
                mean + std
                for mean, std in zip(plot_means, plot_stds, strict=True)
            )
        series_by_profile.append(profile_series)
    ymax = max(all_upper_values, default=0.0) * 1.08

    for index, (ax, profile, profile_series) in enumerate(
        zip(axes, profiles, series_by_profile, strict=True)
    ):
        for (_, epochs, means, stds), color in zip(
            profile_series,
            Q_PALETTE,
            strict=True,
        ):
            ax.plot(epochs, means, color=color, linewidth=1.45)
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
                    alpha=0.10,
                    linewidth=0,
                )
        add_facet_title(ax, profile.spec)
        ax.set_xlim(1, max_epoch)
        ax.set_ylim(0.0, ymax)
        ax.set_ylabel(r"$H_{q,T}$" if index in (0, 2) else "")
        ax.set_xlabel(r"$T$" if index in (2, 3) else "")
        if index not in (0, 2):
            ax.tick_params(axis="y", labelleft=False)
        if index not in (2, 3):
            ax.tick_params(axis="x", labelbottom=False)
        style_axis(ax)

    draw_q_key(key_ax)
    # fig.suptitle(
    #     "Cumulative recovery entropy",
    #     x=0.075,
    #     y=0.995,
    #     ha="left",
    #     va="top",
    #     fontsize=11,
    #     fontweight="bold",
    # )
    fig.subplots_adjust(top=0.91, bottom=0.08, left=0.09, right=0.985)
    add_aligned_panel_labels(fig, axes, y_axes=1.125)
    return save_figure(fig, output_dir, "appendix_recovery_entropy")


def parse_args() -> argparse.Namespace:
    """Parse the portable sweep and output locations."""
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
            "cora_np64_sweep/appendix_figures_three_part"
        ),
    )
    parser.add_argument("--max-epoch", type=int, default=200)
    return parser.parse_args()


def main() -> None:
    """Validate the sweep and render the appendix layout alternatives."""
    args = parse_args()
    output_dir = Path(args.output_dir)
    profiles = discover_profiles(Path(args.sweep_root))

    summary = plot_structural_summary(profiles, output_dir)
    entropy = plot_entropy(
        profiles,
        output_dir,
        max_epoch=args.max_epoch,
    )
    recovery = plot_recovery_dynamics(
        profiles,
        output_dir,
        max_epoch=args.max_epoch,
    )
    local_recovery = plot_recovery_pair(
        profiles,
        output_dir,
        max_epoch=args.max_epoch,
        cellular=False,
    )
    cellular_recovery = plot_recovery_pair(
        profiles,
        output_dir,
        max_epoch=args.max_epoch,
        cellular=True,
    )
    quadrant_recovery = plot_recovery_quadrants(
        profiles,
        output_dir,
        max_epoch=args.max_epoch,
    )
    source_data = write_source_data(
        profiles,
        output_dir,
        max_epoch=args.max_epoch,
    )

    print(f"Wrote structural summary to {summary}")
    print(f"Wrote recovery entropy to {entropy}")
    print(f"Wrote recovery dynamics to {recovery}")
    print(f"Wrote local recovery pair to {local_recovery}")
    print(f"Wrote cellular recovery pair to {cellular_recovery}")
    print(f"Wrote recovery quadrants to {quadrant_recovery}")
    print(f"Wrote source data to {source_data}")


if __name__ == "__main__":
    main()
