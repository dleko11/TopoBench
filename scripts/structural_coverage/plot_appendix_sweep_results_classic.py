"""Render a compact checkpoint-based alternative for the Cora sweep."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter

from scripts.structural_coverage.coverage import per_epoch_probability
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

INK = "#111827"
BLUE = "#2563EB"
ORANGE = "#D97706"
MID = "#6B7280"
LIGHT = "#CBD5E1"
GRID = "#E5E7EB"
CHECKPOINTS = (10, 50, 200)
SAMPLE_EVERY = 1
GRID_Q_VALUES = (2, 4, 8, 16)
DEFAULT_DECAY_FRACTION = 0.01
DEFAULT_COMBINED_ENTROPY_MAX_EPOCH = 1_000_000
COMBINED_ENTROPY_GRID_SIZE = 1500


def configure_style() -> None:
    """Apply a plain, compact experimental-paper style."""
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 7.3,
            "axes.titlesize": 8.3,
            "axes.labelsize": 7.8,
            "xtick.labelsize": 6.8,
            "ytick.labelsize": 6.8,
            "legend.fontsize": 6.5,
            "axes.edgecolor": INK,
            "axes.labelcolor": INK,
            "text.color": INK,
            "xtick.color": INK,
            "ytick.color": INK,
            "axes.linewidth": 0.75,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def style_axis(ax: plt.Axes) -> None:
    """Add only horizontal reference lines and short outward ticks."""
    ax.grid(axis="y", color=GRID, linewidth=0.55)
    ax.tick_params(direction="out", length=3, width=0.7)
    ax.set_axisbelow(True)


def add_title(ax: plt.Axes, letter: str, title: str) -> None:
    """Add an aligned, conventional subplot heading."""
    ax.set_title(f"({letter})  {title}", loc="left", fontweight="semibold", pad=6)


def statistic_at_epoch(
    profile: ProfileData,
    q: int,
    *,
    source: str,
    field: str,
    epoch: int,
) -> tuple[float, float]:
    """Return the mean and standard deviation at an exact checkpoint."""
    epochs, means, stds, _ = epoch_stats(
        profile.runs_by_q[q],
        source,
        field,
    )
    for observed_epoch, mean, std in zip(epochs, means, stds, strict=True):
        if observed_epoch == epoch:
            return mean, std
    raise ValueError(f"Missing epoch {epoch} for {profile.spec.label}, q={q}")


def plot_recovery_checkpoints(
    profiles: list[ProfileData],
    output_dir: Path,
) -> Path:
    """Compare recovery across q at three fixed training checkpoints."""
    fig, axes_array = plt.subplots(
        2,
        2,
        figsize=(FIGURE_WIDTH, 4.35),
        sharex=True,
        sharey=True,
    )
    axes = list(axes_array.flat)
    positions = list(range(len(Q_VALUES)))
    checkpoint_style = {
        10: (LIGHT, "o", "--", -0.10),
        50: (BLUE, "s", "-", 0.00),
        200: (INK, "o", "-", 0.10),
    }

    for index, (ax, profile, letter) in enumerate(
        zip(axes, profiles, "abcd", strict=True)
    ):
        empirical_field = f"realized_coverage_{profile.spec.group}"
        theory_field = f"expected_coverage_{profile.spec.group}"
        for checkpoint in CHECKPOINTS:
            color, marker, linestyle, offset = checkpoint_style[checkpoint]
            means = []
            stds = []
            for q in Q_VALUES:
                mean, std = statistic_at_epoch(
                    profile,
                    q,
                    source="empirical",
                    field=empirical_field,
                    epoch=checkpoint,
                )
                means.append(mean)
                stds.append(std)
            x_values = [position + offset for position in positions]
            ax.plot(
                x_values,
                means,
                color=color,
                marker=marker,
                linestyle=linestyle,
                linewidth=1.35,
                markersize=3.4,
                markerfacecolor="white" if checkpoint == 50 else color,
                markeredgewidth=0.8,
                zorder=3,
            )
            ax.errorbar(
                x_values,
                means,
                yerr=stds,
                fmt="none",
                ecolor=color,
                elinewidth=0.7,
                capsize=1.7,
                capthick=0.7,
                zorder=2,
            )

        theory_means = [
            statistic_at_epoch(
                profile,
                q,
                source="theory",
                field=theory_field,
                epoch=CHECKPOINTS[-1],
            )[0]
            for q in Q_VALUES
        ]
        ax.scatter(
            [position + 0.22 for position in positions],
            theory_means,
            color=ORANGE,
            marker="x",
            linewidths=1.0,
            s=18,
            zorder=4,
        )

        add_title(ax, letter, profile.spec.label)
        ax.set_xlim(-0.35, len(Q_VALUES) - 0.55)
        ax.set_ylim(0.0, 1.025)
        ax.set_yticks((0.0, 0.25, 0.5, 0.75, 1.0))
        ax.set_xticks(positions)
        ax.set_xticklabels([str(q) for q in Q_VALUES])
        ax.set_ylabel(r"$R_{q,T}$" if index in (0, 2) else "")
        ax.set_xlabel(r"$q$" if index in (2, 3) else "")
        if index not in (0, 2):
            ax.tick_params(axis="y", labelleft=False)
        if index not in (2, 3):
            ax.tick_params(axis="x", labelbottom=False)
        style_axis(ax)

    handles = [
        Line2D([0], [0], color=LIGHT, marker="o", linestyle="--", label=r"$T=10$"),
        Line2D(
            [0],
            [0],
            color=BLUE,
            marker="s",
            markerfacecolor="white",
            label=r"$T=50$",
        ),
        Line2D([0], [0], color=INK, marker="o", label=r"$T=200$"),
        Line2D(
            [0],
            [0],
            color=ORANGE,
            marker="x",
            linestyle="none",
            label=r"Expected at $T=200$",
        ),
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        ncol=4,
        columnspacing=1.5,
        handlelength=2.1,
    )
    fig.subplots_adjust(
        left=0.09,
        right=0.985,
        top=0.96,
        bottom=0.16,
        wspace=0.18,
        hspace=0.38,
    )
    return save_figure(fig, output_dir, "classic_recovery_checkpoints")


def plot_recovery_single_q(
    profiles: list[ProfileData],
    output_dir: Path,
    *,
    q: int,
    max_epoch: int,
    sample_every: int = SAMPLE_EVERY,
) -> Path:
    """Plot one q recovery trajectory with sampled standard-deviation bars."""
    if q not in Q_VALUES:
        raise ValueError(f"q must be one of {Q_VALUES}, received {q}")

    fig, axes_array = plt.subplots(
        2,
        2,
        figsize=(FIGURE_WIDTH, 4.35),
        sharex=True,
        sharey=True,
    )
    axes = list(axes_array.flat)

    for index, (ax, profile, letter) in enumerate(
        zip(axes, profiles, "abcd", strict=True)
    ):
        theory_field = f"expected_coverage_{profile.spec.group}"
        empirical_field = f"realized_coverage_{profile.spec.group}"
        theory_epochs, theory_means, _, _ = epoch_stats(
            profile.runs_by_q[q],
            "theory",
            theory_field,
        )
        empirical_epochs, empirical_means, empirical_stds, _ = epoch_stats(
            profile.runs_by_q[q],
            "empirical",
            empirical_field,
        )
        theory = [
            (epoch, mean)
            for epoch, mean in zip(theory_epochs, theory_means, strict=True)
            if epoch <= max_epoch and epoch % sample_every == 0
        ]
        empirical = [
            (epoch, mean, std)
            for epoch, mean, std in zip(
                empirical_epochs,
                empirical_means,
                empirical_stds,
                strict=True,
            )
            if epoch <= max_epoch and epoch % sample_every == 0
        ]

        ax.plot(
            [epoch for epoch, _ in theory],
            [mean for _, mean in theory],
            color=MID,
            linestyle="--",
            linewidth=1.25,
            zorder=2,
        )
        ax.errorbar(
            [epoch for epoch, _, _ in empirical],
            [mean for _, mean, _ in empirical],
            yerr=[std for _, _, std in empirical],
            color=BLUE,
            marker="o",
            markersize=2.8,
            markerfacecolor="white",
            markeredgewidth=0.8,
            linewidth=1.35,
            elinewidth=0.75,
            capsize=1.8,
            capthick=0.75,
            zorder=3,
        )

        add_title(ax, letter, profile.spec.label)
        ax.set_xlim(sample_every, max_epoch)
        ax.set_ylim(0.0, 1.025)
        ax.set_yticks((0.0, 0.25, 0.5, 0.75, 1.0))
        ax.set_ylabel(r"$R_{q,T}$" if index in (0, 2) else "")
        ax.set_xlabel(r"$T$" if index in (2, 3) else "")
        if index not in (0, 2):
            ax.tick_params(axis="y", labelleft=False)
        if index not in (2, 3):
            ax.tick_params(axis="x", labelbottom=False)
        style_axis(ax)

    seed_count = len(profiles[0].runs_by_q[q])
    handles = [
        Line2D(
            [0],
            [0],
            color=BLUE,
            marker="o",
            markerfacecolor="white",
            linewidth=1.35,
            label="Empirical mean ± 1 s.d.",
        ),
        Line2D(
            [0],
            [0],
            color=MID,
            linestyle="--",
            linewidth=1.25,
            label="Expected",
        ),
    ]
    fig.legend(
        handles=handles,
        title=rf"$q={q}$, $n={seed_count}$",
        loc="lower center",
        bbox_to_anchor=(0.5, 0.005),
        ncol=2,
        columnspacing=1.7,
        handlelength=2.5,
    )
    fig.subplots_adjust(
        left=0.09,
        right=0.985,
        top=0.96,
        bottom=0.17,
        wspace=0.18,
        hspace=0.38,
    )
    return save_figure(
        fig,
        output_dir,
        f"classic_recovery_q{q}_errorbars_every{sample_every}",
    )


def plot_recovery_q_grid(
    profiles: list[ProfileData],
    output_dir: Path,
    *,
    max_epoch: int,
    sample_every: int = SAMPLE_EVERY,
) -> Path:
    """Plot four q values by four lifting families on one page."""
    fig, axes = plt.subplots(
        len(GRID_Q_VALUES),
        len(profiles),
        figsize=(FIGURE_WIDTH, 7.15),
        sharex=True,
        sharey=True,
        squeeze=False,
    )

    for row, q in enumerate(GRID_Q_VALUES):
        for column, profile in enumerate(profiles):
            ax = axes[row, column]
            theory_field = f"expected_coverage_{profile.spec.group}"
            empirical_field = f"realized_coverage_{profile.spec.group}"
            theory_epochs, theory_means, _, _ = epoch_stats(
                profile.runs_by_q[q],
                "theory",
                theory_field,
            )
            empirical_epochs, empirical_means, empirical_stds, _ = epoch_stats(
                profile.runs_by_q[q],
                "empirical",
                empirical_field,
            )
            theory = [
                (epoch, mean)
                for epoch, mean in zip(theory_epochs, theory_means, strict=True)
                if epoch <= max_epoch and epoch % sample_every == 0
            ]
            empirical = [
                (epoch, mean, std)
                for epoch, mean, std in zip(
                    empirical_epochs,
                    empirical_means,
                    empirical_stds,
                    strict=True,
                )
                if epoch <= max_epoch and epoch % sample_every == 0
            ]

            ax.plot(
                [epoch for epoch, _ in theory],
                [mean for _, mean in theory],
                color=MID,
                linestyle="--",
                linewidth=1.0,
                zorder=2,
            )
            ax.errorbar(
                [epoch for epoch, _, _ in empirical],
                [mean for _, mean, _ in empirical],
                yerr=[std for _, _, std in empirical],
                color=BLUE,
                marker="o",
                markersize=1.9,
                markerfacecolor="white",
                markeredgewidth=0.65,
                linewidth=1.05,
                elinewidth=0.55,
                capsize=1.2,
                capthick=0.55,
                zorder=3,
            )
            ax.axhline(1.0, color=GRID, linewidth=0.65, zorder=0)
            ax.set_xlim(sample_every, max_epoch)
            ax.set_ylim(0.0, 1.02)
            ax.set_yticks((0.0, 0.5, 1.0))
            ax.tick_params(direction="out", length=2.5, width=0.65)

            if row == 0:
                ax.set_title(
                    profile.spec.label,
                    fontsize=7.6,
                    fontweight="semibold",
                    pad=6,
                )
            if column == 0:
                ax.set_ylabel(rf"$R_{{{q},T}}$")
            else:
                ax.tick_params(axis="y", labelleft=False)
            if row == len(GRID_Q_VALUES) - 1:
                ax.set_xlabel(r"$T$")
            else:
                ax.tick_params(axis="x", labelbottom=False)

    handles = [
        Line2D(
            [0],
            [0],
            color=BLUE,
            marker="o",
            markerfacecolor="white",
            linewidth=1.1,
            label="Empirical mean\n± 1 s.d.",
        ),
        Line2D(
            [0],
            [0],
            color=MID,
            linestyle="--",
            linewidth=1.0,
            label="Expectation value",
        ),
    ]
    axes[0, -1].legend(
        handles=handles,
        loc="upper right",
        bbox_to_anchor=(0.97, 0.90),
        fontsize=5.9,
        handlelength=2.2,
        labelspacing=0.7,
        borderaxespad=0.0,
    )
    fig.subplots_adjust(
        left=0.085,
        right=0.99,
        top=0.955,
        bottom=0.07,
        wspace=0.13,
        hspace=0.22,
    )
    return save_figure(
        fig,
        output_dir,
        f"classic_recovery_q2_q4_q8_q16_every{sample_every}",
    )


def plot_entropy_q_grid(
    profiles: list[ProfileData],
    output_dir: Path,
    *,
    max_epoch: int,
    sample_every: int = SAMPLE_EVERY,
) -> Path:
    """Plot entropy for four q values by four lifting families."""
    series: dict[tuple[int, str], list[tuple[int, float]]] = {}
    ymax = 0.0
    for q in GRID_Q_VALUES:
        for profile in profiles:
            field = f"normalized_entropy_nats_{profile.spec.group}"
            epochs, means, _, _ = epoch_stats(
                profile.runs_by_q[q],
                "theory",
                field,
            )
            values = [
                (epoch, mean)
                for epoch, mean in zip(epochs, means, strict=True)
                if epoch <= max_epoch and epoch % sample_every == 0
            ]
            series[q, profile.spec.key] = values
            ymax = max(ymax, max((mean for _, mean in values), default=0.0))
    ymax *= 1.05

    fig, axes = plt.subplots(
        len(GRID_Q_VALUES),
        len(profiles),
        figsize=(FIGURE_WIDTH, 7.15),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    for row, q in enumerate(GRID_Q_VALUES):
        for column, profile in enumerate(profiles):
            ax = axes[row, column]
            values = series[q, profile.spec.key]
            ax.plot(
                [epoch for epoch, _ in values],
                [mean for _, mean in values],
                color=BLUE,
                marker="o",
                markersize=1.9,
                markerfacecolor="white",
                markeredgewidth=0.65,
                linewidth=1.05,
            )
            ax.axhline(ymax, color=GRID, linewidth=0.65, zorder=0)
            ax.set_xlim(sample_every, max_epoch)
            ax.set_ylim(0.0, ymax)
            ax.set_yticks((0.0, 0.25, 0.5))
            ax.tick_params(direction="out", length=2.5, width=0.65)

            if row == 0:
                ax.set_title(
                    profile.spec.label,
                    fontsize=7.6,
                    fontweight="semibold",
                    pad=6,
                )
            if column == 0:
                ax.set_ylabel(rf"$H_{{{q},T}}$")
            else:
                ax.tick_params(axis="y", labelleft=False)
            if row == len(GRID_Q_VALUES) - 1:
                ax.set_xlabel(r"$T$")
            else:
                ax.tick_params(axis="x", labelbottom=False)

    fig.subplots_adjust(
        left=0.085,
        right=0.99,
        top=0.955,
        bottom=0.07,
        wspace=0.13,
        hspace=0.22,
    )
    return save_figure(
        fig,
        output_dir,
        f"classic_entropy_q2_q4_q8_q16_every{sample_every}",
    )


EntropyTerms = tuple[list[tuple[float, int]], int]


def entropy_terms(profile: ProfileData, q: int) -> EntropyTerms:
    """Return per-span probabilities and the observable normalization count."""
    run = profile.runs_by_q[q][0]
    rank = int(profile.spec.group.removeprefix("rank"))
    k_eff = int(run["metadata"]["K_eff"])
    terms: list[tuple[float, int]] = []
    observable = 0
    for row in run["spans"]:
        if int(row["rank"]) != rank:
            continue
        count = int(float(row["count"]))
        probability = per_epoch_probability(int(row["span"]), q, k_eff)
        if probability <= 0.0:
            continue
        observable += count
        if probability < 1.0:
            terms.append((probability, count))
    return terms, observable


def normalized_entropy_at_epoch(
    epoch: float,
    terms: list[tuple[float, int]],
    observable: int,
) -> float:
    """Evaluate the exact normalized entropy at an arbitrary epoch."""
    if observable == 0:
        return 0.0
    entropy_nats = 0.0
    for probability, count in terms:
        log_survival = epoch * math.log1p(-probability)
        if log_survival < -745.0:
            continue
        survival = math.exp(log_survival)
        rho = -math.expm1(log_survival)
        if rho <= 0.0 or survival <= 0.0:
            continue
        entropy_nats += count * (
            -rho * math.log(rho) - survival * math.log(survival)
        )
    return entropy_nats / observable


def logarithmic_integer_epochs(
    max_epoch: int,
    *,
    grid_size: int = COMBINED_ENTROPY_GRID_SIZE,
) -> list[int]:
    """Return unique logarithmically spaced integer epochs including endpoints."""
    if max_epoch < 1:
        raise ValueError("max_epoch must be at least 1")
    if grid_size < 2:
        raise ValueError("grid_size must be at least 2")
    log_upper = math.log(max_epoch)
    return sorted(
        {
            1,
            max_epoch,
            *(
                max(
                    1,
                    int(round(math.exp(log_upper * index / (grid_size - 1)))),
                )
                for index in range(grid_size)
            ),
        }
    )


def entropy_peak_decay_milestone(
    profile: ProfileData,
    q: int,
    *,
    decay_fraction: float,
) -> tuple[int, float, int]:
    """Return analytic peak and final decay epochs beyond the saved horizon."""
    terms, observable = entropy_terms(profile, q)
    if observable == 0 or not terms:
        return 1, 0.0, 1

    def entropy_at(epoch: float) -> float:
        return normalized_entropy_at_epoch(epoch, terms, observable)

    half_epochs = [
        math.log(0.5) / math.log1p(-probability)
        for probability, _ in terms
    ]
    latest_component_peak = max(1.0, max(half_epochs))

    peak_grid_size = 6000
    peak_log_upper = math.log(latest_component_peak)
    peak_grid = [
        math.exp(peak_log_upper * index / (peak_grid_size - 1))
        for index in range(peak_grid_size)
    ]
    peak_values = [entropy_at(epoch) for epoch in peak_grid]
    peak_grid_index = max(
        range(peak_grid_size),
        key=lambda index: peak_values[index],
    )

    left_index = max(0, peak_grid_index - 1)
    right_index = min(peak_grid_size - 1, peak_grid_index + 1)
    left = math.log(peak_grid[left_index])
    right = math.log(peak_grid[right_index])
    golden_ratio = (math.sqrt(5.0) - 1.0) / 2.0
    x1 = right - golden_ratio * (right - left)
    x2 = left + golden_ratio * (right - left)
    for _ in range(80):
        if entropy_at(math.exp(x1)) < entropy_at(math.exp(x2)):
            left = x1
            x1 = x2
            x2 = left + golden_ratio * (right - left)
        else:
            right = x2
            x2 = x1
            x1 = right - golden_ratio * (right - left)
    continuous_peak = math.exp((left + right) / 2.0)
    peak_candidates = {
        max(1, int(math.floor(continuous_peak)) + offset)
        for offset in range(-2, 4)
    }
    peak_epoch = max(peak_candidates, key=entropy_at)
    peak_value = entropy_at(peak_epoch)
    threshold = peak_value * decay_fraction

    upper = max(peak_epoch + 1, int(math.ceil(latest_component_peak)))
    while entropy_at(upper) > threshold:
        upper *= 2

    crossing_grid_size = 16000
    crossing_log_upper = math.log(upper)
    crossing_grid = [
        math.exp(crossing_log_upper * index / (crossing_grid_size - 1))
        for index in range(crossing_grid_size)
    ]
    crossing_values = [entropy_at(epoch) for epoch in crossing_grid]
    last_above = max(
        index
        for index, value in enumerate(crossing_values)
        if value > threshold
    )
    lower_epoch = max(1, int(math.floor(crossing_grid[last_above])))
    upper_epoch = int(math.ceil(crossing_grid[last_above + 1]))
    while entropy_at(lower_epoch) <= threshold and lower_epoch > 1:
        lower_epoch -= 1
    while lower_epoch + 1 < upper_epoch:
        midpoint = (lower_epoch + upper_epoch) // 2
        if entropy_at(midpoint) > threshold:
            lower_epoch = midpoint
        else:
            upper_epoch = midpoint
    return peak_epoch, peak_value, upper_epoch


EntropyMilestones = dict[tuple[str, int], tuple[int, float, int]]


def calculate_entropy_milestones(
    profiles: list[ProfileData],
    *,
    decay_fraction: float,
) -> EntropyMilestones:
    """Calculate peak and final decay epochs once for all figure variants."""
    if not 0.0 < decay_fraction < 1.0:
        raise ValueError("decay_fraction must lie strictly between 0 and 1")
    return {
        (profile.spec.key, q): entropy_peak_decay_milestone(
            profile,
            q,
            decay_fraction=decay_fraction,
        )
        for profile in profiles
        for q in GRID_Q_VALUES
    }


def print_entropy_peak_epochs(
    profiles: list[ProfileData],
    milestones: EntropyMilestones,
    *,
    max_epoch: int,
    sample_every: int,
) -> None:
    """Print exact entropy peaks beside the maxima visible in the trace grid."""
    print("Entropy peak epochs:")
    print(
        f"{'profile':<23} {'q':>3} {'exact T':>8} "
        f"{'plotted T':>10}  note"
    )
    for profile in profiles:
        field = f"normalized_entropy_nats_{profile.spec.group}"
        for q in GRID_Q_VALUES:
            exact_peak_epoch, _, _ = milestones[profile.spec.key, q]
            epochs, means, _, _ = epoch_stats(
                profile.runs_by_q[q],
                "theory",
                field,
            )
            plotted_values = [
                (epoch, mean)
                for epoch, mean in zip(epochs, means, strict=True)
                if epoch <= max_epoch and epoch % sample_every == 0
            ]
            if not plotted_values:
                raise ValueError(
                    "No plotted entropy values for "
                    f"{profile.spec.label}, q={q}"
                )
            plotted_peak_epoch, _ = max(
                plotted_values,
                key=lambda item: item[1],
            )
            if exact_peak_epoch > max_epoch:
                note = f"exact peak is beyond T={max_epoch}"
            elif plotted_peak_epoch != exact_peak_epoch:
                note = f"trace is sampled every {sample_every} epochs"
            else:
                note = "aligned"
            print(
                f"{profile.spec.label:<23} {q:>3} "
                f"{exact_peak_epoch:>8} {plotted_peak_epoch:>10}  {note}"
            )


def entropy_milestone_handles(decay_fraction: float) -> list[Line2D]:
    """Return the shared legend for the analytic entropy intervals."""
    threshold_percent = 100.0 * decay_fraction
    return [
        Line2D(
            [0],
            [0],
            marker="o",
            markerfacecolor=INK,
            markeredgecolor=INK,
            color="none",
            label="Peak entropy",
        ),
        Line2D(
            [0],
            [0],
            marker="s",
            markerfacecolor=INK,
            markeredgecolor=INK,
            color="none",
            label=rf"$H_{{q,T}} \leq {threshold_percent:g}\%$ of peak",
        ),
    ]


def draw_entropy_peak_decay_axis(
    ax: plt.Axes,
    profile: ProfileData,
    milestones: EntropyMilestones,
    *,
    show_title: bool,
    show_ylabel: bool,
    right_padding: float = 2.8,
    x_limits: tuple[float, float] | None = None,
) -> None:
    """Draw one lifting family's analytic peak-to-decay intervals."""
    y_positions = list(range(len(GRID_Q_VALUES)))
    profile_milestones = [
        milestones[profile.spec.key, q] for q in GRID_Q_VALUES
    ]
    for y, (peak_epoch, _, decay_epoch) in zip(
        y_positions,
        profile_milestones,
        strict=True,
    ):
        ax.hlines(
            y,
            peak_epoch,
            decay_epoch,
            color=BLUE,
            linewidth=1.45,
            zorder=2,
        )
        ax.scatter(
            peak_epoch,
            y,
            s=32,
            facecolor=INK,
            edgecolor=INK,
            linewidth=0.9,
            zorder=4,
        )
        ax.scatter(
            decay_epoch,
            y,
            s=24,
            marker="s",
            color=INK,
            zorder=4,
        )
        peak_label = (
            f"{peak_epoch:,}"
            if peak_epoch < 100_000
            else f"{peak_epoch:.1e}"
        )
        endpoint_label = (
            f"{decay_epoch:,}"
            if decay_epoch < 100_000
            else f"{decay_epoch:.1e}"
        )
        ax.annotate(
            peak_label,
            xy=(peak_epoch, y),
            xytext=(0, -6),
            textcoords="offset points",
            fontsize=5.6,
            color=MID,
            ha="center",
            va="top",
        )
        ax.annotate(
            endpoint_label,
            xy=(decay_epoch, y),
            xytext=(0, 5),
            textcoords="offset points",
            fontsize=5.6,
            color=MID,
            ha="center",
            va="bottom",
        )

    if show_title:
        ax.set_title(
            profile.spec.label,
            loc="left",
            fontweight="semibold",
            pad=6,
        )
    ax.set_xscale("log")
    if x_limits is None:
        minimum_peak = min(peak for peak, _, _ in profile_milestones)
        maximum_decay = max(decay for _, _, decay in profile_milestones)
        x_limits = (
            max(0.8, minimum_peak / 1.8),
            maximum_decay * right_padding,
        )
    ax.set_xlim(*x_limits)
    ax.set_ylim(-0.55, len(GRID_Q_VALUES) - 0.45)
    ax.set_yticks(y_positions)
    ax.set_yticklabels([str(q) for q in GRID_Q_VALUES])
    ax.set_xlabel(r"$T$")
    ax.set_ylabel(r"$q$" if show_ylabel else "")
    if not show_ylabel:
        ax.tick_params(axis="y", labelleft=False)
    ax.grid(False)
    ax.set_facecolor("white")
    ax.tick_params(direction="out", length=3, width=0.7)


def plot_entropy_peak_decay(
    profiles: list[ProfileData],
    output_dir: Path,
    milestones: EntropyMilestones,
    *,
    decay_fraction: float,
) -> Path:
    """Summarize analytic peak-to-decay intervals on independent log scales."""
    fig, axes_array = plt.subplots(
        2,
        2,
        figsize=(FIGURE_WIDTH, 4.25),
        sharey=True,
    )
    axes = list(axes_array.flat)
    for index, (ax, profile) in enumerate(zip(axes, profiles, strict=True)):
        draw_entropy_peak_decay_axis(
            ax,
            profile,
            milestones,
            show_title=True,
            show_ylabel=index in (0, 2),
        )

    axes[1].legend(
        handles=entropy_milestone_handles(decay_fraction),
        loc="upper right",
        borderaxespad=0.4,
        handletextpad=0.4,
    )
    fig.subplots_adjust(
        left=0.09,
        right=0.97,
        top=0.95,
        bottom=0.12,
        wspace=0.18,
        hspace=0.38,
    )
    threshold_percent = 100.0 * decay_fraction
    threshold_tag = f"{threshold_percent:g}".replace(".", "p")
    return save_figure(
        fig,
        output_dir,
        f"classic_entropy_peak_to_{threshold_tag}pct_analytic",
    )


def plot_entropy_combined(
    profiles: list[ProfileData],
    output_dir: Path,
    milestones: EntropyMilestones,
    *,
    max_epoch: int,
    decay_fraction: float,
) -> Path:
    """Stack exact entropy traces over aligned peak-to-decay summaries."""
    if max_epoch < 1:
        raise ValueError("max_epoch must be at least 1")
    base_epochs = logarithmic_integer_epochs(max_epoch)
    series: dict[tuple[int, str], list[tuple[int, float]]] = {}
    ymax = 0.0
    for q in GRID_Q_VALUES:
        for profile in profiles:
            terms, observable = entropy_terms(profile, q)
            peak_epoch = milestones[profile.spec.key, q][0]
            epochs = base_epochs
            if peak_epoch <= max_epoch and peak_epoch not in base_epochs:
                epochs = sorted((*base_epochs, peak_epoch))
            values = [
                (
                    epoch,
                    normalized_entropy_at_epoch(epoch, terms, observable),
                )
                for epoch in epochs
            ]
            series[q, profile.spec.key] = values
            ymax = max(ymax, max((mean for _, mean in values), default=0.0))
    ymax *= 1.05

    fig = plt.figure(figsize=(FIGURE_WIDTH, 8.65))
    grid = fig.add_gridspec(
        5,
        len(profiles),
        height_ratios=(1.0, 1.0, 1.0, 1.0, 1.25),
        left=0.085,
        right=0.99,
        top=0.975,
        bottom=0.06,
        wspace=0.13,
        hspace=0.27,
    )
    entropy_axes: list[list[plt.Axes]] = []
    for row, q in enumerate(GRID_Q_VALUES):
        row_axes: list[plt.Axes] = []
        for column, profile in enumerate(profiles):
            ax = fig.add_subplot(grid[row, column])
            row_axes.append(ax)
            values = series[q, profile.spec.key]
            ax.plot(
                [epoch for epoch, _ in values],
                [mean for _, mean in values],
                color=BLUE,
                linewidth=1.25,
            )
            ax.set_xscale("log")
            ax.set_xlim(1, max_epoch)
            ax.set_ylim(0.0, ymax)
            ax.set_yticks((0.0, 0.25, 0.5))
            ax.tick_params(direction="out", length=2.5, width=0.65)
            if row == 0:
                ax.set_title(
                    profile.spec.label,
                    fontsize=7.6,
                    fontweight="semibold",
                    pad=6,
                )
            if column == 0:
                ax.set_ylabel(rf"$H_{{{q},T}}$")
            else:
                ax.tick_params(axis="y", labelleft=False)
            if row == len(GRID_Q_VALUES) - 1:
                ax.set_xlabel(r"$T$")
            else:
                ax.tick_params(axis="x", labelbottom=False)
        entropy_axes.append(row_axes)

    interval_axes: list[plt.Axes] = []
    minimum_peak = min(
        peak
        for peak, _, _ in milestones.values()
    )
    maximum_decay = max(
        decay
        for _, _, decay in milestones.values()
    )
    shared_interval_limits = (
        max(0.8, minimum_peak / 1.8),
        maximum_decay * 10.0,
    )
    for column, profile in enumerate(profiles):
        ax = fig.add_subplot(grid[4, column])
        interval_axes.append(ax)
        draw_entropy_peak_decay_axis(
            ax,
            profile,
            milestones,
            show_title=False,
            show_ylabel=column == 0,
            right_padding=10.0,
            x_limits=shared_interval_limits,
        )
    bottom_position = interval_axes[0].get_position()
    interval_axes[-1].legend(
        handles=entropy_milestone_handles(decay_fraction),
        loc="lower right",
        bbox_to_anchor=(0.99, 0.02),
        borderaxespad=0.2,
        handletextpad=0.25,
        labelspacing=0.55,
        fontsize=5.4,
    )

    top_position = entropy_axes[0][0].get_position()
    fig.text(
        0.012,
        top_position.y1,
        "a",
        fontsize=8.5,
        fontweight="bold",
        va="top",
    )
    fig.text(
        0.012,
        bottom_position.y1,
        "b",
        fontsize=8.5,
        fontweight="bold",
        va="top",
    )
    threshold_percent = 100.0 * decay_fraction
    threshold_tag = f"{threshold_percent:g}".replace(".", "p")
    return save_figure(
        fig,
        output_dir,
        f"classic_entropy_combined_q_grid_and_{threshold_tag}pct",
    )


def entropy_summary(
    profile: ProfileData,
    q: int,
    *,
    max_epoch: int,
) -> tuple[float, float]:
    """Return peak entropy and entropy at the last plotted epoch."""
    field = f"normalized_entropy_nats_{profile.spec.group}"
    epochs, means, _, _ = epoch_stats(profile.runs_by_q[q], "theory", field)
    filtered = [
        (epoch, mean)
        for epoch, mean in zip(epochs, means, strict=True)
        if epoch <= max_epoch
    ]
    if not filtered:
        raise ValueError(f"No entropy values for {profile.spec.label}, q={q}")
    return max(mean for _, mean in filtered), filtered[-1][1]


def plot_entropy_dumbbells(
    profiles: list[ProfileData],
    output_dir: Path,
    *,
    max_epoch: int,
) -> Path:
    """Show the peak-to-late entropy change as compact dumbbell plots."""
    summaries = {
        profile.spec.key: [
            entropy_summary(profile, q, max_epoch=max_epoch) for q in Q_VALUES
        ]
        for profile in profiles
    }
    ymax = max(
        peak
        for profile_values in summaries.values()
        for peak, _ in profile_values
    ) * 1.08

    fig, axes_array = plt.subplots(
        2,
        2,
        figsize=(FIGURE_WIDTH, 4.15),
        sharex=True,
        sharey=True,
    )
    axes = list(axes_array.flat)
    positions = list(range(len(Q_VALUES)))

    for index, (ax, profile, letter) in enumerate(
        zip(axes, profiles, "abcd", strict=True)
    ):
        values = summaries[profile.spec.key]
        peaks = [peak for peak, _ in values]
        late = [final for _, final in values]
        ax.vlines(positions, late, peaks, color=LIGHT, linewidth=2.0, zorder=1)
        ax.scatter(
            positions,
            peaks,
            facecolor="white",
            edgecolor=BLUE,
            linewidth=1.1,
            s=24,
            zorder=3,
        )
        ax.scatter(
            positions,
            late,
            color=INK,
            marker="s",
            s=16,
            zorder=3,
        )

        add_title(ax, letter, profile.spec.label)
        ax.set_xlim(-0.4, len(Q_VALUES) - 0.6)
        ax.set_ylim(0.0, ymax)
        ax.set_xticks(positions)
        ax.set_xticklabels([str(q) for q in Q_VALUES])
        ax.set_ylabel(r"$H_{q,T}$" if index in (0, 2) else "")
        ax.set_xlabel(r"$q$" if index in (2, 3) else "")
        if index not in (0, 2):
            ax.tick_params(axis="y", labelleft=False)
        if index not in (2, 3):
            ax.tick_params(axis="x", labelbottom=False)
        style_axis(ax)

    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            markerfacecolor="white",
            markeredgecolor=BLUE,
            color="none",
            label="Peak",
        ),
        Line2D(
            [0],
            [0],
            marker="s",
            markerfacecolor=INK,
            markeredgecolor=INK,
            color="none",
            label=rf"At $T={max_epoch}$",
        ),
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        ncol=2,
        columnspacing=1.7,
    )
    fig.subplots_adjust(
        left=0.09,
        right=0.985,
        top=0.96,
        bottom=0.16,
        wspace=0.18,
        hspace=0.40,
    )
    return save_figure(fig, output_dir, "classic_entropy_peak_to_late")


def observable_values(profile: ProfileData) -> list[float]:
    """Return the q-observable fraction for one lifting profile."""
    field = f"observable_ceiling_{profile.spec.group}"
    values = []
    for q in Q_VALUES:
        value = to_float(profile.runs_by_q[q][0]["theory"][0], field)
        if value is None:
            raise ValueError(f"Missing {field} for {profile.spec.label}, q={q}")
        values.append(value)
    return values


def threshold_q(values: list[float], threshold: float) -> int | None:
    """Return the first q meeting an observability threshold."""
    return next(
        (q for q, value in zip(Q_VALUES, values, strict=True) if value >= threshold),
        None,
    )


def plot_structural_thresholds(
    profiles: list[ProfileData],
    output_dir: Path,
) -> Path:
    """Combine observability thresholds and global counts in a table-like plot."""
    fig, ax = plt.subplots(figsize=(FIGURE_WIDTH, 2.3))
    positions = {q: index for index, q in enumerate(Q_VALUES)}
    y_positions = list(reversed(range(len(profiles))))

    for y, profile in zip(y_positions, profiles, strict=True):
        values = observable_values(profile)
        q95 = threshold_q(values, 0.95)
        q99 = threshold_q(values, 0.99)
        x95 = positions[q95] if q95 is not None else len(Q_VALUES)
        x99 = positions[q99] if q99 is not None else len(Q_VALUES)
        ax.plot([x95, x99], [y, y], color=LIGHT, linewidth=2.0, zorder=1)
        ax.scatter(
            x95,
            y,
            s=28,
            facecolor="white",
            edgecolor=BLUE,
            linewidth=1.1,
            zorder=3,
        )
        ax.scatter(x99, y, s=21, color=INK, marker="s", zorder=3)
        ax.text(
            5.75,
            y,
            f"{target_count(profile):,}",
            ha="left",
            va="center",
            fontsize=7.0,
            color=INK,
        )

    ax.axvline(5.45, color=GRID, linewidth=0.8)
    ax.text(
        5.75,
        y_positions[0] + 0.62,
        r"Global $|S^\ast|$",
        ha="left",
        va="bottom",
        fontsize=7.0,
        fontweight="semibold",
    )
    ax.set_xlim(-0.35, 7.1)
    ax.set_ylim(-0.55, len(profiles) - 0.25)
    ax.set_xticks(list(range(len(Q_VALUES))))
    ax.set_xticklabels([str(q) for q in Q_VALUES])
    ax.set_yticks(y_positions)
    ax.set_yticklabels([profile.spec.label for profile in profiles])
    ax.set_xlabel(r"Minimum $q$ meeting the observability threshold")
    ax.grid(axis="x", color=GRID, linewidth=0.55)
    ax.tick_params(axis="both", direction="out", length=3, width=0.7)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.set_axisbelow(True)

    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            markerfacecolor="white",
            markeredgecolor=BLUE,
            color="none",
            label="95% observable",
        ),
        Line2D(
            [0],
            [0],
            marker="s",
            markerfacecolor=INK,
            markeredgecolor=INK,
            color="none",
            label="99% observable",
        ),
    ]
    ax.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.38, 1.01),
        ncol=2,
        columnspacing=1.6,
    )
    fig.subplots_adjust(left=0.25, right=0.98, top=0.83, bottom=0.23)
    return save_figure(fig, output_dir, "classic_structural_thresholds")


def plot_structural_observability_redesign(
    profiles: list[ProfileData],
    output_dir: Path,
) -> Path:
    """Show three higher-order observability curves in a compact panel."""
    plotted_q = tuple(q for q in Q_VALUES if q <= 16)
    profiles_by_key = {profile.spec.key: profile for profile in profiles}
    series = (
        ("Hyperedges", profiles_by_key["hypergraph"], "#2563EB"),
        ("Cells", profiles_by_key["cell_basis"], "#D97706"),
        ("Simplices", profiles_by_key["simplicial"], "#7C3AED"),
    )
    fig, ax = plt.subplots(figsize=(3.55, 2.05))
    for label, profile, color in series:
        values = observable_values(profile)[: len(plotted_q)]
        ax.plot(
            plotted_q,
            values,
            color=color,
            linewidth=1.45,
            marker="o",
            markersize=4.0,
            markerfacecolor="white",
            markeredgecolor=color,
            markeredgewidth=1.0,
            label=label,
            zorder=3,
        )
    ax.set_xscale("log", base=2)
    ax.set_xlim(0.85, 18.0)
    ax.set_ylim(0.5, 1.015)
    ax.set_xticks(plotted_q)
    ax.set_xticklabels([str(q) for q in plotted_q])
    ax.set_yticks((0.5, 0.75, 1.0))
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    ax.set_xlabel(r"$q$")
    ax.axhline(1.0, color=GRID, linewidth=0.65, zorder=0)
    ax.grid(False)
    ax.tick_params(direction="out", length=2.7, width=0.65)
    ax.legend(
        loc="lower right",
        fontsize=5.7,
        handlelength=2.0,
        handletextpad=0.45,
        labelspacing=0.45,
        borderaxespad=0.55,
    )
    fig.subplots_adjust(left=0.13, right=0.985, top=0.98, bottom=0.20)
    return save_figure(
        fig,
        output_dir,
        "classic_structural_observability_redesign",
    )


def parse_args() -> argparse.Namespace:
    """Parse sweep and output paths."""
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
            "cora_np64_sweep/appendix_figures_classic"
        ),
    )
    parser.add_argument("--max-epoch", type=int, default=200)
    parser.add_argument(
        "--combined-entropy-max-epoch",
        type=int,
        default=DEFAULT_COMBINED_ENTROPY_MAX_EPOCH,
    )
    parser.add_argument("--focus-q", type=int, default=8)
    parser.add_argument("--sample-every", type=int, default=SAMPLE_EVERY)
    parser.add_argument(
        "--entropy-decay-fraction",
        type=float,
        default=DEFAULT_DECAY_FRACTION,
    )
    return parser.parse_args()


def main() -> None:
    """Render the checkpoint-based alternative figure set."""
    args = parse_args()
    configure_style()
    profiles = discover_profiles(Path(args.sweep_root))
    output_dir = Path(args.output_dir)

    structural = plot_structural_thresholds(profiles, output_dir)
    structural_redesign = plot_structural_observability_redesign(
        profiles,
        output_dir,
    )
    recovery = plot_recovery_checkpoints(profiles, output_dir)
    single_q_recovery = plot_recovery_single_q(
        profiles,
        output_dir,
        q=args.focus_q,
        max_epoch=args.max_epoch,
        sample_every=args.sample_every,
    )
    recovery_grid = plot_recovery_q_grid(
        profiles,
        output_dir,
        max_epoch=args.max_epoch,
        sample_every=args.sample_every,
    )
    entropy = plot_entropy_dumbbells(
        profiles,
        output_dir,
        max_epoch=args.max_epoch,
    )
    entropy_grid = plot_entropy_q_grid(
        profiles,
        output_dir,
        max_epoch=args.max_epoch,
        sample_every=args.sample_every,
    )
    entropy_milestones = calculate_entropy_milestones(
        profiles,
        decay_fraction=args.entropy_decay_fraction,
    )
    print_entropy_peak_epochs(
        profiles,
        entropy_milestones,
        max_epoch=args.max_epoch,
        sample_every=args.sample_every,
    )
    entropy_peak_decay = plot_entropy_peak_decay(
        profiles,
        output_dir,
        entropy_milestones,
        decay_fraction=args.entropy_decay_fraction,
    )
    entropy_combined = plot_entropy_combined(
        profiles,
        output_dir,
        entropy_milestones,
        max_epoch=args.combined_entropy_max_epoch,
        decay_fraction=args.entropy_decay_fraction,
    )
    print(f"Wrote structural thresholds to {structural}")
    print(f"Wrote structural observability redesign to {structural_redesign}")
    print(f"Wrote recovery checkpoints to {recovery}")
    print(f"Wrote single-q recovery to {single_q_recovery}")
    print(f"Wrote 16-panel recovery grid to {recovery_grid}")
    print(f"Wrote entropy summary to {entropy}")
    print(f"Wrote 16-panel entropy grid to {entropy_grid}")
    print(f"Wrote entropy peak-decay intervals to {entropy_peak_decay}")
    print(f"Wrote combined entropy figure to {entropy_combined}")


if __name__ == "__main__":
    main()
