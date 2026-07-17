"""Render the saved combined appendix figures from a portable sweep.

The input layout is the one produced by ``scripts.structural_coverage.sweep``::

    SWEEP_ROOT/{profile}/q{q:02d}/seed{seed:02d}/

All empirical and theoretical curves are aggregated across seeds. No missing
profile, q value, seed, epoch, or coverage artifact is silently imputed.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.ticker import PercentFormatter

from scripts.structural_coverage.plot_results import (
    FIGURE_WIDTH,
    epoch_stats,
    load_run,
    save_figure,
    to_float,
)

Q_VALUES = (1, 2, 4, 8, 16, 32)
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


@dataclass(frozen=True)
class ProfileSpec:
    """Plotting metadata for one lifting profile."""

    key: str
    label: str
    group: str
    color: str
    marker: str
    linestyle: str = "-"


@dataclass(frozen=True)
class ProfileData:
    """All real sweep runs for one lifting profile."""

    spec: ProfileSpec
    runs_by_q: dict[int, list[dict[str, Any]]]


PROFILE_SPECS = (
    ProfileSpec(
        key="hypergraph",
        label="1-hop hyperedges",
        group="rank1",
        color="#3B6F9B",
        marker="o",
    ),
    ProfileSpec(
        key="simplicial",
        label="2-simplices",
        group="rank2",
        color="#7763A5",
        marker="D",
    ),
    ProfileSpec(
        key="cell_basis",
        label="Cycle-basis 2-cells",
        group="rank2",
        color="#D8872E",
        marker="s",
        linestyle="--",
    ),
    ProfileSpec(
        key="cell_simple_coverage",
        label="Cycle-span 2-cells",
        group="rank2",
        color="#C5534B",
        marker="^",
    ),
)

REQUIRED_ARTIFACTS = (
    "empirical_coverage.csv",
    "theory_curves.csv",
    "span_histogram.csv",
    "run_metadata.json",
)

plt.rcParams.update(
    {
        "text.color": INK_COLOR,
        "axes.labelcolor": INK_COLOR,
        "axes.edgecolor": INK_COLOR,
        "xtick.color": INK_COLOR,
        "ytick.color": INK_COLOR,
    }
)


def validate_manifest(sweep_root: Path) -> None:
    """Require a manifest that reports a completely successful sweep."""
    path = sweep_root / "manifest_summary.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing sweep manifest summary: {path}")
    summary = json.loads(path.read_text())
    total = int(summary.get("total", 0))
    successes = int(summary.get("counts", {}).get("success", 0))
    if total <= 0 or successes != total:
        raise ValueError(
            f"Sweep is incomplete: {successes} successful tasks out of {total}."
        )


def discover_profiles(sweep_root: Path) -> list[ProfileData]:
    """Load and validate all profile/q/seed runs in a portable sweep."""
    validate_manifest(sweep_root)
    profiles: list[ProfileData] = []
    expected_seeds: set[int] | None = None

    for spec in PROFILE_SPECS:
        runs_by_q: dict[int, list[dict[str, Any]]] = {}
        for q in Q_VALUES:
            q_dir = sweep_root / spec.key / f"q{q:02d}"
            seed_dirs = sorted(q_dir.glob("seed[0-9][0-9]"))
            if not seed_dirs:
                raise FileNotFoundError(f"No seed directories found in {q_dir}")

            q_runs: list[dict[str, Any]] = []
            for seed_dir in seed_dirs:
                missing = [
                    name
                    for name in REQUIRED_ARTIFACTS
                    if not (seed_dir / name).exists()
                ]
                if missing:
                    raise FileNotFoundError(
                        f"Incomplete run {seed_dir}: missing {', '.join(missing)}"
                    )
                run = load_run(seed_dir)
                if int(run["q"]) != q:
                    raise ValueError(
                        f"Directory q={q} but metadata q={run['q']}: {seed_dir}"
                    )
                q_runs.append(run)

            seeds = {int(run["seed"]) for run in q_runs}
            if len(seeds) != len(q_runs):
                raise ValueError(f"Duplicate seed in {q_dir}")
            if expected_seeds is None:
                expected_seeds = seeds
            elif seeds != expected_seeds:
                raise ValueError(
                    f"Unbalanced seeds in {q_dir}: {sorted(seeds)}; "
                    f"expected {sorted(expected_seeds)}"
                )
            runs_by_q[q] = q_runs
        profiles.append(ProfileData(spec=spec, runs_by_q=runs_by_q))

    return profiles


def style_axis(ax: plt.Axes) -> None:
    """Apply the compact shared axis treatment."""
    ax.tick_params(direction="out", length=2.8, width=0.7)
    ax.grid(False)


def add_facet_title(ax: plt.Axes, spec: ProfileSpec) -> None:
    """Add a concise family-coloured facet title."""
    ax.add_patch(
        Rectangle(
            (-0.005, 1.075),
            0.018,
            0.105,
            transform=ax.transAxes,
            facecolor=spec.color,
            edgecolor="none",
            clip_on=False,
        )
    )
    ax.text(
        0.035,
        1.125,
        spec.label,
        transform=ax.transAxes,
        fontsize=8.5,
        fontweight="bold",
        ha="left",
        va="center",
    )


def draw_q_key(ax: plt.Axes, *, entropy_only: bool = False) -> None:
    """Draw the shared discrete q colour key."""
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
        ax.text(0.47, 0.13, "Expected", fontsize=6.2, va="center")


def add_curve_key(ax: plt.Axes) -> None:
    """Place the expected/empirical convention inside the first facet."""
    q8_color = Q_PALETTE[Q_VALUES.index(8)]
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
            label="Empirical",
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


def draw_recovery_facet(
    ax: plt.Axes,
    profile: ProfileData,
    *,
    max_epoch: int,
    show_ylabel: bool,
    show_xlabel: bool,
) -> None:
    """Draw all-q real theory and empirical seed aggregates."""
    for q, color in zip(Q_VALUES, Q_PALETTE, strict=True):
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
                theory_epochs, theory_means, strict=True
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

        ax.plot(
            [epoch for epoch, _ in theory],
            [value for _, value in theory],
            color=color,
            linestyle="--",
            linewidth=1.0,
            alpha=0.95,
            zorder=1,
        )
        ax.plot(
            [epoch for epoch, _, _ in empirical],
            [mean for _, mean, _ in empirical],
            color=color,
            linewidth=1.65,
            marker="o",
            markevery=max(1, len(empirical) // 8),
            markersize=2.5,
            markeredgecolor="white",
            markeredgewidth=0.35,
            zorder=3,
        )
        if any(std > 0.0 for _, _, std in empirical):
            ax.fill_between(
                [epoch for epoch, _, _ in empirical],
                [max(0.0, mean - std) for _, mean, std in empirical],
                [min(1.0, mean + std) for _, mean, std in empirical],
                color=color,
                alpha=0.10,
                linewidth=0,
                zorder=2,
            )

    add_facet_title(ax, profile.spec)
    seed_count = len(profile.runs_by_q[Q_VALUES[0]])
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


def observable_stats(
    profile: ProfileData,
    q: int,
) -> tuple[float, float]:
    """Return mean and sample s.d. of the q-observable fraction."""
    field = f"observable_ceiling_{profile.spec.group}"
    values = []
    for run in profile.runs_by_q[q]:
        value = to_float(run["theory"][0], field)
        if value is None:
            raise ValueError(f"Missing {field} in {run['path']}")
        values.append(value)
    mean = statistics.fmean(values)
    std = statistics.stdev(values) if len(values) > 1 else 0.0
    return mean, std


def draw_observable_structures_panel(
    ax: plt.Axes,
    profiles: list[ProfileData],
) -> None:
    """Draw the real q-observable fraction for each structure family."""
    for profile in profiles:
        means = []
        for q in Q_VALUES:
            mean, _ = observable_stats(profile, q)
            means.append(mean)
        ax.plot(
            Q_VALUES,
            means,
            color=profile.spec.color,
            linestyle=profile.spec.linestyle,
            linewidth=1.55,
            marker=profile.spec.marker,
            markersize=3.8,
            markeredgecolor="white",
            markeredgewidth=0.45,
            label=profile.spec.label,
            zorder=3,
        )
    ax.set_title("Observable structures", loc="left", fontweight="bold")
    ax.set_xscale("log", base=2)
    ax.set_xlim(0.85, 39)
    ax.set_ylim(0.0, 1.025)
    ax.set_xticks(Q_VALUES)
    ax.set_xticklabels([str(q) for q in Q_VALUES])
    ax.set_yticks((0.0, 0.25, 0.5, 0.75, 1.0))
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    ax.set_xlabel(r"$q$")
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


def target_count(profile: ProfileData) -> int:
    """Return the invariant global target-structure count."""
    field = f"total_count_{profile.spec.group}"
    values: set[int] = set()
    for runs in profile.runs_by_q.values():
        for run in runs:
            value = to_float(run["empirical"][0], field)
            if value is None:
                raise ValueError(f"Missing {field} in {run['path']}")
            values.add(int(value))
    if len(values) != 1:
        raise ValueError(
            f"Global structure count varies for {profile.spec.key}: "
            f"{sorted(values)}"
        )
    return values.pop()


def draw_structure_count_panel(
    ax: plt.Axes,
    profiles: list[ProfileData],
) -> None:
    """Draw exact global target counts on a logarithmic scale."""
    totals = [target_count(profile) for profile in profiles]
    for y, (profile, total) in enumerate(
        zip(profiles, totals, strict=True)
    ):
        ax.barh(
            y,
            total - 1,
            left=1,
            height=0.56,
            color=profile.spec.color,
            alpha=0.88,
            edgecolor="none",
        )
        ax.text(
            total * 1.10,
            y,
            f"{total:,}",
            fontsize=6.5,
            color=profile.spec.color,
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
    ax.set_yticklabels([profile.spec.label for profile in profiles])
    ax.invert_yaxis()
    ax.tick_params(axis="y", length=0, pad=3)
    style_axis(ax)


def plot_structural_coverage(
    profiles: list[ProfileData],
    output_dir: Path,
    *,
    max_epoch: int,
) -> Path:
    """Render recovery, observability, and global structure counts."""
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
    observable_ax = fig.add_subplot(diagnostics[0, 0])
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
    draw_observable_structures_panel(observable_ax, profiles)
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
    for ax, label in ((observable_ax, "b"), (count_ax, "c")):
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
    fig.subplots_adjust(top=0.88, bottom=0.075, left=0.09, right=0.985)
    return save_figure(
        fig,
        output_dir,
        "appendix_structural_coverage",
    )


def plot_entropy(
    profiles: list[ProfileData],
    output_dir: Path,
    *,
    max_epoch: int,
) -> Path:
    """Render the real all-q recovery-entropy block."""
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
                    epochs, means, stds, strict=True
                )
                if epoch <= max_epoch
            ]
            plot_epochs = [epoch for epoch, _, _ in filtered]
            plot_means = [mean for _, mean, _ in filtered]
            plot_stds = [std for _, _, std in filtered]
            profile_series.append((q, plot_epochs, plot_means, plot_stds))
            all_upper_values.extend(
                mean + std for mean, std in zip(plot_means, plot_stds, strict=True)
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
                    [max(0.0, mean - std) for mean, std in zip(means, stds, strict=True)],
                    [mean + std for mean, std in zip(means, stds, strict=True)],
                    color=color,
                    alpha=0.10,
                    linewidth=0,
                )
        add_facet_title(ax, profile.spec)
        ax.set_xlim(1, max_epoch)
        ax.set_ylim(0.0, ymax)
        ax.set_ylabel(
            r"$H_{q,T}/|S^\ast|$" if index in (0, 2) else ""
        )
        ax.set_xlabel(r"$T$" if index in (2, 3) else "")
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
    fig.subplots_adjust(top=0.86, bottom=0.08, left=0.09, right=0.985)
    return save_figure(fig, output_dir, "appendix_recovery_entropy")


def write_source_data(
    profiles: list[ProfileData],
    output_dir: Path,
    *,
    max_epoch: int,
) -> Path:
    """Write the aggregated values used by the quantitative panels."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "appendix_structural_coverage_source_data.csv"
    fieldnames = [
        "panel",
        "profile",
        "structure",
        "q",
        "epoch",
        "n",
        "empirical_mean",
        "empirical_sd",
        "theory_mean",
        "theory_sd",
        "observable_mean",
        "observable_sd",
        "structure_count",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for profile in profiles:
            for q in Q_VALUES:
                runs = profile.runs_by_q[q]
                theory_epochs, theory_means, theory_stds, _ = epoch_stats(
                    runs,
                    "theory",
                    f"expected_coverage_{profile.spec.group}",
                )
                empirical_epochs, empirical_means, empirical_stds, _ = (
                    epoch_stats(
                        runs,
                        "empirical",
                        f"realized_coverage_{profile.spec.group}",
                    )
                )
                empirical_by_epoch = {
                    epoch: (mean, std)
                    for epoch, mean, std in zip(
                        empirical_epochs,
                        empirical_means,
                        empirical_stds,
                        strict=True,
                    )
                }
                for epoch, theory_mean, theory_std in zip(
                    theory_epochs,
                    theory_means,
                    theory_stds,
                    strict=True,
                ):
                    if epoch > max_epoch or epoch not in empirical_by_epoch:
                        continue
                    empirical_mean, empirical_std = empirical_by_epoch[epoch]
                    writer.writerow(
                        {
                            "panel": "a",
                            "profile": profile.spec.key,
                            "structure": profile.spec.label,
                            "q": q,
                            "epoch": epoch,
                            "n": len(runs),
                            "empirical_mean": empirical_mean,
                            "empirical_sd": empirical_std,
                            "theory_mean": theory_mean,
                            "theory_sd": theory_std,
                        }
                    )
                observable_mean, observable_sd = observable_stats(profile, q)
                writer.writerow(
                    {
                        "panel": "b",
                        "profile": profile.spec.key,
                        "structure": profile.spec.label,
                        "q": q,
                        "n": len(runs),
                        "observable_mean": observable_mean,
                        "observable_sd": observable_sd,
                    }
                )
            writer.writerow(
                {
                    "panel": "c",
                    "profile": profile.spec.key,
                    "structure": profile.spec.label,
                    "structure_count": target_count(profile),
                }
            )
    return path


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
            "cora_np64_sweep/appendix_figures"
        ),
    )
    parser.add_argument("--max-epoch", type=int, default=200)
    return parser.parse_args()


def main() -> None:
    """Validate the real sweep and render the appendix export bundle."""
    args = parse_args()
    sweep_root = Path(args.sweep_root)
    output_dir = Path(args.output_dir)
    profiles = discover_profiles(sweep_root)
    coverage = plot_structural_coverage(
        profiles,
        output_dir,
        max_epoch=args.max_epoch,
    )
    entropy = plot_entropy(
        profiles,
        output_dir,
        max_epoch=args.max_epoch,
    )
    source_data = write_source_data(
        profiles,
        output_dir,
        max_epoch=args.max_epoch,
    )
    print(f"Wrote structural coverage figure to {coverage}")
    print(f"Wrote recovery entropy figure to {entropy}")
    print(f"Wrote source data to {source_data}")


if __name__ == "__main__":
    main()
