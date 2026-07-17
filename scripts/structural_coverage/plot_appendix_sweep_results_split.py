"""Render the saved local-versus-cellular split appendix figures.

The local-lifting figure tests whether containment theory predicts empirical
recovery. The cellular-lifting figure isolates the reconstruction discrepancy
between cycle-basis and cycle-span constructions. The saved combined design
remains available in ``plot_appendix_sweep_results_combined.py``.
"""

from __future__ import annotations

import argparse
import csv
import statistics
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

from scripts.structural_coverage.plot_appendix_sweep_results_combined import (
    Q_VALUES,
    REFERENCE_COLOR,
    ProfileData,
    add_curve_key,
    discover_profiles,
    draw_q_key,
    draw_recovery_facet,
    plot_entropy,
    style_axis,
    target_count,
    write_source_data,
)
from scripts.structural_coverage.plot_results import (
    FIGURE_WIDTH,
    save_figure,
    to_float,
)


def draw_observable_panel(
    ax: plt.Axes,
    profiles: list[ProfileData],
) -> None:
    """Draw q-observable fractions for a subset of lifting profiles."""
    for profile in profiles:
        values = []
        field = f"observable_ceiling_{profile.spec.group}"
        for q in Q_VALUES:
            value = to_float(profile.runs_by_q[q][0]["theory"][0], field)
            if value is None:
                raise ValueError(
                    f"Missing {field} in {profile.runs_by_q[q][0]['path']}"
                )
            values.append(value)
        ax.plot(
            Q_VALUES,
            values,
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
        fontsize=5.8,
        handlelength=2.2,
        handletextpad=0.55,
        labelspacing=0.35,
        borderaxespad=0.4,
    )
    style_axis(ax)


def value_at_epoch(
    run: dict[str, Any],
    source: str,
    field: str,
    epoch: int,
) -> float:
    """Return one run value at an exact epoch."""
    for row in run[source]:
        if int(float(row["epoch"])) != epoch:
            continue
        value = to_float(row, field)
        if value is None:
            break
        return value
    raise ValueError(
        f"Missing {field} at epoch {epoch} in {run['path']}"
    )


def deficit_stats(
    profile: ProfileData,
    q: int,
    epoch: int,
) -> tuple[float, float, float, float]:
    """Return paired theory-minus-empirical deficit statistics."""
    theory_field = f"expected_coverage_{profile.spec.group}"
    empirical_field = f"realized_coverage_{profile.spec.group}"
    theory_values = []
    empirical_values = []
    differences = []
    for run in profile.runs_by_q[q]:
        theory = value_at_epoch(run, "theory", theory_field, epoch)
        empirical = value_at_epoch(run, "empirical", empirical_field, epoch)
        theory_values.append(theory)
        empirical_values.append(empirical)
        differences.append(theory - empirical)
    mean = statistics.fmean(differences)
    std = statistics.stdev(differences) if len(differences) > 1 else 0.0
    return (
        mean,
        std,
        statistics.fmean(theory_values),
        statistics.fmean(empirical_values),
    )


def draw_deficit_panel(
    ax: plt.Axes,
    profiles: list[ProfileData],
    *,
    epoch: int,
) -> None:
    """Show the paired reconstruction deficit across q."""
    all_lower = []
    all_upper = []
    for profile in profiles:
        means = []
        stds = []
        for q in Q_VALUES:
            mean, std, _, _ = deficit_stats(profile, q, epoch)
            means.append(mean)
            stds.append(std)
            all_lower.append(mean - std)
            all_upper.append(mean + std)
        ax.fill_between(
            Q_VALUES,
            [mean - std for mean, std in zip(means, stds, strict=True)],
            [mean + std for mean, std in zip(means, stds, strict=True)],
            color=profile.spec.color,
            alpha=0.12,
            linewidth=0,
        )
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
    margin = 0.012
    lower = min(-0.02, min(all_lower, default=0.0) - margin)
    upper = max(0.13, max(all_upper, default=0.0) + margin)
    ax.set_title(
        rf"Reconstruction deficit at $T={epoch}$",
        loc="left",
        fontweight="bold",
    )
    ax.set_xscale("log", base=2)
    ax.set_xlim(0.85, 39)
    ax.set_ylim(lower, upper)
    ax.set_xticks(Q_VALUES)
    ax.set_xticklabels([str(q) for q in Q_VALUES])
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    ax.set_xlabel(r"$q$")
    ax.set_ylabel(r"$\Delta R_{q,T}$")
    ax.axhline(0.0, color=REFERENCE_COLOR, linewidth=0.75, zorder=0)
    ax.legend(
        loc="upper right",
        fontsize=5.8,
        handlelength=2.2,
        handletextpad=0.55,
        labelspacing=0.35,
        borderaxespad=0.4,
    )
    style_axis(ax)


def draw_count_panel(
    ax: plt.Axes,
    profiles: list[ProfileData],
    *,
    title: str,
) -> None:
    """Draw exact higher-order target counts for a profile subset."""
    totals = [target_count(profile) for profile in profiles]
    for y, (profile, total) in enumerate(
        zip(profiles, totals, strict=True)
    ):
        ax.barh(
            y,
            total - 1,
            left=1,
            height=0.48,
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
    ax.set_title(title, loc="left", fontweight="bold")
    ax.set_xscale("log")
    ax.set_xlim(1, max(totals) * 2.1)
    ax.set_xlabel(r"$|S^\ast|$")
    ax.set_yticks(range(len(profiles)))
    ax.set_yticklabels([profile.spec.label for profile in profiles])
    ax.set_ylim(-0.65, len(profiles) - 0.35)
    ax.invert_yaxis()
    ax.tick_params(axis="y", length=0, pad=3)
    style_axis(ax)


def add_panel_heading(
    fig: plt.Figure,
    hero_axes: list[plt.Axes],
) -> None:
    """Add the shared panel-a label and heading above two recovery facets."""
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


def add_bottom_panel_labels(
    left_ax: plt.Axes,
    right_ax: plt.Axes,
) -> None:
    """Add the b/c labels to the supporting panels."""
    for ax, label in ((left_ax, "b"), (right_ax, "c")):
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


def build_split_layout() -> tuple[
    plt.Figure,
    list[plt.Axes],
    plt.Axes,
    plt.Axes,
    plt.Axes,
]:
    """Create the shared two-facet hero plus two-panel support layout."""
    fig = plt.figure(figsize=(FIGURE_WIDTH, 5.35))
    outer = fig.add_gridspec(
        2,
        1,
        height_ratios=(1.62, 1.0),
        hspace=0.48,
    )
    hero = outer[0].subgridspec(
        1,
        3,
        width_ratios=(1.0, 1.0, 0.22),
        wspace=0.26,
    )
    hero_axes = [fig.add_subplot(hero[0, 0]), fig.add_subplot(hero[0, 1])]
    key_ax = fig.add_subplot(hero[0, 2])
    support = outer[1].subgridspec(1, 2, wspace=0.34)
    left_ax = fig.add_subplot(support[0, 0])
    right_ax = fig.add_subplot(support[0, 1])
    return fig, hero_axes, key_ax, left_ax, right_ax


def plot_local_liftings(
    profiles: list[ProfileData],
    output_dir: Path,
    *,
    max_epoch: int,
) -> Path:
    """Render the local-lifting recovery validation figure."""
    fig, hero_axes, key_ax, observable_ax, count_ax = build_split_layout()
    for index, (ax, profile) in enumerate(
        zip(hero_axes, profiles, strict=True)
    ):
        draw_recovery_facet(
            ax,
            profile,
            max_epoch=max_epoch,
            show_ylabel=index == 0,
            show_xlabel=True,
        )
    add_curve_key(hero_axes[0])
    draw_q_key(key_ax)
    draw_observable_panel(observable_ax, profiles)
    draw_count_panel(
        count_ax,
        profiles,
        title="Global higher-order structures",
    )
    add_panel_heading(fig, hero_axes)
    add_bottom_panel_labels(observable_ax, count_ax)
    fig.suptitle(
        "Structural recovery of local liftings",
        x=0.075,
        y=0.995,
        ha="left",
        va="top",
        fontsize=11,
        fontweight="bold",
    )
    fig.subplots_adjust(top=0.82, bottom=0.08, left=0.09, right=0.985)
    return save_figure(fig, output_dir, "appendix_local_structural_recovery")


def plot_cellular_liftings(
    profiles: list[ProfileData],
    output_dir: Path,
    *,
    max_epoch: int,
) -> Path:
    """Render the cell-lifting reconstruction discrepancy figure."""
    fig, hero_axes, key_ax, deficit_ax, count_ax = build_split_layout()
    for index, (ax, profile) in enumerate(
        zip(hero_axes, profiles, strict=True)
    ):
        draw_recovery_facet(
            ax,
            profile,
            max_epoch=max_epoch,
            show_ylabel=index == 0,
            show_xlabel=True,
        )
    add_curve_key(hero_axes[0])
    draw_q_key(key_ax)
    draw_deficit_panel(deficit_ax, profiles, epoch=max_epoch)
    draw_count_panel(count_ax, profiles, title="Global 2-cells")
    add_panel_heading(fig, hero_axes)
    add_bottom_panel_labels(deficit_ax, count_ax)
    fig.suptitle(
        "Structural recovery of cellular liftings",
        x=0.075,
        y=0.995,
        ha="left",
        va="top",
        fontsize=11,
        fontweight="bold",
    )
    fig.subplots_adjust(top=0.82, bottom=0.08, left=0.09, right=0.985)
    return save_figure(
        fig,
        output_dir,
        "appendix_cellular_structural_recovery",
    )


def write_deficit_source_data(
    profiles: list[ProfileData],
    output_dir: Path,
    *,
    epoch: int,
) -> Path:
    """Write paired cell-lifting deficit statistics."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "appendix_cellular_deficit_source_data.csv"
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "profile",
                "structure",
                "q",
                "epoch",
                "n",
                "expected_mean",
                "empirical_mean",
                "deficit_mean",
                "deficit_sd",
            ),
        )
        writer.writeheader()
        for profile in profiles:
            for q in Q_VALUES:
                deficit, std, expected, empirical = deficit_stats(
                    profile,
                    q,
                    epoch,
                )
                writer.writerow(
                    {
                        "profile": profile.spec.key,
                        "structure": profile.spec.label,
                        "q": q,
                        "epoch": epoch,
                        "n": len(profile.runs_by_q[q]),
                        "expected_mean": expected,
                        "empirical_mean": empirical,
                        "deficit_mean": deficit,
                        "deficit_sd": std,
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
            "cora_np64_sweep/appendix_figures_split"
        ),
    )
    parser.add_argument("--max-epoch", type=int, default=200)
    return parser.parse_args()


def main() -> None:
    """Validate the sweep and render the split appendix export bundle."""
    args = parse_args()
    output_dir = Path(args.output_dir)
    profiles = discover_profiles(Path(args.sweep_root))
    by_key = {profile.spec.key: profile for profile in profiles}
    local = [by_key["hypergraph"], by_key["simplicial"]]
    cellular = [by_key["cell_basis"], by_key["cell_simple_coverage"]]

    local_figure = plot_local_liftings(
        local,
        output_dir,
        max_epoch=args.max_epoch,
    )
    cellular_figure = plot_cellular_liftings(
        cellular,
        output_dir,
        max_epoch=args.max_epoch,
    )
    entropy_figure = plot_entropy(
        profiles,
        output_dir,
        max_epoch=args.max_epoch,
    )
    source_data = write_source_data(
        profiles,
        output_dir,
        max_epoch=args.max_epoch,
    )
    deficit_data = write_deficit_source_data(
        cellular,
        output_dir,
        epoch=args.max_epoch,
    )

    print(f"Wrote local-lifting figure to {local_figure}")
    print(f"Wrote cellular-lifting figure to {cellular_figure}")
    print(f"Wrote recovery entropy figure to {entropy_figure}")
    print(f"Wrote structural source data to {source_data}")
    print(f"Wrote deficit source data to {deficit_data}")


if __name__ == "__main__":
    main()
