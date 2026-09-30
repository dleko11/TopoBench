from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D

REPO = Path(__file__).resolve().parents[2]
OUTPUT_DIR = REPO / "outputs/figure3_pareto"

DATASETS = ("cora_full", "amazon_ratings", "questions")
DATASET_LABELS = {
    "cora_full": "Cora Full",
    "amazon_ratings": "Amazon Ratings",
    "questions": "Questions",
}
MODELS = ("gcn", "edgnn", "unignn", "cwn", "topotune", "scn", "sccnn")
MODEL_LABELS = {
    "gcn": "GCN",
    "edgnn": "EDHNN",
    "unignn": "UniGNN",
    "cwn": "CWN",
    "topotune": "TopoTune",
    "scn": "SCN",
    "sccnn": "SCCNN",
}

# Current ensemble-only Table 1 means, in percentage points.
PERFORMANCE = {
    ("cora_full", "gcn"): (70.18, 69.19),
    ("cora_full", "edgnn"): (69.74, 67.06),
    ("cora_full", "unignn"): (68.44, 67.61),
    ("cora_full", "cwn"): (60.50, 62.66),
    ("cora_full", "topotune"): (46.40, 56.45),
    ("cora_full", "scn"): (70.45, 70.32),
    ("cora_full", "sccnn"): (70.81, 70.20),
    ("amazon_ratings", "gcn"): (48.29, 49.62),
    ("amazon_ratings", "edgnn"): (48.41, 51.41),
    ("amazon_ratings", "unignn"): (47.05, 47.24),
    ("amazon_ratings", "cwn"): (44.07, 46.25),
    ("amazon_ratings", "topotune"): (43.18, 43.27),
    ("amazon_ratings", "scn"): (50.08, 50.65),
    ("amazon_ratings", "sccnn"): (51.20, 51.24),
    ("questions", "gcn"): (76.69, 73.83),
    ("questions", "edgnn"): (75.73, 76.23),
    ("questions", "unignn"): (69.44, 71.09),
    ("questions", "cwn"): (68.21, 70.20),
    ("questions", "topotune"): (71.53, 73.61),
    ("questions", "scn"): (74.77, 73.04),
    ("questions", "sccnn"): (77.14, 74.10),
}

LABEL_OFFSETS = {
    ("cora_full", "gcn"): (4, -9),
    ("cora_full", "edgnn"): (-5, -13),
    ("cora_full", "unignn"): (5, 9),
    ("cora_full", "cwn"): (4, 5),
    ("cora_full", "topotune"): (4, 5),
    ("cora_full", "scn"): (4, -8),
    ("cora_full", "sccnn"): (4, 5),
    ("amazon_ratings", "gcn"): (4, 5),
    ("amazon_ratings", "edgnn"): (4, 5),
    ("amazon_ratings", "unignn"): (4, -9),
    ("amazon_ratings", "cwn"): (-6, -1),
    ("amazon_ratings", "topotune"): (6, 0),
    ("amazon_ratings", "scn"): (5, -11),
    ("amazon_ratings", "sccnn"): (4, 5),
    ("questions", "gcn"): (7, 11),
    ("questions", "edgnn"): (-7, -6),
    ("questions", "unignn"): (7, 1),
    ("questions", "cwn"): (4, 5),
    ("questions", "topotune"): (4, 5),
    ("questions", "scn"): (-6, 11),
    ("questions", "sccnn"): (7, 10),
}

LEADER_LINE_KEYS = {
    ("cora_full", "unignn"),
    ("cora_full", "edgnn"),
    ("amazon_ratings", "scn"),
    ("questions", "sccnn"),
    ("questions", "scn"),
    ("questions", "edgnn"),
}


def _fresh_preprocessing() -> pd.DataFrame:
    phases = pd.read_csv(REPO / "outputs/table8_partition_build/phase_metrics.csv")
    phases["events_complete"] = (
        phases["events_complete"].astype(str).str.lower() == "true"
    )
    phases = phases[
        (phases["mode"] == "partitioning") & (phases["model"] == "gcn")
    ].copy()
    records = []
    keys = ["dataset", "num_parts", "seed", "run_id"]
    for (dataset, num_parts, seed, _), group in phases.groupby(keys, sort=False):
        load = group[group["phase"] == "dataset_load"]
        trainer = group[group["phase"] == "trainer_init"]
        if len(load) != 1 or len(trainer) != 1:
            raise ValueError(f"Incomplete fresh preprocessing phases for {dataset}, K={num_parts}, seed={seed}")
        if not group["events_complete"].all():
            raise ValueError(f"Incomplete markers for {dataset}, K={num_parts}, seed={seed}")
        records.append(
            {
                "dataset": dataset,
                "num_parts": int(num_parts),
                "seed": int(seed),
                "fresh_preprocessing_sec": (
                    float(trainer.iloc[0]["last_end_timestamp"])
                    - float(load.iloc[0]["first_start_timestamp"])
                ),
            }
        )
    return pd.DataFrame(records)


def build_source_data() -> pd.DataFrame:
    runs = pd.read_csv(REPO / "outputs/table1_wall_clock/run_components.csv")
    fresh = _fresh_preprocessing()
    runs["num_parts"] = pd.to_numeric(runs["num_parts"], errors="coerce")
    partitioned = runs[runs["mode"] == "partitioning"].merge(
        fresh,
        on=["dataset", "num_parts", "seed"],
        how="left",
        validate="many_to_one",
    )
    if partitioned["fresh_preprocessing_sec"].isna().any():
        missing = partitioned.loc[
            partitioned["fresh_preprocessing_sec"].isna(),
            ["dataset", "model", "num_parts", "seed"],
        ]
        raise ValueError(f"Missing fresh preprocessing measurements:\n{missing}")
    partitioned["corrected_total_sec"] = (
        partitioned["fresh_preprocessing_sec"]
        + partitioned["training_sec"]
        + partitioned["final_evaluation_sec"]
    )
    full = runs[runs["mode"] == "full"].copy()
    full = full[
        [
            "dataset",
            "model",
            "seed",
            "total_wall_clock_sec",
            "gpu_peak_allocated_gib",
        ]
    ].rename(
        columns={
            "total_wall_clock_sec": "full_total_sec",
            "gpu_peak_allocated_gib": "full_gpu_gib",
        }
    )
    partitioned = partitioned[
        [
            "dataset",
            "model",
            "seed",
            "corrected_total_sec",
            "gpu_peak_allocated_gib",
        ]
    ].rename(columns={"gpu_peak_allocated_gib": "partitioned_gpu_gib"})
    paired = full.merge(
        partitioned,
        on=["dataset", "model", "seed"],
        how="inner",
        validate="one_to_one",
    )
    paired["memory_ratio"] = paired["partitioned_gpu_gib"] / paired["full_gpu_gib"]
    paired["runtime_ratio"] = paired["corrected_total_sec"] / paired["full_total_sec"]

    records = []
    for (dataset, model), group in paired.groupby(["dataset", "model"]):
        full_perf, partitioned_perf = PERFORMANCE[(dataset, model)]
        records.append(
            {
                "dataset": dataset,
                "model": model,
                "n": len(group),
                "memory_ratio_mean": group["memory_ratio"].mean(),
                "memory_ratio_median": group["memory_ratio"].median(),
                "memory_ratio_q1": group["memory_ratio"].quantile(0.25),
                "memory_ratio_q3": group["memory_ratio"].quantile(0.75),
                "runtime_ratio_mean": group["runtime_ratio"].mean(),
                "runtime_ratio_median": group["runtime_ratio"].median(),
                "runtime_ratio_q1": group["runtime_ratio"].quantile(0.25),
                "runtime_ratio_q3": group["runtime_ratio"].quantile(0.75),
                "full_performance": full_perf,
                "partitioned_performance": partitioned_perf,
                "performance_delta_pp": partitioned_perf - full_perf,
            }
        )
    source = pd.DataFrame(records)
    if len(source) != 21 or not (source["n"] == 5).all():
        raise ValueError(f"Expected 21 five-seed comparisons, got:\n{source[['dataset', 'model', 'n']]}")
    source["dataset"] = pd.Categorical(source["dataset"], DATASETS, ordered=True)
    source["model"] = pd.Categorical(source["model"], MODELS, ordered=True)
    return source.sort_values(["dataset", "model"]).reset_index(drop=True)


def marker_size(delta: float, scale: str = "soft") -> float:
    magnitude = min(abs(delta), 10.0)
    if scale == "strong":
        return 20.0 + 22.0 * magnitude
    return 30.0 + 30.0 * np.sqrt(magnitude)


def _style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 7,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.linewidth": 0.75,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
            "hatch.linewidth": 0.35,
        }
    )


def _base_axes(
    *,
    quadrant_tint: bool = False,
    legend_layout: str | None = None,
    balanced_axis_spacing: bool = False,
    light_grid: bool = False,
    normal_grid: bool = False,
):
    height = 2.62 if legend_layout != "split" else 2.82
    fig, axes = plt.subplots(1, 3, figsize=(7.25, height), sharex=True, sharey=True)
    for ax, dataset in zip(axes, DATASETS, strict=True):
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(0.009, 1.25)
        ax.set_ylim(0.07, 12.5)
        if quadrant_tint:
            ax.axvspan(0.009, 1.0, ymin=0.0, ymax=np.log10(1 / 0.07) / np.log10(12.5 / 0.07), color="#edf5ed", zorder=0)
        ax.set_title(DATASET_LABELS[dataset], fontweight="bold", pad=5)
        if normal_grid:
            ax.grid(
                True,
                which="major",
                color="#dddddd",
                linewidth=0.45,
                alpha=0.70,
                zorder=0,
            )
            ax.grid(
                True,
                which="minor",
                color="#eeeeee",
                linewidth=0.30,
                alpha=0.75,
                zorder=0,
            )
        elif light_grid:
            ax.grid(
                True,
                which="major",
                color="#e5e5e5",
                linewidth=0.45,
                alpha=0.65,
                zorder=0,
            )
        else:
            ax.grid(False)
        if normal_grid or light_grid:
            ax.axvline(1.0, color="white", linewidth=1.7, zorder=1)
            ax.axhline(1.0, color="white", linewidth=1.7, zorder=1)
        ax.axvline(1.0, color="#777777", linewidth=0.75, linestyle=(0, (3, 2)), zorder=2)
        ax.axhline(1.0, color="#777777", linewidth=0.75, linestyle=(0, (3, 2)), zorder=2)
        ax.tick_params(length=2.8, width=0.65)
    axes[0].set_ylabel("Total-runtime ratio", loc="center")
    xlabel_y = 0.067 if balanced_axis_spacing else 0.095
    top = 0.82 if legend_layout != "split" else 0.72
    bottom = 0.22 if legend_layout != "split" else 0.21
    fig.subplots_adjust(left=0.078, right=0.985, top=top, bottom=bottom, wspace=0.12)
    fig.supxlabel(
        "Peak GPU-memory ratio",
        x=(fig.subplotpars.left + fig.subplotpars.right) / 2,
        y=xlabel_y,
        fontsize=9,
    )
    return fig, axes


def _add_labels(
    ax,
    panel: pd.DataFrame,
    *,
    include_delta: bool,
    labels_right: bool = False,
    bold_labels: bool = False,
    selective_leader_lines: bool = False,
    summary_stat: str = "median",
) -> None:
    for row in panel.itertuples(index=False):
        key = (str(row.dataset), str(row.model))
        dx, dy = LABEL_OFFSETS[(str(row.dataset), str(row.model))]
        ha = "left" if dx >= 0 else "right"
        if labels_right:
            dx = abs(dx)
            ha = "left"
            left_overrides = {
                ("cora_full", "topotune"): (-12, 5),
                ("cora_full", "cwn"): (-6, 5),
                ("cora_full", "unignn"): (-11, -4),
                ("amazon_ratings", "topotune"): (-6, 0),
                ("amazon_ratings", "cwn"): (-6, -1),
                ("amazon_ratings", "edgnn"): (-6, -8),
                ("questions", "edgnn"): (-8, 6),
            }
            if key in left_overrides:
                dx, dy = left_overrides[key]
                ha = "right"
            elif key == ("questions", "gcn"):
                dy = -5
            elif key == ("questions", "scn"):
                dy = 13
            elif key == ("questions", "unignn"):
                dy = -4
            elif key == ("questions", "sccnn"):
                dx, dy = 5, 21
        label = MODEL_LABELS[str(row.model)]
        if include_delta:
            label = f"{label} ({row.performance_delta_pp:+.2f})"
        arrowprops = None
        if not selective_leader_lines or key in LEADER_LINE_KEYS:
            arrowprops = {
                "arrowstyle": "-",
                "color": "#777777",
                "linewidth": 0.35,
                "shrinkA": 1.0,
                "shrinkB": 3.0,
            }
        ax.annotate(
            label,
            (
                getattr(row, f"memory_ratio_{summary_stat}"),
                getattr(row, f"runtime_ratio_{summary_stat}"),
            ),
            xytext=(dx, dy),
            textcoords="offset points",
            ha=ha,
            va="center",
            fontsize=5.6 if include_delta else 5.8,
            fontweight="bold" if bold_labels else "normal",
            color="#252525",
            arrowprops=arrowprops,
            zorder=5,
        )


def _size_legend(ax) -> None:
    handles = [
        ax.scatter([], [], s=marker_size(value), facecolor="#dddddd", edgecolor="#666666", linewidth=0.6)
        for value in (0.5, 2.0, 10.0)
    ]
    ax.legend(
        handles,
        ["0.5", "2", "10"],
        title=r"$|\Delta|$ (pp)",
        loc="lower right",
        frameon=False,
        fontsize=5.6,
        title_fontsize=5.8,
        handletextpad=0.5,
        labelspacing=0.55,
        borderpad=0.2,
    )


def _size_legend_handles(
    scale: str = "soft",
    values: tuple[float, ...] = (0.5, 2.0, 10.0),
) -> list[Line2D]:
    return [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor="#e2e2e2",
            markeredgecolor="#666666",
            markeredgewidth=0.65,
            markersize=np.sqrt(marker_size(value, scale)),
        )
        for value in values
    ]


def plot_discrete(
    source: pd.DataFrame,
    filename: str,
    *,
    include_delta: bool = False,
    quadrant_tint: bool = False,
    neutral_band: bool = True,
    vary_size: bool = True,
    legend_layout: str | None = None,
    size_scale: str = "soft",
    hatch_mode: str | None = None,
    monochrome_hatch: str | None = None,
    colors_override: dict[str, str] | None = None,
    size_legend_first: bool = False,
    marker_alpha: float | None = None,
    delta_label_first: bool = False,
    color_legend_first: bool = False,
    labels_right: bool = False,
    bold_labels: bool = False,
    selective_leader_lines: bool = False,
    balanced_axis_spacing: bool = False,
    legend_values: tuple[float, ...] = (0.5, 2.0, 10.0),
    tight_crop: bool = False,
    light_grid: bool = False,
    normal_grid: bool = False,
    summary_stat: str = "median",
) -> None:
    fig, axes = _base_axes(
        quadrant_tint=quadrant_tint,
        legend_layout=legend_layout,
        balanced_axis_spacing=balanced_axis_spacing,
        light_grid=light_grid,
        normal_grid=normal_grid,
    )
    if hatch_mode is not None and monochrome_hatch is not None:
        raise ValueError("hatch_mode and monochrome_hatch are mutually exclusive")
    colors = colors_override or {
        "gain": "#a9d6b2",
        "loss": "#efb1ab",
        "neutral": "#dedede",
    }
    if monochrome_hatch is not None:
        colors = {"gain": "#fafafa", "loss": "#fafafa", "neutral": "#dedede"}

    def facecolor(key: str):
        if marker_alpha is None:
            return colors[key]
        return mpl.colors.to_rgba(colors[key], marker_alpha)

    for ax, dataset in zip(axes, DATASETS, strict=True):
        panel = source[source["dataset"] == dataset]
        for row in panel.itertuples(index=False):
            delta = row.performance_delta_pp
            if neutral_band:
                key = "gain" if delta > 0.05 else "loss" if delta < -0.05 else "neutral"
            else:
                key = "gain" if delta >= 0 else "loss"
            hatch = None
            if hatch_mode == "loss" and key == "loss":
                hatch = "////"
            elif hatch_mode == "dense_gain" and key == "gain":
                hatch = "////////"
            elif hatch_mode == "opposing":
                hatch = "////" if key == "gain" else "\\\\\\\\"
            elif monochrome_hatch == key:
                hatch = "////"
            ax.scatter(
                getattr(row, f"memory_ratio_{summary_stat}"),
                getattr(row, f"runtime_ratio_{summary_stat}"),
                s=marker_size(delta, size_scale) if vary_size else 55,
                facecolor=facecolor(key),
                edgecolor="#4f4f4f",
                linewidth=0.65,
                alpha=0.95 if marker_alpha is None else None,
                hatch=hatch,
                zorder=4,
            )
        _add_labels(
            ax,
            panel,
            include_delta=include_delta,
            labels_right=labels_right,
            bold_labels=bold_labels,
            selective_leader_lines=selective_leader_lines,
            summary_stat=summary_stat,
        )
    if monochrome_hatch is not None:
        sign_handles = [
            axes[0].scatter(
                [],
                [],
                s=45,
                facecolor=facecolor("gain"),
                edgecolor="#3f3f3f",
                linewidth=0.7,
                hatch="////" if monochrome_hatch == "gain" else None,
                label="Performance gain",
            ),
            axes[0].scatter(
                [],
                [],
                s=45,
                facecolor=facecolor("loss"),
                edgecolor="#3f3f3f",
                linewidth=0.7,
                hatch="////" if monochrome_hatch == "loss" else None,
                label="Performance loss",
            ),
        ]
    elif hatch_mode is None:
        sign_handles = [
            Line2D([0], [0], marker="o", color="none", markerfacecolor=facecolor("gain"), markeredgecolor="#4f4f4f", markersize=6, label="Performance gain"),
            Line2D([0], [0], marker="o", color="none", markerfacecolor=facecolor("loss"), markeredgecolor="#4f4f4f", markersize=6, label="Performance loss"),
        ]
    else:
        gain_hatch = (
            "////////"
            if hatch_mode == "dense_gain"
            else "////"
            if hatch_mode == "opposing"
            else None
        )
        loss_hatch = "\\\\\\\\" if hatch_mode == "opposing" else "////" if hatch_mode == "loss" else None
        sign_handles = [
            axes[0].scatter([], [], s=45, facecolor=facecolor("gain"), edgecolor="#4f4f4f", linewidth=0.65, hatch=gain_hatch, label="Performance gain"),
            axes[0].scatter([], [], s=45, facecolor=facecolor("loss"), edgecolor="#4f4f4f", linewidth=0.65, hatch=loss_hatch, label="Performance loss"),
        ]
    if neutral_band:
        sign_handles.append(
            Line2D([0], [0], marker="o", color="none", markerfacecolor=colors["neutral"], markeredgecolor="#4f4f4f", markersize=6, label=r"$|\Delta| \leq 0.05$ pp")
        )
    legend_kwargs = {
        "frameon": False,
        "fontsize": 8,
        "handletextpad": 0.4,
        "columnspacing": 1.1,
    }
    if legend_layout == "single":
        size_handles = _size_legend_handles(size_scale, legend_values)
        handles = [*sign_handles, *size_handles]
        labels = [
            "Performance gain",
            "Performance loss",
            r"$|\Delta|=0.5$ pp",
            "2 pp",
            "10 pp",
        ]
        if size_legend_first:
            handles = [*size_handles, *sign_handles]
            labels = [
                r"$|\Delta|=0.5$ pp",
                "2 pp",
                "10 pp",
                "Performance gain",
                "Performance loss",
            ]
        if delta_label_first:
            delta_handle = Line2D(
                [], [], linestyle="none", marker=None, color="none"
            )
            if color_legend_first:
                handles = [*sign_handles, delta_handle, *size_handles]
                labels = [
                    "Performance gain",
                    "Performance loss",
                    "Performance change (p.p.):",
                    *(f"{value:g}" for value in legend_values),
                ]
            else:
                handles = [delta_handle, *size_handles, *sign_handles]
                labels = [
                    "Performance change (p.p.):",
                    *(f"{value:g}" for value in legend_values),
                    "Performance gain",
                    "Performance loss",
                ]
        legend = fig.legend(
            handles=handles,
            labels=labels,
            loc="upper center",
            ncol=len(handles),
            bbox_to_anchor=(0.5, 0.995),
            **legend_kwargs,
        )
        if delta_label_first and color_legend_first:
            fig.canvas.draw()
            renderer = fig.canvas.get_renderer()
            title_top = max(ax.title.get_window_extent(renderer).y1 for ax in axes)
            figure_bounds = fig.get_tightbbox(renderer)
            center = (figure_bounds.x0 + figure_bounds.x1) / (2 * fig.get_figwidth())
            legend_y = title_top / fig.bbox.height
            legend.set_loc("lower center")
            legend.set_bbox_to_anchor((center, legend_y), transform=fig.transFigure)
    elif legend_layout == "split":
        fig.legend(
            handles=sign_handles,
            loc="upper center",
            ncol=len(sign_handles),
            bbox_to_anchor=(0.5, 0.995),
            **legend_kwargs,
        )
        fig.legend(
            handles=_size_legend_handles(size_scale, legend_values),
            labels=[f"{value:g} pp" for value in legend_values],
            loc="upper center",
            ncol=3,
            bbox_to_anchor=(0.5, 0.925),
            **legend_kwargs,
        )
    else:
        fig.legend(
            handles=sign_handles,
            loc="upper center",
            ncol=len(sign_handles),
            bbox_to_anchor=(0.48, 0.985),
            **legend_kwargs,
        )
    if vary_size and legend_layout is None:
        _size_legend(axes[-1])
    fig.savefig(
        OUTPUT_DIR / filename,
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.06 if tight_crop else 0.1,
        facecolor="white",
    )
    plt.close(fig)


def plot_continuous(source: pd.DataFrame, filename: str) -> None:
    fig, axes = _base_axes()
    cmap = mpl.colors.LinearSegmentedColormap.from_list(
        "soft_performance", ["#d98b82", "#f4f1ed", "#75b88a"]
    )
    norm = TwoSlopeNorm(vmin=-3.2, vcenter=0.0, vmax=10.1)
    for ax, dataset in zip(axes, DATASETS, strict=True):
        panel = source[source["dataset"] == dataset]
        ax.scatter(
            panel["memory_ratio_median"],
            panel["runtime_ratio_median"],
            s=[marker_size(value) for value in panel["performance_delta_pp"]],
            c=panel["performance_delta_pp"],
            cmap=cmap,
            norm=norm,
            edgecolor="#4f4f4f",
            linewidth=0.65,
            zorder=4,
        )
        _add_labels(ax, panel, include_delta=False)
    cax = fig.add_axes([0.35, 0.145, 0.30, 0.022])
    colorbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, orientation="horizontal")
    colorbar.set_label(r"Test-performance change, $P-FG$ (pp)", labelpad=2)
    colorbar.ax.tick_params(labelsize=5.8, length=2)
    _size_legend(axes[-1])
    fig.subplots_adjust(bottom=0.34)
    fig.savefig(OUTPUT_DIR / filename, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def render_mean_variants(source: pd.DataFrame) -> None:
    for normal_grid, stem in (
        (False, "pareto_mean_size5_no_grid_final"),
        (True, "pareto_mean_size5_normal_light_grid_final"),
    ):
        for extension in ("pdf", "png"):
            plot_discrete(
                source,
                f"{stem}.{extension}",
                neutral_band=False,
                legend_layout="single",
                size_scale="strong",
                colors_override={
                    "gain": "#648FFF",
                    "loss": "#DC267F",
                    "neutral": "#DEDEDE",
                },
                marker_alpha=0.45,
                delta_label_first=True,
                color_legend_first=True,
                labels_right=True,
                bold_labels=True,
                selective_leader_lines=True,
                balanced_axis_spacing=True,
                legend_values=(0.5, 5.0, 10.0),
                tight_crop=True,
                normal_grid=normal_grid,
                summary_stat="mean",
            )


def main() -> None:
    _style()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    source = build_source_data()
    source.to_csv(OUTPUT_DIR / "pareto_source_data.csv", index=False)
    render_mean_variants(source)
    plot_discrete(source, "v1_sign_and_size.png")
    plot_discrete(source, "v2_sign_size_delta_labels.png", include_delta=True)
    plot_discrete(source, "v3_sign_size_quadrant_tint.png", quadrant_tint=True)
    plot_continuous(source, "v4_continuous_color_and_size.png")
    plot_discrete(source, "v5_recommended_sign_size.png")
    plot_discrete(source, "v6_recommended_with_delta.png", include_delta=True)
    plot_continuous(source, "v7_continuous_color_size.png")
    plot_discrete(
        source,
        "v8_binary_sign_and_size.png",
        neutral_band=False,
    )
    plot_discrete(
        source,
        "v9_binary_sign_constant_size.png",
        neutral_band=False,
        vary_size=False,
    )
    plot_discrete(
        source,
        "v10_external_legend_single_row.png",
        neutral_band=False,
        legend_layout="single",
    )
    plot_discrete(
        source,
        "v11_external_legend_two_rows.png",
        neutral_band=False,
        legend_layout="split",
    )
    plot_discrete(
        source,
        "v12_stronger_sizes.png",
        neutral_band=False,
        legend_layout="single",
        size_scale="strong",
    )
    plot_discrete(
        source,
        "v13_stronger_sizes_loss_hatch.png",
        neutral_band=False,
        legend_layout="single",
        size_scale="strong",
        hatch_mode="loss",
    )
    plot_discrete(
        source,
        "v14_stronger_sizes_opposing_hatches.png",
        neutral_band=False,
        legend_layout="single",
        size_scale="strong",
        hatch_mode="opposing",
    )
    plot_discrete(
        source,
        "v15_monochrome_blank_gain_hatched_loss.png",
        neutral_band=False,
        legend_layout="single",
        size_scale="strong",
        monochrome_hatch="loss",
    )
    plot_discrete(
        source,
        "v16_monochrome_hatched_gain_blank_loss.png",
        neutral_band=False,
        legend_layout="single",
        size_scale="strong",
        monochrome_hatch="gain",
    )
    plot_discrete(
        source,
        "v17_colored_dense_hatched_gain.png",
        neutral_band=False,
        legend_layout="single",
        size_scale="strong",
        hatch_mode="dense_gain",
    )
    plot_discrete(
        source,
        "v18_pale_blue_magenta_delta_first.png",
        neutral_band=False,
        legend_layout="single",
        size_scale="strong",
        hatch_mode="dense_gain",
        colors_override={
            "gain": "#A3CEFF",
            "loss": "#EEA6BD",
            "neutral": "#DEDEDE",
        },
        size_legend_first=True,
    )
    plot_discrete(
        source,
        "v19_pale_blue_magenta_solid.png",
        neutral_band=False,
        legend_layout="single",
        size_scale="strong",
        colors_override={
            "gain": "#A3CEFF",
            "loss": "#EEA6BD",
            "neutral": "#DEDEDE",
        },
        size_legend_first=True,
    )
    for alpha, filename in (
        (0.55, "v20_teal_rose_alpha55.png"),
        (0.70, "v21_teal_rose_alpha70.png"),
    ):
        plot_discrete(
            source,
            filename,
            neutral_band=False,
            legend_layout="single",
            size_scale="strong",
            colors_override={
                "gain": "#44AA99",
                "loss": "#CC6677",
                "neutral": "#DEDEDE",
            },
            size_legend_first=True,
            marker_alpha=alpha,
        )
    plot_discrete(
        source,
        "v22_pale_orange_purple_delta_leading.png",
        neutral_band=False,
        legend_layout="single",
        size_scale="strong",
        colors_override={
            "gain": "#E66100",
            "loss": "#5D3A9B",
            "neutral": "#DEDEDE",
        },
        marker_alpha=0.55,
        delta_label_first=True,
    )
    plot_discrete(
        source,
        "v23_pale_green_purple_colors_first.png",
        neutral_band=False,
        legend_layout="single",
        size_scale="strong",
        colors_override={
            "gain": "#1AFF1A",
            "loss": "#4B0092",
            "neutral": "#DEDEDE",
        },
        marker_alpha=0.45,
        delta_label_first=True,
        color_legend_first=True,
    )
    plot_discrete(
        source,
        "v24_deeper_green_purple_colors_first.png",
        neutral_band=False,
        legend_layout="single",
        size_scale="strong",
        colors_override={
            "gain": "#0FB50F",
            "loss": "#4B0092",
            "neutral": "#DEDEDE",
        },
        marker_alpha=0.45,
        delta_label_first=True,
        color_legend_first=True,
    )
    plot_discrete(
        source,
        "v25_periwinkle_magenta_colors_first.png",
        neutral_band=False,
        legend_layout="single",
        size_scale="strong",
        colors_override={
            "gain": "#648FFF",
            "loss": "#DC267F",
            "neutral": "#DEDEDE",
        },
        marker_alpha=0.45,
        delta_label_first=True,
        color_legend_first=True,
    )
    plot_discrete(
        source,
        "v26_periwinkle_magenta_bold_right_labels.png",
        neutral_band=False,
        legend_layout="single",
        size_scale="strong",
        colors_override={
            "gain": "#648FFF",
            "loss": "#DC267F",
            "neutral": "#DEDEDE",
        },
        marker_alpha=0.45,
        delta_label_first=True,
        color_legend_first=True,
        labels_right=True,
        bold_labels=True,
    )
    for middle_value, filename in (
        (1.0, "v27_selective_lines_size1.png"),
        (5.0, "v28_selective_lines_size5.png"),
    ):
        plot_discrete(
            source,
            filename,
            neutral_band=False,
            legend_layout="single",
            size_scale="strong",
            colors_override={
                "gain": "#648FFF",
                "loss": "#DC267F",
                "neutral": "#DEDEDE",
            },
            marker_alpha=0.45,
            delta_label_first=True,
            color_legend_first=True,
            labels_right=True,
            bold_labels=True,
            selective_leader_lines=True,
            balanced_axis_spacing=True,
            legend_values=(0.5, middle_value, 10.0),
            tight_crop=True,
        )
    for light_grid, filename in (
        (False, "pareto_size5_no_grid.pdf"),
        (True, "pareto_size5_light_grid.pdf"),
    ):
        plot_discrete(
            source,
            filename,
            neutral_band=False,
            legend_layout="single",
            size_scale="strong",
            colors_override={
                "gain": "#648FFF",
                "loss": "#DC267F",
                "neutral": "#DEDEDE",
            },
            marker_alpha=0.45,
            delta_label_first=True,
            color_legend_first=True,
            labels_right=True,
            bold_labels=True,
            selective_leader_lines=True,
            balanced_axis_spacing=True,
            legend_values=(0.5, 5.0, 10.0),
            tight_crop=True,
            light_grid=light_grid,
        )
    for filename in (
        "pareto_size5_normal_light_grid.pdf",
        "pareto_size5_normal_light_grid.png",
    ):
        plot_discrete(
            source,
            filename,
            neutral_band=False,
            legend_layout="single",
            size_scale="strong",
            colors_override={
                "gain": "#648FFF",
                "loss": "#DC267F",
                "neutral": "#DEDEDE",
            },
            marker_alpha=0.45,
            delta_label_first=True,
            color_legend_first=True,
            labels_right=True,
            bold_labels=True,
            selective_leader_lines=True,
            balanced_axis_spacing=True,
            legend_values=(0.5, 5.0, 10.0),
            tight_crop=True,
            normal_grid=True,
        )
    print(source.to_string(index=False))


if __name__ == "__main__":
    main()
