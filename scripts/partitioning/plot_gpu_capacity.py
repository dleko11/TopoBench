#!/usr/bin/env python3
"""Plot completed GPU capacity measurements copied from the cluster.

CUDA OOM markers indicate failed configurations, never measured memory values.
The figure uses peak reserved memory by default; allocated peaks are exported
in the same source-data CSV and can be plotted with --metric allocated.
Use --depth-width and --width-depth to select slices of larger-model probes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

COLORS = {"full": "#333333", "partitioned": "#2166AC"}
LABELS = {"full": "Full graph", "partitioned": "Partitioned (K=64, q=8)"}


def load_results(input_dir):
    """Require every planned configuration and preserve all failure states."""
    plan = json.loads((input_dir / "plan.json").read_text())
    rows = []
    for job in plan["jobs"]:
        path = input_dir / "runs" / f"{job['id']}.json"
        row = (
            json.loads(path.read_text())
            if path.exists()
            else {**job, "status": "pending"}
        )
        if any(row.get(key) != value for key, value in job.items()):
            raise ValueError(
                f"Result configuration does not match the plan: {path}"
            )
        rows.append(row)
    frame = pd.DataFrame(rows)
    unfinished = frame.loc[~frame.status.isin(["success", "cuda_oom"])]
    if len(unfinished):
        counts = unfinished.status.value_counts().to_dict()
        raise ValueError(
            f"Sweep has unfinished or unexpected failures: {counts}"
        )
    if (
        frame.gpu_name.nunique() != 1
        or frame.gpu_total_gib.max() - frame.gpu_total_gib.min() > 0.01
    ):
        raise ValueError(
            "Do not combine results from different GPU models/capacities."
        )
    successful = frame.loc[frame.status == "success"]
    for metric in ("peak_allocated_gib", "peak_reserved_gib"):
        if metric not in frame:
            frame[metric] = np.nan
        values = successful.get(metric, pd.Series(dtype=float))
        if (
            len(values) != len(successful)
            or not np.isfinite(values).all()
            or (values <= 0).any()
        ):
            raise ValueError(
                f"Successful runs have missing or invalid {metric}."
            )
    if (
        not successful.empty
        and not (successful.completed_epochs == plan["epochs"]).all()
    ):
        raise ValueError("Successful runs did not finish all planned epochs.")
    return frame


def plot_results(
    frame, output_dir, metric="reserved", *, depth_width=128, width_depth=4
):
    """Compare width and depth at fixed graph size with separate OOM marks."""
    if depth_width <= 0 or width_depth <= 0:
        raise ValueError("Slice width and depth must be positive.")
    slices = [
        (axis, fixed_axis, value)
        for axis, fixed_axis, value in (
            ("depth", "width", depth_width),
            ("width", "depth", width_depth),
        )
        if (frame[fixed_axis] == value).any()
    ]
    if not slices:
        raise ValueError(
            "No measurements in the requested slices. Choose --depth-width "
            "or --width-depth present in the results."
        )
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
    models = [m for m in ("cwn", "sccnn") if m in set(frame.model)]
    total = float(frame.gpu_total_gib.iloc[0])
    metric_key = f"peak_{metric}_gib"
    fig, axes = plt.subplots(
        len(slices),
        len(models),
        figsize=(3.5 * len(models), 5.3 if len(slices) == 2 else 3.3),
        squeeze=False,
        sharey=True,
    )
    for col, model in enumerate(models):
        for row, (axis, fixed_axis, fixed_value) in enumerate(slices):
            ax = axes[row, col]
            panel = frame.loc[
                (frame.model == model) & (frame[fixed_axis] == fixed_value)
            ]
            if panel.empty:
                ax.set_visible(False)
                continue
            ticks = sorted(panel[axis].unique())
            for mode, color in COLORS.items():
                selected = panel.loc[panel["mode"] == mode]
                peaks, lower, upper = [], [], []
                for value in ticks:
                    point = selected.loc[selected[axis] == value]
                    successes = point.loc[
                        point.status == "success", metric_key
                    ]
                    # Do not average a mixed success/OOM setting into a fit claim.
                    all_fit = len(successes) == len(point) and len(point) > 0
                    peaks.append(successes.mean() if all_fit else np.nan)
                    lower.append(successes.min() if all_fit else np.nan)
                    upper.append(successes.max() if all_fit else np.nan)
                    if (point.status == "cuda_oom").any():
                        ax.plot(
                            value,
                            1.04 if mode == "full" else 1.11,
                            "x",
                            color=color,
                            ms=6,
                            transform=ax.get_xaxis_transform(),
                            clip_on=False,
                        )
                ax.plot(
                    ticks,
                    peaks,
                    marker="o" if mode == "full" else "s",
                    ms=4,
                    color=color,
                    lw=1.4,
                )
                if selected.seed.nunique() > 1:
                    ax.fill_between(
                        ticks,
                        lower,
                        upper,
                        color=color,
                        alpha=0.15,
                        linewidth=0,
                    )
            ax.axhline(total, color="#777777", ls=":", lw=1)
            ax.set_ylim(0, total * 1.03)
            ax.set_xscale("log", base=2)
            ax.set_xticks(ticks, [str(t) for t in ticks])
            ax.set_xlabel(
                f"Layers (hidden channels = {fixed_value})"
                if axis == "depth"
                else f"Hidden channels (layers = {fixed_value})"
            )
            ax.set_title(
                f"{'abcd'[row * len(models) + col]}  {model.upper()}",
                loc="left",
                pad=30,
                fontweight="bold",
            )
            ax.spines[["top", "right"]].set_visible(False)
            ax.grid(axis="y", color="#E5E5E5", lw=0.5)
            ax.set_axisbelow(True)
            if col == 0:
                ax.set_ylabel(f"Peak GPU {metric} memory (GiB)")
    handles = [
        Line2D(
            [],
            [],
            color=COLORS[mode],
            marker="o" if mode == "full" else "s",
            label=label,
        )
        for mode, label in LABELS.items()
    ]
    handles += [
        Line2D(
            [],
            [],
            color="#777777",
            ls=":",
            label=f"GPU capacity ({total:.1f} GiB)",
        ),
        Line2D(
            [], [], color="#333333", marker="x", ls="none", label="CUDA OOM"
        ),
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, 1),
    )
    fig.text(
        0.5,
        0.018,
        "Cora Full · 3 training epochs · OOM markers above axes are not memory measurements",
        ha="center",
        fontsize=8,
    )
    fig.subplots_adjust(
        left=0.10,
        right=0.98,
        top=0.79 if len(slices) == 2 else 0.65,
        bottom=0.13 if len(slices) == 2 else 0.22,
        hspace=0.85,
        wspace=0.24,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_dir / "gpu_capacity_source.csv", index=False)
    for suffix in ("pdf", "svg", "png"):
        fig.savefig(
            output_dir / f"gpu_capacity_{metric}.{suffix}",
            dpi=300,
            bbox_inches="tight",
        )
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--metric", choices=["allocated", "reserved"], default="reserved"
    )
    parser.add_argument(
        "--depth-width",
        type=int,
        default=128,
        help="Hold this many channels fixed in the depth panel (default: 128)",
    )
    parser.add_argument(
        "--width-depth",
        type=int,
        default=4,
        help="Hold this many layers fixed in the width panel (default: 4)",
    )
    args = parser.parse_args()
    frame = load_results(args.input_dir)
    figure = plot_results(
        frame,
        args.output_dir or args.input_dir / "analysis",
        args.metric,
        depth_width=args.depth_width,
        width_depth=args.width_depth,
    )
    plt.close(figure)
    print(frame.groupby(["model", "mode", "status"]).size().to_string())


if __name__ == "__main__":
    main()
