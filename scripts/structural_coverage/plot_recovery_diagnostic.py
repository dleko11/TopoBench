"""Plot validated structural recovery without reinterpreting the raw runs.

The plotted means and sample-SD intervals come from the checked result bundle.
Analytical entropy is calculated from the same frozen span histogram, including
epochs beyond the simulated horizon when requested.
"""

from __future__ import annotations

import argparse
import csv
import ctypes
import errno
import hashlib
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter

from scripts.structural_coverage.plot_appendix_sweep_results_classic import (
    BLUE,
    MID,
    ORANGE,
    configure_style,
    style_axis,
)
from scripts.structural_coverage.recovery_core import (
    EntropyMilestone,
    recovery_entropy,
    span_entropy_milestone,
)
from scripts.structural_coverage.recovery_io import (
    load_recovery_bundle,
    validate_recovery_bundle,
)

FAMILY_ORDER = ("hypergraph", "simplicial", "cellular")
FAMILY_LABELS = {
    "hypergraph": "Neighbourhood hypergraph",
    "simplicial": "Clique simplicial",
    "cellular": "Cycle-based cellular",
}
RECOVERY_FAMILY_LABELS = {
    "hypergraph": "Hyperedges",
    "simplicial": "Simplicial 2-cells",
    "cellular": "Cellular 2-cells",
}
RECOVERY_FAMILY_COLORS = {
    "hypergraph": "#0072B2",
    "simplicial": "#CC79A7",
    "cellular": "#E69F00",
}
Q_COLORS = {2: BLUE, 4: ORANGE, 8: "#7C3AED", 16: "#059669"}
SOURCE_FIELDS = (
    "dataset",
    "manifest_sha256",
    "figure",
    "series_id",
    "K",
    "q",
    "family",
    "measurement",
    "stat",
    "epoch",
    "value",
    "normalization",
    "n_repetitions",
)
# The Cora Full P rows of Table \ref{tab:optuna_hparams}; GCN is graph-native.
CORA_FULL_SELECTED = {
    "GCN": (32, 8, "graph"),
    "EDHNN": (32, 4, "hypergraph"),
    "UniGNN": (32, 4, "hypergraph"),
    "CWN": (32, 4, "cellular"),
    "Cell TopoTune": (64, 8, "cellular"),
    "SCN": (32, 16, "simplicial"),
    "SCCNN": (32, 8, "simplicial"),
}


def _checked_bundle(
    bundle: dict[str, Any] | str | Path, *, publication: bool
) -> dict[str, Any]:
    if isinstance(bundle, (str, Path)):
        return load_recovery_bundle(bundle, publication_only=publication)
    return validate_recovery_bundle(bundle, publication_only=publication)


def _manifest_sha256(manifest: dict[str, Any]) -> str:
    canonical = json.dumps(
        manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _configs(bundle: dict[str, Any], K: int) -> tuple[list[int], list[str]]:
    schedules = [
        row
        for row in bundle["manifest"]["configuration_matrix"]
        if row["K"] == K
    ]
    if not schedules:
        raise ValueError(f"bundle has no configuration for K={K}")
    qs = sorted({row["q"] for row in schedules})
    families = [
        family
        for family in FAMILY_ORDER
        if any(family in row["families"] for row in schedules)
    ]
    return qs, families


def _display_epochs(maximum: int, every: int) -> list[int]:
    return sorted({0, maximum, *range(0, maximum + 1, every)})


def _rows_by_key(rows: list[dict[str, Any]]) -> dict[tuple, dict[str, Any]]:
    return {
        (
            row["K"],
            row["q"],
            row["family"],
            row.get("measurement"),
            row["epoch"],
        ): row
        for row in rows
    }


def _source_row(
    bundle: dict[str, Any],
    *,
    figure: str,
    K: int,
    q: int,
    family: str,
    measurement: str,
    stat: str,
    epoch: int,
    value: float,
    repetitions: int | None,
) -> dict[str, Any]:
    return {
        "dataset": bundle["manifest"]["dataset"],
        "manifest_sha256": _manifest_sha256(bundle["manifest"]),
        "figure": figure,
        "series_id": (f"{figure}:k{K}:q{q}:{family}:{measurement}:{stat}"),
        "K": K,
        "q": q,
        "family": family,
        "measurement": measurement,
        "stat": stat,
        "epoch": epoch,
        "value": value,
        "normalization": bundle["manifest"]["entropy_normalization"],
        "n_repetitions": repetitions if repetitions is not None else "",
    }


def _add_series(
    rows: list[dict[str, Any]],
    bundle: dict[str, Any],
    *,
    figure: str,
    K: int,
    q: int,
    family: str,
    measurement: str,
    stat: str,
    epochs: list[int],
    values: list[float],
    repetitions: int | None = None,
) -> None:
    rows.extend(
        _source_row(
            bundle,
            figure=figure,
            K=K,
            q=q,
            family=family,
            measurement=measurement,
            stat=stat,
            epoch=epoch,
            value=value,
            repetitions=repetitions,
        )
        for epoch, value in zip(epochs, values, strict=True)
    )


def _selected_models(K: int, q: int, family: str) -> list[str]:
    return [
        model
        for model, setting in CORA_FULL_SELECTED.items()
        if setting == (K, q, family)
    ]


def make_recovery_figure(
    bundle: dict[str, Any] | str | Path,
    *,
    K: int,
    publication: bool = True,
) -> tuple[plt.Figure, dict[tuple, Line2D | LineCollection]]:
    """Draw empirical recovery and theory; return artists keyed by protocol."""
    checked = _checked_bundle(bundle, publication=publication)
    configure_style()
    qs, families = _configs(checked, K)
    maximum = checked["manifest"]["epochs"]
    # Every measured epoch defines the curve. Sampling affects only markers and SD.
    epochs = list(range(maximum + 1))
    display_epochs = _display_epochs(
        maximum, checked["manifest"].get("sample_every", 1)
    )
    theory = _rows_by_key(checked["theory"])
    summary = _rows_by_key(checked["summary"])
    fig, axes = plt.subplots(
        len(qs),
        len(families),
        figsize=(7.0, max(3.5, 1.78 * len(qs) + 0.65)),
        squeeze=False,
        sharex=True,
        sharey=True,
    )
    artists: dict[tuple, Line2D | LineCollection] = {}
    source: list[dict[str, Any]] = []
    interval_min = 0.0
    interval_max = 1.0
    for row_index, q in enumerate(qs):
        for col_index, family in enumerate(families):
            ax = axes[row_index, col_index]
            key = (K, q, family)
            if key not in {
                (schedule["K"], schedule["q"], listed)
                for schedule in checked["manifest"]["configuration_matrix"]
                for listed in schedule["families"]
            }:
                ax.set_visible(False)
                continue
            expected = [
                theory[K, q, family, None, epoch]["expected_coverage"]
                for epoch in epochs
            ]
            line = ax.plot(
                epochs,
                expected,
                color=MID,
                linewidth=1.0,
                linestyle="--",
                label="Analytical support expectation",
            )[0]
            artists[(*key, "theory_support", "mean")] = line
            _add_series(
                source,
                checked,
                figure="recovery",
                K=K,
                q=q,
                family=family,
                measurement="theory_support",
                stat="mean",
                epochs=epochs,
                values=expected,
            )
            measures = ["support_available"]
            if family == "cellular":
                measures.append("actual_basis_recovery")
            for measure in measures:
                points = [
                    summary[K, q, family, measure, epoch] for epoch in epochs
                ]
                means = [point["coverage_mean"] for point in points]
                sd = [point["coverage_sample_sd"] for point in points]
                if any(value is None for value in means):
                    raise ValueError(
                        "zero reference count has no recovery fraction"
                    )
                color = RECOVERY_FAMILY_COLORS[family]
                label = (
                    "Empirical support availability"
                    if measure == "support_available"
                    else "Exact cycle-basis recovery"
                )
                empirical = ax.plot(
                    epochs,
                    means,
                    color=color,
                    linewidth=1.4,
                    linestyle="--" if measure == "support_available" else "-",
                    marker="o",
                    markevery=display_epochs,
                    markersize=3.2,
                    markerfacecolor=(
                        "white" if measure == "support_available" else color
                    ),
                    markeredgecolor=color,
                    markeredgewidth=0.9,
                    label=label,
                )[0]
                artists[(*key, measure, "mean")] = empirical
                _add_series(
                    source,
                    checked,
                    figure="recovery",
                    K=K,
                    q=q,
                    family=family,
                    measurement=measure,
                    stat="mean",
                    epochs=epochs,
                    values=means,
                    repetitions=points[0]["n_repetitions"],
                )
                if all(value is not None for value in sd):
                    sampled_means = [means[epoch] for epoch in display_epochs]
                    sampled_sd = [sd[epoch] for epoch in display_epochs]
                    lower = [
                        mean - spread
                        for mean, spread in zip(
                            sampled_means, sampled_sd, strict=True
                        )
                    ]
                    upper = [
                        mean + spread
                        for mean, spread in zip(
                            sampled_means, sampled_sd, strict=True
                        )
                    ]
                    interval_min = min(interval_min, *lower)
                    interval_max = max(interval_max, *upper)
                    interval = ax.vlines(
                        display_epochs,
                        lower,
                        upper,
                        color=color,
                        alpha=0.28,
                        linewidth=1.0,
                    )
                    artists[(*key, measure, "sample_sd_interval")] = interval
                    for stat, values in (
                        ("sample_sd_low", lower),
                        ("sample_sd_high", upper),
                    ):
                        _add_series(
                            source,
                            checked,
                            figure="recovery",
                            K=K,
                            q=q,
                            family=family,
                            measurement=measure,
                            stat=stat,
                            epochs=display_epochs,
                            values=values,
                            repetitions=points[0]["n_repetitions"],
                        )
            ceiling_value = summary[
                K, q, family, "support_available", epochs[0]
            ]["observable_fraction"]
            if ceiling_value is None:
                raise ValueError(
                    "zero reference count has no observable ceiling"
                )
            ceiling = ax.plot(
                [0, maximum],
                [ceiling_value, ceiling_value],
                color="#A0A7B0",
                linestyle=":",
                linewidth=0.9,
                label="q-observable fraction",
            )[0]
            artists[(*key, "observable_ceiling", "mean")] = ceiling
            _add_series(
                source,
                checked,
                figure="recovery",
                K=K,
                q=q,
                family=family,
                measurement="observable_ceiling",
                stat="mean",
                epochs=[0, maximum],
                values=[ceiling_value, ceiling_value],
            )
            heading = f"{RECOVERY_FAMILY_LABELS[family]}  |  q={q}"
            ax.set_title(heading, loc="center", fontsize=8.0, pad=5)
            if checked["manifest"]["dataset"] == "cora_full":
                models = _selected_models(K, q, family)
                if models:
                    ax.text(
                        0.98,
                        0.03,
                        "Selected: " + ", ".join(models),
                        transform=ax.transAxes,
                        ha="right",
                        va="bottom",
                        fontsize=6.1,
                        color=MID,
                    )
            ax.set_xlim(0, maximum)
            style_axis(ax)
            ax.grid(axis="y", color="#DCE1E7", linewidth=0.55, linestyle=":")
            ax.yaxis.set_major_formatter(
                PercentFormatter(xmax=1.0, decimals=0)
            )
            ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
            if col_index == 0:
                ax.set_ylabel("Fraction of full-graph structures")
            if row_index == len(qs) - 1:
                ax.set_xlabel("Epoch")
    dataset = checked["manifest"]["dataset"].replace("_", " ").title()
    diagnostic_label = (
        " · SMOKE / diagnostic only"
        if checked["manifest"]["run_mode"] == "smoke"
        else ""
    )
    repetitions = len(checked["manifest"]["seeds"])
    reshuffling = "reshuffling" if repetitions == 1 else "reshufflings"
    fig.suptitle(
        f"{dataset}\nK={K} · {repetitions} independent {reshuffling}{diagnostic_label}",
        fontsize=9.0,
        y=0.98 if len(qs) == 1 else 0.998,
    )
    for ax in axes.flat:
        if ax.get_visible():
            ax.set_ylim(
                min(-0.02, interval_min - 0.02),
                max(1.05, interval_max + 0.02),
            )
    legend_entries: dict[str, Any] = {}
    for ax in axes.flat:
        if ax.get_visible():
            handles, labels = ax.get_legend_handles_labels()
            for handle, label in zip(handles, labels, strict=True):
                legend_entries.setdefault(label, handle)
    if legend_entries:
        fig.legend(
            list(legend_entries.values()),
            list(legend_entries),
            loc="lower center",
            ncol=min(len(legend_entries), 5),
            bbox_to_anchor=(0.5, 0.09 if len(qs) == 1 else 0.035),
        )
    if any(key[-1] == "sample_sd_interval" for key in artists):
        fig.text(
            0.5,
            0.03 if len(qs) == 1 else 0.012,
            "Whiskers show mean ±1 sample SD across seeds (not a confidence interval).",
            ha="center",
            fontsize=6.5,
            color=MID,
        )
    fig.subplots_adjust(
        left=0.10,
        right=0.99,
        top=0.80 if len(qs) == 1 else 0.94,
        bottom=0.30 if len(qs) == 1 else 0.14,
        hspace=0.42,
        wspace=0.18,
    )
    fig._recovery_source_rows = source
    return fig, artists


def make_combined_support_figure(
    bundle: dict[str, Any] | str | Path,
    *,
    K: int,
    q: int,
    publication: bool = True,
) -> tuple[plt.Figure, dict[str, Line2D]]:
    """Compare cumulative support availability across the three domains."""
    checked = _checked_bundle(bundle, publication=publication)
    families = ("hypergraph", "cellular", "simplicial")
    scheduled = {
        family
        for row in checked["manifest"]["configuration_matrix"]
        if row["K"] == K and row["q"] == q
        for family in row["families"]
    }
    if not set(families) <= scheduled:
        raise ValueError(f"K={K}, q={q} lacks a three-domain schedule")

    configure_style()
    maximum = checked["manifest"]["epochs"]
    epochs = list(range(maximum + 1))
    display_epochs = _display_epochs(
        maximum, checked["manifest"].get("sample_every", 1)
    )
    summary = _rows_by_key(checked["summary"])
    fig, ax = plt.subplots(figsize=(5.3, 3.5))
    lines: dict[str, Line2D] = {}
    for family in families:
        values = [
            summary[K, q, family, "support_available", epoch]["coverage_mean"]
            for epoch in epochs
        ]
        if any(value is None for value in values):
            raise ValueError(f"{family} has no defined support fraction")
        color = RECOVERY_FAMILY_COLORS[family]
        lines[family] = ax.plot(
            epochs,
            values,
            color=color,
            linewidth=1.5,
            linestyle="--",
            marker="o",
            markevery=display_epochs,
            markersize=3.8,
            markerfacecolor="white",
            markeredgecolor=color,
            markeredgewidth=1.0,
            label=RECOVERY_FAMILY_LABELS[family],
        )[0]

    ax.set_xlim(0, maximum)
    ax.set_ylim(-0.015, 1.025)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Full-graph references")
    style_axis(ax)
    ax.grid(axis="y", color="#DCE1E7", linewidth=0.55, linestyle=":")
    ax.legend(loc="lower right", frameon=False)

    dataset = checked["manifest"]["dataset"].replace("_", " ").title()
    repetitions = len(checked["manifest"]["seeds"])
    reshuffling = "reshuffling" if repetitions == 1 else "reshufflings"
    ax.set_title(
        f"Cumulative support availability  |  K={K}, q={q}",
        fontsize=8.1,
        pad=7,
    )
    fig.suptitle(dataset, fontsize=10.0, y=0.98)
    run_label = (
        "pilot diagnostic only"
        if checked["manifest"]["run_mode"] == "smoke"
        else "completed run"
    )
    fig.text(
        0.5,
        0.025,
        f"{repetitions} {reshuffling} · "
        f"{checked['manifest']['partition_source']} partition · {run_label}",
        ha="center",
        fontsize=6.5,
        color=MID,
    )
    fig.subplots_adjust(left=0.13, right=0.98, bottom=0.20, top=0.80)
    return fig, lines


def _entropy_epochs(maximum: int) -> list[int]:
    grid = np.geomspace(1, max(1_000_000, maximum), num=160)
    return sorted({0, maximum, *(int(round(value)) for value in grid)})


def _milestone_metadata(
    milestone: EntropyMilestone, simulated_horizon: int
) -> dict[str, Any]:
    """Keep analytical milestones distinct from simulated recovery results."""
    return {
        "status": milestone.status,
        "peak_epoch": milestone.peak_epoch,
        "peak_nats_per_reference": milestone.peak_value,
        "final_decay_epoch": milestone.final_decay_epoch,
        "relative_decay_fraction": 0.01,
        "basis": "analytical_structural_recovery",
        "extrapolated_beyond_epoch": simulated_horizon,
        "reason": milestone.reason,
    }


def make_entropy_figure(
    bundle: dict[str, Any] | str | Path,
    *,
    K: int,
    epochs: list[int] | None = None,
    publication: bool = True,
) -> tuple[plt.Figure, dict[tuple, Line2D]]:
    """Draw analytical entropy per frozen reference, never seed uncertainty."""
    checked = _checked_bundle(bundle, publication=publication)
    configure_style()
    qs, families = _configs(checked, K)
    maximum = checked["manifest"]["epochs"]
    histograms: dict[tuple[int, str], dict[int, int]] = {}
    for row in checked["span_histogram"]:
        histograms.setdefault((row["K"], row["family"]), {})[row["span"]] = (
            row["count"]
        )
    milestones: dict[tuple[int, int, str], EntropyMilestone] = {
        (K, q, family): span_entropy_milestone(histograms[K, family], K, q)
        for family in families
        for q in qs
        if any(
            row["K"] == K and row["q"] == q and family in row["families"]
            for row in checked["manifest"]["configuration_matrix"]
        )
    }
    if epochs is None:
        last_milestone = max(
            (
                milestone.final_decay_epoch
                for milestone in milestones.values()
                if milestone.status == "resolved"
            ),
            default=maximum,
        )
        chosen = _entropy_epochs(max(maximum, last_milestone))
    else:
        chosen = list(epochs)
    if not chosen or any(
        type(epoch) is not int or epoch < 0 for epoch in chosen
    ):
        raise ValueError("entropy epochs must be nonnegative integers")
    if chosen != sorted(set(chosen)):
        raise ValueError("entropy epochs must be sorted and distinct")
    # A resolved milestone marker must be an actual point on its plotted line.
    last_chosen = chosen[-1]
    chosen = sorted(
        {
            *chosen,
            *(
                epoch
                for milestone in milestones.values()
                if milestone.status == "resolved"
                for epoch in (
                    milestone.peak_epoch,
                    milestone.final_decay_epoch,
                )
                if epoch is not None and epoch <= last_chosen
            ),
        }
    )
    fig, axes = plt.subplots(
        1, len(families), figsize=(7.0, 2.5), squeeze=False
    )
    artists: dict[tuple, Line2D] = {}
    source: list[dict[str, Any]] = []
    for col_index, family in enumerate(families):
        ax = axes[0, col_index]
        for q in qs:
            if not any(
                row["K"] == K and row["q"] == q and family in row["families"]
                for row in checked["manifest"]["configuration_matrix"]
            ):
                continue
            values = [
                recovery_entropy(histograms[K, family], K, q, epoch)
                for epoch in chosen
            ]
            if any(value is None for value in values):
                raise ValueError("zero reference count has no entropy curve")
            line = ax.plot(
                chosen,
                values,
                color=Q_COLORS.get(q, BLUE),
                linewidth=1.2,
                label=f"q={q}",
            )[0]
            artists[(K, q, family, "entropy")] = line
            _add_series(
                source,
                checked,
                figure="entropy",
                K=K,
                q=q,
                family=family,
                measurement="analytical_entropy",
                stat="nats_per_reference",
                epochs=chosen,
                values=values,
            )
            milestone = milestones[(K, q, family)]
            if milestone.status == "resolved":
                for suffix, epoch, marker, filled in (
                    (
                        "entropy_peak",
                        milestone.peak_epoch,
                        "o",
                        True,
                    ),
                    (
                        "entropy_final_decay",
                        milestone.final_decay_epoch,
                        "v",
                        False,
                    ),
                ):
                    if epoch > chosen[-1]:
                        continue
                    value = recovery_entropy(
                        histograms[K, family], K, q, epoch
                    )
                    point = ax.plot(
                        [epoch],
                        [value],
                        linestyle="none",
                        marker=marker,
                        markersize=3.5,
                        markerfacecolor=(
                            Q_COLORS.get(q, BLUE) if filled else "white"
                        ),
                        markeredgecolor=Q_COLORS.get(q, BLUE),
                    )[0]
                    artists[(K, q, family, suffix)] = point
                    _add_series(
                        source,
                        checked,
                        figure="entropy",
                        K=K,
                        q=q,
                        family=family,
                        measurement=suffix,
                        stat="nats_per_reference",
                        epochs=[epoch],
                        values=[value],
                    )
        ax.set_xscale("symlog", linthresh=1)
        ax.set_title(FAMILY_LABELS[family], loc="left")
        ax.set_xlabel(f"Epoch (T>{maximum}: extrapolated)")
        if col_index == 0:
            ax.set_ylabel("Entropy (nats per reference)")
        style_axis(ax)
    dataset = checked["manifest"]["dataset"].replace("_", " ").title()
    diagnostic_label = (
        " · SMOKE / diagnostic only"
        if checked["manifest"]["run_mode"] == "smoke"
        else ""
    )
    fig.suptitle(
        f"{dataset} · K={K} · analytical structural-recovery entropy{diagnostic_label}",
        fontsize=8.6,
        y=1.04,
    )
    fig.text(
        0.5,
        0.01,
        "Circle: entropy peak. Open triangle: final decay below 1% of peak.\n"
        f"T>{maximum}: analytical extrapolation from frozen structure "
        "spans, not observed recovery or training convergence",
        ha="center",
        fontsize=6.0,
        color=MID,
    )
    legend_entries: dict[str, Any] = {}
    for ax in axes.flat:
        if ax.get_visible():
            handles, labels = ax.get_legend_handles_labels()
            for handle, label in zip(handles, labels, strict=True):
                legend_entries.setdefault(label, handle)
    if legend_entries:
        fig.legend(
            list(legend_entries.values()),
            list(legend_entries),
            loc="lower center",
            ncol=len(legend_entries),
            bbox_to_anchor=(0.5, -0.10),
        )
    fig.subplots_adjust(
        left=0.10, right=0.99, top=0.79, bottom=0.30, wspace=0.28
    )
    fig._recovery_source_rows = source
    fig._entropy_milestones = milestones
    return fig, artists


def _validate_staged_export(
    stage: Path,
    stems: list[str],
    rows: list[dict[str, Any]],
    metadata: dict[str, Any],
) -> None:
    expected_names = {
        *(
            f"{stem}.{suffix}"
            for stem in stems
            for suffix in ("pdf", "png", "svg")
        ),
        "plot_source_data.csv",
        "figure_metadata.json",
    }
    if {path.name for path in stage.iterdir()} != expected_names:
        raise ValueError("staged export files are incomplete or unexpected")
    signatures = {
        "pdf": (b"%PDF",),
        "png": (b"\x89PNG\r\n\x1a\n",),
        "svg": (b"<?xml", b"<svg"),
    }
    for stem in stems:
        for suffix, accepted in signatures.items():
            path = stage / f"{stem}.{suffix}"
            if not path.is_file() or path.stat().st_size == 0:
                raise ValueError(
                    f"staged figure missing or empty: {path.name}"
                )
            with path.open("rb") as stream:
                header = stream.read(8)
            if not any(header.startswith(signature) for signature in accepted):
                raise ValueError(
                    f"staged figure format is invalid: {path.name}"
                )
    with (stage / "plot_source_data.csv").open(
        encoding="utf-8", newline=""
    ) as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != SOURCE_FIELDS:
            raise ValueError("staged plot-source CSV has unexpected columns")
        reloaded_rows = list(reader)
    expected_rows = [
        {field: str(row[field]) for field in SOURCE_FIELDS} for row in rows
    ]
    if reloaded_rows != expected_rows:
        raise ValueError("staged plot-source CSV differs from plotted values")
    with (stage / "figure_metadata.json").open(encoding="utf-8") as stream:
        reloaded_metadata = json.load(stream)
    if reloaded_metadata != metadata:
        raise ValueError("staged figure metadata differs from the bundle")


def _publish_staged_directory(stage: Path, target: Path) -> None:
    """Atomically publish without replacing a concurrent destination."""
    if sys.platform == "darwin":
        rename = ctypes.CDLL(None, use_errno=True).renamex_np
        rename.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint]
        rename.restype = ctypes.c_int
        result = rename(os.fsencode(stage), os.fsencode(target), 0x00000004)
    elif sys.platform.startswith("linux"):
        library = ctypes.CDLL(None, use_errno=True)
        try:
            rename = library.renameat2
        except AttributeError as exc:
            raise RuntimeError(
                "atomic no-replace rename is unavailable"
            ) from exc
        rename.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        rename.restype = ctypes.c_int
        result = rename(-100, os.fsencode(stage), -100, os.fsencode(target), 1)
    else:
        raise RuntimeError("atomic no-replace rename is unavailable")
    if result != 0:
        error = ctypes.get_errno()
        if error == errno.EEXIST:
            raise FileExistsError(
                error, "plot output already exists", str(target)
            )
        raise OSError(error, os.strerror(error), str(target))


def export_recovery_figures(
    bundle: dict[str, Any] | str | Path,
    output_dir: str | Path,
    *,
    publication: bool = True,
) -> dict[str, Any]:
    """Export vector/raster figures and their exact numeric plot-source rows."""
    checked = _checked_bundle(bundle, publication=publication)
    output = Path(output_dir)
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"plot output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=output.parent)
    )
    all_rows: list[dict[str, Any]] = []
    stems: list[str] = []
    figure_configurations: dict[str, dict[str, Any]] = {}
    entropy_milestones: dict[str, dict[str, Any]] = {}
    try:
        dataset = checked["manifest"]["dataset"]
        dataset_slug = dataset.replace("_", "")
        for K in sorted(
            {row["K"] for row in checked["manifest"]["configuration_matrix"]}
        ):
            qs, families = _configs(checked, K)
            recovery_stem = f"recovery_{dataset_slug}_k{K}"
            if dataset == "cora_full" and (K, qs, families) == (
                64,
                [8],
                ["cellular"],
            ):
                recovery_stem = "recovery_corafull_topotune_k64_q8"
            makers = [(recovery_stem, make_recovery_figure)]
            if (
                min(
                    row["K"]
                    for row in checked["manifest"]["configuration_matrix"]
                )
                == K
            ):
                makers.append(
                    (f"entropy_{dataset_slug}_k{K}", make_entropy_figure)
                )
            for stem, maker in makers:
                fig, _artists = maker(checked, K=K, publication=publication)
                try:
                    all_rows.extend(fig._recovery_source_rows)
                    if maker is make_entropy_figure:
                        entropy_milestones.update(
                            {
                                f"k{K_value}:q{q}:{family}": _milestone_metadata(
                                    milestone, checked["manifest"]["epochs"]
                                )
                                for (K_value, q, family), milestone in (
                                    fig._entropy_milestones.items()
                                )
                            }
                        )
                    for suffix in ("pdf", "png", "svg"):
                        fig.savefig(
                            stage / f"{stem}.{suffix}",
                            dpi=300 if suffix == "png" else None,
                            bbox_inches="tight",
                            pad_inches=0.04,
                        )
                finally:
                    plt.close(fig)
                stems.append(stem)
                figure_configurations[stem] = {
                    "dataset": checked["manifest"]["dataset"],
                    "K": K,
                    "q_values": qs,
                    "families": families,
                }
        with (stage / "plot_source_data.csv").open(
            "w", encoding="utf-8", newline=""
        ) as stream:
            writer = csv.DictWriter(stream, fieldnames=SOURCE_FIELDS)
            writer.writeheader()
            writer.writerows(all_rows)
        metadata = {
            "dataset": checked["manifest"]["dataset"],
            "manifest_sha256": _manifest_sha256(checked["manifest"]),
            "run_mode": checked["manifest"]["run_mode"],
            "normalization": checked["manifest"]["entropy_normalization"],
            "figures": stems,
            "figure_configurations": figure_configurations,
            "entropy_milestones": entropy_milestones,
            "source_rows": len(all_rows),
        }
        (stage / "figure_metadata.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        _validate_staged_export(stage, stems, all_rows, metadata)
        _publish_staged_directory(stage, output)
        return metadata
    finally:
        if stage.exists():
            shutil.rmtree(stage)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir",
        required=True,
        type=Path,
        help="validated recovery-v2 bundle directory",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="new figure output directory",
    )
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument(
        "--publication",
        action="store_true",
        help="require the complete pinned Cora Full publication protocol",
    )
    modes.add_argument(
        "--diagnostic",
        action="store_true",
        help="allow explicitly labelled incomplete smoke results",
    )
    args = parser.parse_args(argv)
    export_recovery_figures(
        args.results_dir, args.output_dir, publication=args.publication
    )


if __name__ == "__main__":
    main()
