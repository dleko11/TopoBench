"""Numerical artist tests for the validated structural-recovery figures."""

import csv
import hashlib
import json
import math
import statistics
from copy import deepcopy

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pytest

from scripts.structural_coverage import plot_recovery_diagnostic as plot_module
from scripts.structural_coverage.plot_recovery_diagnostic import (
    export_recovery_figures,
    make_entropy_figure,
    make_recovery_figure,
)
from scripts.structural_coverage.recovery_core import (
    EntropyMilestone,
    recovery_entropy,
)
from scripts.structural_coverage.recovery_io import (
    CANONICAL_FAMILY_DEFINITIONS,
    write_recovery_bundle,
)
from test.scripts.test_recovery_io import tiny_bundle as _base_bundle


@pytest.fixture
def three_seed_bundle():
    """Independent counts: support T1 is 1, 1, 2 out of three."""
    bundle = deepcopy(_base_bundle.__wrapped__())
    bundle["manifest"]["seeds"] = [0, 1, 2]
    for row in bundle["observations"]:
        if (
            row["seed"] == 1
            and row["epoch"] == 1
            and row["measurement"] == "support_available"
        ):
            row["recovered_count"] = 1
            row["coverage"] = 1 / 3
    for measurement, counts in (
        ("support_available", [0, 2, 2]),
        ("actual_basis_recovery", [0, 1, 1]),
    ):
        for epoch, count in enumerate(counts):
            bundle["observations"].append(
                {
                    "dataset": "toy",
                    "K": 4,
                    "q": 2,
                    "seed": 2,
                    "epoch": epoch,
                    "family": "cellular",
                    "measurement": measurement,
                    "recovered_count": count,
                    "reference_count": 3,
                    "observable_count": 2,
                    "coverage": count / 3,
                }
            )
    for row in bundle["summary"]:
        row["n_repetitions"] = 3
        if row["epoch"] == 1:
            row["coverage_sample_sd"] = math.sqrt(1 / 27)
            row["coverage_mean"] = (
                4 / 9 if row["measurement"] == "support_available" else 2 / 9
            )
    bundle["manifest"]["completed"] = True
    return bundle


@pytest.fixture
def layout_bundle():
    """Small hand-defined multi-panel bundle with the publication K,q layout."""
    bundle = deepcopy(_base_bundle.__wrapped__())
    manifest = bundle["manifest"]
    manifest.update(
        {
            "dataset": "cora_full",
            "num_nodes": 64,
            "seeds": [0, 1, 2],
            "epochs": 3,
            "sample_every": 2,
            "configuration_matrix": [
                {
                    "K": 32,
                    "q": q,
                    "families": ["hypergraph", "simplicial", "cellular"],
                }
                for q in (2, 4, 8, 16)
            ]
            + [{"K": 64, "q": 8, "families": ["cellular"]}],
            "partition_hashes": {
                "32": "c" * 64 + ":" + "e" * 64,
                "64": "d" * 64 + ":" + "f" * 64,
            },
            "active_cluster_counts": {"32": 32, "64": 64},
            "family_definitions": {
                family: CANONICAL_FAMILY_DEFINITIONS[family]
                for family in ("hypergraph", "simplicial", "cellular")
            },
            "reference_hashes": {
                "hypergraph": "1" * 64,
                "simplicial": "2" * 64,
                "cellular": "3" * 64,
            },
            "completed": True,
        }
    )
    bundle["partition_metadata"] = {
        str(K): {
            **bundle["partition_metadata"]["4"],
            "num_nodes": 64,
            "num_parts": K,
            "partition_assignment_sha256": manifest["partition_hashes"][
                str(K)
            ],
        }
        for K in (32, 64)
    }
    families = ("hypergraph", "simplicial", "cellular")
    bundle["reference_summary"] = [
        {
            "dataset": "cora_full",
            "family": family,
            "reference_count": 3,
            "reference_hash": manifest["reference_hashes"][family],
        }
        for family in families
    ]
    bundle["span_histogram"] = [
        {
            "dataset": "cora_full",
            "K": K,
            "family": family,
            "span": span,
            "count": 1,
            "reference_hash": manifest["reference_hashes"][family],
        }
        for K in (32, 64)
        for family in families
        if K == 32 or family == "cellular"
        for span in (1, 2, 3)
    ]
    bundle["theory"] = []
    bundle["observations"] = []
    bundle["summary"] = []
    for configuration in manifest["configuration_matrix"]:
        K, q = configuration["K"], configuration["q"]
        probabilities = (
            1.0,
            (q - 1) / (K - 1),
            (q - 1) * (q - 2) / ((K - 1) * (K - 2)),
        )
        observable = 2 if q == 2 else 3
        for family in configuration["families"]:
            measures = ["support_available"]
            if family == "cellular":
                measures.append("actual_basis_recovery")
            for epoch in range(4):
                recovered_probabilities = [
                    1 - (1 - probability) ** epoch
                    for probability in probabilities
                ]
                entropy = (
                    sum(
                        -p * math.log(p) - (1 - p) * math.log(1 - p)
                        if 0 < p < 1
                        else 0.0
                        for p in recovered_probabilities
                    )
                    / 3
                )
                bundle["theory"].append(
                    {
                        "dataset": "cora_full",
                        "K": K,
                        "q": q,
                        "epoch": epoch,
                        "family": family,
                        "reference_count": 3,
                        "observable_count": observable,
                        "expected_coverage": sum(recovered_probabilities) / 3,
                        "entropy_nats_per_reference": entropy,
                    }
                )
                for measurement in measures:
                    per_seed_counts = []
                    for seed in (0, 1, 2):
                        count = (
                            (0, 1, 2, 2)[epoch]
                            if measurement == "support_available"
                            else (0, 0, 1, 1)[epoch]
                        )
                        per_seed_counts.append(count)
                        bundle["observations"].append(
                            {
                                "dataset": "cora_full",
                                "K": K,
                                "q": q,
                                "seed": seed,
                                "epoch": epoch,
                                "family": family,
                                "measurement": measurement,
                                "recovered_count": count,
                                "reference_count": 3,
                                "observable_count": observable,
                                "coverage": count / 3,
                            }
                        )
                    per_seed_fractions = [
                        count / 3 for count in per_seed_counts
                    ]
                    bundle["summary"].append(
                        {
                            "dataset": "cora_full",
                            "K": K,
                            "q": q,
                            "epoch": epoch,
                            "family": family,
                            "measurement": measurement,
                            "n_repetitions": 3,
                            "coverage_mean": statistics.mean(
                                per_seed_fractions
                            ),
                            "coverage_sample_sd": statistics.stdev(
                                per_seed_fractions
                            ),
                            "reference_count": 3,
                            "observable_fraction": observable / 3,
                        }
                    )
    return bundle


def test_recovery_artists_match_hand_authored_three_seed_counts(
    three_seed_bundle,
):
    fig, artists = make_recovery_figure(
        three_seed_bundle, K=4, publication=False
    )
    try:
        line = artists[(4, 2, "cellular", "support_available", "mean")]
        assert list(line.get_xdata()) == [0, 1, 2]
        assert list(line.get_ydata()) == pytest.approx([0, 4 / 9, 2 / 3])
        actual = artists[(4, 2, "cellular", "actual_basis_recovery", "mean")]
        assert list(actual.get_ydata()) == pytest.approx([0, 2 / 9, 1 / 3])
        expected = artists[(4, 2, "cellular", "theory_support", "mean")]
        assert list(expected.get_ydata()) == pytest.approx([0, 4 / 9, 14 / 27])
        ceiling = artists[(4, 2, "cellular", "observable_ceiling", "mean")]
        assert list(ceiling.get_ydata()) == pytest.approx([2 / 3, 2 / 3])
        assert "fraction" in fig.axes[0].get_ylabel().lower()
        assert "Cora" not in fig._suptitle.get_text()
        assert "toy" in fig._suptitle.get_text().lower()
        assert "3" in fig._suptitle.get_text()
        assert "SMOKE" in fig._suptitle.get_text()
    finally:
        plt.close(fig)


def test_single_seed_diagnostic_does_not_claim_multiple_runs_or_sd():
    bundle = deepcopy(_base_bundle.__wrapped__())
    bundle["manifest"]["seeds"] = [0]
    bundle["manifest"]["completed"] = True
    bundle["observations"] = [
        row for row in bundle["observations"] if row["seed"] == 0
    ]
    counts = {
        (row["epoch"], row["measurement"]): row["recovered_count"]
        for row in bundle["observations"]
    }
    for row in bundle["summary"]:
        row["n_repetitions"] = 1
        row["coverage_mean"] = (
            counts[row["epoch"], row["measurement"]] / row["reference_count"]
        )
        row["coverage_sample_sd"] = None
    fig, artists = make_recovery_figure(bundle, K=4, publication=False)
    try:
        assert "1 independent reshuffling" in fig._suptitle.get_text()
        assert "reshufflings" not in fig._suptitle.get_text()
        assert "SMOKE" in fig._suptitle.get_text()
        assert not any("Whiskers" in item.get_text() for item in fig.texts)
        assert not any(key[-1] == "sample_sd_interval" for key in artists)
    finally:
        plt.close(fig)


def test_sample_sd_segments_are_not_population_sd_or_standard_error(
    three_seed_bundle,
):
    fig, artists = make_recovery_figure(
        three_seed_bundle, K=4, publication=False
    )
    try:
        interval = artists[
            (4, 2, "cellular", "support_available", "sample_sd_interval")
        ]
        segments = interval.get_segments()
        assert len(segments) == 3
        assert segments[1][:, 0] == pytest.approx([1, 1])
        assert segments[1][:, 1] == pytest.approx(
            [4 / 9 - math.sqrt(1 / 27), 4 / 9 + math.sqrt(1 / 27)]
        )
        assert any("Whiskers" in item.get_text() for item in fig.texts)
    finally:
        plt.close(fig)


def test_entropy_artist_uses_all_three_references(three_seed_bundle):
    fig, artists = make_entropy_figure(
        three_seed_bundle, K=4, epochs=[0, 1, 2], publication=False
    )
    try:
        line = artists[(4, 2, "cellular", "entropy")]
        h = -(1 / 3) * math.log(1 / 3) - (2 / 3) * math.log(2 / 3)
        assert list(line.get_xdata()) == [0, 1, 2]
        assert list(line.get_ydata())[1] == pytest.approx(h / 3)
        assert "nats per reference" in fig.axes[0].get_ylabel().lower()
        assert fig.axes[0].get_title(loc="left") == "Cycle-based cellular"
        assert "analytical" in fig._suptitle.get_text().lower()
        assert "empirical" not in fig._suptitle.get_text().lower()
        assert "SMOKE" in fig._suptitle.get_text()
    finally:
        plt.close(fig)


def test_entropy_figure_marks_exact_analytical_peak_and_final_decay(
    three_seed_bundle,
):
    fig, artists = make_entropy_figure(
        three_seed_bundle, K=4, publication=False
    )
    try:
        key = (4, 2, "cellular")
        milestone = fig._entropy_milestones[key]
        assert milestone.status == "resolved"
        for suffix, epoch in (
            ("entropy_peak", milestone.peak_epoch),
            ("entropy_final_decay", milestone.final_decay_epoch),
        ):
            marker = artists[(*key, suffix)]
            assert list(marker.get_xdata()) == [epoch]
            assert list(marker.get_ydata()) == pytest.approx(
                [recovery_entropy({1: 1, 2: 1, 3: 1}, 4, 2, epoch)]
            )
            source = [
                row
                for row in fig._recovery_source_rows
                if row["series_id"]
                == f"entropy:k4:q2:cellular:{suffix}:nats_per_reference"
            ]
            assert len(source) == 1
            assert source[0]["epoch"] == epoch
            assert source[0]["value"] == pytest.approx(marker.get_ydata()[0])
            curve = artists[(*key, "entropy")]
            matching_curve_points = [
                value
                for plotted_epoch, value in zip(
                    curve.get_xdata(), curve.get_ydata(), strict=True
                )
                if plotted_epoch == epoch
            ]
            assert matching_curve_points == pytest.approx(
                [marker.get_ydata()[0]]
            )
            curve_source = [
                row
                for row in fig._recovery_source_rows
                if row["series_id"]
                == "entropy:k4:q2:cellular:analytical_entropy:nats_per_reference"
                and row["epoch"] == epoch
            ]
            assert [row["value"] for row in curve_source] == pytest.approx(
                [marker.get_ydata()[0]]
            )
        labels = " ".join(text.get_text() for text in fig.texts)
        assert "T>2" in labels
        assert "analytical extrapolation" in labels
        assert "training convergence" in labels
        assert "circle: entropy peak" in labels.lower()
        assert "open triangle: final decay below 1%" in labels.lower()
        assert not fig.axes[0].collections
    finally:
        plt.close(fig)


def test_entropy_export_metadata_agrees_with_marker_source_rows(
    tmp_path, three_seed_bundle
):
    metadata = export_recovery_figures(
        three_seed_bundle, tmp_path / "entropy_export", publication=False
    )
    milestone = metadata["entropy_milestones"]["k4:q2:cellular"]
    assert milestone["status"] == "resolved"
    assert milestone["basis"] == "analytical_structural_recovery"
    assert milestone["extrapolated_beyond_epoch"] == 2
    with (tmp_path / "entropy_export" / "plot_source_data.csv").open(
        newline=""
    ) as stream:
        rows = list(csv.DictReader(stream))
    peak = next(
        row
        for row in rows
        if row["series_id"]
        == "entropy:k4:q2:cellular:entropy_peak:nats_per_reference"
    )
    final = next(
        row
        for row in rows
        if row["series_id"]
        == "entropy:k4:q2:cellular:entropy_final_decay:nats_per_reference"
    )
    assert int(peak["epoch"]) == milestone["peak_epoch"]
    assert float(peak["value"]) == pytest.approx(
        milestone["peak_nats_per_reference"]
    )
    assert int(final["epoch"]) == milestone["final_decay_epoch"]
    assert float(final["value"]) < 0.01 * float(peak["value"])


def test_entropy_default_axis_includes_resolved_milestone_after_a_million(
    three_seed_bundle, monkeypatch
):
    actual = plot_module.span_entropy_milestone

    def long_horizon_milestone(histogram, K, q):
        if (K, q) == (4, 2):
            return EntropyMilestone(
                status="resolved",
                peak_epoch=138_629,
                peak_value=recovery_entropy(histogram, K, q, 138_629),
                final_decay_epoch=1_411_664,
            )
        return actual(histogram, K, q)

    monkeypatch.setattr(
        plot_module, "span_entropy_milestone", long_horizon_milestone
    )
    fig, artists = make_entropy_figure(
        three_seed_bundle, K=4, publication=False
    )
    try:
        curve_epochs = list(artists[(4, 2, "cellular", "entropy")].get_xdata())
        assert curve_epochs[-1] >= 1_411_664
        assert 138_629 in curve_epochs
        assert 1_411_664 in curve_epochs
        assert (4, 2, "cellular", "entropy_final_decay") in artists
        assert fig.axes[0].get_xlim()[1] >= 1_411_664
        source_epochs = {
            row["epoch"]
            for row in fig._recovery_source_rows
            if row["series_id"]
            == "entropy:k4:q2:cellular:analytical_entropy:nats_per_reference"
        }
        assert {138_629, 1_411_664} <= source_epochs
    finally:
        plt.close(fig)


def test_publication_rejects_smoke_bundle(three_seed_bundle):
    with pytest.raises(ValueError, match="smoke"):
        make_recovery_figure(three_seed_bundle, K=4)


def test_plot_cli_requires_explicit_paths_and_mode(capsys):
    with pytest.raises(SystemExit) as absent:
        plot_module.main([])
    assert absent.value.code == 2
    error = capsys.readouterr().err
    assert "--results-dir" in error


def test_plot_cli_diagnostic_exports_validated_smoke_bundle(
    tmp_path, three_seed_bundle
):
    bundle_dir = tmp_path / "smoke-input"
    write_recovery_bundle(bundle_dir, **three_seed_bundle)
    output_dir = tmp_path / "diagnostic-figures"
    plot_module.main(
        [
            "--results-dir",
            str(bundle_dir),
            "--output-dir",
            str(output_dir),
            "--diagnostic",
        ]
    )
    names = {path.name for path in output_dir.iterdir()}
    assert "recovery_toy_k4.pdf" in names
    assert "entropy_toy_k4.svg" in names
    assert "plot_source_data.csv" in names
    assert "figure_metadata.json" in names


def test_plot_cli_publication_rejects_smoke_before_creating_output(
    tmp_path, three_seed_bundle
):
    bundle_dir = tmp_path / "smoke-input"
    write_recovery_bundle(bundle_dir, **three_seed_bundle)
    output_dir = tmp_path / "not-published"
    with pytest.raises(ValueError, match="smoke"):
        plot_module.main(
            [
                "--results-dir",
                str(bundle_dir),
                "--output-dir",
                str(output_dir),
                "--publication",
            ]
        )
    assert not output_dir.exists()


def test_incomplete_and_legacy_normalization_rejected(three_seed_bundle):
    incomplete = deepcopy(three_seed_bundle)
    incomplete["manifest"]["completed"] = False
    with pytest.raises(ValueError, match="incomplete"):
        make_recovery_figure(incomplete, K=4, publication=False)
    legacy = deepcopy(three_seed_bundle)
    legacy["manifest"]["entropy_normalization"] = "observable_reference"
    with pytest.raises(ValueError, match="normalization"):
        make_entropy_figure(legacy, K=4, publication=False)


def test_export_source_csv_matches_summary_and_artists(
    tmp_path, three_seed_bundle
):
    output = tmp_path / "plots"
    result = export_recovery_figures(
        three_seed_bundle, output, publication=False
    )
    assert set(result["figures"]) == {"recovery_toy_k4", "entropy_toy_k4"}
    for stem in result["figures"]:
        for suffix in ("pdf", "png", "svg"):
            assert (output / f"{stem}.{suffix}").is_file()
    with (output / "plot_source_data.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    support = [
        row
        for row in rows
        if row["series_id"] == "recovery:k4:q2:cellular:support_available:mean"
    ]
    assert [(int(row["epoch"]), float(row["value"])) for row in support] == [
        (0, 0.0),
        (1, 4 / 9),
        (2, 2 / 3),
    ]
    assert all(row["dataset"] == "toy" for row in rows)
    assert all(row["normalization"] == "all_reference" for row in rows)
    assert all(len(row["manifest_sha256"]) == 64 for row in rows)
    assert not any("cycle-span" in row["series_id"] for row in rows)


def test_export_failure_leaves_no_partial_publication_directory(
    tmp_path, three_seed_bundle, monkeypatch
):
    destination = tmp_path / "figures"
    original_savefig = plt.Figure.savefig

    def fail_png(self, filename, *args, **kwargs):
        if str(filename).endswith(".png"):
            raise RuntimeError("injected PNG write failure")
        return original_savefig(self, filename, *args, **kwargs)

    with monkeypatch.context() as patcher:
        patcher.setattr(plt.Figure, "savefig", fail_png)
        with pytest.raises(RuntimeError, match="injected PNG"):
            export_recovery_figures(
                three_seed_bundle, destination, publication=False
            )
    assert not destination.exists()
    assert not list(tmp_path.glob(".figures.staging-*"))
    assert set(
        export_recovery_figures(
            three_seed_bundle, destination, publication=False
        )["figures"]
    ) == {"recovery_toy_k4", "entropy_toy_k4"}


def test_publish_does_not_replace_target_created_after_staging(
    tmp_path, three_seed_bundle, monkeypatch
):
    destination = tmp_path / "figures"
    original_publish = plot_module._publish_staged_directory

    def create_competing_target(staged, target):
        target.mkdir()
        return original_publish(staged, target)

    monkeypatch.setattr(
        plot_module, "_publish_staged_directory", create_competing_target
    )
    with pytest.raises(FileExistsError):
        export_recovery_figures(
            three_seed_bundle, destination, publication=False
        )
    assert destination.is_dir()
    assert list(destination.iterdir()) == []
    assert not list(tmp_path.glob(".figures.staging-*"))


def test_staged_csv_is_verified_before_publication(
    tmp_path, three_seed_bundle, monkeypatch
):
    destination = tmp_path / "figures"
    original_validate = plot_module._validate_staged_export

    def corrupt_then_validate(stage, stems, rows, metadata):
        (stage / "plot_source_data.csv").write_text("corrupt\n")
        return original_validate(stage, stems, rows, metadata)

    monkeypatch.setattr(
        plot_module, "_validate_staged_export", corrupt_then_validate
    )
    with pytest.raises(ValueError, match="CSV"):
        export_recovery_figures(
            three_seed_bundle, destination, publication=False
        )
    assert not destination.exists()
    assert not list(tmp_path.glob(".figures.staging-*"))


def test_k32_layout_orders_q_rows_and_domain_columns(layout_bundle):
    fig, artists = make_recovery_figure(layout_bundle, K=32, publication=False)
    try:
        assert len(fig.axes) == 12
        assert [
            fig.axes[index].get_title(loc="center") for index in (0, 3, 6, 9)
        ] == [f"Hyperedges  |  q={q}" for q in (2, 4, 8, 16)]
        assert "Simplicial 2-cells" in fig.axes[1].get_title(loc="center")
        assert "Cellular 2-cells" in fig.axes[2].get_title(loc="center")
        assert list(
            artists[
                (32, 4, "hypergraph", "theory_support", "mean")
            ].get_xdata()
        ) == [0, 1, 2, 3]
        q4 = artists[(32, 4, "hypergraph", "theory_support", "mean")]
        q16 = artists[(32, 16, "hypergraph", "theory_support", "mean")]
        assert list(q16.get_ydata())[1] > list(q4.get_ydata())[1]
        selected = [text.get_text() for ax in fig.axes for text in ax.texts]
        assert "Selected: EDHNN, UniGNN" in selected
        assert "Selected: CWN" in selected
        assert "Selected: SCN" in selected
        assert "Selected: SCCNN" in selected
        labels = [text.get_text() for text in fig.legends[0].get_texts()]
        assert "Analytical support expectation" in labels
        assert "Empirical support availability" in labels
        assert "Exact cycle-basis recovery" in labels
        assert "Empirical recovery" not in labels
        assert "q-observable fraction" in labels
        assert not any("Cycle-span" in label for label in labels)
    finally:
        plt.close(fig)


def test_recovery_figure_matches_observability_visual_language(layout_bundle):
    fig, artists = make_recovery_figure(layout_bundle, K=32, publication=False)
    try:
        assert fig._suptitle.get_text().splitlines()[0] == "Cora Full"
        for family, color in (
            ("hypergraph", "#0072B2"),
            ("simplicial", "#CC79A7"),
            ("cellular", "#E69F00"),
        ):
            support = artists[(32, 4, family, "support_available", "mean")]
            assert support.get_color() == color
            assert support.get_linestyle() == "--"
            assert support.get_markerfacecolor() == "white"
        actual = artists[(32, 4, "cellular", "actual_basis_recovery", "mean")]
        assert actual.get_color() == "#E69F00"
        assert actual.get_linestyle() == "-"
        assert actual.get_markerfacecolor() == "#E69F00"
        ceiling = artists[(32, 4, "cellular", "observable_ceiling", "mean")]
        assert ceiling.get_linestyle() == ":"
        assert fig.axes[0].get_title(loc="center").startswith("Hyperedges")
        fig.canvas.draw()
        assert any(
            "%" in tick.get_text() for tick in fig.axes[0].get_yticklabels()
        )
        labels = [item.get_text() for item in fig.legends[0].get_texts()]
        assert "q-observable fraction" in labels
        assert "Exact cycle-basis recovery" in labels
    finally:
        plt.close(fig)


def test_combined_support_plot_uses_only_matching_domain_support_rows(
    layout_bundle,
):
    fig, lines = plot_module.make_combined_support_figure(
        layout_bundle, K=32, q=4, publication=False
    )
    try:
        assert len(fig.axes) == 1
        assert len(fig.axes[0].lines) == 3
        assert set(lines) == {"hypergraph", "cellular", "simplicial"}
        for family, color in (
            ("hypergraph", "#0072B2"),
            ("cellular", "#E69F00"),
            ("simplicial", "#CC79A7"),
        ):
            expected = [
                row["coverage_mean"]
                for row in layout_bundle["summary"]
                if row["K"] == 32
                and row["q"] == 4
                and row["family"] == family
                and row["measurement"] == "support_available"
            ]
            assert list(lines[family].get_ydata()) == pytest.approx(expected)
            assert lines[family].get_color() == color
            assert lines[family].get_markerfacecolor() == "white"
            assert lines[family].get_linestyle() == "--"
        assert fig._suptitle.get_text() == "Cora Full"
        assert "K=32" in fig.axes[0].get_title()
        assert "q=4" in fig.axes[0].get_title()
        assert "support availability" in fig.axes[0].get_title().lower()
        assert any(
            "diagnostic" in item.get_text().lower() for item in fig.texts
        )
        assert "Exact cycle-basis recovery" not in [
            item.get_text() for item in fig.axes[0].get_legend().get_texts()
        ]
        fig.canvas.draw()
        assert any(
            "%" in tick.get_text() for tick in fig.axes[0].get_yticklabels()
        )
    finally:
        plt.close(fig)


def test_k64_cellular_panel_has_distinct_support_and_basis(layout_bundle):
    fig, artists = make_recovery_figure(layout_bundle, K=64, publication=False)
    try:
        assert len(fig.axes) == 1
        assert "K=64" in fig._suptitle.get_text()
        assert "q=8" in fig.axes[0].get_title(loc="center")
        assert "Selected: Cell TopoTune" in [
            text.get_text() for text in fig.axes[0].texts
        ]
        assert fig.get_figheight() >= 3.4
        assert fig.subplotpars.bottom >= 0.25
        support = artists[(64, 8, "cellular", "support_available", "mean")]
        actual = artists[(64, 8, "cellular", "actual_basis_recovery", "mean")]
        assert support.get_ydata()[1] > actual.get_ydata()[1]
        assert (
            artists[
                (64, 8, "cellular", "observable_ceiling", "mean")
            ].get_ydata()[0]
            == 1.0
        )
    finally:
        plt.close(fig)


def test_entropy_is_analytical_without_seed_band(layout_bundle):
    fig, artists = make_entropy_figure(
        layout_bundle, K=32, epochs=[0, 1, 2, 3, 201], publication=False
    )
    try:
        assert len(fig.axes) == 3
        assert len([key for key in artists if key[-1] == "entropy"]) == 12
        assert all(len(ax.collections) == 0 for ax in fig.axes)
        assert [ax.get_title(loc="left") for ax in fig.axes] == [
            "Neighbourhood hypergraph",
            "Clique simplicial",
            "Cycle-based cellular",
        ]
        assert artists[(32, 2, "cellular", "entropy")].get_xdata()[-1] == 201
        assert "analytical" in fig._suptitle.get_text().lower()
        assert "empirical" not in fig._suptitle.get_text().lower()
    finally:
        plt.close(fig)


def test_entropy_legend_collects_q_from_all_visible_families(layout_bundle):
    mixed = deepcopy(layout_bundle)
    for configuration in mixed["manifest"]["configuration_matrix"]:
        if configuration["K"] == 32 and configuration["q"] == 16:
            configuration["families"].remove("hypergraph")
    for name in ("theory", "observations", "summary"):
        mixed[name] = [
            row
            for row in mixed[name]
            if not (
                row["K"] == 32
                and row["q"] == 16
                and row["family"] == "hypergraph"
            )
        ]
    fig, artists = make_entropy_figure(mixed, K=32, publication=False)
    try:
        assert (32, 16, "hypergraph", "entropy") not in artists
        assert (32, 16, "cellular", "entropy") in artists
        assert "q=16" in [
            text.get_text() for text in fig.legends[0].get_texts()
        ]
    finally:
        plt.close(fig)


def test_hidden_nonlinear_epoch_is_drawn_not_interpolated(layout_bundle):
    nonlinear = deepcopy(layout_bundle)
    for row in nonlinear["observations"]:
        if (
            row["K"],
            row["q"],
            row["family"],
            row["measurement"],
            row["epoch"],
        ) == (32, 2, "hypergraph", "support_available", 1):
            row["recovered_count"] = 0
            row["coverage"] = 0.0
    for row in nonlinear["summary"]:
        if (
            row["K"],
            row["q"],
            row["family"],
            row["measurement"],
            row["epoch"],
        ) == (32, 2, "hypergraph", "support_available", 1):
            row["coverage_mean"] = 0.0
            row["coverage_sample_sd"] = 0.0
    fig, artists = make_recovery_figure(nonlinear, K=32, publication=False)
    try:
        line = artists[(32, 2, "hypergraph", "support_available", "mean")]
        assert list(line.get_xdata()) == [0, 1, 2, 3]
        assert list(line.get_ydata()) == [0.0, 0.0, 2 / 3, 2 / 3]
        assert list(line.get_markevery()) == [0, 2, 3]
        intervals = artists[
            (32, 2, "hypergraph", "support_available", "sample_sd_interval")
        ]
        assert [segment[0, 0] for segment in intervals.get_segments()] == [
            0,
            2,
            3,
        ]
        plotted_rows = [
            row
            for row in fig._recovery_source_rows
            if row["series_id"]
            == "recovery:k32:q2:hypergraph:support_available:mean"
        ]
        assert [row["epoch"] for row in plotted_rows] == [0, 1, 2, 3]
        assert [row["value"] for row in plotted_rows] == [
            0.0,
            0.0,
            2 / 3,
            2 / 3,
        ]
    finally:
        plt.close(fig)


def test_all_exported_series_match_artists_and_checked_rows(
    tmp_path, layout_bundle
):
    figures = {}
    for K in (32, 64):
        fig, artists = make_recovery_figure(
            layout_bundle, K=K, publication=False
        )
        figures[("recovery", K)] = (fig, artists)
    entropy_fig, entropy_artists = make_entropy_figure(
        layout_bundle, K=32, publication=False
    )
    figures[("entropy", 32)] = (entropy_fig, entropy_artists)
    try:
        output = tmp_path / "all-panels"
        export_recovery_figures(layout_bundle, output, publication=False)
        metadata = json.loads((output / "figure_metadata.json").read_text())
        expected_manifest_digest = hashlib.sha256(
            json.dumps(
                layout_bundle["manifest"],
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
            ).encode("utf-8")
        ).hexdigest()
        assert metadata["manifest_sha256"] == expected_manifest_digest
        assert metadata["figure_configurations"][
            "recovery_corafull_topotune_k64_q8"
        ] == {
            "dataset": "cora_full",
            "K": 64,
            "q_values": [8],
            "families": ["cellular"],
        }
        with (output / "plot_source_data.csv").open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        assert metadata["figures"] == [
            "recovery_corafull_k32",
            "entropy_corafull_k32",
            "recovery_corafull_topotune_k64_q8",
        ]
        assert {path.name for path in output.iterdir()} == {
            *(
                f"{stem}.{suffix}"
                for stem in metadata["figures"]
                for suffix in ("pdf", "png", "svg")
            ),
            "plot_source_data.csv",
            "figure_metadata.json",
        }
        for (kind, K), (_fig, artists) in figures.items():
            matching = [
                row
                for row in rows
                if row["figure"] == kind and int(row["K"]) == K
            ]
            assert matching
            for key, artist in artists.items():
                if kind == "entropy":
                    artist_K, q, family, suffix = key
                    measure = (
                        "analytical_entropy" if suffix == "entropy" else suffix
                    )
                    stat = "nats_per_reference"
                else:
                    artist_K, q, family, measure, stat = key
                if stat == "sample_sd_interval":
                    for side, endpoint in (
                        ("sample_sd_low", 0),
                        ("sample_sd_high", 1),
                    ):
                        source = [
                            row
                            for row in matching
                            if (
                                int(row["q"]),
                                row["family"],
                                row["measurement"],
                                row["stat"],
                            )
                            == (q, family, measure, side)
                        ]
                        assert [
                            float(row["value"]) for row in source
                        ] == pytest.approx(
                            [
                                segment[endpoint, 1]
                                for segment in artist.get_segments()
                            ]
                        )
                    continue
                source = [
                    row
                    for row in matching
                    if (
                        int(row["q"]),
                        row["family"],
                        row["measurement"],
                        row["stat"],
                    )
                    == (q, family, measure, stat)
                ]
                assert [int(row["epoch"]) for row in source] == list(
                    artist.get_xdata()
                )
                assert [
                    float(row["value"]) for row in source
                ] == pytest.approx(list(artist.get_ydata()))
                if kind == "recovery" and measure in {
                    "support_available",
                    "actual_basis_recovery",
                }:
                    for row in source:
                        checked = next(
                            item
                            for item in layout_bundle["summary"]
                            if (
                                item["K"],
                                item["q"],
                                item["family"],
                                item["measurement"],
                                item["epoch"],
                            )
                            == (
                                artist_K,
                                q,
                                family,
                                measure,
                                int(row["epoch"]),
                            )
                        )
                        assert float(row["value"]) == pytest.approx(
                            checked["coverage_mean"]
                        )
        assert all(row["normalization"] == "all_reference" for row in rows)
        assert all(row["dataset"] == "cora_full" for row in rows)
        assert {row["manifest_sha256"] for row in rows} == {
            expected_manifest_digest
        }
    finally:
        for fig, _artists in figures.values():
            plt.close(fig)
