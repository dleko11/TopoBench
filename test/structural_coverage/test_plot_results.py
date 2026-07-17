"""Tests for structural-coverage plotting summaries."""

from __future__ import annotations

import pytest

from scripts.structural_coverage.plot_appendix_design_preview import (
    theory_from_histogram,
)
from scripts.structural_coverage.plot_results import (
    entropy_density_stats,
    threshold_epoch,
)
from scripts.structural_coverage.plot_sweep_design_preview import (
    ProfileView,
    observable_ceiling,
)


def _run(
    empirical_values: list[float],
    theory_values: list[float],
) -> dict:
    return {
        "empirical": [
            {
                "epoch": str(epoch),
                "total_count_rank1_2": "10",
                "realized_coverage_rank1": str(value),
            }
            for epoch, value in enumerate(empirical_values)
        ],
        "theory": [
            {
                "epoch": str(epoch),
                "entropy_nats_rank1_2": str(entropy),
                "expected_coverage_rank1": str(value),
            }
            for epoch, (entropy, value) in enumerate(
                zip(theory_values, empirical_values, strict=True)
            )
        ],
    }


def test_entropy_density_uses_global_structure_denominator():
    run = _run([0.0, 0.5, 0.9], [0.0, 5.0, 2.0])

    epochs, means, stds, counts = entropy_density_stats([run])

    assert epochs == [1, 2]
    assert means == pytest.approx([0.5, 0.2])
    assert stds == [0.0, 0.0]
    assert counts == [1, 1]


def test_threshold_epoch_uses_the_across_run_mean_curve():
    first = _run([0.0, 0.90, 0.96], [0.0, 1.0, 0.5])
    second = _run([0.0, 0.92, 0.98], [0.0, 1.0, 0.5])

    assert (
        threshold_epoch(
            [first, second],
            "empirical",
            "realized_coverage_rank1",
        )
        == 2
    )


def test_observable_ceiling_uses_rank_specific_span_counts():
    run = {
        "spans": [
            {"rank": "1", "span": "1", "count": "3"},
            {"rank": "1", "span": "2", "count": "1"},
            {"rank": "2", "span": "1", "count": "99"},
        ]
    }

    assert observable_ceiling(run, rank=1, q_values=(1, 2, 4)) == [
        0.75,
        1.0,
        1.0,
    ]


def test_appendix_preview_theory_uses_target_rank_histogram():
    profile = ProfileView(
        key="hypergraph",
        heading="Hypergraph",
        structure_label="rank-1 hyperedges",
        legend_label="Hyperedges",
        group="rank1",
        runs=[
            {
                "metadata": {"K_eff": 4},
                "spans": [
                    {"rank": "1", "span": "1", "count": "1"},
                    {"rank": "1", "span": "2", "count": "1"},
                    {"rank": "2", "span": "1", "count": "100"},
                ],
            }
        ],
    )

    epochs, coverage, entropy = theory_from_histogram(
        profile,
        q=1,
        max_epoch=2,
    )

    assert epochs == [1, 2]
    assert coverage == [0.5, 0.5]
    assert entropy == [0.0, 0.0]
