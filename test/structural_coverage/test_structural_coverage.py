"""Tests for structural coverage experiment helpers."""

from __future__ import annotations

import math
from collections import Counter
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from scripts.structural_coverage.coverage import (
    build_theory_curve_rows,
    cell_cycles_from_edges,
    compute_mean_pair_cooccurrence,
    compute_structure_spans,
    empirical_coverage_row,
    expected_pair_cooccurrence,
    extract_cell_simple_cycle_structures_from_batch,
    extract_cell_structures_from_batch,
    extract_hypergraph_structures_from_batch,
    extract_simplicial_structures_from_edge_index,
    induced_edge_mismatch,
    per_epoch_probability,
    simple_cycle_cells_from_edges,
)

SQUARE_WITH_DIAGONAL_EDGES = {
    ("e", 0, 1),
    ("e", 1, 2),
    ("e", 2, 3),
    ("e", 0, 3),
    ("e", 0, 2),
}
TRIANGLE_012_CELL = ("c", 0, 1, 0, 2, 1, 2)
TRIANGLE_023_CELL = ("c", 0, 2, 0, 3, 2, 3)
SQUARE_0123_CELL = ("c", 0, 1, 0, 3, 1, 2, 2, 3)


def test_extract_simplicial_structures_triangle_with_duplicate_edges():
    edge_index = torch.tensor(
        [
            [0, 1, 1, 2, 0, 2],
            [1, 0, 2, 1, 2, 0],
        ],
        dtype=torch.long,
    )
    global_nid = torch.tensor([10, 11, 12], dtype=torch.long)

    structures = extract_simplicial_structures_from_edge_index(
        edge_index, global_nid
    )

    assert structures[0] == {("v", 10), ("v", 11), ("v", 12)}
    assert structures[1] == {
        ("e", 10, 11),
        ("e", 10, 12),
        ("e", 11, 12),
    }
    assert structures[2] == {("t", 10, 11, 12)}


def test_extract_cell_structures_from_incidences():
    incidence_1 = torch.sparse_coo_tensor(
        torch.tensor(
            [
                [0, 1, 1, 2, 0, 2],
                [0, 0, 1, 1, 2, 2],
            ],
            dtype=torch.long,
        ),
        torch.ones(6),
        size=(3, 3),
    ).coalesce()
    incidence_2 = torch.sparse_coo_tensor(
        torch.tensor([[0, 1, 2], [0, 0, 0]], dtype=torch.long),
        torch.ones(3),
        size=(3, 1),
    ).coalesce()
    batch = SimpleNamespace(
        global_nid=torch.tensor([10, 11, 12], dtype=torch.long),
        incidence_1=incidence_1,
        incidence_2=incidence_2,
    )

    structures = extract_cell_structures_from_batch(batch)

    assert structures[0] == {("v", 10), ("v", 11), ("v", 12)}
    assert structures[1] == {
        ("e", 10, 11),
        ("e", 10, 12),
        ("e", 11, 12),
    }
    assert structures[2] == {("c", 10, 11, 10, 12, 11, 12)}


def test_simple_cycle_cells_respect_support_node_cap():
    active_nodes = np.array([0, 1, 2, 3], dtype=np.int64)

    cells_n3 = simple_cycle_cells_from_edges(
        active_nodes=active_nodes,
        edges=SQUARE_WITH_DIAGONAL_EDGES,
        max_support_nodes=3,
    )
    cells_n4 = simple_cycle_cells_from_edges(
        active_nodes=active_nodes,
        edges=SQUARE_WITH_DIAGONAL_EDGES,
        max_support_nodes=4,
    )

    assert cells_n3 == {TRIANGLE_012_CELL, TRIANGLE_023_CELL}
    assert cells_n4 == {
        TRIANGLE_012_CELL,
        TRIANGLE_023_CELL,
        SQUARE_0123_CELL,
    }


def test_simple_cycle_cells_include_generated_non_basis_cycles():
    active_nodes = np.array([0, 1, 2, 3], dtype=np.int64)

    basis_cells = cell_cycles_from_edges(
        active_nodes=active_nodes,
        edges=SQUARE_WITH_DIAGONAL_EDGES,
    )
    generated_cells = simple_cycle_cells_from_edges(
        active_nodes=active_nodes,
        edges=SQUARE_WITH_DIAGONAL_EDGES,
        max_support_nodes=4,
    )

    assert len(basis_cells) == 2
    assert SQUARE_0123_CELL not in basis_cells
    assert SQUARE_0123_CELL in generated_cells


def test_extract_cell_simple_cycles_from_batch_uses_induced_edges():
    edge_index = torch.tensor(
        [
            [0, 1, 2, 3, 0],
            [1, 2, 3, 0, 2],
        ],
        dtype=torch.long,
    )
    batch = SimpleNamespace(
        global_nid=torch.tensor([10, 11, 12, 13], dtype=torch.long),
        edge_index=edge_index,
    )

    structures = extract_cell_simple_cycle_structures_from_batch(
        batch,
        max_support_nodes=4,
    )

    assert structures[2] == {
        ("c", 10, 11, 10, 12, 11, 12),
        ("c", 10, 12, 10, 13, 12, 13),
        ("c", 10, 11, 10, 13, 11, 12, 12, 13),
    }


def test_extract_hypergraph_structures_from_incidence():
    incidence_hyperedges = torch.sparse_coo_tensor(
        torch.tensor([[0, 2, 1, 2], [0, 0, 1, 1]], dtype=torch.long),
        torch.ones(4),
        size=(3, 2),
    ).coalesce()
    batch = SimpleNamespace(
        global_nid=torch.tensor([5, 6, 7], dtype=torch.long),
        incidence_hyperedges=incidence_hyperedges,
    )

    structures = extract_hypergraph_structures_from_batch(batch)

    assert structures[0] == {("v", 5), ("v", 6), ("v", 7)}
    assert structures[1] == {("h", 5, 7), ("h", 6, 7)}
    assert structures[2] == set()


def test_compute_structure_spans_from_partptr():
    structures = {
        0: {("v", 0), ("v", 3)},
        1: {("e", 0, 1), ("e", 1, 3), ("h", 0, 3, 5)},
        2: {
            ("t", 0, 2, 4),
            ("c", 0, 1, 1, 3, 3, 4, 0, 4),
            SQUARE_0123_CELL,
        },
    }
    partptr = np.array([0, 2, 4, 6], dtype=np.int64)

    spans = compute_structure_spans(structures, partptr)

    assert spans[("v", 0)] == 1
    assert spans[("v", 3)] == 1
    assert spans[("e", 0, 1)] == 1
    assert spans[("e", 1, 3)] == 2
    assert spans[("h", 0, 3, 5)] == 3
    assert spans[("t", 0, 2, 4)] == 3
    assert spans[("c", 0, 1, 1, 3, 3, 4, 0, 4)] == 3
    assert spans[SQUARE_0123_CELL] == 2


def test_per_epoch_probability_small_analytic_cases():
    assert per_epoch_probability(span=1, q=1, k_eff=6) == 1.0
    assert per_epoch_probability(span=2, q=3, k_eff=6) == pytest.approx(2 / 5)
    assert per_epoch_probability(span=3, q=3, k_eff=6) == pytest.approx(1 / 10)
    assert per_epoch_probability(span=4, q=3, k_eff=6) == 0.0
    assert per_epoch_probability(span=2, q=2, k_eff=31) == pytest.approx(
        1 / 31
    )


def test_theory_curve_epoch_zero_and_entropy_units():
    structures = {
        0: {("v", 0)},
        1: {("e", 0, 2)},
        2: {("t", 0, 2, 4)},
    }
    spans = {
        ("v", 0): 1,
        ("e", 0, 2): 2,
        ("t", 0, 2, 4): 3,
    }

    rows = build_theory_curve_rows(
        structures=structures,
        spans=spans,
        q=3,
        k_eff=6,
        max_epochs=1,
    )

    assert rows[0]["epoch"] == 0
    assert rows[0]["expected_coverage_all"] == 0.0
    assert rows[0]["entropy_bits_all"] == 0.0
    assert rows[1]["entropy_nats_all"] == pytest.approx(
        rows[1]["entropy_bits_all"] * math.log(2.0)
    )


def test_grouping_probability_matches_monte_carlo():
    rng = np.random.default_rng(123)
    k = 6
    q = 3
    trials = 20000
    support_2 = {0, 1}
    support_3 = {0, 1, 2}
    hits_2 = 0
    hits_3 = 0

    for _ in range(trials):
        perm = rng.permutation(k)
        groups = [set(perm[i : i + q].tolist()) for i in range(0, k, q)]
        hits_2 += any(support_2.issubset(group) for group in groups)
        hits_3 += any(support_3.issubset(group) for group in groups)

    assert hits_2 / trials == pytest.approx(
        per_epoch_probability(2, q, k), abs=0.02
    )
    assert hits_3 / trials == pytest.approx(
        per_epoch_probability(3, q, k), abs=0.02
    )


def test_q_observable_ceiling_for_triangle_spanning_three_parts():
    structures = {0: set(), 1: set(), 2: {("t", 0, 2, 4)}}
    spans = {("t", 0, 2, 4): 3}

    q2 = build_theory_curve_rows(
        structures=structures,
        spans=spans,
        q=2,
        k_eff=3,
        max_epochs=10,
    )
    q3 = build_theory_curve_rows(
        structures=structures,
        spans=spans,
        q=3,
        k_eff=3,
        max_epochs=10,
    )

    assert q2[-1]["observable_ceiling_rank2"] == 0.0
    assert q2[-1]["expected_coverage_rank2"] == 0.0
    assert q3[-1]["observable_ceiling_rank2"] == 1.0
    assert q3[-1]["expected_coverage_rank2"] == 1.0


def test_q_observable_ceiling_for_generated_cell():
    structures = {0: set(), 1: set(), 2: {SQUARE_0123_CELL}}
    spans = {SQUARE_0123_CELL: 4}

    q3 = build_theory_curve_rows(
        structures=structures,
        spans=spans,
        q=3,
        k_eff=4,
        max_epochs=10,
    )
    q4 = build_theory_curve_rows(
        structures=structures,
        spans=spans,
        q=4,
        k_eff=4,
        max_epochs=10,
    )

    assert q3[-1]["observable_ceiling_rank2"] == 0.0
    assert q3[-1]["expected_coverage_rank2"] == 0.0
    assert q4[-1]["observable_ceiling_rank2"] == 1.0
    assert q4[-1]["expected_coverage_rank2"] == 1.0


def test_induced_edge_audit_detects_missing_cross_edge():
    full_edges = {("e", 0, 1), ("e", 1, 2)}
    global_nid = torch.tensor([0, 1, 2], dtype=torch.long)
    missing_edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    full_edge_index = torch.tensor(
        [[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long
    )

    mismatch = induced_edge_mismatch(
        full_edges=full_edges,
        batch_edge_index=missing_edge_index,
        batch_global_nid=global_nid,
    )
    assert mismatch is not None
    assert mismatch["expected_num_edges"] == 2
    assert mismatch["observed_num_edges"] == 1

    assert (
        induced_edge_mismatch(
            full_edges=full_edges,
            batch_edge_index=full_edge_index,
            batch_global_nid=global_nid,
        )
        is None
    )


def test_empirical_row_includes_rank_1_2_aggregate():
    global_structures = {
        0: {("v", 0)},
        1: {("e", 0, 1)},
        2: {("t", 0, 1, 2)},
    }
    observed = {0: {("v", 0)}, 1: {("e", 0, 1)}, 2: set()}

    row = empirical_coverage_row(
        epoch=1,
        global_structures=global_structures,
        observed=observed,
    )

    assert row["total_count_rank1_2"] == 2
    assert row["observed_count_rank1_2"] == 1
    assert row["realized_coverage_rank1_2"] == 0.5


def test_pair_cooccurrence_audit_statistic():
    part_ids = np.array([0, 1, 2, 3], dtype=np.int64)
    counts = Counter({(0, 1): 2, (2, 3): 2})

    assert compute_mean_pair_cooccurrence(counts, part_ids, 2) == pytest.approx(
        2 / 6
    )
    assert expected_pair_cooccurrence(q=2, k_eff=4) == pytest.approx(1 / 3)
