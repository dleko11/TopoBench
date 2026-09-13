"""Tests for staged full-graph structure counting."""

import argparse

import networkx as nx
import pytest
import torch

from scripts.partitioning import count_clique_simplices
from scripts.partitioning.count_clique_simplices import (
    _parse_count_types,
    _upsert_dataset_row,
    count_filtered_cycle_basis,
    count_graph_structures,
    count_triangles_from_edge_index,
)


def _toy_edge_index() -> torch.Tensor:
    return torch.tensor(
        [
            [0, 1, 2, 2, 3, 4, 5],
            [1, 2, 0, 3, 4, 5, 2],
        ]
    )


def test_selective_simplex_count_checkpoints_completed_stages():
    checkpoints = []

    result = count_graph_structures(
        _toy_edge_index(),
        num_nodes=6,
        count_types=("simplices",),
        checkpoint=lambda stage, _result: checkpoints.append(stage),
    )

    assert result["num_edges"] == 7
    assert result["num_2_simplices"] == 1
    assert result["num_2_cells"] is None
    assert checkpoints == ["graph", "simplices"]


def test_selective_cell_count_checkpoints_completed_stages():
    checkpoints = []

    result = count_graph_structures(
        _toy_edge_index(),
        num_nodes=6,
        count_types=("cells",),
        checkpoint=lambda stage, _result: checkpoints.append(stage),
    )

    assert result["num_edges"] == 7
    assert result["num_2_simplices"] is None
    assert result["num_2_cells"] == 2
    assert checkpoints == ["graph", "cells"]


@pytest.mark.parametrize("max_cycle_length", [2, 3, 4, 9])
def test_streamed_cycle_count_matches_networkx(max_cycle_length):
    graph = nx.Graph()
    graph.add_nodes_from(range(9))
    graph.add_edges_from(
        [
            (0, 1),
            (1, 2),
            (2, 0),
            (2, 3),
            (3, 4),
            (4, 5),
            (5, 2),
            (6, 7),
            (7, 8),
            (8, 6),
            (8, 8),
        ]
    )
    expected = sum(
        1
        for cycle in nx.cycle_basis(graph)
        if 1 < len(cycle) <= max_cycle_length
    )

    assert count_filtered_cycle_basis(graph, max_cycle_length) == expected


def test_direct_edge_index_triangle_count_simplifies_graph():
    edge_index = torch.tensor(
        [
            [0, 1, 2, 1, 0, 3, 3],
            [1, 2, 0, 0, 1, 3, 4],
        ]
    )

    assert count_triangles_from_edge_index(edge_index, num_nodes=5) == 1


def test_graph_only_count_skips_networkx_construction(monkeypatch):
    edge_index = torch.tensor(
        [
            [0, 1, 0, 1, 2, 2],
            [1, 0, 1, 0, 2, 2],
        ]
    )
    checkpoints = []
    monkeypatch.setattr(
        count_clique_simplices,
        "build_simple_undirected_graph",
        lambda *_args: pytest.fail("NetworkX graph should not be constructed"),
    )

    result = count_graph_structures(
        edge_index,
        num_nodes=3,
        count_types=("graph",),
        checkpoint=lambda stage, _result: checkpoints.append(stage),
    )

    assert result["num_nodes"] == 3
    assert result["num_edges"] == 2
    assert result["num_hyperedges"] == 3
    assert result["num_2_simplices"] is None
    assert result["num_2_cells"] is None
    assert checkpoints == ["graph"]


def test_count_type_parser_is_ordered_and_rejects_unknown_values():
    assert _parse_count_types("cells,graph,simplices") == (
        "graph",
        "simplices",
        "cells",
    )
    with pytest.raises(
        argparse.ArgumentTypeError, match="Unknown count types"
    ):
        _parse_count_types("edges")


def test_upsert_preserves_results_from_a_separate_count_process():
    rows = [{"dataset": "ogbn_products", "num_2_simplices": "123"}]

    _upsert_dataset_row(
        rows,
        {
            "dataset": "ogbn_products",
            "num_2_simplices": None,
            "num_2_cells": 45,
        },
    )

    assert rows == [
        {
            "dataset": "ogbn_products",
            "num_2_simplices": "123",
            "num_2_cells": 45,
        }
    ]
