"""Tests for exact structural observability counters."""

from collections import Counter
from types import SimpleNamespace

import networkx as nx
import numpy as np

from scripts.partitioning.plot_structural_observability import (
    download_histograms,
    plot_observability,
)
from scripts.partitioning.structural_observability import (
    cell_span_histogram,
    hypergraph_span_histogram,
    observability_rows,
    partition_labels_from_partition,
    simplicial_span_histogram,
)


def test_partition_labels_are_mapped_to_original_node_ids():
    partptr = np.asarray([0, 2, 5])
    node_perm = np.asarray([3, 0, 4, 2, 1])

    labels = partition_labels_from_partition(partptr, node_perm)

    assert labels.tolist() == [0, 1, 1, 0, 1]


def test_hypergraph_span_histogram_counts_unique_neighborhoods():
    graph = nx.path_graph(3)
    labels = np.asarray([0, 0, 1])

    histogram = hypergraph_span_histogram(graph, labels)

    assert histogram == Counter({2: 2, 1: 1})


def test_cell_span_histogram_uses_filtered_cycle_basis():
    graph = nx.Graph()
    graph.add_edges_from(
        [
            (0, 1),
            (1, 2),
            (2, 0),
            (3, 4),
            (4, 5),
            (5, 3),
            (6, 7),
            (7, 8),
            (8, 9),
            (9, 6),
        ]
    )
    labels = np.asarray([0, 0, 0, 0, 0, 1, 0, 1, 2, 3])

    histogram = cell_span_histogram(graph, labels, max_cell_length=3)

    assert histogram == Counter({1: 1, 2: 1})


def test_simplicial_histogram_separates_all_triangle_spans():
    graph = nx.Graph()
    graph.add_edges_from(
        [
            (0, 1),
            (1, 2),
            (2, 0),
            (3, 4),
            (4, 5),
            (5, 3),
            (6, 7),
            (7, 8),
            (8, 6),
        ]
    )
    labels = np.asarray([0, 0, 0, 0, 0, 1, 0, 1, 2])

    histogram = simplicial_span_histogram(graph, labels)

    assert histogram == Counter({1: 1, 2: 1, 3: 1})


def test_observability_rows_are_cumulative_and_report_thresholds():
    histogram = [
        {
            "dataset": "toy",
            "family": "cell",
            "definition": "toy cells",
            "num_parts": 4,
            "span": 1,
            "count": 6,
            "structure_count": 10,
        },
        {
            "dataset": "toy",
            "family": "cell",
            "definition": "toy cells",
            "num_parts": 4,
            "span": 2,
            "count": 3,
            "structure_count": 10,
        },
        {
            "dataset": "toy",
            "family": "cell",
            "definition": "toy cells",
            "num_parts": 4,
            "span": 4,
            "count": 1,
            "structure_count": 10,
        },
    ]

    curves, threshold = observability_rows(histogram)

    assert [row["observable_fraction"] for row in curves] == [
        0.6,
        0.9,
        0.9,
        1.0,
    ]
    assert threshold["q_95"] == 4
    assert threshold["q_99"] == 4


def test_plot_observability_exports_all_formats(tmp_path):
    histogram = [
        {
            "dataset": "cora_full",
            "family": "cell",
            "definition": "toy cells",
            "num_parts": 4,
            "span": 1,
            "count": 9,
            "structure_count": 10,
        },
        {
            "dataset": "cora_full",
            "family": "cell",
            "definition": "toy cells",
            "num_parts": 4,
            "span": 2,
            "count": 1,
            "structure_count": 10,
        },
    ]
    curves, threshold = observability_rows(histogram)
    output_stem = tmp_path / "structural_observability"

    plot_observability(curves, [threshold], output_stem)

    assert output_stem.with_suffix(".pdf").exists()
    assert output_stem.with_suffix(".svg").exists()
    assert output_stem.with_suffix(".png").exists()


def test_download_histograms_uses_latest_artifact(tmp_path, mocker):
    api = mocker.Mock()
    api.runs.return_value = [
        SimpleNamespace(
            job_type="structure_span_count",
            config={
                "dataset": "cora_full",
                "family": "cell",
                "num_parts": 32,
            },
        )
    ]
    artifact = mocker.Mock()
    api.artifact.return_value = artifact
    mocker.patch(
        "scripts.partitioning.plot_structural_observability.wandb.Api",
        return_value=api,
    )

    count = download_histograms(
        entity="entity",
        project="project",
        output_dir=tmp_path,
    )

    assert count == 1
    api.artifact.assert_called_once_with(
        "entity/project/structural-observability-cora_full-cell-k32:latest",
        type="structural-observability",
    )
    artifact.download.assert_called_once_with(
        root=str(tmp_path / "cora_full_k32")
    )
