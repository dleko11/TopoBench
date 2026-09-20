"""Server-entry tests against small explicit structural references."""

import importlib
import importlib.util
import networkx as nx
import numpy as np
import pytest
from scipy import sparse

MODULE = "scripts.structural_coverage.run_reddit_support_recovery"


def runner():
    assert importlib.util.find_spec(MODULE) is not None, "Reddit server entry point is missing"
    return importlib.import_module(MODULE)


def graph():
    g = nx.Graph()
    g.add_nodes_from(range(8))
    g.add_edges_from([(0, 1), (1, 2), (2, 0), (2, 3), (3, 4),
                      (4, 2), (4, 5), (5, 6), (6, 7), (7, 4)])
    return g


def test_streamed_signatures_preserve_reference_counts_and_memberships():
    from scripts.structural_coverage.recovery_core import (
        neighbourhood_references, cycle_basis_references, triangle_references,
    )
    from scripts.structural_coverage.support_recovery_multidataset import build_signature_groups
    api = runner()
    labels = np.arange(8) // 2
    g = graph()
    refs = {"hypergraph": neighbourhood_references(g),
            "cellular": cycle_basis_references(g),
            "simplicial": triangle_references(g)}
    for family, items in refs.items():
        actual, histogram = api.prepare_signatures(g, labels, family)
        expected, wanted_histogram = build_signature_groups(items, labels, q=4)
        assert histogram == wanted_histogram
        for span in expected:
            as_dict = lambda pair: {tuple(row): int(weight) for row, weight in zip(*pair)}
            assert as_dict(actual[span]) == as_dict(expected[span])
    # In K3 every closed neighbourhood is identical, but all three centres count.
    _, histogram = api.prepare_signatures(nx.complete_graph(3), np.array([0, 1, 1]), "hypergraph")
    assert sum(histogram.values()) == 3


def test_counts_match_existing_empirical_analyzer():
    from scripts.structural_coverage.recovery_core import triangle_references
    from scripts.structural_coverage.support_recovery_multidataset import analyze_q_sweep
    api = runner()
    labels = np.arange(8) // 2
    groups, histogram = api.prepare_signatures(graph(), labels, "simplicial")
    expected = analyze_q_sweep({"simplicial": triangle_references(graph())},
                              labels=labels, K=4, q_values=[1, 2, 4], seeds=[0, 1], epochs=5)
    for q in [1, 2, 4]:
        got = api.measure(groups, histogram, K=4, q=q, epochs=5, seeds=[0, 1])
        want = expected[q]["families"]["simplicial"]
        assert got["counts_by_seed"] == want["counts_by_seed"]
        assert got["reference_count"] == want["reference_count"]
        assert got["coverage_mean"] == want["coverage_mean"]
        assert got["coverage_sample_sd"] == want["coverage_sample_sd"]


def test_adjacency_is_topology_only_and_validated(tmp_path):
    api = runner()
    path = tmp_path / "reddit_graph.npz"
    sparse.save_npz(path, sparse.csr_matrix(nx.to_scipy_sparse_array(graph())))
    adj = api.load_adjacency(path, expected_nodes=8, expected_edges=10)
    assert adj.shape == (8, 8)
    assert len(list(tmp_path.iterdir())) == 1
    with pytest.raises(ValueError, match="node count"):
        api.load_adjacency(path, expected_nodes=9, expected_edges=10)
    with pytest.raises(ValueError, match="edge count"):
        api.load_adjacency(path, expected_nodes=8, expected_edges=12)


def test_topology_input_removes_self_loops_and_rejects_asymmetry(tmp_path):
    api = runner()
    adjacency = sparse.csr_matrix(nx.to_scipy_sparse_array(graph()))
    adjacency.setdiag(1)
    path = tmp_path / "graph.npz"
    sparse.save_npz(path, adjacency)
    assert api.load_adjacency(path, expected_nodes=8, expected_edges=10).nnz == 20
    adjacency[0, 1] = 0
    sparse.save_npz(path, adjacency)
    with pytest.raises(ValueError, match="symmetric"):
        api.load_adjacency(path, expected_nodes=8, expected_edges=10)


def test_featureless_metis_covers_all_nodes():
    adjacency = sparse.csr_matrix(nx.to_scipy_sparse_array(graph()))
    labels = runner().make_partition(adjacency, K=2)
    assert labels.shape == (8,)
    assert set(labels) == {0, 1}


@pytest.mark.parametrize("labels", [[0, 0, 0], [0, 1, 2], [-1, 0, 1], [0.5, 1, 0]])
def test_reject_invalid_partition(labels):
    with pytest.raises(ValueError):
        runner().validate_labels(np.array(labels), nodes=3, K=2)


def test_checkpoint_rejects_other_graph_or_partition(tmp_path):
    api = runner()
    manifest = {"graph_sha256": "a", "partition_sha256": "b", "K": 4}
    api.check_manifest(tmp_path, manifest, resume=False)
    api.check_manifest(tmp_path, manifest, resume=True)
    with pytest.raises(ValueError, match="manifest"):
        api.check_manifest(tmp_path, dict(manifest, partition_sha256="c"), resume=True)


def test_cli_smoke_and_resume(tmp_path):
    api = runner()
    output = tmp_path / "smoke"
    args = ["--smoke", "--output-dir", str(output)]
    api.main(args)
    assert (output / "q_recovery.pdf").is_file()
    assert (output / "q_recovery_source_data.csv").is_file()
    before = (output / "q_recovery_source_data.csv").read_bytes()
    api.main(args + ["--resume"])
    assert (output / "q_recovery_source_data.csv").read_bytes() == before
    with pytest.raises(ValueError, match="exists"):
        api.main(args)
