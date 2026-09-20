"""Parity checks for the structure-only adapter and native TopoBench path."""

import hashlib
from collections import defaultdict

import numpy as np
import pytest
import torch

from scripts.structural_coverage.recovery_core import (
    ReferenceStructure,
    local_cycle_basis_ids,
    neighbourhood_references,
    triangle_references,
)
from scripts.structural_coverage.recovery_io import load_recovery_config
from scripts.structural_coverage.run_recovery_diagnostic import (
    RecoveryGraphAdapter,
    audit_native_full_batch_order,
    local_basis_ids_in_global_coordinates,
    parse_manuscript_model_annotations,
    preflight_cora_full_run,
    preflight_model_annotations,
    preflight_partition,
    preflight_provenance,
    reference_fingerprints,
    verify_reference_snapshot,
)
from topobench.dataloader.dataload_cluster import (
    BlockCSRBatchCollator,
    _HandleAdapter,
)
from topobench.transforms.liftings.graph2cell.cycle_lifting import (
    CellCycleLiftingSelective,
)
from topobench.transforms.liftings.graph2hypergraph.khop_lifting_large_scale import (
    HypergraphKHopLiftingLargeScale,
)
from topobench.transforms.liftings.graph2simplicial.clique_lifting_fast import (
    SimplicialCliqueLiftingFast,
)


def make_partitioned_graph(num_nodes, edges, clusters, global_ids=None):
    """Make the exact CSR array conventions read by BlockCSRBatchCollator."""
    global_ids = tuple(range(num_nodes)) if global_ids is None else global_ids
    perm = tuple(node for cluster in clusters for node in cluster)
    assert sorted(perm) == list(range(num_nodes))
    inverse = {node: pos for pos, node in enumerate(perm)}
    adjacency = defaultdict(list)
    for source, target in edges:
        adjacency[inverse[source]].append(inverse[target])
    indptr = [0]
    indices = []
    for node in range(num_nodes):
        indices.extend(adjacency[node])
        indptr.append(len(indices))
    partptr = [0]
    for cluster in clusters:
        partptr.append(partptr[-1] + len(cluster))
    arrays = (
        np.asarray(partptr, dtype=np.int64),
        np.asarray(indptr, dtype=np.int64),
        np.asarray(indices, dtype=np.int64),
        np.asarray([global_ids[node] for node in perm], dtype=np.int64),
    )
    return RecoveryGraphAdapter(*arrays), arrays


def native_batch(tmp_path, arrays, parts):
    memmap = tmp_path / "perm_memmap"
    memmap.mkdir()
    for name, array in zip(
        ("partptr", "indptr", "indices", "perm_to_global"),
        arrays,
        strict=True,
    ):
        np.save(memmap / f"{name}.npy", array)
    for split in ("train", "val", "test"):
        np.save(
            memmap / f"{split}_mask_perm.npy", np.ones(len(arrays[3]), bool)
        )
    handle = _HandleAdapter(
        {
            "processed_dir": str(tmp_path),
            "num_parts": len(arrays[0]) - 1,
            "sparse_format": "csr",
        }
    )
    return BlockCSRBatchCollator(handle)(parts)


def undirected_global_edges(batch):
    return {
        tuple(sorted((batch.global_nodes[int(u)], batch.global_nodes[int(v)])))
        for u, v in batch.edge_index.T.tolist()
        if u != v
    }


def incidence_edges(incidence_1):
    matrix = incidence_1.to_dense()
    return [
        tuple(torch.where(matrix[:, col] != 0)[0].tolist())
        for col in range(matrix.shape[1])
    ]


def test_adapter_retains_every_cross_cluster_edge_and_native_order(tmp_path):
    # Triangle: each support node belongs to a separate cluster.
    edges = [(0, 1), (1, 0), (1, 2), (2, 1), (0, 2), (2, 0)]
    adapter, arrays = make_partitioned_graph(3, edges, [[2], [0], [1]])
    batch = adapter.batch_for_clusters([2, 0, 1])
    native = native_batch(tmp_path, arrays, [2, 0, 1])
    assert batch.global_nodes == (2, 0, 1)
    assert torch.equal(batch.edge_index, native.edge_index)
    assert torch.equal(torch.tensor(batch.permuted_nodes), native.global_nid)
    assert undirected_global_edges(batch) == {(0, 1), (1, 2), (0, 2)}


def test_nonidentity_permutation_and_noncontiguous_external_ids(tmp_path):
    adapter, arrays = make_partitioned_graph(
        4,
        [(0, 1), (1, 0), (0, 2), (2, 0), (1, 3), (3, 1)],
        [[2, 0], [3], [1]],
        global_ids=(10, 40, 20, 70),
    )
    batch = adapter.batch_for_clusters([2, 0])
    native = native_batch(tmp_path, arrays, [2, 0])
    assert batch.global_nodes == (20, 10, 40)
    assert torch.equal(batch.edge_index, native.edge_index)
    assert undirected_global_edges(batch) == {(10, 20), (10, 40)}


def test_two_centres_with_identical_support_remain_distinct_hyperedges():
    adapter, _ = make_partitioned_graph(2, [(0, 1), (1, 0)], [[0], [1]])
    batch = adapter.batch_for_clusters([0, 1])
    references = neighbourhood_references(batch.global_graph())
    assert [reference.identity for reference in references] == [
        ("hyperedge", 0),
        ("hyperedge", 1),
    ]
    assert references[0].support == references[1].support == {0, 1}
    native = HypergraphKHopLiftingLargeScale(k_value=1).lift_topology(
        batch.as_data()
    )
    incidence = native["incidence_hyperedges"].to_dense()
    assert incidence.shape == (2, 2)
    for col, reference in enumerate(references):
        assert {
            batch.global_nodes[row]
            for row in torch.where(incidence[:, col] != 0)[0].tolist()
        } == reference.support


def test_isolated_nodes_are_retained_by_batch_and_hypergraph_lifting():
    adapter, _ = make_partitioned_graph(3, [(0, 1), (1, 0)], [[0], [1, 2]])
    batch = adapter.batch_for_clusters([1])
    assert batch.global_nodes == (1, 2)
    assert batch.edge_index.shape == (2, 0)
    native = HypergraphKHopLiftingLargeScale(k_value=1).lift_topology(
        batch.as_data()
    )
    assert torch.equal(native["incidence_hyperedges"].to_dense(), torch.eye(2))


@pytest.mark.parametrize("cycle_length, expected", [(9, 1), (10, 0)])
def test_cycle_filter_matches_selective_lifting(cycle_length, expected):
    edges = [
        pair
        for node in range(cycle_length)
        for pair in (
            (node, (node + 1) % cycle_length),
            ((node + 1) % cycle_length, node),
        )
    ]
    adapter, _ = make_partitioned_graph(
        cycle_length, edges, [list(range(cycle_length))]
    )
    batch = adapter.batch_for_clusters([0])
    extracted = local_basis_ids_in_global_coordinates(batch, max_length=9)
    native = CellCycleLiftingSelective(
        max_cell_length=9, neighborhoods=[]
    ).lift_topology(batch.as_data())
    assert len(extracted) == native["incidence_2"].shape[1] == expected


def test_diamond_cycle_boundaries_match_selective_lifting():
    undirected = [(0, 1), (1, 2), (2, 0), (1, 3), (3, 2)]
    directed = [pair for u, v in undirected for pair in ((u, v), (v, u))]
    adapter, _ = make_partitioned_graph(4, directed, [[0, 2], [1, 3]])
    batch = adapter.batch_for_clusters([0, 1])
    extracted = local_basis_ids_in_global_coordinates(batch, max_length=9)
    native = CellCycleLiftingSelective(
        max_cell_length=9, neighborhoods=[]
    ).lift_topology(batch.as_data())
    edge_columns = incidence_edges(native["incidence_1"])
    incidence_2 = native["incidence_2"].to_dense()
    native_ids = {
        (
            "cycle",
            tuple(
                sorted(
                    tuple(
                        sorted((batch.global_nodes[a], batch.global_nodes[b]))
                    )
                    for edge_col, (a, b) in enumerate(edge_columns)
                    if incidence_2[edge_col, cell_col] != 0
                )
            ),
        )
        for cell_col in range(incidence_2.shape[1])
    }
    assert extracted == native_ids


def test_chorded_cycle_basis_is_selected_before_mapping_local_ids_to_global():
    permutation = (1, 0, 2, 3, 4)
    local_edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0), (0, 2), (1, 3)]
    directed_global = [
        pair
        for source, target in local_edges
        for pair in (
            (permutation[source], permutation[target]),
            (permutation[target], permutation[source]),
        )
    ]
    adapter, _ = make_partitioned_graph(
        5, directed_global, [[1], [0], [2], [3], [4]]
    )
    batch = adapter.batch_for_clusters([4, 0, 3, 2, 1])
    assert batch.global_nodes == permutation
    native = CellCycleLiftingSelective(
        max_cell_length=9, neighborhoods=[]
    ).lift_topology(batch.as_data())
    edge_columns = incidence_edges(native["incidence_1"])
    incidence_2 = native["incidence_2"].to_dense()
    native_ids = {
        (
            "cycle",
            tuple(
                sorted(
                    tuple(
                        sorted((batch.global_nodes[a], batch.global_nodes[b]))
                    )
                    for edge_col, (a, b) in enumerate(edge_columns)
                    if incidence_2[edge_col, cell_col] != 0
                )
            ),
        )
        for cell_col in range(incidence_2.shape[1])
    }
    assert (
        local_cycle_basis_ids(batch.global_graph(), max_length=9) != native_ids
    )
    assert (
        local_basis_ids_in_global_coordinates(batch, max_length=9)
        == native_ids
    )


def test_triangle_2_cell_matches_fast_clique_lifting():
    directed = [(0, 1), (1, 0), (1, 2), (2, 1), (0, 2), (2, 0)]
    adapter, _ = make_partitioned_graph(3, directed, [[2], [0], [1]])
    batch = adapter.batch_for_clusters([0, 1, 2])
    extracted = {
        reference.identity
        for reference in triangle_references(batch.global_graph())
    }
    native = SimplicialCliqueLiftingFast(
        complex_dim=2, neighborhoods=[]
    ).lift_topology(batch.as_data())
    assert native["incidence_2"].shape[1] == len(extracted) == 1
    edge_columns = incidence_edges(native["incidence_1"])
    incidence_2 = native["incidence_2"].to_dense()
    actual_nodes = {
        batch.global_nodes[node]
        for edge_col, edge_nodes in enumerate(edge_columns)
        if incidence_2[edge_col, 0] != 0
        for node in edge_nodes
    }
    assert (
        extracted
        == {("triangle", *sorted(actual_nodes))}
        == {("triangle", 0, 1, 2)}
    )


def test_preflight_rejects_missing_nodes_empty_clusters_and_inactive_clusters():
    adapter, arrays = make_partitioned_graph(3, [], [[0], [1], [2]])
    assert (
        preflight_partition(
            adapter, expected_nodes=3, K=3, train_mask=np.ones(3, bool)
        )["active_clusters"]
        == 3
    )
    with pytest.raises(ValueError, match="inactive"):
        preflight_partition(
            adapter, expected_nodes=3, K=3, train_mask=np.array([1, 1, 0])
        )
    with pytest.raises(ValueError, match="node count"):
        preflight_partition(
            adapter, expected_nodes=4, K=3, train_mask=np.ones(3, bool)
        )
    with pytest.raises(ValueError, match="nonempty"):
        RecoveryGraphAdapter(np.array([0, 1, 1, 3]), *arrays[1:])


def test_preflight_provenance_accepts_matching_graph_and_partition():
    adapter, arrays = make_partitioned_graph(
        3, [(0, 1), (1, 0), (1, 2), (2, 1)], [[2], [0, 1]]
    )
    edge_index = np.array([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=np.int64)

    def digest(array):
        return hashlib.sha256(np.asarray(array).tobytes()).hexdigest()

    topology_digest = hashlib.sha256()
    topology_digest.update(np.array([3, 4], dtype=np.int64).tobytes())
    topology_digest.update(edge_index.tobytes())
    partition_meta = {
        "num_nodes": 3,
        "num_parts": 2,
        "num_input_edges": 4,
        "topology_sha256": topology_digest.hexdigest(),
        "csr_order_sha256": digest(arrays[1]) + ":" + digest(arrays[2]),
    }
    train_mask = np.array([True, True, True])
    manifest = {
        "dataset": "cora_full",
        "num_nodes": 3,
        "num_parts": 2,
        "num_edges_directed": 4,
        "edge_order_sha256": digest(edge_index),
        "partition_assignment_sha256": (
            digest(arrays[3]) + ":" + digest(arrays[0])
        ),
        "train_mask_sha256": digest(train_mask),
        "active_cluster_count": 2,
        "max_cell_length": 9,
    }
    result = preflight_provenance(
        adapter,
        edge_index=edge_index,
        partition_metadata=partition_meta,
        manifest=manifest,
        train_mask_perm=train_mask,
        expected_dataset="cora_full",
        expected_cycle_cap=9,
    )
    assert result["graph_order_sha256"] == digest(edge_index)
    assert result["directed_csr_entries"] == 4


def test_preflight_provenance_rejects_edge_order_drift_even_with_same_edge_set():
    adapter, arrays = make_partitioned_graph(2, [(0, 1), (1, 0)], [[0], [1]])
    original = np.array([[0, 1], [1, 0]], dtype=np.int64)
    reversed_order = original[:, ::-1].copy()

    def digest(array):
        return hashlib.sha256(np.asarray(array).tobytes()).hexdigest()

    topology_digest = hashlib.sha256()
    topology_digest.update(np.array([2, 2], dtype=np.int64).tobytes())
    topology_digest.update(reversed_order.tobytes())
    metadata = {
        "num_nodes": 2,
        "num_parts": 2,
        "num_input_edges": 2,
        "topology_sha256": topology_digest.hexdigest(),
        "csr_order_sha256": digest(arrays[1]) + ":" + digest(arrays[2]),
    }
    manifest = {
        "dataset": "cora_full",
        "num_nodes": 2,
        "num_parts": 2,
        "num_edges_directed": 2,
        "edge_order_sha256": digest(original),
        "partition_assignment_sha256": digest(arrays[3])
        + ":"
        + digest(arrays[0]),
        "train_mask_sha256": digest(np.ones(2, bool)),
        "active_cluster_count": 2,
        "max_cell_length": 9,
    }
    with pytest.raises(ValueError, match="edge order"):
        preflight_provenance(
            adapter,
            edge_index=reversed_order,
            partition_metadata=metadata,
            manifest=manifest,
            train_mask_perm=np.ones(2, bool),
            expected_dataset="cora_full",
            expected_cycle_cap=9,
        )


def test_preflight_provenance_rejects_corrupted_saved_csr_edges():
    adapter, arrays = make_partitioned_graph(
        3, [(0, 1), (1, 0), (1, 2), (2, 1)], [[2], [0, 1]]
    )
    edge_index = np.array([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=np.int64)

    def digest(array):
        return hashlib.sha256(np.asarray(array).tobytes()).hexdigest()

    topology_digest = hashlib.sha256()
    topology_digest.update(np.array([3, 4], dtype=np.int64).tobytes())
    topology_digest.update(edge_index.tobytes())
    metadata = {
        "num_nodes": 3,
        "num_parts": 2,
        "num_input_edges": 4,
        "topology_sha256": topology_digest.hexdigest(),
        "csr_order_sha256": digest(arrays[1]) + ":" + digest(arrays[2]),
    }
    manifest = {
        "dataset": "cora_full",
        "num_nodes": 3,
        "num_parts": 2,
        "num_edges_directed": 4,
        "edge_order_sha256": digest(edge_index),
        "partition_assignment_sha256": digest(arrays[3])
        + ":"
        + digest(arrays[0]),
        "train_mask_sha256": digest(np.ones(3, bool)),
        "active_cluster_count": 2,
        "max_cell_length": 9,
    }
    corrupted = RecoveryGraphAdapter(
        arrays[0], arrays[1], np.array([0, 1, 1, 1]), arrays[3]
    )
    metadata["csr_order_sha256"] = corrupted.csr_order_sha256()
    with pytest.raises(ValueError, match="CSR graph"):
        preflight_provenance(
            corrupted,
            edge_index=edge_index,
            partition_metadata=metadata,
            manifest=manifest,
            train_mask_perm=np.ones(3, bool),
            expected_dataset="cora_full",
            expected_cycle_cap=9,
        )


def manuscript_fixture(topotune_k=64, gcn_k=32):
    lines = [r"\label{tab:optuna_hparams}"]
    for model, k, q in (
        ("GCN", gcn_k, 8),
        ("EDHNN", 32, 4),
        ("UniGNN", 32, 4),
        ("CWN", 32, 4),
        ("TopoTune", topotune_k, 8),
        ("SCN", 32, 16),
        ("SCCNN", 32, 8),
    ):
        lines.extend(
            (
                rf"\multicolumn{{9}}{{c}}{{\textbf{{{model}}}}} \\*",
                r"\multirow{2}{*}{Cora Full} & FG & $10^{-3}$ & $10^{-4}$ & $128$ & $0.3$ & -- & -- & $0$ \\*",
                rf"& P & $10^{{-3}}$ & $10^{{-4}}$ & $128$ & $0.2$ & ${k}$ & ${q}$ & $0$ \\*",
            )
        )
    lines.append(r"\end{longtable}")
    return "\n".join(lines)


def test_model_annotation_preflight_parses_current_table_and_rejects_drift():
    text = manuscript_fixture()
    assert parse_manuscript_model_annotations(text)["TopoTune"] == (64, 8)
    preflight_model_annotations(text)
    with pytest.raises(ValueError, match="TopoTune"):
        preflight_model_annotations(manuscript_fixture(topotune_k=32))
    with pytest.raises(ValueError, match="GCN"):
        preflight_model_annotations(manuscript_fixture(gcn_k=64))


def test_pinned_cora_full_preflight_rejects_toy_graph_and_invalid_config():
    adapter, _ = make_partitioned_graph(3, [], [[0], [1], [2]])
    config = load_recovery_config()
    inputs = {
        "edge_index": np.empty((2, 0), dtype=np.int64),
        "partition_sidecar": {},
        "manifest": {},
        "train_mask_perm": np.ones(3, bool),
        "manuscript_text": manuscript_fixture(),
    }
    with pytest.raises(ValueError, match="node count"):
        preflight_cora_full_run(
            adapter, config=config, schedule="primary", **inputs
        )
    changed = dict(config)
    changed["expected_num_nodes"] = 2_708
    with pytest.raises(ValueError, match="expected_num_nodes"):
        preflight_cora_full_run(
            adapter, config=changed, schedule="primary", **inputs
        )


def test_order_sensitive_csr_hash_rejects_neighbor_reordering():
    adapter, arrays = make_partitioned_graph(
        3, [(0, 1), (0, 2), (1, 0), (2, 0)], [[0], [1, 2]]
    )
    reordered = np.array(arrays[2], copy=True)
    reordered[:2] = reordered[:2][::-1]
    altered = RecoveryGraphAdapter(arrays[0], arrays[1], reordered, arrays[3])
    assert set(
        adapter.batch_for_clusters([0, 1]).global_graph().edges()
    ) == set(altered.batch_for_clusters([0, 1]).global_graph().edges())
    assert adapter.csr_order_sha256() != altered.csr_order_sha256()
    edge_index = np.array([[0, 0, 1, 2], [1, 2, 0, 0]], dtype=np.int64)

    def digest(value):
        return hashlib.sha256(np.asarray(value).tobytes()).hexdigest()

    topology_digest = hashlib.sha256()
    topology_digest.update(np.array([3, 4], dtype=np.int64).tobytes())
    topology_digest.update(edge_index.tobytes())
    sidecar = {
        "num_nodes": 3,
        "num_parts": 2,
        "num_input_edges": 4,
        "topology_sha256": topology_digest.hexdigest(),
        "csr_order_sha256": adapter.csr_order_sha256(),
    }
    manifest = {
        "dataset": "cora_full",
        "num_nodes": 3,
        "num_parts": 2,
        "num_edges_directed": 4,
        "edge_order_sha256": digest(edge_index),
        "partition_assignment_sha256": digest(arrays[3])
        + ":"
        + digest(arrays[0]),
        "train_mask_sha256": digest(np.ones(3, bool)),
        "active_cluster_count": 2,
        "max_cell_length": 9,
    }
    with pytest.raises(ValueError, match="CSR order"):
        preflight_provenance(
            altered,
            edge_index=edge_index,
            partition_metadata=sidecar,
            manifest=manifest,
            train_mask_perm=np.ones(3, bool),
            expected_dataset="cora_full",
            expected_cycle_cap=9,
        )


def test_q_equals_k_native_order_audit_reports_permutation_not_equivalence():
    original = np.array([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=np.int64)
    adapter, _ = make_partitioned_graph(
        3, list(zip(original[0], original[1], strict=True)), [[2], [0, 1]]
    )
    report = audit_native_full_batch_order(
        adapter, original, cluster_order=(1, 0)
    )
    assert report["same_directed_edge_multiset"] is True
    assert report["same_insertion_order"] is False
    assert isinstance(report["same_cycle_basis"], bool)


def test_frozen_reference_snapshot_rejects_count_and_identity_drift():
    adapter, _ = make_partitioned_graph(2, [(0, 1), (1, 0)], [[0], [1]])
    references = neighbourhood_references(
        adapter.batch_for_clusters([0, 1]).global_graph()
    )
    expected = reference_fingerprints({"hypergraph": references})
    verify_reference_snapshot({"hypergraph": references}, expected)
    with pytest.raises(ValueError, match="hypergraph"):
        verify_reference_snapshot({"hypergraph": references[:1]}, expected)
    changed_support = [
        ReferenceStructure(references[0].identity, frozenset({0})),
        references[1],
    ]
    with pytest.raises(ValueError, match="hypergraph"):
        verify_reference_snapshot({"hypergraph": changed_support}, expected)


def test_reference_snapshot_hashes_indexed_identities_and_supports():
    adapter, _ = make_partitioned_graph(2, [(0, 1), (1, 0)], [[0], [1]])
    references = neighbourhood_references(
        adapter.batch_for_clusters([0, 1]).global_graph()
    )
    first = reference_fingerprints({"hypergraph": references})
    assert first["hypergraph"]["count"] == 2
    assert len(first["hypergraph"]["sha256"]) == 64
    assert (
        reference_fingerprints({"hypergraph": list(reversed(references))})
        == first
    )
    assert reference_fingerprints({"hypergraph": references[:1]}) != first
