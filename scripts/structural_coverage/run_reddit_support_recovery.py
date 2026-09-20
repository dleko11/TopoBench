"""Topology-only Reddit support recovery by epoch 200 versus q.

No training or entropy experiment. Cycles measure support of a fixed full-graph
NetworkX basis, not identity recovery by a recomputed local basis. Reference
definitions match the existing diagnostic, but references are streamed into
cluster-signature counts instead of retaining billions of triangle objects.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import random
import statistics
import subprocess
import time
from pathlib import Path

import networkx as nx
import numpy as np
from scipy import sparse

from scripts.structural_coverage.recovery_core import (
    ReferenceStructure, expected_coverage, generate_epoch_schedules,
)
from scripts.structural_coverage.support_recovery_multidataset import (
    build_signature_groups, count_support_by_epoch,
)

FAMILIES = ("hypergraph", "cellular", "simplicial")
Q_VALUES = [1, 2, 4, 8, 10, 20, 40, 100, 200, 500, 1000, 2000, 5000, 10000]


def digest(*arrays):
    """Hash array shapes, dtypes and bytes without a full-size byte copy."""
    value = hashlib.sha256()
    for array in arrays:
        array = np.ascontiguousarray(array)
        value.update(str((array.shape, array.dtype.str)).encode())
        value.update(memoryview(array).cast("B"))
    return value.hexdigest()


def write_json(path, data):
    """Publish a completed checkpoint atomically on the same filesystem."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def load_adjacency(path, *, expected_nodes, expected_edges):
    """Read only a SciPy graph NPZ, with no Reddit node-data/features file."""
    adjacency = sparse.csr_matrix(sparse.load_npz(path))
    if adjacency.shape != (expected_nodes, expected_nodes):
        raise ValueError("unexpected adjacency node count or shape")
    adjacency.eliminate_zeros()
    adjacency = adjacency.astype(bool)
    adjacency.sum_duplicates()
    adjacency.setdiag(False)
    adjacency.eliminate_zeros()
    adjacency.sort_indices()
    if (adjacency != adjacency.T).nnz:
        raise ValueError("expected symmetric undirected Reddit adjacency")
    if adjacency.nnz != 2 * expected_edges:
        raise ValueError("unexpected undirected edge count")
    return adjacency


def validate_labels(labels, *, nodes, K):
    """Require one integer assignment per node and K nonempty clusters."""
    labels = np.asarray(labels)
    if (labels.shape != (nodes,) or labels.dtype.kind not in "iu"
            or not np.array_equal(np.unique(labels), np.arange(K))):
        raise ValueError("partition must assign every node to one of K nonempty clusters")
    return labels.astype(np.int32, copy=False)


def make_partition(adjacency, K):
    """Partition a featureless graph once. All clusters participate."""
    import torch
    from torch_geometric.data import Data
    from torch_geometric.loader import ClusterData

    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)
    coo = adjacency.tocoo()
    edge_index = torch.from_numpy(np.stack((coo.row, coo.col)).astype(np.int64))
    partition = ClusterData(
        Data(edge_index=edge_index, num_nodes=adjacency.shape[0]),
        num_parts=K, recursive=False, keep_inter_cluster_edges=False,
        sparse_format="csr", save_dir=None, log=True,
    ).partition
    ptr = partition.partptr.cpu().numpy()
    perm = partition.node_perm.cpu().numpy()
    labels = np.empty(adjacency.shape[0], dtype=np.int32)
    for cluster in range(K):
        labels[perm[ptr[cluster]:ptr[cluster + 1]]] = cluster
    return validate_labels(labels, nodes=len(labels), K=K)


def references(graph, family):
    """Yield unchanged full-graph reference supports, without feature tensors."""
    if family == "hypergraph":
        for centre in graph:
            yield ReferenceStructure(("hyperedge", centre),
                                     frozenset([centre, *graph[centre]]))
    elif family == "cellular":
        # The length cap is applied AFTER NetworkX constructs the basis.
        # Basis selection can change with graph traversal order.
        for index, cycle in enumerate(nx.cycle_basis(graph)):
            if 1 < len(cycle) <= 9:
                yield ReferenceStructure(("basis_cycle", index), frozenset(cycle))
    elif family == "simplicial":
        # Every triangle appears once, u < v < w. No list of all triangles.
        for u in graph:
            neighbours = set(graph[u])
            for v in neighbours:
                if v <= u:
                    continue
                for w in neighbours.intersection(graph[v]):
                    if w > v:
                        yield ReferenceStructure(("triangle", u, v, w), frozenset((u, v, w)))
    else:
        raise ValueError(f"unknown reference family: {family}")


def prepare_signatures(graph, labels, family):
    """Aggregate supports exactly, retaining multiplicities across identities."""
    def with_progress():
        for count, reference in enumerate(references(graph, family), start=1):
            if count % 1_000_000 == 0:
                print(f"{family}: processed {count:,} reference supports", flush=True)
            yield reference
    return build_signature_groups(with_progress(), labels, q=int(max(labels)) + 1)


def measure(groups, histogram, *, K, q, epochs, seeds):
    """Use the same tested empirical counter as the four-dataset plots."""
    total = sum(histogram.values())
    if not total:
        raise ValueError("empty reference family")
    observable = sum(count for span, count in histogram.items() if span <= q)
    selected = {span: pair for span, pair in groups.items() if span <= q}
    counts = {}
    for seed in seeds:
        if q in (1, K):
            counts[seed] = [0] + [observable if q == 1 else total] * epochs
        else:
            counts[seed] = count_support_by_epoch(
                selected, K=K, q=q,
                schedules=generate_epoch_schedules(K, q, epochs, seed),
            )
    rows = list(zip(*(counts[seed] for seed in seeds)))
    return {
        "reference_count": total, "observable_count": observable,
        "expected_coverage": [expected_coverage(histogram, K, q, t) for t in range(epochs + 1)],
        "counts_by_seed": counts,
        "coverage_mean": [statistics.mean(row) / total for row in rows],
        "coverage_sample_sd": [statistics.stdev(row) / total for row in rows],
    }


def check_manifest(directory, manifest, *, resume):
    path = directory / "run_manifest.json"
    if path.exists():
        if not resume:
            raise ValueError("output already exists, use a new directory or --resume")
        if json.loads(path.read_text()) != manifest:
            raise ValueError("resume manifest differs: graph, partition, settings or environment changed")
    else:
        directory.mkdir(parents=True, exist_ok=True)
        write_json(path, manifest)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adjacency", type=Path, help="Topology-only raw reddit_graph.npz")
    parser.add_argument("--partition-labels", type=Path, help="Optional node-ID-order labels.npy with K=10000")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--check-input", action="store_true", help="Check adjacency only, no partition or enumeration")
    parser.add_argument("--smoke", action="store_true", help="Synthetic 8-node diagnostic, NOT Reddit evidence")
    args = parser.parse_args(argv)
    started = time.monotonic()
    if args.smoke:
        graph = nx.Graph()
        graph.add_nodes_from(range(8))
        graph.add_edges_from([(0, 1), (1, 2), (2, 0), (2, 3), (3, 4),
                              (4, 2), (4, 5), (5, 6), (6, 7), (7, 4)])
        adjacency = sparse.csr_matrix(nx.to_scipy_sparse_array(graph, dtype=bool))
        K, q_values, epochs, seeds = 4, [1, 2, 4], 3, [0, 1]
    else:
        if not args.adjacency:
            parser.error("--adjacency is required except for --smoke")
        adjacency = load_adjacency(args.adjacency, expected_nodes=232965, expected_edges=57307946)
        graph = None
        K, q_values, epochs, seeds = 10000, Q_VALUES, 200, list(range(10))
    graph_hash = digest(adjacency.indptr.astype(np.int64), adjacency.indices.astype(np.int64))
    print(json.dumps({"stage": "input_checked", "nodes": adjacency.shape[0],
                      "edges": adjacency.nnz // 2, "graph_sha256": graph_hash}), flush=True)
    if args.check_input:
        return
    output = args.output_dir
    if output.exists() and any(output.iterdir()) and not args.resume:
        raise ValueError("output already exists, choose a new directory or --resume")
    output.mkdir(parents=True, exist_ok=True)
    label_path = output / "partition_labels.npy"
    if args.partition_labels:
        labels = np.load(args.partition_labels, allow_pickle=False)
    elif label_path.exists() and args.resume:
        labels = np.load(label_path, allow_pickle=False)
    elif args.smoke:
        labels = np.arange(8) // 2
    else:
        print("Building one featureless METIS partition", flush=True)
        labels = make_partition(adjacency, K)
    labels = validate_labels(labels, nodes=adjacency.shape[0], K=K)
    manifest = {
        "schema": 1, "dataset": "synthetic_smoke" if args.smoke else "reddit",
        "nodes": adjacency.shape[0], "undirected_edges": adjacency.nnz // 2,
        "graph_sha256": graph_hash, "partition_sha256": digest(labels),
        "K": K, "q_values": q_values, "epochs": epochs, "seeds": seeds,
        "cycle_length_cap": 9, "hyperedge_identity": "centre_indexed",
        "cluster_policy": "all_partition_clusters",
        "cellular_metric": "fixed_full_graph_basis_support_not_local_basis_identity",
        "graph_order": "canonical_sorted_CSR", "networkx": nx.__version__,
        "numpy": np.__version__, "python": platform.python_version(),
    }
    check_manifest(output, manifest, resume=args.resume)
    if not label_path.exists():
        np.save(label_path, labels, allow_pickle=False)
    revision = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True)
    if not (output / "provenance.json").exists():
        write_json(output / "provenance.json", {
            "git_revision": revision.stdout.strip(),
            "partition_source": "provided_labels" if args.partition_labels else "new_metis" if not args.smoke else "synthetic",
        })
    sweep = {q: {"K": K, "q": q, "epochs": epochs, "seeds": seeds, "families": {}}
             for q in q_values}
    for family in FAMILIES:
        cache = output / f"{family}_signatures.npz"
        if cache.exists():
            with np.load(cache, allow_pickle=False) as saved:
                spans = saved["spans"].tolist()
                groups = {int(span): (saved[f"s{span}"], saved[f"w{span}"]) for span in spans}
                histogram = {span: int(weights.sum()) for span, (_, weights) in groups.items()}
        else:
            if graph is None:
                print("Constructing full-graph topology only", flush=True)
                graph = nx.from_scipy_sparse_array(adjacency)
            print(f"Enumerating and aggregating {family} reference supports", flush=True)
            groups, histogram = prepare_signatures(graph, labels, family)
            total = sum(histogram.values())
            if not args.smoke and family in ("hypergraph", "simplicial"):
                expected = 232965 if family == "hypergraph" else 8360338411
                if total != expected:
                    raise ValueError(f"{family} reference count {total} differs from expected {expected}")
            arrays = {"spans": np.array(sorted(groups), dtype=np.int32)}
            for span, (signatures, weights) in groups.items():
                arrays[f"s{span}"], arrays[f"w{span}"] = signatures, weights
            temporary = cache.with_suffix(".tmp.npz")
            np.savez(temporary, **arrays)
            temporary.replace(cache)
            del arrays
        print(json.dumps({"family": family, "reference_count": sum(histogram.values()),
                          "distinct_cluster_signatures": sum(len(w) for _, w in groups.values())}), flush=True)
        for q in q_values:
            path = output / f"{family}_q{q}.json"
            if path.exists():
                data = json.loads(path.read_text())
                data["counts_by_seed"] = {int(seed): counts for seed, counts in data["counts_by_seed"].items()}
            else:
                print(f"Measuring {family}, q={q}, {epochs} epochs, {len(seeds)} seeds", flush=True)
                data = measure(groups, histogram, K=K, q=q, epochs=epochs, seeds=seeds)
                write_json(path, data)
            sweep[q]["families"][family] = data
        del groups
    from scripts.structural_coverage.run_support_recovery_multidataset import TITLES, export_q_sweep
    TITLES["reddit"] = "Synthetic smoke (not Reddit)" if args.smoke else "Reddit"
    export_q_sweep(sweep, dataset="reddit", output_dir=output, input_manifest=manifest)
    write_json(output / "completion.json", {"complete": True, "invocation_seconds": time.monotonic() - started})
    print(f"Complete: {output / 'q_recovery.pdf'}", flush=True)


if __name__ == "__main__":
    main()
