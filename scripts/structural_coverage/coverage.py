"""Structural coverage helpers for partitioned TopoBench experiments."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import shutil
import time
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
import torch
from lightning import Callback

StructureKey = tuple[Any, ...]
RankedStructureSet = dict[int, set[StructureKey]]

STRUCTURE_NODE_ID_SPACE = "partition_permuted"
RANKS = (0, 1, 2)
GROUPS = ("all", "rank0", "rank1", "rank2", "rank1_2")
DEFAULT_STRUCTURE_FAMILY = "simplicial_clique"
SUPPORTED_STRUCTURE_FAMILIES = (
    "simplicial_clique",
    "cell_cycle",
    "cell_simple_cycles",
    "hypergraph_khop",
    "hypergraph_incidence",
)


def _json_default(value: Any) -> Any:
    """Convert common NumPy/PyTorch scalar values for JSON output."""
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if isinstance(value, Path):
        return str(value)
    return str(value)


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    """Write a JSON file with stable formatting."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(payload, f, indent=2, sort_keys=True, default=_json_default)
        f.write("\n")


def write_csv_rows(
    path: str | Path,
    rows: list[dict[str, Any]],
    fieldnames: list[str],
    *,
    append: bool = False,
) -> None:
    """Write or append rows to a CSV file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not append or not path.exists()
    mode = "a" if append else "w"
    with path.open(mode, newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def canonical_edge(u: int, v: int) -> StructureKey | None:
    """Return a canonical undirected edge key or ``None`` for self-loops."""
    u = int(u)
    v = int(v)
    if u == v:
        return None
    if u > v:
        u, v = v, u
    return ("e", u, v)


def _edge_vertices(edge: StructureKey) -> tuple[int, int]:
    return int(edge[1]), int(edge[2])


def _triangle_vertices(triangle: StructureKey) -> tuple[int, int, int]:
    return int(triangle[1]), int(triangle[2]), int(triangle[3])


def _nodes_for_structure(structure: StructureKey) -> tuple[int, ...]:
    if structure[0] == "v":
        return (int(structure[1]),)
    if structure[0] == "e":
        return _edge_vertices(structure)
    if structure[0] == "t":
        return _triangle_vertices(structure)
    if structure[0] == "c":
        if (len(structure) - 1) % 2 != 0:
            raise ValueError(f"Malformed cell key: {structure!r}")
        nodes = set()
        for idx in range(1, len(structure), 2):
            nodes.add(int(structure[idx]))
            nodes.add(int(structure[idx + 1]))
        return tuple(sorted(nodes))
    if structure[0] == "h":
        return tuple(int(node) for node in structure[1:])
    raise ValueError(f"Unsupported structure key: {structure!r}")


def canonical_cell(boundary_edges: set[StructureKey]) -> StructureKey | None:
    """Return a canonical 2-cell key from its undirected boundary edges."""
    if not boundary_edges:
        return None
    flat: list[int] = []
    for edge in sorted(boundary_edges):
        u, v = _edge_vertices(edge)
        flat.extend([u, v])
    return ("c", *flat)


def canonical_hyperedge(nodes: list[int] | tuple[int, ...]) -> StructureKey | None:
    """Return a canonical hyperedge key from incident node IDs."""
    unique_nodes = tuple(sorted({int(node) for node in nodes}))
    if not unique_nodes:
        return None
    return ("h", *unique_nodes)


def edge_index_to_undirected_edges(
    edge_index: torch.Tensor,
    global_nid: torch.Tensor,
) -> set[StructureKey]:
    """Convert local ``edge_index`` columns into canonical global edge keys."""
    edge_index = edge_index.detach().cpu().to(torch.long)
    global_nid = global_nid.detach().cpu().to(torch.long)
    edges: set[StructureKey] = set()
    if edge_index.numel() == 0:
        return edges

    src = edge_index[0].tolist()
    dst = edge_index[1].tolist()
    node_ids = global_nid.tolist()
    for local_u, local_v in zip(src, dst, strict=False):
        edge = canonical_edge(node_ids[local_u], node_ids[local_v])
        if edge is not None:
            edges.add(edge)
    return edges


def triangles_from_edges(edges: set[StructureKey]) -> set[StructureKey]:
    """Return all triangle clique keys induced by an undirected edge set."""
    adjacency: dict[int, set[int]] = defaultdict(set)
    for edge in edges:
        u, v = _edge_vertices(edge)
        adjacency[u].add(v)
        adjacency[v].add(u)

    triangles: set[StructureKey] = set()
    for u in sorted(adjacency):
        for v in sorted(n for n in adjacency[u] if n > u):
            common = adjacency[u].intersection(adjacency[v])
            for w in sorted(n for n in common if n > v):
                triangles.add(("t", u, v, w))
    return triangles


def extract_simplicial_structures_from_edge_index(
    edge_index: torch.Tensor,
    global_nid: torch.Tensor,
) -> RankedStructureSet:
    """Extract rank 0/1/2 clique structures from a batch graph."""
    nodes = {
        ("v", int(node_id))
        for node_id in global_nid.detach().cpu().to(torch.long).tolist()
    }
    edges = edge_index_to_undirected_edges(edge_index, global_nid)
    triangles = triangles_from_edges(edges)
    return {0: nodes, 1: edges, 2: triangles}


def _incidence_indices_by_column(
    incidence: torch.Tensor,
    *,
    expected_rows: int,
) -> dict[int, list[int]]:
    """Return nonzero row indices grouped by incidence-matrix column."""
    matrix = incidence.detach().cpu()
    if matrix.layout != torch.sparse_coo:
        matrix = matrix.to_sparse_coo()
    matrix = matrix.coalesce()

    indices = matrix.indices().to(torch.long)
    shape = matrix.shape
    if shape[0] == expected_rows:
        rows = indices[0].tolist()
        cols = indices[1].tolist()
    elif shape[1] == expected_rows:
        rows = indices[1].tolist()
        cols = indices[0].tolist()
    else:
        raise ValueError(
            "Incidence matrix does not align with batch.global_nid: "
            f"shape={tuple(shape)}, expected_rows={expected_rows}."
        )

    grouped: dict[int, list[int]] = defaultdict(list)
    for row, col in zip(rows, cols, strict=False):
        grouped[int(col)].append(int(row))
    return grouped


def _edge_keys_from_incidence_1(
    incidence_1: torch.Tensor,
    global_nid: torch.Tensor,
) -> dict[int, StructureKey]:
    """Map local rank-1 cell IDs to canonical edge keys."""
    node_ids = global_nid.detach().cpu().to(torch.long).tolist()
    columns = _incidence_indices_by_column(
        incidence_1,
        expected_rows=len(node_ids),
    )
    edges: dict[int, StructureKey] = {}
    for edge_id, local_nodes in columns.items():
        unique_local_nodes = sorted(set(local_nodes))
        if len(unique_local_nodes) != 2:
            continue
        edge = canonical_edge(
            node_ids[unique_local_nodes[0]],
            node_ids[unique_local_nodes[1]],
        )
        if edge is not None:
            edges[int(edge_id)] = edge
    return edges


def extract_cell_structures_from_batch(batch: Any) -> RankedStructureSet:
    """Extract rank 0/1/2 cell-cycle structures from batch incidences."""
    if not hasattr(batch, "incidence_1"):
        raise ValueError("Cell structural coverage requires batch.incidence_1.")
    if not hasattr(batch, "incidence_2"):
        raise ValueError("Cell structural coverage requires batch.incidence_2.")

    global_nid = batch.global_nid.detach().cpu().to(torch.long)
    nodes = {("v", int(node_id)) for node_id in global_nid.tolist()}
    edge_by_id = _edge_keys_from_incidence_1(batch.incidence_1, global_nid)
    edges = set(edge_by_id.values())

    cell_columns = _incidence_indices_by_column(
        batch.incidence_2,
        expected_rows=int(batch.incidence_1.shape[1]),
    )
    cells: set[StructureKey] = set()
    for edge_ids in cell_columns.values():
        boundary_edges = {
            edge_by_id[int(edge_id)]
            for edge_id in edge_ids
            if int(edge_id) in edge_by_id
        }
        cell = canonical_cell(boundary_edges)
        if cell is not None:
            cells.add(cell)
    return {0: nodes, 1: edges, 2: cells}


def extract_cell_simple_cycle_structures_from_batch(
    batch: Any,
    *,
    max_support_nodes: int = 8,
) -> RankedStructureSet:
    """Extract generated simple-cycle cells from a batch induced graph."""
    global_nid = batch.global_nid.detach().cpu().to(torch.long)
    nodes = {("v", int(node_id)) for node_id in global_nid.tolist()}
    edges = edge_index_to_undirected_edges(batch.edge_index, global_nid)
    cells = simple_cycle_cells_from_edges(
        active_nodes=global_nid.numpy(),
        edges=edges,
        max_support_nodes=max_support_nodes,
    )
    return {0: nodes, 1: edges, 2: cells}


def extract_hypergraph_structures_from_batch(batch: Any) -> RankedStructureSet:
    """Extract rank 0 nodes and rank 1 hyperedges from batch incidence."""
    if not hasattr(batch, "incidence_hyperedges"):
        raise ValueError(
            "Hypergraph structural coverage requires "
            "batch.incidence_hyperedges."
        )

    global_nid = batch.global_nid.detach().cpu().to(torch.long)
    node_ids = global_nid.tolist()
    nodes = {("v", int(node_id)) for node_id in node_ids}
    hyperedge_columns = _incidence_indices_by_column(
        batch.incidence_hyperedges,
        expected_rows=len(node_ids),
    )

    hyperedges: set[StructureKey] = set()
    for local_nodes in hyperedge_columns.values():
        hyperedge = canonical_hyperedge(
            [node_ids[int(local_node)] for local_node in local_nodes]
        )
        if hyperedge is not None:
            hyperedges.add(hyperedge)
    return {0: nodes, 1: hyperedges, 2: set()}


def extract_structures_from_batch(
    *,
    batch: Any,
    structure_family: str,
    structure_params: dict[str, Any] | None = None,
) -> RankedStructureSet:
    """Dispatch batch extraction for the configured structure family."""
    structure_params = structure_params or {}
    if structure_family == "simplicial_clique":
        return extract_simplicial_structures_from_edge_index(
            batch.edge_index,
            batch.global_nid,
        )
    if structure_family == "cell_cycle":
        return extract_cell_structures_from_batch(batch)
    if structure_family == "cell_simple_cycles":
        return extract_cell_simple_cycle_structures_from_batch(
            batch,
            max_support_nodes=int(
                structure_params.get("max_support_nodes", 8)
            ),
        )
    if structure_family in ("hypergraph_khop", "hypergraph_incidence"):
        return extract_hypergraph_structures_from_batch(batch)
    raise ValueError(f"Unsupported structure family: {structure_family!r}")


def load_part_ids_for_split(
    handle: dict[str, Any],
    split: str = "train",
) -> np.ndarray:
    """Load partition IDs used by a split, falling back to all partitions."""
    paths = handle.get("paths", {})
    path = paths.get(f"parts_with_{split}")
    if path and os.path.exists(path):
        part_ids = np.load(path).astype(np.int64)
        if part_ids.size > 0:
            return part_ids
    return np.arange(int(handle["num_parts"]), dtype=np.int64)


def nodes_for_part_ids(
    partptr: np.ndarray,
    part_ids: np.ndarray,
) -> np.ndarray:
    """Return all permuted node IDs belonging to the selected partitions."""
    chunks = [
        np.arange(int(partptr[part]), int(partptr[part + 1]), dtype=np.int64)
        for part in part_ids
    ]
    if not chunks:
        return np.empty((0,), dtype=np.int64)
    return np.concatenate(chunks)


def part_ids_for_nodes(partptr: np.ndarray, nodes: np.ndarray) -> np.ndarray:
    """Map permuted node IDs to partition IDs."""
    nodes = np.asarray(nodes, dtype=np.int64)
    return np.searchsorted(partptr, nodes, side="right") - 1


def csr_to_undirected_edges(
    indptr: np.ndarray,
    indices: np.ndarray,
    active_nodes: np.ndarray | None = None,
) -> set[StructureKey]:
    """Extract canonical undirected edge keys from global CSR arrays."""
    if active_nodes is None:
        active_nodes = np.arange(indptr.shape[0] - 1, dtype=np.int64)
    active = {int(node) for node in active_nodes.tolist()}

    edges: set[StructureKey] = set()
    for u in active:
        start = int(indptr[u])
        end = int(indptr[u + 1])
        for v_raw in indices[start:end]:
            v = int(v_raw)
            if v not in active:
                continue
            edge = canonical_edge(u, v)
            if edge is not None:
                edges.add(edge)
    return edges


def graph_from_nodes_and_edges(
    nodes: np.ndarray,
    edges: set[StructureKey],
) -> nx.Graph:
    """Build a NetworkX graph in partition-permuted node coordinates."""
    graph = nx.Graph()
    graph.add_nodes_from(int(node) for node in nodes.tolist())
    graph.add_edges_from(_edge_vertices(edge) for edge in sorted(edges))
    return graph


def _cycle_boundary_edges(cycle: list[int]) -> set[StructureKey]:
    """Return canonical boundary edges for one ordered cycle."""
    boundary_edges: set[StructureKey] = set()
    if len(cycle) < 2:
        return boundary_edges
    for idx, u in enumerate(cycle):
        v = cycle[(idx + 1) % len(cycle)]
        edge = canonical_edge(int(u), int(v))
        if edge is not None:
            boundary_edges.add(edge)
    return boundary_edges


def cell_cycles_from_edges(
    *,
    active_nodes: np.ndarray,
    edges: set[StructureKey],
    max_cell_length: int | None = None,
) -> set[StructureKey]:
    """Compute canonical 2-cell keys from NetworkX cycle-basis cells."""
    graph = graph_from_nodes_and_edges(active_nodes, edges)
    cycles = sorted(
        nx.cycle_basis(graph),
        key=lambda cycle: (len(cycle), tuple(sorted(int(node) for node in cycle))),
    )
    cells: set[StructureKey] = set()
    for cycle in cycles:
        if len(cycle) == 1:
            continue
        if max_cell_length is not None and len(cycle) > max_cell_length:
            continue
        cell = canonical_cell(_cycle_boundary_edges(cycle))
        if cell is not None:
            cells.add(cell)
    return cells


def simple_cycle_cells_from_edges(
    *,
    active_nodes: np.ndarray,
    edges: set[StructureKey],
    max_support_nodes: int = 8,
) -> set[StructureKey]:
    """Enumerate all undirected simple-cycle cells up to a node-support cap."""
    max_support_nodes = int(max_support_nodes)
    if max_support_nodes < 3 or not edges:
        return set()

    graph = graph_from_nodes_and_edges(active_nodes, edges)
    adjacency = {
        int(node): sorted(int(neighbor) for neighbor in graph.neighbors(node))
        for node in graph.nodes
    }
    cells: set[StructureKey] = set()

    def dfs(start: int, path: list[int], visited: set[int]) -> None:
        current = path[-1]
        for neighbor in adjacency[current]:
            if neighbor < start:
                continue
            if neighbor == start:
                if len(path) >= 3:
                    cell = canonical_cell(_cycle_boundary_edges(path))
                    if cell is not None:
                        cells.add(cell)
                continue
            if neighbor in visited or len(path) >= max_support_nodes:
                continue
            visited.add(neighbor)
            path.append(neighbor)
            dfs(start, path, visited)
            path.pop()
            visited.remove(neighbor)

    for start in sorted(adjacency):
        dfs(start, [start], {start})
    return cells


def khop_hyperedges_from_edges(
    *,
    active_nodes: np.ndarray,
    edges: set[StructureKey],
    k_value: int,
) -> set[StructureKey]:
    """Compute k-hop-neighborhood hyperedge keys from an undirected graph."""
    graph = graph_from_nodes_and_edges(active_nodes, edges)
    hyperedges: set[StructureKey] = set()
    for node in active_nodes.tolist():
        lengths = nx.single_source_shortest_path_length(
            graph,
            int(node),
            cutoff=int(k_value),
        )
        hyperedge = canonical_hyperedge(tuple(lengths.keys()))
        if hyperedge is not None:
            hyperedges.add(hyperedge)
    return hyperedges


def compute_global_simplicial_structures(
    *,
    memmap_dir: str | Path,
    train_part_ids: np.ndarray,
) -> tuple[RankedStructureSet, np.ndarray, np.ndarray, set[StructureKey]]:
    """Compute full-graph rank 0/1/2 structures in partition coordinates."""
    memmap_dir = Path(memmap_dir)
    partptr = np.load(memmap_dir / "partptr.npy", mmap_mode="r")
    indptr = np.load(memmap_dir / "indptr.npy", mmap_mode="r")
    indices = np.load(memmap_dir / "indices.npy", mmap_mode="r")

    active_nodes = nodes_for_part_ids(partptr, train_part_ids)
    nodes = {("v", int(node)) for node in active_nodes.tolist()}
    edges = csr_to_undirected_edges(indptr, indices, active_nodes)
    triangles = triangles_from_edges(edges)
    structures = {0: nodes, 1: edges, 2: triangles}
    return structures, np.asarray(partptr), active_nodes, edges


def compute_global_cell_structures(
    *,
    memmap_dir: str | Path,
    train_part_ids: np.ndarray,
    max_cell_length: int | None = None,
) -> tuple[RankedStructureSet, np.ndarray, np.ndarray, set[StructureKey]]:
    """Compute full-graph rank 0/1/2 cycle-cell structures."""
    memmap_dir = Path(memmap_dir)
    partptr = np.load(memmap_dir / "partptr.npy", mmap_mode="r")
    indptr = np.load(memmap_dir / "indptr.npy", mmap_mode="r")
    indices = np.load(memmap_dir / "indices.npy", mmap_mode="r")

    active_nodes = nodes_for_part_ids(partptr, train_part_ids)
    nodes = {("v", int(node)) for node in active_nodes.tolist()}
    edges = csr_to_undirected_edges(indptr, indices, active_nodes)
    cells = cell_cycles_from_edges(
        active_nodes=active_nodes,
        edges=edges,
        max_cell_length=max_cell_length,
    )
    structures = {0: nodes, 1: edges, 2: cells}
    return structures, np.asarray(partptr), active_nodes, edges


def compute_global_cell_simple_cycle_structures(
    *,
    memmap_dir: str | Path,
    train_part_ids: np.ndarray,
    max_support_nodes: int = 8,
) -> tuple[RankedStructureSet, np.ndarray, np.ndarray, set[StructureKey]]:
    """Compute rank 0/1 and generated simple-cycle rank-2 cell structures."""
    memmap_dir = Path(memmap_dir)
    partptr = np.load(memmap_dir / "partptr.npy", mmap_mode="r")
    indptr = np.load(memmap_dir / "indptr.npy", mmap_mode="r")
    indices = np.load(memmap_dir / "indices.npy", mmap_mode="r")

    active_nodes = nodes_for_part_ids(partptr, train_part_ids)
    nodes = {("v", int(node)) for node in active_nodes.tolist()}
    edges = csr_to_undirected_edges(indptr, indices, active_nodes)
    cells = simple_cycle_cells_from_edges(
        active_nodes=active_nodes,
        edges=edges,
        max_support_nodes=max_support_nodes,
    )
    structures = {0: nodes, 1: edges, 2: cells}
    return structures, np.asarray(partptr), active_nodes, edges


def compute_global_hypergraph_structures(
    *,
    memmap_dir: str | Path,
    train_part_ids: np.ndarray,
    k_value: int = 1,
) -> tuple[RankedStructureSet, np.ndarray, np.ndarray, set[StructureKey]]:
    """Compute full-graph k-hop hyperedge structures."""
    memmap_dir = Path(memmap_dir)
    partptr = np.load(memmap_dir / "partptr.npy", mmap_mode="r")
    indptr = np.load(memmap_dir / "indptr.npy", mmap_mode="r")
    indices = np.load(memmap_dir / "indices.npy", mmap_mode="r")

    active_nodes = nodes_for_part_ids(partptr, train_part_ids)
    nodes = {("v", int(node)) for node in active_nodes.tolist()}
    edges = csr_to_undirected_edges(indptr, indices, active_nodes)
    hyperedges = khop_hyperedges_from_edges(
        active_nodes=active_nodes,
        edges=edges,
        k_value=k_value,
    )
    structures = {0: nodes, 1: hyperedges, 2: set()}
    return structures, np.asarray(partptr), active_nodes, edges


def compute_global_structures(
    *,
    memmap_dir: str | Path,
    train_part_ids: np.ndarray,
    structure_family: str,
    structure_params: dict[str, Any] | None = None,
) -> tuple[RankedStructureSet, np.ndarray, np.ndarray, set[StructureKey]]:
    """Dispatch global structure extraction for the configured family."""
    structure_params = structure_params or {}
    if structure_family == "simplicial_clique":
        return compute_global_simplicial_structures(
            memmap_dir=memmap_dir,
            train_part_ids=train_part_ids,
        )
    if structure_family == "cell_cycle":
        max_cell_length = structure_params.get("max_cell_length")
        return compute_global_cell_structures(
            memmap_dir=memmap_dir,
            train_part_ids=train_part_ids,
            max_cell_length=(
                int(max_cell_length)
                if max_cell_length not in (None, "null")
                else None
            ),
        )
    if structure_family == "cell_simple_cycles":
        return compute_global_cell_simple_cycle_structures(
            memmap_dir=memmap_dir,
            train_part_ids=train_part_ids,
            max_support_nodes=int(
                structure_params.get("max_support_nodes", 8)
            ),
        )
    if structure_family == "hypergraph_khop":
        return compute_global_hypergraph_structures(
            memmap_dir=memmap_dir,
            train_part_ids=train_part_ids,
            k_value=int(structure_params.get("k_value", 1)),
        )
    if structure_family == "hypergraph_incidence":
        raise ValueError(
            "hypergraph_incidence can extract batch hyperedges, but global "
            "S* needs a concrete lifting rule. Use hypergraph_khop for the "
            "current graph2hypergraph khop lifting."
        )
    raise ValueError(f"Unsupported structure family: {structure_family!r}")


def flatten_structures(structures: RankedStructureSet) -> set[StructureKey]:
    """Flatten a ranked structure mapping into a single set."""
    out: set[StructureKey] = set()
    for rank in RANKS:
        out.update(structures.get(rank, set()))
    return out


def filter_structures_to_universe(
    structures: RankedStructureSet,
    allowed_structures: RankedStructureSet,
) -> RankedStructureSet:
    """Intersect ranked structures with the global structure universe."""
    return {
        rank: set(structures.get(rank, set())).intersection(
            allowed_structures.get(rank, set())
        )
        for rank in RANKS
    }


def compute_structure_spans(
    structures: RankedStructureSet,
    partptr: np.ndarray,
) -> dict[StructureKey, int]:
    """Compute cluster span for every structure."""
    spans: dict[StructureKey, int] = {}
    for ranked in structures.values():
        for structure in ranked:
            nodes = np.asarray(_nodes_for_structure(structure), dtype=np.int64)
            parts = part_ids_for_nodes(partptr, nodes)
            spans[structure] = int(np.unique(parts).shape[0])
    return spans


def effective_batch_sizes(k_eff: int, q: int) -> list[int]:
    """Return the epoch batch sizes induced by ``drop_last=False`` chunking."""
    k_eff = int(k_eff)
    q = int(q)
    if k_eff <= 0:
        return []
    if q <= 0:
        raise ValueError(f"q must be positive, got {q}.")
    full_batches, remainder = divmod(k_eff, q)
    sizes = [q] * full_batches
    if remainder:
        sizes.append(remainder)
    return sizes


def per_epoch_probability(span: int, q: int, k_eff: int) -> float:
    """Return the theoretical per-epoch recovery probability.

    When ``k_eff`` is divisible by ``q`` this is exactly the appendix formula.
    Otherwise it computes the same co-location probability for the actual
    ``drop_last=False`` batch-size multiset: full ``q`` batches plus one
    remainder batch.
    """
    span = int(span)
    q = int(q)
    k_eff = int(k_eff)
    if span <= 0:
        return 0.0
    if k_eff <= 0 or span > k_eff:
        return 0.0
    if span == 1:
        return 1.0

    denominator = math.comb(k_eff - 1, span - 1)
    if denominator == 0:
        return 0.0

    probability = 0.0
    for batch_size in effective_batch_sizes(k_eff, q):
        if batch_size < span:
            continue
        reference_probability = batch_size / k_eff
        colocate_probability = (
            math.comb(batch_size - 1, span - 1) / denominator
        )
        probability += reference_probability * colocate_probability
    return probability


def bernoulli_entropy_bits(probability: float) -> float:
    """Return Bernoulli entropy in bits."""
    p = float(probability)
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return -p * math.log2(p) - (1.0 - p) * math.log2(1.0 - p)


def build_span_histogram_rows(
    *,
    structures: RankedStructureSet,
    spans: dict[StructureKey, int],
    q: int,
    k_eff: int,
) -> list[dict[str, Any]]:
    """Build rows for ``span_histogram.csv``."""
    total = sum(len(structures.get(rank, set())) for rank in RANKS)
    rows: list[dict[str, Any]] = []
    for rank in RANKS:
        counter = Counter(spans[s] for s in structures.get(rank, set()))
        for span in sorted(counter):
            count = int(counter[span])
            rows.append(
                {
                    "rank": rank,
                    "span": span,
                    "count": count,
                    "fraction": count / total if total else 0.0,
                    "per_epoch_probability": per_epoch_probability(
                        span, q, k_eff
                    ),
                    "asymptotic_observable": int(span <= q),
                }
            )
    return rows


def _group_ranks(group: str) -> tuple[int, ...]:
    if group == "all":
        return RANKS
    if group == "rank0":
        return (0,)
    if group == "rank1":
        return (1,)
    if group == "rank2":
        return (2,)
    if group == "rank1_2":
        return (1, 2)
    raise ValueError(f"Unsupported coverage group: {group}")


def _structures_for_group(
    structures: RankedStructureSet,
    group: str,
) -> set[StructureKey]:
    out: set[StructureKey] = set()
    for rank in _group_ranks(group):
        out.update(structures.get(rank, set()))
    return out


def build_theory_curve_rows(
    *,
    structures: RankedStructureSet,
    spans: dict[StructureKey, int],
    q: int,
    k_eff: int,
    max_epochs: int,
) -> list[dict[str, Any]]:
    """Build rows for expected coverage and entropy curves."""
    probabilities = {
        structure: per_epoch_probability(span, q, k_eff)
        for structure, span in spans.items()
    }
    rows: list[dict[str, Any]] = []
    for epoch in range(int(max_epochs) + 1):
        row: dict[str, Any] = {"epoch": epoch}
        for group in GROUPS:
            group_structures = _structures_for_group(structures, group)
            total = len(group_structures)
            observable = sum(
                1 for structure in group_structures if probabilities[structure] > 0
            )
            expected = 0.0
            entropy_bits = 0.0
            for structure in group_structures:
                p = probabilities[structure]
                rho = 1.0 - (1.0 - p) ** epoch
                expected += rho
                if p > 0:
                    entropy_bits += bernoulli_entropy_bits(rho)

            expected_coverage = expected / total if total else 0.0
            row[f"expected_coverage_{group}"] = expected_coverage
            row[f"observable_ceiling_{group}"] = (
                observable / total if total else 0.0
            )
            row[f"entropy_bits_{group}"] = entropy_bits
            row[f"entropy_nats_{group}"] = entropy_bits * math.log(2.0)
            row[f"normalized_entropy_bits_{group}"] = (
                entropy_bits / observable if observable else 0.0
            )
            row[f"normalized_entropy_nats_{group}"] = (
                entropy_bits * math.log(2.0) / observable
                if observable
                else 0.0
            )
        rows.append(row)
    return rows


def empirical_coverage_row(
    *,
    epoch: int,
    global_structures: RankedStructureSet,
    observed: RankedStructureSet,
    new_by_rank: RankedStructureSet | None = None,
) -> dict[str, Any]:
    """Build one row for ``empirical_coverage.csv``."""
    new_by_rank = new_by_rank or {rank: set() for rank in RANKS}
    row: dict[str, Any] = {"epoch": int(epoch)}
    for group in GROUPS:
        total_set = _structures_for_group(global_structures, group)
        observed_set = _structures_for_group(observed, group)
        new_set = _structures_for_group(new_by_rank, group)
        total = len(total_set)
        observed_count = len(observed_set)
        row[f"observed_count_{group}"] = observed_count
        row[f"total_count_{group}"] = total
        row[f"realized_coverage_{group}"] = (
            observed_count / total if total else 0.0
        )
        row[f"new_count_{group}"] = len(new_set)
    return row


def grouping_signature(groups: list[tuple[int, ...]]) -> tuple[tuple[int, ...], ...]:
    """Return an order-invariant signature for an epoch grouping."""
    return tuple(sorted(tuple(sorted(group)) for group in groups))


def compute_mean_pair_cooccurrence(
    pair_counts: Counter[tuple[int, int]],
    part_ids: np.ndarray,
    num_epochs: int,
) -> float:
    """Return mean pair co-occurrence frequency across epochs."""
    if num_epochs <= 0:
        return 0.0
    parts = [int(part) for part in sorted(part_ids.tolist())]
    all_pairs = list(combinations(parts, 2))
    if not all_pairs:
        return 0.0
    values = [pair_counts.get(pair, 0) / num_epochs for pair in all_pairs]
    return float(np.mean(values)) if values else 0.0


def expected_pair_cooccurrence(q: int, k_eff: int) -> float:
    """Return expected pair co-occurrence probability per epoch."""
    return per_epoch_probability(span=2, q=q, k_eff=k_eff)


def induced_edge_mismatch(
    *,
    full_edges: set[StructureKey],
    batch_edge_index: torch.Tensor,
    batch_global_nid: torch.Tensor,
) -> dict[str, Any] | None:
    """Compare batch edges with the full-graph induced edge set."""
    observed_edges = edge_index_to_undirected_edges(
        batch_edge_index, batch_global_nid
    )
    node_set = {
        int(node)
        for node in batch_global_nid.detach().cpu().to(torch.long).tolist()
    }
    expected_edges = {
        edge
        for edge in full_edges
        if edge[1] in node_set and edge[2] in node_set
    }
    if observed_edges == expected_edges:
        return None

    return {
        "expected_num_edges": len(expected_edges),
        "observed_num_edges": len(observed_edges),
        "missing_edges": sorted(expected_edges - observed_edges)[:10],
        "extra_edges": sorted(observed_edges - expected_edges)[:10],
    }


def _set_hash(values: set[StructureKey]) -> str:
    payload = json.dumps(sorted(values), separators=(",", ":"))
    return hashlib.sha1(payload.encode()).hexdigest()


class StructuralCoverageCallback(Callback):
    """Track cumulative structural recovery during clustered training."""

    def __init__(
        self,
        *,
        handle: dict[str, Any],
        q: int,
        results_dir: str | Path,
        max_epochs: int,
        structure_family: str = DEFAULT_STRUCTURE_FAMILY,
        structure_params: dict[str, Any] | None = None,
        cfg_snapshot: dict[str, Any] | None = None,
        save_batch_events: bool = False,
        audit_induced_edges: bool = True,
        audit_max_batches: int = 10,
        require_equal_batches: bool = True,
    ) -> None:
        super().__init__()
        self.handle = handle
        self.q = int(q)
        self.results_dir = Path(results_dir)
        self.max_epochs = int(max_epochs)
        self.structure_family = str(structure_family)
        if self.structure_family not in SUPPORTED_STRUCTURE_FAMILIES:
            raise ValueError(
                f"Unsupported structure_family={self.structure_family!r}. "
                f"Expected one of {SUPPORTED_STRUCTURE_FAMILIES}."
            )
        self.structure_params = structure_params or {}
        self.cfg_snapshot = cfg_snapshot or {}
        self.save_batch_events = bool(save_batch_events)
        self.audit_induced_edges = bool(audit_induced_edges)
        self.audit_max_batches = int(audit_max_batches)
        self.require_equal_batches = bool(require_equal_batches)

        self.global_structures: RankedStructureSet = {
            rank: set() for rank in RANKS
        }
        self.observed: RankedStructureSet = {rank: set() for rank in RANKS}
        self.spans: dict[StructureKey, int] = {}
        self.partptr: np.ndarray | None = None
        self.train_part_ids: np.ndarray | None = None
        self.full_edges: set[StructureKey] = set()
        self.active_node_count: int | None = None

        self.epoch_groups: list[tuple[int, ...]] = []
        self.epoch_batch_rows: list[dict[str, Any]] = []
        self.epoch_new_by_rank: RankedStructureSet = {
            rank: set() for rank in RANKS
        }
        self.epoch_grouping_signatures: set[tuple[tuple[int, ...], ...]] = set()
        self.pair_counts: Counter[tuple[int, int]] = Counter()
        self.completed_epochs = 0
        self.audited_batches = 0
        self.setup_complete = False

    @property
    def empirical_path(self) -> Path:
        return self.results_dir / "empirical_coverage.csv"

    @property
    def batch_events_path(self) -> Path:
        return self.results_dir / "batch_events.csv"

    def setup(
        self,
        trainer: Any,
        pl_module: Any,
        stage: str | None = None,
    ) -> None:
        """Prepare global structures and static artifacts."""
        if self.setup_complete or stage not in (None, "fit"):
            return

        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.train_part_ids = load_part_ids_for_split(self.handle, "train")
        self.global_structures, self.partptr, active_nodes, self.full_edges = (
            compute_global_structures(
                memmap_dir=self.handle["memmap_dir"],
                train_part_ids=self.train_part_ids,
                structure_family=self.structure_family,
                structure_params=self.structure_params,
            )
        )
        self.active_node_count = int(active_nodes.shape[0])
        self.spans = compute_structure_spans(
            self.global_structures, self.partptr
        )
        k_eff = int(self.train_part_ids.shape[0])
        if self.require_equal_batches and k_eff % self.q != 0:
            raise ValueError(
                "Structural coverage theory assumes equal-sized partition "
                f"batches, but K_eff={k_eff} is not divisible by q={self.q}."
            )

        span_rows = build_span_histogram_rows(
            structures=self.global_structures,
            spans=self.spans,
            q=self.q,
            k_eff=k_eff,
        )
        write_csv_rows(
            self.results_dir / "span_histogram.csv",
            span_rows,
            [
                "rank",
                "span",
                "count",
                "fraction",
                "per_epoch_probability",
                "asymptotic_observable",
            ],
        )

        theory_rows = build_theory_curve_rows(
            structures=self.global_structures,
            spans=self.spans,
            q=self.q,
            k_eff=k_eff,
            max_epochs=self.max_epochs,
        )
        theory_fields = ["epoch"]
        for group in GROUPS:
            theory_fields.extend(
                [
                    f"expected_coverage_{group}",
                    f"observable_ceiling_{group}",
                    f"entropy_bits_{group}",
                    f"entropy_nats_{group}",
                    f"normalized_entropy_bits_{group}",
                    f"normalized_entropy_nats_{group}",
                ]
            )
        write_csv_rows(
            self.results_dir / "theory_curves.csv",
            theory_rows,
            theory_fields,
        )

        empirical_fields = ["epoch"]
        for group in GROUPS:
            empirical_fields.extend(
                [
                    f"observed_count_{group}",
                    f"total_count_{group}",
                    f"realized_coverage_{group}",
                    f"new_count_{group}",
                ]
            )
        write_csv_rows(
            self.empirical_path,
            [
                empirical_coverage_row(
                    epoch=0,
                    global_structures=self.global_structures,
                    observed=self.observed,
                )
            ],
            empirical_fields,
        )

        if self.save_batch_events:
            write_csv_rows(
                self.batch_events_path,
                [],
                [
                    "epoch",
                    "batch_idx",
                    "part_ids",
                    "num_nodes",
                    "num_edges",
                    "observed_global_structures",
                    "new_global_structures",
                    "observed_rank0",
                    "observed_rank1",
                    "observed_rank2",
                    "new_rank0",
                    "new_rank1",
                    "new_rank2",
                ],
            )

        self._write_metadata(active_nodes=active_nodes)
        self.setup_complete = True

    def on_train_epoch_start(
        self,
        trainer: Any,
        pl_module: Any,
    ) -> None:
        """Reset per-epoch audit buffers."""
        self.epoch_groups = []
        self.epoch_batch_rows = []
        self.epoch_new_by_rank = {rank: set() for rank in RANKS}

    def on_train_batch_end(
        self,
        trainer: Any,
        pl_module: Any,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        """Update cumulative structural coverage from one training batch."""
        if self.partptr is None:
            raise RuntimeError("StructuralCoverageCallback.setup was not run.")
        if not hasattr(batch, "global_nid"):
            raise ValueError("Structural coverage requires batch.global_nid.")
        if not hasattr(batch, "edge_index"):
            raise ValueError("Structural coverage requires batch.edge_index.")

        if (
            self.audit_induced_edges
            and self.audited_batches < self.audit_max_batches
        ):
            mismatch = induced_edge_mismatch(
                full_edges=self.full_edges,
                batch_edge_index=batch.edge_index,
                batch_global_nid=batch.global_nid,
            )
            self.audited_batches += 1
            if mismatch is not None:
                raise ValueError(
                    "Cluster batch is not an induced full-graph subgraph: "
                    f"{mismatch}"
                )

        structures = extract_structures_from_batch(
            batch=batch,
            structure_family=self.structure_family,
            structure_params=self.structure_params,
        )
        structures = filter_structures_to_universe(
            structures, self.global_structures
        )

        new_by_rank: RankedStructureSet = {rank: set() for rank in RANKS}
        for rank in RANKS:
            new_by_rank[rank] = structures[rank] - self.observed[rank]
            self.observed[rank].update(structures[rank])
            self.epoch_new_by_rank[rank].update(new_by_rank[rank])

        global_nid = batch.global_nid.detach().cpu().to(torch.long).numpy()
        part_ids = tuple(
            int(part)
            for part in sorted(np.unique(part_ids_for_nodes(self.partptr, global_nid)))
        )
        self.epoch_groups.append(part_ids)
        for pair in combinations(part_ids, 2):
            self.pair_counts[pair] += 1

        if self.save_batch_events:
            observed_count = sum(len(structures[rank]) for rank in RANKS)
            new_count = sum(len(new_by_rank[rank]) for rank in RANKS)
            self.epoch_batch_rows.append(
                {
                    "epoch": int(trainer.current_epoch) + 1,
                    "batch_idx": int(batch_idx),
                    "part_ids": json.dumps(part_ids),
                    "num_nodes": int(batch.global_nid.numel()),
                    "num_edges": len(
                        edge_index_to_undirected_edges(
                            batch.edge_index,
                            batch.global_nid,
                        )
                    ),
                    "observed_global_structures": observed_count,
                    "new_global_structures": new_count,
                    "observed_rank0": len(structures[0]),
                    "observed_rank1": len(structures[1]),
                    "observed_rank2": len(structures[2]),
                    "new_rank0": len(new_by_rank[0]),
                    "new_rank1": len(new_by_rank[1]),
                    "new_rank2": len(new_by_rank[2]),
                }
            )

    def on_train_epoch_end(
        self,
        trainer: Any,
        pl_module: Any,
    ) -> None:
        """Append empirical coverage and optional batch audit rows."""
        self.completed_epochs += 1
        self.epoch_grouping_signatures.add(grouping_signature(self.epoch_groups))

        empirical_fields = ["epoch"]
        for group in GROUPS:
            empirical_fields.extend(
                [
                    f"observed_count_{group}",
                    f"total_count_{group}",
                    f"realized_coverage_{group}",
                    f"new_count_{group}",
                ]
            )
        write_csv_rows(
            self.empirical_path,
            [
                empirical_coverage_row(
                    epoch=int(trainer.current_epoch) + 1,
                    global_structures=self.global_structures,
                    observed=self.observed,
                    new_by_rank=self.epoch_new_by_rank,
                )
            ],
            empirical_fields,
            append=True,
        )
        if self.save_batch_events and self.epoch_batch_rows:
            write_csv_rows(
                self.batch_events_path,
                self.epoch_batch_rows,
                [
                    "epoch",
                    "batch_idx",
                    "part_ids",
                    "num_nodes",
                    "num_edges",
                    "observed_global_structures",
                    "new_global_structures",
                    "observed_rank0",
                    "observed_rank1",
                    "observed_rank2",
                    "new_rank0",
                    "new_rank1",
                    "new_rank2",
                ],
                append=True,
            )

    def on_fit_end(self, trainer: Any, pl_module: Any) -> None:
        """Rewrite metadata with final reshuffling audit statistics."""
        if self.partptr is not None:
            self._write_metadata(active_nodes=np.asarray([], dtype=np.int64))

    def _metadata_payload(self, active_nodes: np.ndarray) -> dict[str, Any]:
        """Build run metadata payload."""
        if self.train_part_ids is None:
            train_part_ids = np.asarray([], dtype=np.int64)
        else:
            train_part_ids = self.train_part_ids
        k_eff = int(train_part_ids.shape[0])
        all_structures = flatten_structures(self.global_structures)
        observable_count = sum(
            1 for structure, span in self.spans.items() if span <= self.q
        )
        payload = {
            "structure_node_id_space": STRUCTURE_NODE_ID_SPACE,
            "structure_family": self.structure_family,
            "structure_params": self.structure_params,
            "structure_ranks": list(RANKS),
            "q": self.q,
            "K": int(self.handle["num_parts"]),
            "K_eff": k_eff,
            "train_part_ids": train_part_ids.tolist(),
            "effective_train_batch_sizes": effective_batch_sizes(
                k_eff, self.q
            ),
            "uses_remainder_batch": bool(k_eff % self.q != 0)
            if self.q > 0
            else False,
            "partition_hash": self.handle.get("config_hash"),
            "partition_processed_dir": self.handle.get("processed_dir"),
            "memmap_dir": self.handle.get("memmap_dir"),
            "perm_to_global_path": str(
                Path(self.handle["memmap_dir"]) / "perm_to_global.npy"
            ),
            "global_to_perm_path": str(
                Path(self.handle["memmap_dir"]) / "global_to_perm.npy"
            ),
            "global_structure_count": len(all_structures),
            "global_structure_hash": _set_hash(all_structures),
            "observable_structure_count": observable_count,
            "observable_ceiling_all": (
                observable_count / len(all_structures)
                if all_structures
                else 0.0
            ),
            "active_node_count": int(active_nodes.shape[0])
            if active_nodes.size
            else self.active_node_count,
            "audit_induced_edges": self.audit_induced_edges,
            "audit_max_batches": self.audit_max_batches,
            "audited_batches": self.audited_batches,
            "require_equal_batches": self.require_equal_batches,
            "unique_epoch_groupings": len(self.epoch_grouping_signatures),
            "mean_pair_cooccurrence": compute_mean_pair_cooccurrence(
                self.pair_counts, train_part_ids, self.completed_epochs
            ),
            "expected_pair_cooccurrence": expected_pair_cooccurrence(
                self.q, k_eff
            ),
            "completed_epochs": self.completed_epochs,
            "created_at_unix": time.time(),
            "config": self.cfg_snapshot,
        }
        if self.structure_family == "cell_simple_cycles":
            payload.update(
                {
                    "max_support_nodes": int(
                        self.structure_params.get("max_support_nodes", 8)
                    ),
                    "cell_object": "simple_cycle",
                    "cell_generation_field": "F2_cycle_space_simple_cycles",
                }
            )
        return payload

    def _write_metadata(self, active_nodes: np.ndarray) -> None:
        write_json(
            self.results_dir / "run_metadata.json",
            self._metadata_payload(active_nodes),
        )


def copy_csv_logger_metrics(
    *,
    loggers: list[Any],
    destination: str | Path,
) -> str | None:
    """Copy the first Lightning CSVLogger metrics file to destination."""
    destination = Path(destination)
    for logger in loggers:
        log_dir = getattr(logger, "log_dir", None)
        if log_dir is None:
            continue
        metrics_path = Path(log_dir) / "metrics.csv"
        if metrics_path.exists():
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(metrics_path, destination)
            return str(destination)
    return None
