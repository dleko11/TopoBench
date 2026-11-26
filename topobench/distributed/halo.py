"""Halo-augmented partitioned clique lifting with unique cell ownership.

Each simulated worker owns the nodes a partition assigns to it (its core). It
lifts the subgraph induced on its core and one-hop halo, then keeps the cells
whose minimum vertex is a core node. Every clique lies in the one-hop
neighbourhood of its minimum vertex, so the cells owned across workers
partition the global clique complex exactly, while no worker builds it.

Faces of owned cells that another worker owns are *ghost faces*. Their
owner-side index is resolved once at setup by vertex-tuple key, which stands
in for a one-time index request to the owner.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import igraph as ig
import networkx as nx
import torch
from torch_geometric.data import Data
from torch_geometric.loader import ClusterData
from torch_geometric.utils import coalesce, remove_self_loops, to_undirected

from topobench.transforms.liftings.graph2simplicial._clique_utils import (
    build_clique_complex_incidences,
)

_INT64_MAX = 2**63 - 1


def canonical_edges(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """Return unique undirected edges as ``(2, m)`` with ``src < dst``.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge index of any orientation, possibly with duplicates.
    num_nodes : int
        Number of nodes.

    Returns
    -------
    torch.Tensor
        Canonical edge index sorted lexicographically.
    """
    edge_index, _ = remove_self_loops(edge_index)
    edge_index = coalesce(
        to_undirected(edge_index, num_nodes=num_nodes), num_nodes=num_nodes
    )
    return edge_index[:, edge_index[0] < edge_index[1]]


def encode_simplices(simplices: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """Encode sorted vertex tuples as order-preserving int64 keys.

    Parameters
    ----------
    simplices : torch.Tensor
        Simplices of one rank as ``(n, k + 1)`` sorted vertex tuples.
    num_nodes : int
        Number of graph nodes, used as the radix.

    Returns
    -------
    torch.Tensor
        Keys whose order matches the lexicographic order of the tuples.
    """
    width = simplices.size(1)
    if num_nodes**width > _INT64_MAX:
        raise ValueError(
            f"Cannot encode rank-{width - 1} simplices of a {num_nodes}-node "
            "graph in int64."
        )
    keys = torch.zeros(simplices.size(0), dtype=torch.long)
    for column in range(width):
        keys = keys * num_nodes + simplices[:, column]
    return keys


def drop_one_faces(cells: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """List the codimension-one faces of each cell.

    Parameters
    ----------
    cells : torch.Tensor
        Cells as ``(n, k + 1)`` sorted vertex tuples, with ``k >= 1``.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        Faces as ``((k + 1) n, k)`` sorted tuples and the index of the cell
        each face belongs to.
    """
    width = cells.size(1)
    faces = [
        cells[:, [col for col in range(width) if col != drop]]
        for drop in range(width)
    ]
    owners = torch.arange(cells.size(0)).repeat(width)
    return torch.cat(faces), owners


def vertex_weights(
    edge_index: torch.Tensor, num_nodes: int, scheme: str = "cells"
) -> torch.Tensor:
    """Estimate per-node lifted work, to balance METIS by cells, not nodes.

    Triangle counts come from local clustering coefficients, so no triangle
    list is built.

    Parameters
    ----------
    edge_index : torch.Tensor
        Graph edges.
    num_nodes : int
        Number of nodes.
    scheme : {"triangles", "cells"}, optional
        ``"triangles"``: ``1 + t(v)``. ``"cells"``: every node, edge and
        triangle split evenly over its vertices, times 6 to stay integral:
        ``6 + 3 deg(v) + 2 t(v)``.

    Returns
    -------
    torch.Tensor
        Integer weight of every node.
    """
    edges = canonical_edges(edge_index, num_nodes)
    graph = ig.Graph(n=num_nodes, edges=edges.t().tolist())
    degree = torch.tensor(graph.degree(), dtype=torch.long)
    clustering = torch.tensor(
        graph.transitivity_local_undirected(mode="zero"), dtype=torch.float64
    )
    triangles = (clustering * degree * (degree - 1) / 2).round().long()
    if scheme == "triangles":
        return 1 + triangles
    if scheme == "cells":
        return 6 + 3 * degree + 2 * triangles
    raise ValueError(f"Unknown weight scheme {scheme!r}.")


def partition_nodes(
    edge_index: torch.Tensor,
    num_nodes: int,
    num_parts: int,
    method: str = "metis",
    seed: int = 0,
    node_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """Assign each node to a worker.

    Parameters
    ----------
    edge_index : torch.Tensor
        Graph edges.
    num_nodes : int
        Number of nodes.
    num_parts : int
        Number of workers.
    method : {"metis", "random"}, optional
        METIS (as in Cluster-TNN), or a balanced random assignment used as a
        control.
    seed : int, optional
        Seed for the random assignment.
    node_weight : torch.Tensor, optional
        Integer METIS vertex weights. Without them METIS balances node
        counts, as PyG ``ClusterData`` does.

    Returns
    -------
    torch.Tensor
        Worker id of every node.
    """
    if num_parts < 1:
        raise ValueError(f"num_parts must be positive, got {num_parts}.")
    if num_parts == 1:
        return torch.zeros(num_nodes, dtype=torch.long)
    if method == "random":
        generator = torch.Generator().manual_seed(seed)
        order = torch.randperm(num_nodes, generator=generator)
        part = torch.empty(num_nodes, dtype=torch.long)
        part[order] = torch.arange(num_nodes) % num_parts
        return part
    if method != "metis":
        raise ValueError(f"Unknown partition method {method!r}.")

    undirected = to_undirected(edge_index, num_nodes=num_nodes)
    if node_weight is None:
        cluster = ClusterData(
            Data(edge_index=undirected, num_nodes=num_nodes),
            num_parts=num_parts,
            recursive=False,
            log=False,
        )
        partptr = cluster.partition.partptr
        node_perm = cluster.partition.node_perm
    else:
        from torch_sparse import SparseTensor
        from torch_sparse.metis import partition as metis_partition

        adjacency = SparseTensor.from_edge_index(
            undirected, sparse_sizes=(num_nodes, num_nodes)
        )
        _, partptr, node_perm = metis_partition(
            adjacency, num_parts, node_weight=node_weight.long()
        )
    part = torch.empty(num_nodes, dtype=torch.long)
    for worker in range(num_parts):
        part[node_perm[partptr[worker] : partptr[worker + 1]]] = worker
    return part


def lift_clique_complex(
    edges: torch.Tensor, nodes: torch.Tensor, complex_dim: int
) -> tuple[list[torch.Tensor], float]:
    """Lift the subgraph induced on ``nodes`` with TopoBench's clique code.

    Parameters
    ----------
    edges : torch.Tensor
        Canonical global edges.
    nodes : torch.Tensor
        Sorted global ids of the nodes to keep.
    complex_dim : int
        Maximum simplex rank.

    Returns
    -------
    tuple[list[torch.Tensor], float]
        Simplices per rank as sorted global vertex tuples, and the wall time
        spent building the graph and lifting it.
    """
    num_nodes = int(edges.max()) + 1 if edges.numel() else 0
    num_nodes = max(num_nodes, int(nodes.max()) + 1 if nodes.numel() else 0)
    keep = torch.zeros(num_nodes, dtype=torch.bool)
    keep[nodes] = True
    induced = edges[:, keep[edges[0]] & keep[edges[1]]]
    to_local = torch.full((num_nodes,), -1, dtype=torch.long)
    to_local[nodes] = torch.arange(nodes.numel())

    start = time.perf_counter()
    graph = nx.Graph()
    graph.add_nodes_from(range(nodes.numel()))
    graph.add_edges_from(to_local[induced].t().tolist())
    _, simplices, _, _ = build_clique_complex_incidences(
        graph,
        complex_dim=complex_dim,
        signed=False,
        num_nodes=nodes.numel(),
    )
    seconds = time.perf_counter() - start
    # ``nodes`` is sorted, so mapping local ids back keeps tuples sorted.
    return [nodes[rank_simplices] for rank_simplices in simplices], seconds


@dataclass
class WorkerShard:
    """Cells one worker lifted and owns.

    Attributes
    ----------
    worker : int
        Worker id.
    core : torch.Tensor
        Sorted global ids of the nodes the partition assigns to the worker.
    halo : torch.Tensor
        Sorted global ids of one-hop neighbours outside the core.
    owned : list[torch.Tensor]
        Owned cells per rank: cells whose minimum vertex is in the core.
    lifted_counts : list[int]
        Cells per rank in the worker's full local lifting, before filtering
        by ownership.
    lift_seconds : float
        Wall time of the local lifting.
    """

    worker: int
    core: torch.Tensor
    halo: torch.Tensor
    owned: list[torch.Tensor]
    lifted_counts: list[int]
    lift_seconds: float


def build_shard(
    worker: int,
    edges: torch.Tensor,
    part: torch.Tensor,
    complex_dim: int,
) -> WorkerShard:
    """Lift a worker's core plus one-hop halo and keep the cells it owns.

    Parameters
    ----------
    worker : int
        Worker id.
    edges : torch.Tensor
        Canonical global edges.
    part : torch.Tensor
        Worker id of every node.
    complex_dim : int
        Maximum simplex rank.

    Returns
    -------
    WorkerShard
        The worker's lifted and owned cells.
    """
    in_core = part == worker
    src, dst = edges
    touches_core = in_core[src] | in_core[dst]
    in_local = in_core.clone()
    in_local[src[touches_core]] = True
    in_local[dst[touches_core]] = True
    core = in_core.nonzero().flatten()
    halo = (in_local & ~in_core).nonzero().flatten()

    if core.numel() == 0:
        empty = [
            torch.empty((0, rank + 1), dtype=torch.long)
            for rank in range(complex_dim + 1)
        ]
        return WorkerShard(
            worker, core, halo, empty, [0] * (complex_dim + 1), 0.0
        )

    simplices, seconds = lift_clique_complex(
        edges, in_local.nonzero().flatten(), complex_dim
    )
    owned = [cells[part[cells[:, 0]] == worker] for cells in simplices]
    return WorkerShard(
        worker=worker,
        core=core,
        halo=halo,
        owned=owned,
        lifted_counts=[cells.size(0) for cells in simplices],
        lift_seconds=seconds,
    )


@dataclass
class IncidencePlan:
    """Per-worker pieces of one unsigned boundary matrix ``B_k``.

    Rows of ``B_k`` are rank ``k - 1`` faces and columns are rank ``k``
    cofaces. Worker ``p`` stores the columns of its owned cofaces. Its local
    row space is its owned faces followed by its ghost faces.

    Attributes
    ----------
    rank : int
        Coface rank ``k``.
    local : list[torch.Tensor]
        Sparse ``(n_owned_faces + n_ghosts, n_owned_cofaces)`` matrices.
    ghosts : list[torch.Tensor]
        Owner-major global indices of each worker's ghost faces.
    face_offsets : torch.Tensor
        Owner-major offsets of rank ``k - 1`` cells, length ``P + 1``.
    coface_offsets : torch.Tensor
        Owner-major offsets of rank ``k`` cells, length ``P + 1``.
    ghost_pairs : int
        Number of (worker, owner) pairs that exchange ghost values.
    """

    rank: int
    local: list[torch.Tensor]
    ghosts: list[torch.Tensor]
    face_offsets: torch.Tensor
    coface_offsets: torch.Tensor
    ghost_pairs: int


@dataclass
class HaloPlan:
    """Owner-partitioned simplicial complex built without global lifting.

    Attributes
    ----------
    num_nodes : int
        Number of graph nodes.
    num_parts : int
        Number of workers.
    complex_dim : int
        Maximum simplex rank.
    part : torch.Tensor
        Worker id of every node.
    shards : list[WorkerShard]
        Per-worker lifted and owned cells.
    offsets : list[torch.Tensor]
        Owner-major offsets per rank, each of length ``P + 1``.
    incidences : dict[int, IncidencePlan]
        Distributed boundary matrices for ranks ``1..complex_dim``.
    resolve_seconds : float
        Wall time spent resolving ghost faces.
    """

    num_nodes: int
    num_parts: int
    complex_dim: int
    part: torch.Tensor
    shards: list[WorkerShard]
    offsets: list[torch.Tensor]
    incidences: dict[int, IncidencePlan]
    resolve_seconds: float

    def cells(self, rank: int) -> torch.Tensor:
        """Return all owned cells of a rank in owner-major order.

        Parameters
        ----------
        rank : int
            Simplex rank.

        Returns
        -------
        torch.Tensor
            Concatenated owned cells of every worker.
        """
        return torch.cat([shard.owned[rank] for shard in self.shards])

    def num_cells(self, rank: int) -> int:
        """Return the number of cells of a rank across all workers.

        Parameters
        ----------
        rank : int
            Simplex rank.

        Returns
        -------
        int
            Total owned cells of that rank.
        """
        return int(self.offsets[rank][-1])


def _resolve_at_owners(
    faces: torch.Tensor,
    part: torch.Tensor,
    owned_keys: list[torch.Tensor],
    offsets: torch.Tensor,
    num_nodes: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Find each face's owner-major index by looking it up at its owner.

    Parameters
    ----------
    faces : torch.Tensor
        Faces as sorted vertex tuples.
    part : torch.Tensor
        Worker id of every node.
    owned_keys : list[torch.Tensor]
        Sorted keys of each worker's owned cells of the face rank.
    offsets : torch.Tensor
        Owner-major offsets of the face rank.
    num_nodes : int
        Number of graph nodes.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        Owner-major index and owner of every face.

    Raises
    ------
    RuntimeError
        If a face is missing at its owner, i.e. the halo was insufficient.
    """
    owners = part[faces[:, 0]]
    keys = encode_simplices(faces, num_nodes)
    index = torch.empty(faces.size(0), dtype=torch.long)
    for owner in owners.unique().tolist():
        selected = owners == owner
        query = keys[selected]
        table = owned_keys[owner]
        position = torch.searchsorted(table, query)
        clamped = position.clamp(max=max(table.numel() - 1, 0))
        if table.numel() == 0 or not torch.equal(table[clamped], query):
            raise RuntimeError(
                f"A face is missing at its owner (worker {owner}); the halo "
                "does not cover every face of the owned cells."
            )
        index[selected] = offsets[owner] + position
    return index, owners


def build_halo_plan(
    edge_index: torch.Tensor,
    num_nodes: int,
    part: torch.Tensor,
    complex_dim: int = 2,
) -> HaloPlan:
    """Lift every worker locally and wire up the distributed incidences.

    Parameters
    ----------
    edge_index : torch.Tensor
        Graph edges.
    num_nodes : int
        Number of nodes.
    part : torch.Tensor
        Worker id of every node.
    complex_dim : int, optional
        Maximum simplex rank.

    Returns
    -------
    HaloPlan
        Owner-partitioned complex with per-worker boundary matrices.
    """
    edges = canonical_edges(edge_index, num_nodes)
    num_parts = int(part.max()) + 1
    shards = [
        build_shard(worker, edges, part, complex_dim)
        for worker in range(num_parts)
    ]

    offsets = []
    keys = []
    for rank in range(complex_dim + 1):
        counts = torch.tensor([shard.owned[rank].size(0) for shard in shards])
        offsets.append(
            torch.cat([torch.zeros(1, dtype=torch.long), counts.cumsum(0)])
        )
        rank_keys = [
            encode_simplices(shard.owned[rank], num_nodes) for shard in shards
        ]
        for worker_keys in rank_keys:
            if worker_keys.numel() > 1 and not bool(
                (worker_keys[1:] > worker_keys[:-1]).all()
            ):
                raise RuntimeError("Owned cells must be sorted and unique.")
        keys.append(rank_keys)

    start = time.perf_counter()
    incidences = {}
    for rank in range(1, complex_dim + 1):
        local, ghosts, ghost_pairs = [], [], 0
        face_offsets = offsets[rank - 1]
        for worker, shard in enumerate(shards):
            cofaces = shard.owned[rank]
            faces, columns = drop_one_faces(cofaces)
            index, owners = _resolve_at_owners(
                faces, part, keys[rank - 1], face_offsets, num_nodes
            )
            is_ghost = owners != worker
            ghost = index[is_ghost].unique()
            num_owned_faces = int(
                face_offsets[worker + 1] - face_offsets[worker]
            )
            rows = index - face_offsets[worker]
            rows[is_ghost] = num_owned_faces + torch.searchsorted(
                ghost, index[is_ghost]
            )
            local.append(
                torch.sparse_coo_tensor(
                    torch.stack([rows, columns]),
                    torch.ones(rows.numel()),
                    size=(num_owned_faces + ghost.numel(), cofaces.size(0)),
                ).coalesce()
            )
            ghosts.append(ghost)
            ghost_pairs += owners[is_ghost].unique().numel()
        incidences[rank] = IncidencePlan(
            rank=rank,
            local=local,
            ghosts=ghosts,
            face_offsets=face_offsets,
            coface_offsets=offsets[rank],
            ghost_pairs=ghost_pairs,
        )

    return HaloPlan(
        num_nodes=num_nodes,
        num_parts=num_parts,
        complex_dim=complex_dim,
        part=part,
        shards=shards,
        offsets=offsets,
        incidences=incidences,
        resolve_seconds=time.perf_counter() - start,
    )


def lift_input_features(
    plan: HaloPlan, x_0: torch.Tensor, chunk_size: int = 8192
) -> list[torch.Tensor]:
    """Lift node features to owned cells, worker by worker.

    Matches TopoBench's ``ProjectionSum`` lifting. Applying
    ``x_k = |B_k|^T x_{k-1}`` from rank 0 gives
    ``x_k(sigma) = k! * sum of x_0 over the vertices of sigma``, so each worker
    only needs the features of its core and halo nodes. Like
    ``ProjectionSum``, the sums are computed in float32; they are exact for
    integer-valued features.

    Parameters
    ----------
    plan : HaloPlan
        Owner-partitioned complex.
    x_0 : torch.Tensor
        Node features of the whole graph; each worker reads only its rows.
    chunk_size : int, optional
        Rows processed at a time, to bound temporary memory.

    Returns
    -------
    list[torch.Tensor]
        Owner-major float32 features per rank.
    """
    x_0 = x_0.float()
    lifted = [[] for _ in range(plan.complex_dim + 1)]
    for shard in plan.shards:
        local = torch.zeros(plan.num_nodes, dtype=torch.bool)
        local[shard.core] = True
        local[shard.halo] = True
        for rank, cells in enumerate(shard.owned):
            if not bool(local[cells].all()):
                raise RuntimeError(
                    f"Worker {shard.worker} needs features outside its halo."
                )
            out = torch.empty((cells.size(0), x_0.size(1)), dtype=x_0.dtype)
            for start in range(0, cells.size(0), chunk_size):
                block = cells[start : start + chunk_size]
                values = x_0[block[:, 0]].clone()
                for column in range(1, block.size(1)):
                    values += x_0[block[:, column]]
                out[start : start + chunk_size] = values * math.factorial(rank)
            lifted[rank].append(out)
    return [torch.cat(blocks) for blocks in lifted]


def plan_statistics(plan: HaloPlan) -> dict[str, float]:
    """Summarize replication, ownership balance, and ghost counts.

    Parameters
    ----------
    plan : HaloPlan
        Owner-partitioned complex.

    Returns
    -------
    dict[str, float]
        Flat statistics, suitable for one row of a results table.
    """
    shards = [shard for shard in plan.shards if shard.core.numel() > 0]
    core = torch.tensor(
        [shard.core.numel() for shard in shards], dtype=torch.float
    )
    halo = torch.tensor(
        [shard.halo.numel() for shard in shards], dtype=torch.float
    )
    stats = {
        "workers": plan.num_parts,
        "empty_workers": plan.num_parts - len(shards),
        "halo_ratio_mean": float((halo / core).mean()),
        "halo_ratio_max": float((halo / core).max()),
        "local_nodes_max_frac": float((core + halo).max() / plan.num_nodes),
        "lift_seconds_max": max(shard.lift_seconds for shard in shards),
        "lift_seconds_sum": sum(shard.lift_seconds for shard in shards),
        "resolve_seconds": plan.resolve_seconds,
    }
    held_all = torch.zeros(plan.num_parts)
    for rank in range(plan.complex_dim + 1):
        total = plan.num_cells(rank)
        owned = torch.tensor(
            [shard.owned[rank].size(0) for shard in shards], dtype=torch.float
        )
        lifted = sum(shard.lifted_counts[rank] for shard in shards)
        ghosts = (
            torch.tensor(
                [g.numel() for g in plan.incidences[rank + 1].ghosts],
                dtype=torch.float,
            )
            if rank + 1 in plan.incidences
            else torch.zeros(plan.num_parts)
        )
        held = (
            torch.tensor(
                [shard.owned[rank].size(0) for shard in plan.shards],
                dtype=torch.float,
            )
            + ghosts
        )
        stats[f"cells_{rank}"] = total
        stats[f"lifted_{rank}_max"] = max(
            shard.lifted_counts[rank] for shard in shards
        )
        stats[f"held_{rank}_max"] = float(held.max())
        stats[f"owned_{rank}_max"] = float(owned.max())
        stats[f"owned_{rank}_imbalance"] = float(owned.max() / owned.mean())
        stats[f"lift_duplication_{rank}"] = lifted / max(total, 1)
        stats[f"ghosts_{rank}_total"] = float(ghosts.sum())
        stats[f"held_{rank}_max_frac"] = float(held.max() / max(total, 1))
        held_all += held
    # Per-worker memory scales with all cells it holds, across ranks.
    total_cells = sum(plan.num_cells(r) for r in range(plan.complex_dim + 1))
    stats["held_total_max_frac"] = float(held_all.max() / total_cells)
    stats["held_total_mean_frac"] = float(held_all.mean() / total_cells)
    return stats


def cluster_batch_counts(
    edge_index: torch.Tensor,
    num_nodes: int,
    part: torch.Tensor,
    q: int,
    num_batches: int,
    complex_dim: int = 2,
    seed: int = 0,
) -> list[list[int]]:
    """Count cells in Cluster-TNN mini-batches of ``q`` random clusters.

    Parameters
    ----------
    edge_index : torch.Tensor
        Graph edges.
    num_nodes : int
        Number of nodes.
    part : torch.Tensor
        Cluster id of every node.
    q : int
        Clusters per mini-batch.
    num_batches : int
        Number of random mini-batches to sample.
    complex_dim : int, optional
        Maximum simplex rank.
    seed : int, optional
        Sampling seed.

    Returns
    -------
    list[list[int]]
        Cells per rank for each sampled mini-batch.
    """
    edges = canonical_edges(edge_index, num_nodes)
    num_parts = int(part.max()) + 1
    generator = torch.Generator().manual_seed(seed)
    counts = []
    for _ in range(num_batches):
        chosen = torch.randperm(num_parts, generator=generator)[:q]
        nodes = torch.isin(part, chosen).nonzero().flatten()
        simplices, _ = lift_clique_complex(edges, nodes, complex_dim)
        counts.append([cells.size(0) for cells in simplices])
    return counts
