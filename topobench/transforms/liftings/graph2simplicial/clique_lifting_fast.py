"""Fast simplicial clique lifting bypassing toponetx object construction."""

from itertools import combinations
from typing import Any

import networkx as nx
import numpy as np
import torch
import torch_geometric

from topobench.data.utils.utils import get_complex_connectivity_from_incidences
from topobench.transforms.liftings.graph2simplicial import (
    Graph2SimplicialLifting,
)


class SimplicialCliqueLiftingFast(Graph2SimplicialLifting):
    r"""Lift graphs to simplicial complex domain (fast, bypasses toponetx).

    The algorithm creates simplices by identifying the cliques and considering
    them as simplices of the same dimension. Connectivity matrices are built
    directly as PyTorch sparse tensors, avoiding the slow toponetx
    SimplicialComplex construction.

    Parameters
    ----------
    **kwargs : optional
        Additional arguments for the class.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def lift_topology(self, data: torch_geometric.data.Data) -> dict:
        r"""Lift the topology of a graph to a simplicial complex.

        Parameters
        ----------
        data : torch_geometric.data.Data
            The input data to be lifted.

        Returns
        -------
        dict
            The lifted topology.
        """
        graph = self._generate_graph_from_data(data)
        num_nodes = graph.number_of_nodes()

        # --- Find all cliques ---
        # igraph is much faster for large graphs if available, but staying with networkx for now
        # to focus on the memory optimization of the storage.
        cliques = list(nx.find_cliques(graph))

        # --- Collect simplices per rank using compact tensor representation ---
        simplices_by_rank = [None] * (self.complex_dim + 1)
        
        # Rank 0: Nodes
        simplices_by_rank[0] = torch.arange(num_nodes).view(-1, 1)
        
        # Rank 1: Edges (Sorted)
        # Directly from graph to avoid redundant combinations
        edges = sorted([tuple(sorted(e)) for e in graph.edges()])
        simplices_by_rank[1] = torch.tensor(edges, dtype=torch.long)

        # Rank >= 2: Higher order simplices
        for rank in range(2, self.complex_dim + 1):
            s_set = set()
            for clique in cliques:
                if len(clique) >= rank + 1:
                    for c in combinations(sorted(clique), rank + 1):
                        s_set.add(c)
            
            if s_set:
                # Convert to sorted tensor for fast indexing later
                simplices_by_rank[rank] = torch.tensor(sorted(list(s_set)), dtype=torch.long)
            else:
                simplices_by_rank[rank] = torch.empty((0, rank + 1), dtype=torch.long)

        # --- Build incidence matrices ---
        incidences = {}

        # incidence_0: (1 x num_nodes)
        incidences[0] = torch.sparse_coo_tensor(
            torch.stack([
                torch.zeros(num_nodes, dtype=torch.long),
                torch.arange(num_nodes, dtype=torch.long)
            ]),
            torch.ones(num_nodes, dtype=torch.float),
            size=(1, num_nodes)
        ).coalesce()

        # incidence_k for k >= 1
        for rank in range(1, self.complex_dim + 1):
            k_simplices = simplices_by_rank[rank]
            km1_simplices = simplices_by_rank[rank - 1]
            
            if k_simplices.shape[0] == 0:
                incidences[rank] = torch.sparse_coo_tensor(
                    size=(km1_simplices.shape[0], 0)
                ).coalesce()
                continue

            # Vectorized Face Generation
            # For each k-simplex (v0, v1, ..., vk), its faces are km1-simplices.
            # There are k+1 faces per simplex.
            num_k = k_simplices.shape[0]
            S = rank + 1
            
            # Generate all faces for all k-simplices in one go
            # k_simplices is (num_k, S)
            # repeating num_k rows, each S times
            all_faces = []
            for i in range(S):
                # Remove i-th vertex
                mask = torch.ones(S, dtype=torch.bool)
                mask[i] = False
                face = k_simplices[:, mask]
                all_faces.append(face)
            
            # faces_tensor: (S * num_k, rank)
            faces_tensor = torch.cat(all_faces, dim=0)
            
            # Now find the index of each face in km1_simplices.
            # Since km1_simplices is SORTED, we can use a lexicographical binary search.
            # Convert tuples to a single value if possible, or use numpy searchsorted.
            
            # Using numpy searchsorted for 2D lexicographical search (very efficient)
            target = km1_simplices.numpy()
            query = faces_tensor.numpy()
            
            # Lexicographical sort key for 2D array
            def get_sort_key(arr):
                # We can use a void view for fast comparison of rows
                return arr.view(np.dtype((np.void, arr.dtype.itemsize * arr.shape[1])))

            target_view = get_sort_key(target)
            query_view = get_sort_key(query)
            
            # target is already sorted because we used sorted() when building it
            row_indices = np.searchsorted(target_view.ravel(), query_view.ravel())
            
            # row_indices: (S * num_k)
            # col_indices: (0, 1, ..., num_k-1) repeated S times
            # Note: repeat gives [0...N, 0...N, ...] which matches cat([face0...faceS])
            col_indices = torch.arange(num_k).repeat(S)
            
            # Boundary values: face index k (removing k-th vertex) gets sign (-1)^k
            if self.signed:
                # signs: 1, -1, 1, -1... for S faces
                single_vals = torch.tensor([(-1.0)**i for i in range(S)])
                # Order of vals must match cat([face0, face1...])
                # so face0 needs its sign, face1 needs its sign...
                vals = single_vals.repeat_interleave(num_k)
            else:
                vals = torch.ones(S * num_k)

            incidences[rank] = torch.sparse_coo_tensor(
                torch.stack([torch.from_numpy(row_indices.astype(np.int64)), col_indices]),
                vals,
                size=(km1_simplices.shape[0], k_simplices.shape[0]),
            ).coalesce()

        # --- Compute connectivity ---
        shape = [s.shape[0] for s in simplices_by_rank]
        lifted_topology = get_complex_connectivity_from_incidences(
            incidences,
            shape,
            self.complex_dim,
            neighborhoods=self.neighborhoods,
            signed=self.signed,
        )

        # --- Features ---
        lifted_topology["x_0"] = data.x

        # Preserve edge attributes if applicable
        if self.contains_edge_attr and simplices_by_rank[1].shape[0] == graph.number_of_edges():
            # Reorder edge_attr to match our canonical sorted edge ordering
            # simplices_by_rank[1] is a tensor (num_edges, 2)
            edge_list = [tuple(e.tolist()) for e in simplices_by_rank[1]]
            lifted_topology["x_1"] = self._reorder_edge_attr(
                data, graph, edge_list
            )

        return lifted_topology

    @staticmethod
    def _reorder_edge_attr(data, graph, sorted_edges):
        """Reorder edge attributes to match canonical sorted edge ordering.

        Parameters
        ----------
        data : torch_geometric.data.Data
            The input data with edge_attr.
        graph : nx.Graph
            The NetworkX graph.
        sorted_edges : list
            Canonically sorted edge list.

        Returns
        -------
        torch.Tensor
            Reordered edge attributes.
        """
        # Build map from edge pair -> index in data.edge_index
        ei = data.edge_index
        data_edge_map = {}
        for idx in range(ei.size(1)):
            u, v = ei[0, idx].item(), ei[1, idx].item()
            key = (min(u, v), max(u, v))
            if key not in data_edge_map:
                data_edge_map[key] = idx

        # Reorder
        reorder_indices = [data_edge_map[e] for e in sorted_edges]
        return data.edge_attr[reorder_indices]
