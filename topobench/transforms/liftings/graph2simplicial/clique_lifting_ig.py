"""Fast simplicial clique lifting using igraph backend."""

from itertools import combinations
import igraph as ig
import numpy as np
import torch
import torch_geometric

from topobench.data.utils.utils import get_complex_connectivity_from_incidences
from topobench.transforms.liftings.graph2simplicial import (
    Graph2SimplicialLifting,
)


class SimplicialCliqueLiftingIG(Graph2SimplicialLifting):
    r"""Lift graphs to simplicial complex domain (fast, igraph backend).

    Connectivity matrices are built directly as PyTorch sparse tensors. 
    It uses `python-igraph` for high-performance clique finding.

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
        num_nodes = data.num_nodes

        # --- Canonical edge ordering and igraph Construction ---
        ei = data.edge_index
        edge_set = set()
        for i in range(ei.size(1)):
            u, v = ei[0, i].item(), ei[1, i].item()
            if u != v:
                edge_set.add((min(u, v), max(u, v)))
        
        sorted_edges = sorted(edge_set)
        num_edges = len(sorted_edges)

        # Create igraph object
        graph = ig.Graph(n=num_nodes, edges=sorted_edges)
        
        # --- Find all cliques via igraph ---
        cliques = graph.maximal_cliques()

        # --- Collect simplices per rank using compact tensor representation ---
        simplices_by_rank = [None] * (self.complex_dim + 1)
        
        # Rank 0: Nodes
        simplices_by_rank[0] = torch.arange(num_nodes).view(-1, 1)
        
        # Rank 1: Edges (Sorted)
        simplices_by_rank[1] = torch.tensor(sorted_edges, dtype=torch.long)

        # Rank >= 2: Higher order simplices from cliques
        for rank in range(2, self.complex_dim + 1):
            s_set = set()
            for clique in cliques:
                clique_sorted = sorted(clique)
                if len(clique_sorted) >= rank + 1:
                    for c in combinations(clique_sorted, rank + 1):
                        s_set.add(c)
            
            if s_set:
                simplices_by_rank[rank] = torch.tensor(sorted(list(s_set)), dtype=torch.long)
            else:
                simplices_by_rank[rank] = torch.empty((0, rank + 1), dtype=torch.long)

        # --- Build incidence matrices ---
        incidences = {}

        # incidence_0
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

            num_k = k_simplices.shape[0]
            S = rank + 1
            
            # Vectorized Face Generation
            all_faces = []
            for i in range(S):
                mask = torch.ones(S, dtype=torch.bool)
                mask[i] = False
                face = k_simplices[:, mask]
                all_faces.append(face)
            
            faces_tensor = torch.cat(all_faces, dim=0)
            
            # Using numpy searchsorted for 2D lexicographical search
            target = km1_simplices.numpy()
            query = faces_tensor.numpy()
            
            def get_sort_key(arr):
                return arr.view(np.dtype((np.void, arr.dtype.itemsize * arr.shape[1])))

            target_view = get_sort_key(target)
            query_view = get_sort_key(query)
            
            row_indices = np.searchsorted(target_view.ravel(), query_view.ravel())
            
            col_indices = torch.arange(num_k).repeat(S)
            
            if self.signed:
                single_vals = torch.tensor([(-1.0)**i for i in range(S)])
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

        lifted_topology["x_0"] = data.x

        has_edge_attr = hasattr(data, "edge_attr") and data.edge_attr is not None
        if has_edge_attr and num_edges == (data.edge_index.size(1) // 2):
            lifted_topology["x_1"] = self._reorder_edge_attr(
                data, sorted_edges
            )

        return lifted_topology

    @staticmethod
    def _reorder_edge_attr(data, sorted_edges):
        ei = data.edge_index
        data_edge_map = {}
        for idx in range(ei.size(1)):
            u, v = ei[0, idx].item(), ei[1, idx].item()
            if u != v:
                key = (min(u, v), max(u, v))
                if key not in data_edge_map:
                    data_edge_map[key] = idx

        reorder_indices = [data_edge_map[e] for e in sorted_edges]
        return data.edge_attr[reorder_indices]
