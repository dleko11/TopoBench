"""Fast cell cycle lifting using igraph backend."""

import igraph as ig
import numpy as np
import torch
import torch_geometric

from topobench.data.utils.utils import get_complex_connectivity_from_incidences
from topobench.transforms.liftings.graph2cell.base import (
    Graph2CellLifting,
)


class CellCycleLiftingIG(Graph2CellLifting):
    r"""Lift graphs to cell complexes (fast, igraph backend).

    Connectivity matrices are built directly as PyTorch sparse tensors. 
    It uses `python-igraph` for high-performance cycle finding.

    Parameters
    ----------
    max_cell_length : int, optional
        The maximum length of the cycles to be lifted. Default is None.
    **kwargs : optional
        Additional arguments for the class.
    """

    def __init__(self, max_cell_length=None, **kwargs):
        super().__init__(**kwargs)
        self.complex_dim = 2
        self.max_cell_length = max_cell_length

    def lift_topology(self, data: torch_geometric.data.Data) -> dict:
        r"""Find the cycles of a graph and lift them to 2-cells.

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
        
        # --- Find fundamental cycles via igraph ---
        cycle_edge_indices = graph.fundamental_cycles()

        # Filter by max_cell_length if needed
        if self.max_cell_length is not None:
            cycle_edge_indices = [c for c in cycle_edge_indices if len(c) <= self.max_cell_length]

        num_cells = len(cycle_edge_indices)

        # --- Build incidence matrices ---
        incidences = {}

        # incidence_0
        incidences[0] = torch.sparse_coo_tensor(
            size=(0, num_nodes)
        ).coalesce()

        # incidence_1
        edge_tensor = torch.tensor(sorted_edges, dtype=torch.long)
        num_k1 = edge_tensor.shape[0]
        rows_1 = edge_tensor.t().contiguous().view(-1)
        cols_1 = torch.arange(num_k1).repeat_interleave(2)
        vals_1 = torch.tensor([-1.0, 1.0]).repeat(num_k1)

        incidences[1] = torch.sparse_coo_tensor(
            torch.stack([rows_1, cols_1]),
            vals_1,
            size=(num_nodes, num_edges),
        ).coalesce()

        # incidence_2: (num_edges x num_cells)
        if num_cells > 0:
            # We need to traverse each cycle to assign orientations.
            # While fundamental_cycles gives edge IDs, they aren't necessarily ordered.
            rows_2, cols_2, vals_2 = [], [], []
            
            # Pre-calculate adjacency for fast traversal
            adj = [[] for _ in range(num_nodes)]
            for e_idx, (u, v) in enumerate(sorted_edges):
                adj[u].append((v, e_idx))
                adj[v].append((u, e_idx))

            for cell_idx, edge_ids in enumerate(cycle_edge_indices):
                if len(edge_ids) < 3: continue
                
                curr_edge_set = set(edge_ids)
                # Find a starting node that has edges in this cycle
                e_start_idx = edge_ids[0]
                u_start, v_start = sorted_edges[e_start_idx]
                
                # Traverse the cycle
                curr_v = v_start
                prev_e = e_start_idx
                
                # First edge orientation
                rows_2.append(e_start_idx)
                cols_2.append(cell_idx)
                vals_2.append(1.0 if u_start < v_start else -1.0)
                
                for _ in range(len(edge_ids) - 1):
                    next_e = -1
                    next_v = -1
                    for neighbor, e_idx in adj[curr_v]:
                        if e_idx != prev_e and e_idx in curr_edge_set:
                            next_e = e_idx
                            next_v = neighbor
                            break
                    if next_e == -1: break
                    
                    rows_2.append(next_e)
                    cols_2.append(cell_idx)
                    vals_2.append(1.0 if curr_v < next_v else -1.0)
                    
                    prev_e = next_e
                    curr_v = next_v

            incidences[2] = torch.sparse_coo_tensor(
                torch.stack([torch.tensor(rows_2, dtype=torch.long), torch.tensor(cols_2, dtype=torch.long)]),
                torch.tensor(vals_2, dtype=torch.float),
                size=(num_edges, num_cells),
            ).coalesce()
        else:
            incidences[2] = torch.sparse_coo_tensor(
                size=(num_edges, 0)
            ).coalesce()

        # --- Compute connectivity ---
        shape = [num_nodes, num_edges, num_cells]
        lifted_topology = get_complex_connectivity_from_incidences(
            incidences,
            shape,
            self.complex_dim,
            neighborhoods=self.neighborhoods,
            signed=False,
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
