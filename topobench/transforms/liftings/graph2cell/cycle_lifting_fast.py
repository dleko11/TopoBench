"""Fast cell cycle lifting bypassing toponetx object construction."""

import networkx as nx
import numpy as np
import torch
import torch_geometric

from topobench.data.utils.utils import get_complex_connectivity_from_incidences
from topobench.transforms.liftings.graph2cell.base import (
    Graph2CellLifting,
)


class CellCycleLiftingFast(Graph2CellLifting):
    r"""Lift graphs to cell complexes (fast, bypasses toponetx).

    The algorithm creates 2-cells by identifying the cycles and considering
    them as 2-cells. Connectivity matrices are built directly as PyTorch
    sparse tensors, avoiding the slow toponetx CellComplex construction.

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
        graph = self._generate_graph_from_data(data)
        num_nodes = graph.number_of_nodes()

        # --- Canonical edge ordering ---
        sorted_edges = sorted(graph.edges())
        edge_map = {e: i for i, e in enumerate(sorted_edges)}
        num_edges = len(sorted_edges)

        # --- Find cycles ---
        cycles = nx.cycle_basis(graph)

        # Filter self-loops
        cycles = [c for c in cycles if len(c) != 1]

        # Filter by max_cell_length
        if self.max_cell_length is not None:
            cycles = [c for c in cycles if len(c) <= self.max_cell_length]

        num_cells = len(cycles)

        # --- Build incidence matrices ---
        incidences = {}

        # incidence_0: placeholder (0 x num_nodes zero)
        incidences[0] = torch.sparse_coo_tensor(
            size=(0, num_nodes)
        ).coalesce()

        # incidence_1: (num_nodes x num_edges)
        # Vectorized construction
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
            # Flatten all cycles and track their lengths/offsets
            cycle_nodes = []
            cycle_ptr = [0]
            for cycle in cycles:
                cycle_nodes.extend(cycle)
                cycle_ptr.append(cycle_ptr[-1] + len(cycle))
            
            nodes_flat = torch.tensor(cycle_nodes, dtype=torch.long)
            ptr = torch.tensor(cycle_ptr, dtype=torch.long)
            
            # For each cycle [v0, v1, ..., vL-1], the edges are (v0, v1), (v1, v2), ..., (vL-1, v0)
            u = nodes_flat # v0, v1, ..., vL-1
            # v calculation: shift nodes_flat within each cycle range
            v = torch.empty_like(nodes_flat)
            for i in range(num_cells):
                start, end = ptr[i], ptr[i+1]
                v[start : end-1] = nodes_flat[start+1 : end]
                v[end-1] = nodes_flat[start]
            
            # Canonical edges for lookup
            e_min = torch.minimum(u, v)
            e_max = torch.maximum(u, v)
            query = torch.stack([e_min, e_max], dim=1).numpy()
            
            # Target edges (Rank 1)
            target = edge_tensor.numpy()
            
            def get_sort_key(arr):
                return arr.view(np.dtype((np.void, arr.dtype.itemsize * arr.shape[1])))
            
            target_view = get_sort_key(target)
            query_view = get_sort_key(query)
            
            row_indices = np.searchsorted(target_view.ravel(), query_view.ravel())
            
            # Column indices: repeat each cell_idx L times
            cell_lengths = ptr[1:] - ptr[:-1]
            col_indices = torch.repeat_interleave(torch.arange(num_cells), cell_lengths)
            
            # Orientation: +1 if u < v (matches canonical order), -1 otherwise
            vals_2 = torch.where(u < v, torch.ones_like(u, dtype=torch.float), -torch.ones_like(u, dtype=torch.float))
            
            incidences[2] = torch.sparse_coo_tensor(
                torch.stack([torch.from_numpy(row_indices.astype(np.int64)), col_indices]),
                vals_2,
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

        # --- Features ---
        lifted_topology["x_0"] = data.x

        # Preserve edge attributes if applicable
        if self.contains_edge_attr and num_edges == graph.number_of_edges():
            lifted_topology["x_1"] = self._reorder_edge_attr(
                data, graph, sorted_edges
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
        ei = data.edge_index
        data_edge_map = {}
        for idx in range(ei.size(1)):
            u, v = ei[0, idx].item(), ei[1, idx].item()
            key = (min(u, v), max(u, v))
            if key not in data_edge_map:
                data_edge_map[key] = idx

        reorder_indices = [data_edge_map[e] for e in sorted_edges]
        return data.edge_attr[reorder_indices]
