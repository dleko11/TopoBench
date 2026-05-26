"""Fast clique lifting bypassing toponetx object construction."""

import torch_geometric

from topobench.data.utils.utils import get_complex_connectivity_from_incidences
from topobench.transforms.liftings.graph2simplicial import (
    Graph2SimplicialLifting,
)
from topobench.transforms.liftings.graph2simplicial._clique_utils import (
    build_clique_complex_incidences,
    get_preserved_edge_features,
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
        incidences, simplices_by_rank, shape, sorted_edges = (
            build_clique_complex_incidences(
                graph=graph,
                complex_dim=self.complex_dim,
                signed=True,
                num_nodes=data.x.shape[0],
            )
        )

        # --- Compute connectivity ---
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
        if self.contains_edge_attr and simplices_by_rank[1].shape[0] == (
            graph.number_of_edges()
        ):
            edge_features = get_preserved_edge_features(graph, sorted_edges)
            if edge_features is not None:
                lifted_topology["x_1"] = edge_features

        return lifted_topology
