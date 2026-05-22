"""Unit tests for large scale k-hop hypergraph lifting."""

import pytest
import torch
from topobench.transforms.liftings.graph2hypergraph import (
    HypergraphKHopLifting,
)
from topobench.transforms.liftings.graph2hypergraph.khop_lifting_large_scale import (
    HypergraphKHopLiftingLargeScale,
)


class TestHypergraphKHopLiftingLargeScale:
    """Test class for HypergraphKHopLiftingLargeScale."""

    def test_compare_lift_topology(self, simple_graph_2):
        """Compare HypergraphKHopLiftingLargeScale against standard HypergraphKHopLifting.

        Parameters
        ----------
        simple_graph_2 : torch_geometric.data.Data
            Synthetic graph dataset fixture.
        """
        data = simple_graph_2

        # 1-hop comparison
        lift_ref_1 = HypergraphKHopLifting(k_value=1)
        lift_ls_1 = HypergraphKHopLiftingLargeScale(k_value=1)

        out_ref_1 = lift_ref_1.forward(data.clone())
        out_ls_1 = lift_ls_1.forward(data.clone())

        assert out_ref_1.num_hyperedges == out_ls_1.num_hyperedges
        assert torch.allclose(
            out_ref_1.incidence_hyperedges.to_dense(),
            out_ls_1.incidence_hyperedges.to_dense(),
        )

        # 2-hop comparison
        lift_ref_2 = HypergraphKHopLifting(k_value=2)
        lift_ls_2 = HypergraphKHopLiftingLargeScale(k_value=2)

        out_ref_2 = lift_ref_2.forward(data.clone())
        out_ls_2 = lift_ls_2.forward(data.clone())

        assert out_ref_2.num_hyperedges == out_ls_2.num_hyperedges
        assert torch.allclose(
            out_ref_2.incidence_hyperedges.to_dense(),
            out_ls_2.incidence_hyperedges.to_dense(),
        )

    def test_repr(self):
        """Verify that the __repr__ method returns the expected string representation."""
        lifting = HypergraphKHopLiftingLargeScale(k_value=3)
        assert repr(lifting) == "HypergraphKHopLiftingLargeScale(k=3)"

    def test_isolated_nodes(self):
        """Verify that the lifting cleanly handles graphs containing completely isolated nodes."""
        # Graph with 5 nodes. Nodes 0, 1, 2 are connected. Nodes 3, 4 are completely isolated.
        edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
        x = torch.randn(5, 3)
        data = Data = torch.sparse_coo_tensor # We'll just construct PyG Data object
        import torch_geometric
        data = torch_geometric.data.Data(x=x, edge_index=edge_index, num_nodes=5)

        lift_ref = HypergraphKHopLifting(k_value=1)
        lift_ls = HypergraphKHopLiftingLargeScale(k_value=1)

        out_ref = lift_ref.forward(data.clone())
        out_ls = lift_ls.forward(data.clone())

        # Verify correct outputs
        assert out_ref.num_hyperedges == out_ls.num_hyperedges
        assert torch.allclose(
            out_ref.incidence_hyperedges.to_dense(),
            out_ls.incidence_hyperedges.to_dense(),
        )

        # Isolated nodes should be in their own k-hop neighborhood (hyperedge of size 1)
        # Check that columns corresponding to hyperedges 3 and 4 have exactly one active entry (themselves)
        dense_ls = out_ls.incidence_hyperedges.to_dense()
        assert dense_ls[3, 3] == 1.0
        assert dense_ls[:, 3].sum() == 1.0
        assert dense_ls[4, 4] == 1.0
        assert dense_ls[:, 4].sum() == 1.0
