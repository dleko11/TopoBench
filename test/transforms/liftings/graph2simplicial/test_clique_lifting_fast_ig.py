"""Unit tests for the fast simplicial clique lifting variants (Fast and IG)."""

import pytest
import torch
import torch_geometric
from topobench.transforms.liftings.graph2simplicial import (
    SimplicialCliqueLifting,
)
from topobench.transforms.liftings.graph2simplicial.clique_lifting_fast import (
    SimplicialCliqueLiftingFast,
)
from topobench.transforms.liftings.graph2simplicial.clique_lifting_ig import (
    SimplicialCliqueLiftingIG,
)


class TestFastSimplicialCliqueLiftings:
    """Test the Fast and IG simplicial clique lifting classes."""

    def test_compare_lift_topology(self, simple_graph_1):
        """Compare and verify that all implementations produce identical topologies.

        Parameters
        ----------
        simple_graph_1 : torch_geometric.data.Data
            Synthetic graph dataset fixture.
        """
        data = simple_graph_1

        # Instantiate liftings with same configuration
        # Use complex_dim=2 to find triangles
        lift_ref = SimplicialCliqueLifting(complex_dim=2, signed=False)
        lift_fast = SimplicialCliqueLiftingFast(complex_dim=2, signed=False)
        lift_ig = SimplicialCliqueLiftingIG(complex_dim=2, signed=False)

        # Forward passes
        out_ref = lift_ref.forward(data.clone())
        out_fast = lift_fast.forward(data.clone())
        out_ig = lift_ig.forward(data.clone())

        # Verify fast implementation matches reference
        assert torch.allclose(out_ref.incidence_1.to_dense(), out_fast.incidence_1.to_dense())
        assert torch.allclose(out_ref.incidence_2.to_dense(), out_fast.incidence_2.to_dense())
        assert torch.allclose(out_ref.adjacency_0.to_dense(), out_fast.adjacency_0.to_dense())
        assert torch.allclose(out_ref.x_0, out_fast.x_0)

        # Verify igraph implementation matches reference
        assert torch.allclose(out_ref.incidence_1.to_dense(), out_ig.incidence_1.to_dense())
        assert torch.allclose(out_ref.incidence_2.to_dense(), out_ig.incidence_2.to_dense())
        assert torch.allclose(out_ref.adjacency_0.to_dense(), out_ig.adjacency_0.to_dense())
        assert torch.allclose(out_ref.x_0, out_ig.x_0)

    def test_signed_boundary_calculations(self, simple_graph_1):
        """Verify that signed=True works and is mathematically consistent between implementations.

        Parameters
        ----------
        simple_graph_1 : torch_geometric.data.Data
            Synthetic graph dataset fixture.
        """
        data = simple_graph_1

        lift_ref = SimplicialCliqueLifting(complex_dim=2, signed=True)
        lift_fast = SimplicialCliqueLiftingFast(complex_dim=2, signed=True)
        lift_ig = SimplicialCliqueLiftingIG(complex_dim=2, signed=True)

        out_ref = lift_ref.forward(data.clone())
        out_fast = lift_fast.forward(data.clone())
        out_ig = lift_ig.forward(data.clone())

        # Check incidence matrices
        assert torch.allclose(out_ref.incidence_1.to_dense(), out_fast.incidence_1.to_dense())
        assert torch.allclose(out_ref.incidence_2.to_dense(), out_fast.incidence_2.to_dense())
        assert torch.allclose(out_ref.incidence_1.to_dense(), out_ig.incidence_1.to_dense())
        assert torch.allclose(out_ref.incidence_2.to_dense(), out_ig.incidence_2.to_dense())

        # Mathematical property of simplicial boundary: d1 * d2 should be all zeros
        d1 = out_fast.incidence_1.to_dense()
        d2 = out_fast.incidence_2.to_dense()
        assert torch.allclose(torch.matmul(d1, d2), torch.zeros(d1.shape[0], d2.shape[1]))

        # Also verify for igraph
        d1_ig = out_ig.incidence_1.to_dense()
        d2_ig = out_ig.incidence_2.to_dense()
        assert torch.allclose(torch.matmul(d1_ig, d2_ig), torch.zeros(d1_ig.shape[0], d2_ig.shape[1]))

    def test_edge_attributes_propagation(self, simple_graph_1):
        """Verify that edge attributes are correctly reordered and preserved when contains_edge_attr=True.

        Parameters
        ----------
        simple_graph_1 : torch_geometric.data.Data
            Synthetic graph dataset fixture.
        """
        data = simple_graph_1.clone()
        num_edges = data.edge_index.size(1)
        data.edge_attr = torch.randn(num_edges, 4)

        lift_ref = SimplicialCliqueLifting(complex_dim=2, signed=False, contains_edge_attr=True)
        lift_fast = SimplicialCliqueLiftingFast(complex_dim=2, signed=False, contains_edge_attr=True)
        lift_ig = SimplicialCliqueLiftingIG(complex_dim=2, signed=False, contains_edge_attr=True)

        out_ref = lift_ref.forward(data.clone())
        out_fast = lift_fast.forward(data.clone())
        out_ig = lift_ig.forward(data.clone())

        assert "x_1" in out_ref
        assert "x_1" in out_fast
        assert "x_1" in out_ig

        assert torch.allclose(out_ref.x_1, out_fast.x_1)
        assert torch.allclose(out_ref.x_1, out_ig.x_1)

    def test_no_higher_order_cliques(self):
        """Verify that the liftings cleanly execute on a graph with no cliques of dimension >= 2 (triangles)."""
        # A simple tree path with no triangles
        edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)
        x = torch.randn(4, 3)
        data = torch_geometric.data.Data(x=x, edge_index=edge_index, num_nodes=4)

        lift_ref = SimplicialCliqueLifting(complex_dim=2, signed=False)
        lift_fast = SimplicialCliqueLiftingFast(complex_dim=2, signed=False)
        lift_ig = SimplicialCliqueLiftingIG(complex_dim=2, signed=False)

        out_ref = lift_ref.forward(data.clone())
        out_fast = lift_fast.forward(data.clone())
        out_ig = lift_ig.forward(data.clone())

        assert out_ref.incidence_2.shape[1] == 0
        assert out_fast.incidence_2.shape[1] == 0
        assert out_ig.incidence_2.shape[1] == 0
