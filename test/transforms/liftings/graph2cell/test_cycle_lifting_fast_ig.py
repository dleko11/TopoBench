"""Unit tests for the fast cycle lifting variants (Fast, IG, Selective)."""

import pytest
import torch
import torch_geometric
from topobench.transforms.liftings.graph2cell import CellCycleLifting
from topobench.transforms.liftings.graph2cell.cycle_lifting import CellCycleLiftingSelective
from topobench.transforms.liftings.graph2cell.cycle_lifting_fast import CellCycleLiftingFast
from topobench.transforms.liftings.graph2cell.cycle_lifting_ig import CellCycleLiftingIG


class TestFastCycleLiftings:
    """Test the Fast, IG, and Selective cycle lifting classes."""

    def test_compare_lift_topology(self, simple_graph_1):
        """Compare and verify that all implementations produce identical topologies.

        Parameters
        ----------
        simple_graph_1 : torch_geometric.data.Data
            Synthetic graph dataset fixture.
        """
        data = simple_graph_1

        # Instantiate liftings
        ref_lift = CellCycleLifting()
        fast_lift = CellCycleLiftingFast()
        ig_lift = CellCycleLiftingIG()
        sel_lift = CellCycleLiftingSelective(neighborhoods=["up-adjacency-1", "down-laplacian-1"])

        # Forward passes
        out_ref = ref_lift.forward(data.clone())
        out_fast = fast_lift.forward(data.clone())
        out_ig = ig_lift.forward(data.clone())
        out_sel = sel_lift.forward(data.clone())

        # Verify fast implementation matches reference
        assert torch.allclose(out_ref.incidence_1.to_dense(), out_fast.incidence_1.to_dense())
        assert torch.allclose(out_ref.incidence_2.to_dense(), out_fast.incidence_2.to_dense())
        assert torch.allclose(out_ref.adjacency_0.to_dense(), out_fast.adjacency_0.to_dense())
        assert torch.allclose(out_ref.x_0, out_fast.x_0)

        # Verify igraph implementation is topologically correct (all cells are valid simple cycles)
        assert torch.allclose(out_ref.incidence_1.to_dense(), out_ig.incidence_1.to_dense())
        assert out_ig.incidence_2.shape == (out_ref.incidence_1.shape[1], out_ref.incidence_2.shape[1])

        import networkx as nx
        ei = data.edge_index
        edge_set = set()
        for i in range(ei.size(1)):
            u, v = ei[0, i].item(), ei[1, i].item()
            if u != v:
                edge_set.add((min(u, v), max(u, v)))
        sorted_edges = sorted(edge_set)

        inc_2_dense = out_ig.incidence_2.to_dense()
        num_cells = inc_2_dense.shape[1]
        for c in range(num_cells):
            edge_indices = torch.where(inc_2_dense[:, c] != 0)[0].tolist()
            assert len(edge_indices) >= 3
            sub_g = nx.Graph()
            for e_idx in edge_indices:
                sub_g.add_edge(*sorted_edges[e_idx])
            assert nx.is_connected(sub_g)
            assert all(deg == 2 for node, deg in sub_g.degree())

        assert torch.allclose(out_ref.adjacency_0.to_dense(), out_ig.adjacency_0.to_dense())
        assert torch.allclose(out_ref.x_0, out_ig.x_0)

        # Verify selective implementation contains requested neighborhoods
        assert "incidence_1" in out_sel
        assert "incidence_2" in out_sel
        assert "adjacency_1" in out_sel or "up_adjacency-1" in out_sel or "up-adjacency-1" in out_sel
        assert "down_laplacian_1" in out_sel or "down-laplacian-1" in out_sel

    def test_edge_attributes_propagation(self, simple_graph_1):
        """Verify that edge attributes are correctly reordered and preserved.

        Parameters
        ----------
        simple_graph_1 : torch_geometric.data.Data
            Synthetic graph dataset fixture.
        """
        data = simple_graph_1.clone()
        num_edges = data.edge_index.size(1)
        data.edge_attr = torch.randn(num_edges, 3)

        ref_lift = CellCycleLifting(contains_edge_attr=True)
        fast_lift = CellCycleLiftingFast(contains_edge_attr=True)
        ig_lift = CellCycleLiftingIG(contains_edge_attr=True)
        sel_lift = CellCycleLiftingSelective(contains_edge_attr=True, neighborhoods=["up-adjacency-1"])

        out_ref = ref_lift.forward(data.clone())
        out_fast = fast_lift.forward(data.clone())
        out_ig = ig_lift.forward(data.clone())
        out_sel = sel_lift.forward(data.clone())

        assert "x_1" in out_ref
        assert "x_1" in out_fast
        assert "x_1" in out_ig
        assert "x_1" in out_sel

        assert torch.allclose(out_ref.x_1, out_fast.x_1)
        assert torch.allclose(out_ref.x_1, out_ig.x_1)
        assert torch.allclose(out_ref.x_1, out_sel.x_1)

    def test_cycle_length_filtering(self, simple_graph_1):
        """Verify that max_cell_length correctly filters cycle representation.

        Parameters
        ----------
        simple_graph_1 : torch_geometric.data.Data
            Synthetic graph dataset fixture.
        """
        data = simple_graph_1

        ref_lift = CellCycleLifting(max_cell_length=3)
        fast_lift = CellCycleLiftingFast(max_cell_length=3)
        ig_lift = CellCycleLiftingIG(max_cell_length=3)
        sel_lift = CellCycleLiftingSelective(max_cell_length=3, neighborhoods=["up-adjacency-1"])

        out_ref = ref_lift.forward(data.clone())
        out_fast = fast_lift.forward(data.clone())
        out_ig = ig_lift.forward(data.clone())
        out_sel = sel_lift.forward(data.clone())

        # Check cycle length <= 3
        ref_lengths = torch.sum(out_ref.incidence_2.to_dense() != 0, dim=0)
        assert torch.all(ref_lengths <= 3)

        fast_lengths = torch.sum(out_fast.incidence_2.to_dense() != 0, dim=0)
        assert torch.all(fast_lengths <= 3)

        ig_lengths = torch.sum(out_ig.incidence_2.to_dense() != 0, dim=0)
        assert torch.all(ig_lengths <= 3)

        assert out_ref.incidence_2.shape == out_fast.incidence_2.shape
        assert out_ref.incidence_2.shape == out_sel.incidence_2.shape
        assert out_ig.incidence_2.shape[0] == out_ref.incidence_2.shape[0]

    def test_no_cycles(self):
        """Verify that liftings run cleanly on graphs with no cycles."""
        edge_index = torch.tensor([[0, 1, 2, 1, 2, 3], [1, 2, 3, 0, 1, 2]], dtype=torch.long)
        x = torch.randn(4, 3)
        data = torch_geometric.data.Data(x=x, edge_index=edge_index, num_nodes=4)

        ref_lift = CellCycleLifting()
        fast_lift = CellCycleLiftingFast()
        ig_lift = CellCycleLiftingIG()
        sel_lift = CellCycleLiftingSelective(neighborhoods=["up-adjacency-1"])

        out_ref = ref_lift.forward(data.clone())
        out_fast = fast_lift.forward(data.clone())
        out_ig = ig_lift.forward(data.clone())
        out_sel = sel_lift.forward(data.clone())

        assert out_fast.incidence_2.shape[1] == 0
        assert out_ig.incidence_2.shape[1] == 0
        assert out_sel.incidence_2.shape[1] == 0

    def test_empty_graph(self):
        """Verify that liftings run cleanly on a graph with zero edges."""
        x = torch.randn(4, 3)
        edge_index = torch.empty((2, 0), dtype=torch.long)
        data = torch_geometric.data.Data(x=x, edge_index=edge_index, num_nodes=4)

        ref_lift = CellCycleLifting()
        fast_lift = CellCycleLiftingFast()
        ig_lift = CellCycleLiftingIG()
        sel_lift = CellCycleLiftingSelective(neighborhoods=["up-adjacency-1"])

        out_ref = ref_lift.forward(data.clone())
        out_fast = fast_lift.forward(data.clone())
        out_ig = ig_lift.forward(data.clone())
        out_sel = sel_lift.forward(data.clone())

        assert out_fast.incidence_1.shape == (4, 0)
        assert out_ig.incidence_1.shape == (4, 0)
        assert out_sel.incidence_1.shape == (4, 0)

        assert out_fast.incidence_2.shape == (0, 0)
        assert out_ig.incidence_2.shape == (0, 0)
        assert out_sel.incidence_2.shape == (0, 0)
