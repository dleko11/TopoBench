"""Test the message passing module."""

from contextlib import contextmanager

import torch

from topobench.transforms.feature_liftings.projection_sum import ProjectionSum
from topobench.transforms.liftings.graph2simplicial import (
    SimplicialCliqueLifting,
)


class TestProjectionSum:
    """Test the ConcatentionLifting class."""

    def setup_method(self):
        """Set up the test."""
        # Initialize a lifting class
        self.lifting = SimplicialCliqueLifting(
            feature_lifting="ProjectionSum", complex_dim=3
        )

    def test_lift_features(self, simple_graph_1):
        """Test the lift_features method.

        Parameters
        ----------
        simple_graph_1 : torch_geometric.data.Data
            A simple graph data object.
        """
        data = simple_graph_1
        # Test the lift_features method
        lifted_data = self.lifting.forward(data.clone())

        expected_x1 = torch.tensor(
            [
                [   6.],
                [  11.],
                [ 101.],
                [5001.],
                [  15.],
                [ 105.],
                [  60.],
                [ 110.],
                [ 510.],
                [5010.],
                [1050.],
                [1500.],
                [5500.]
            ]
        )

        expected_x2 = torch.tensor(
            [
                [   32.],
                [  212.],
                [  222.],
                [10022.],
                [  230.],
                [11020.]
            ]
        )

        expected_x3 = torch.tensor(
            [
                [696.]
            ]
        )

        assert (
            expected_x1 == lifted_data.x_1
        ).all(), "Something is wrong with the lifted features x_1."
        assert (
            expected_x2 == lifted_data.x_2
        ).all(), "Something is wrong with the lifted features x_2."
        assert (
            expected_x3 == lifted_data.x_3
        ).all(), "Something is wrong with the lifted features x_3."

    def test_tracks_each_projected_rank(self, monkeypatch):
        """Track every feature projection independently."""
        phases = []

        @contextmanager
        def record_phase(phase, **_kwargs):
            phases.append(phase)
            yield

        monkeypatch.setattr(
            "topobench.transforms.feature_liftings.projection_sum.track_phase",
            record_phase,
        )
        data = {
            "x_0": torch.tensor([[1.0], [2.0]]),
            "incidence_1": torch.tensor([[1.0], [1.0]]),
            "incidence_2": torch.tensor([[1.0]]),
            "incidence_3": torch.tensor([[1.0]]),
        }

        ProjectionSum()(data)

        assert phases == [
            "feature_projection_rank_1",
            "feature_projection_rank_2",
            "feature_projection_rank_3",
        ]
