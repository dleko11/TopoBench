"""Tests for batches without active node supervision."""

from unittest.mock import MagicMock

import pytest
import torch
from torch_geometric.data import Data

from topobench.model.model import TBModel


def _make_model(task_level: str) -> TBModel:
    """Instantiate a TBModel with mocked dependencies."""
    backbone = MagicMock()
    backbone.parameters.return_value = []
    readout = MagicMock(task_level=task_level)
    readout.parameters.return_value = []
    feature_encoder = MagicMock()
    feature_encoder.parameters.return_value = []
    optimizer = MagicMock()
    optimizer.configure_optimizer.return_value = {"optimizer": MagicMock()}

    return TBModel(
        backbone=backbone,
        readout=readout,
        loss=MagicMock(),
        feature_encoder=feature_encoder,
        evaluator=MagicMock(),
        optimizer=optimizer,
    )


@pytest.mark.parametrize(
    ("step_name", "mask_name", "phase"),
    [
        ("training_step", "train_mask", "train"),
        ("validation_step", "val_mask", "val"),
        ("test_step", "test_mask", "test"),
    ],
)
def test_empty_batch_skips_model_loss_and_evaluator(
    step_name,
    mask_name,
    phase,
):
    """An empty supervision mask bypasses the entire model step."""
    model = _make_model("node")
    model.model_step = MagicMock(
        side_effect=AssertionError("model_step must not be called")
    )
    batch = Data(num_nodes=4)
    setattr(batch, mask_name, torch.zeros(4, dtype=torch.bool))

    result = getattr(model, step_name)(batch, 0)

    assert result is None
    assert model.empty_supervision_batches[phase] == 1
    model.model_step.assert_not_called()
    model.evaluator.update.assert_not_called()


def test_distributed_empty_training_batch_fails_explicitly():
    """Distributed training cannot silently skip a rank-local batch."""
    model = _make_model("node")
    model._trainer = MagicMock(world_size=2)
    batch = Data(
        num_nodes=4,
        train_mask=torch.zeros(4, dtype=torch.bool),
    )

    with pytest.raises(RuntimeError, match="distributed run"):
        model.training_step(batch, 0)


def test_graph_level_batch_is_never_treated_as_empty():
    """Graph-level tasks do not depend on node supervision masks."""
    model = _make_model("graph")

    assert not model._skip_empty_supervision_batch(
        Data(num_nodes=4),
        phase="train",
    )
