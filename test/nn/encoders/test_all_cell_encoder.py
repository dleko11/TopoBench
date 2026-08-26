"""Tests for all-cell feature encoding."""

import torch
from torch_geometric.data import Data

from topobench.nn.encoders.all_cell_encoder import AllCellFeatureEncoder


def test_encode_then_lift_projects_low_dimensional_features():
    """Project encoded node features and derive higher-rank batches."""
    node_features = torch.randn(6, 3, requires_grad=True)
    incidence_1 = torch.sparse_coo_tensor(
        torch.tensor(
            [
                [0, 1, 1, 2, 0, 2, 3, 4, 4, 5, 3, 5],
                [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5],
            ]
        ),
        torch.tensor([-1.0, 1.0] * 6),
        size=(6, 6),
    ).coalesce()
    incidence_2 = torch.sparse_coo_tensor(
        torch.tensor([[0, 1, 2, 3, 4, 5], [0, 0, 0, 1, 1, 1]]),
        torch.tensor([1.0, -1.0, 1.0, -1.0, 1.0, -1.0]),
        size=(6, 2),
    ).coalesce()
    data = Data(
        x=node_features,
        incidence_1=incidence_1,
        incidence_2=incidence_2,
        batch_0=torch.tensor([0, 0, 0, 1, 1, 1]),
    )
    encoder = AllCellFeatureEncoder(
        in_channels=[3],
        out_channels=2,
        selected_dimensions=[0, 1, 2],
        lift_encoded_features=True,
    )

    output = encoder(data)

    expected_x_1 = torch.mm(incidence_1.to_dense().abs().T, output.x_0)
    expected_x_2 = torch.mm(incidence_2.to_dense().abs().T, expected_x_1)
    assert not hasattr(encoder, "encoder_1")
    assert not hasattr(encoder, "encoder_2")
    assert torch.allclose(output.x_1, expected_x_1)
    assert torch.allclose(output.x_2, expected_x_2)
    assert torch.equal(output.batch_1, torch.tensor([0, 0, 0, 1, 1, 1]))
    assert torch.equal(output.batch_2, torch.tensor([0, 1]))

    output.x_2.sum().backward()
    assert node_features.grad is not None
    assert encoder.encoder_0.linear.weight.grad is not None
