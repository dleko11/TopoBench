"""Encoder-first encoding must equal encoding the ProjectionSum-lifted features."""

import copy

import networkx as nx
import pytest
import torch
from torch_geometric.data import Data

from topobench.nn.encoders.all_cell_encoder import AllCellFeatureEncoder
from topobench.transforms.liftings.graph2simplicial.clique_lifting import (
    SimplicialCliqueLiftingSelective,
)


def _union_batch(seed):
    """Two disjoint random graphs lifted together, with graph ids."""
    sizes = (14, 11)
    graphs = [
        nx.gnp_random_graph(n, 0.45, seed=seed + i)
        for i, n in enumerate(sizes)
    ]
    edges, offset, batch = [], 0, []
    for graph_id, graph in enumerate(graphs):
        edges += [(a + offset, b + offset) for a, b in graph.edges()]
        batch += [graph_id] * graph.number_of_nodes()
        offset += graph.number_of_nodes()
    edge_index = torch.tensor(edges, dtype=torch.long).t()
    edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
    generator = torch.Generator().manual_seed(seed)
    x = torch.randint(-3, 4, (offset, 5), generator=generator).float()
    lifting = SimplicialCliqueLiftingSelective(
        complex_dim=2,
        signed=False,
        feature_lifting="ProjectionSum",
        neighborhoods=["incidence-1", "incidence-2"],
    )
    data = lifting(Data(x=x, edge_index=edge_index))
    batch_0 = torch.tensor(batch)
    incidence_1 = data.incidence_1.coalesce()
    rows, cols = incidence_1.indices()
    batch_1 = torch.zeros(incidence_1.size(1), dtype=torch.long)
    batch_1[cols] = batch_0[rows]
    incidence_2 = data.incidence_2.coalesce()
    rows2, cols2 = incidence_2.indices()
    batch_2 = torch.zeros(incidence_2.size(1), dtype=torch.long)
    batch_2[cols2] = batch_1[rows2]
    for rank, vector in enumerate((batch_0, batch_1, batch_2)):
        data[f"batch_{rank}"] = vector
        data[f"x_{rank}"] = data[f"x_{rank}"].double()
    for rank in (1, 2):
        data[f"incidence_{rank}"] = data[f"incidence_{rank}"].double()
    return data


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_encoder_first_matches_lifted_encoding(seed):
    data = _union_batch(seed)
    assert data.incidence_2.size(1) > 0, "test graph needs triangles"
    torch.manual_seed(seed)
    standard = AllCellFeatureEncoder(
        in_channels=[5, 5, 5], out_channels=4, selected_dimensions=[0, 1, 2]
    ).double()
    with torch.no_grad():
        for parameter in standard.parameters():
            parameter.normal_()
    first = copy.deepcopy(standard)
    first.encoder_first = True

    expected = standard(data.clone())
    lean = data.clone()
    del lean.x_1, lean.x_2, lean.batch_1, lean.batch_2
    actual = first(lean)

    loss_expected = sum((expected[f"x_{r}"] ** 2).sum() for r in range(3))
    loss_actual = sum((actual[f"x_{r}"] ** 2).sum() for r in range(3))
    loss_expected.backward()
    loss_actual.backward()
    for rank in range(3):
        torch.testing.assert_close(
            actual[f"x_{rank}"], expected[f"x_{rank}"], rtol=0, atol=1e-9
        )
    for (name, a), b in zip(
        standard.named_parameters(), first.parameters(), strict=True
    ):
        torch.testing.assert_close(
            b.grad, a.grad, rtol=1e-9, atol=1e-9, msg=name
        )
