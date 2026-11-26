"""Halo lifting plus ghost exchange must reproduce full-graph SCN exactly."""

import networkx as nx
import pytest
import torch

from topobench.distributed.halo import (
    _resolve_at_owners,
    build_halo_plan,
    canonical_edges,
    encode_simplices,
    lift_input_features,
    partition_nodes,
)
from topobench.distributed.operators import (
    CommLog,
    DistributedBoundaryOperator,
)
from topobench.distributed.scn import (
    SCNNodeClassifier,
    align_to_reference,
    build_reference_batch,
    cast_batch,
    compare_stages,
    distributed_stages,
    reference_cells,
    reference_stages,
)
from topobench.nn.backbones.simplicial.incidence_operators import (
    BoundaryOperator,
)

TOLERANCE = 1e-10


def _graph(edges, num_nodes):
    edge_index = torch.tensor(edges, dtype=torch.long).t()
    return torch.cat([edge_index, edge_index.flip(0)], dim=1), num_nodes


def _random_graph(num_nodes, p, seed):
    graph = nx.gnp_random_graph(num_nodes, p, seed=seed)
    return _graph(list(graph.edges()), num_nodes)


CASES = {
    # Every simplex of K5 spans several workers.
    "k5": (
        _graph([(a, b) for a in range(5) for b in range(a + 1, 5)], 5),
        torch.tensor([0, 0, 1, 1, 2]),
    ),
    # Two triangles sharing edge (1, 2); the shared edge's faces sit on
    # different workers than the triangles' minimum vertices.
    "shared_edge": (
        _graph([(0, 1), (0, 2), (1, 2), (1, 3), (2, 3)], 4),
        torch.tensor([0, 1, 1, 2]),
    ),
    # Edge (0, 1) is lower adjacent to (1, 2) through vertex 1, and node 2
    # is two hops from worker 0's core: the case a one-hop halo misses.
    "two_hop_lower": (
        _graph([(0, 1), (1, 2), (2, 3), (1, 3)], 4),
        torch.tensor([0, 1, 2, 2]),
    ),
    # Worker 1 owns no nodes.
    "empty_worker": (
        _graph([(0, 1), (1, 2), (0, 2), (2, 3)], 4),
        torch.tensor([0, 0, 2, 2]),
    ),
}


def _random_cases():
    cases = {}
    for seed, num_parts in [(0, 1), (1, 3), (2, 5), (3, 7)]:
        (edge_index, num_nodes) = _random_graph(40, 0.25, seed)
        part = partition_nodes(
            edge_index, num_nodes, num_parts, "random", seed
        )
        cases[f"random_{seed}_p{num_parts}"] = ((edge_index, num_nodes), part)
    (edge_index, num_nodes) = _random_graph(60, 0.15, 11)
    part = partition_nodes(edge_index, num_nodes, 4, "metis")
    cases["metis_p4"] = ((edge_index, num_nodes), part)
    return cases


ALL_CASES = {**CASES, **_random_cases()}


def _setup(case):
    (edge_index, num_nodes), part = ALL_CASES[case]
    generator = torch.Generator().manual_seed(0)
    # Integer features keep ProjectionSum exact under any summation order.
    x_0 = torch.randint(0, 3, (num_nodes, 6), generator=generator).float()
    y = torch.randint(0, 3, (num_nodes,), generator=generator)
    edge_index = canonical_edges(edge_index, num_nodes)
    edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
    return edge_index, num_nodes, part, x_0, y


def _model(in_channels):
    torch.manual_seed(0)
    model = SCNNodeClassifier(in_channels, 8, 3, n_layers=2).double().eval()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.normal_()
    return model


@pytest.mark.parametrize("case", sorted(ALL_CASES))
def test_owned_cells_partition_the_global_complex(case):
    edge_index, num_nodes, part, x_0, y = _setup(case)
    plan = build_halo_plan(edge_index, num_nodes, part)
    reference = build_reference_batch(x_0, edge_index, y)
    align_to_reference(plan, reference_cells(reference))


@pytest.mark.parametrize("case", sorted(ALL_CASES))
def test_boundary_operators_match_centralized(case):
    edge_index, num_nodes, part, x_0, y = _setup(case)
    plan = build_halo_plan(edge_index, num_nodes, part)
    reference = cast_batch(
        build_reference_batch(x_0, edge_index, y), torch.float64
    )
    alignment = align_to_reference(plan, reference_cells(reference))
    generator = torch.Generator().manual_seed(1)
    for rank in (1, 2):
        central = BoundaryOperator(reference[f"incidence_{rank}"])
        distributed = DistributedBoundaryOperator(
            plan.incidences[rank],
            CommLog(),
            f"B{rank}",
            torch.float64,
            torch.device("cpu"),
        )
        faces, cofaces = alignment[rank - 1], alignment[rank]
        x = torch.randn(
            faces.numel(), 4, generator=generator, dtype=torch.float64
        )
        y_cof = torch.randn(
            cofaces.numel(), 4, generator=generator, dtype=torch.float64
        )
        x_ref = torch.empty_like(x)
        x_ref[faces] = x
        y_ref = torch.empty_like(y_cof)
        y_ref[cofaces] = y_cof
        torch.testing.assert_close(
            distributed.up(x), central.up(x_ref)[faces], rtol=0, atol=TOLERANCE
        )
        torch.testing.assert_close(
            distributed.down(y_cof),
            central.down(y_ref)[cofaces],
            rtol=0,
            atol=TOLERANCE,
        )
        torch.testing.assert_close(
            distributed.up_diagonal(),
            central.up_diagonal()[faces],
            rtol=0,
            atol=TOLERANCE,
        )


@pytest.mark.parametrize("case", sorted(ALL_CASES))
def test_distributed_scn_matches_full_graph(case):
    edge_index, num_nodes, part, x_0, y = _setup(case)
    plan = build_halo_plan(edge_index, num_nodes, part)
    reference = cast_batch(
        build_reference_batch(x_0, edge_index, y), torch.float64
    )
    alignment = align_to_reference(plan, reference_cells(reference))
    model = _model(x_0.size(1))

    expected = reference_stages(model, reference)
    with torch.no_grad():
        direct = model(reference.clone())["logits"]
    torch.testing.assert_close(expected["logits"], direct, rtol=0, atol=0)

    features = [f.double() for f in lift_input_features(plan, x_0)]
    torch.testing.assert_close(
        features[2], reference.x_2[alignment[2]], rtol=0, atol=0
    )
    comm = CommLog()
    actual = distributed_stages(model, plan, features, comm)
    diffs = compare_stages(actual, expected, alignment)
    assert max(diffs.values()) <= TOLERANCE, diffs

    if plan.num_parts == 1:
        exchanged = [
            entry["rows"]
            for (phase, operation), entry in comm.records.items()
            if phase == "forward" and "graphnorm" not in operation
        ]
        assert exchanged and not any(exchanged)


def test_missing_face_at_owner_is_detected():
    (edge_index, num_nodes), part = CASES["k5"]
    plan = build_halo_plan(edge_index, num_nodes, part)
    keys = [
        encode_simplices(shard.owned[1], num_nodes) for shard in plan.shards
    ]
    keys[0] = keys[0][1:]  # worker 0 "forgets" its edge (0, 1)
    with pytest.raises(RuntimeError, match="missing at its owner"):
        _resolve_at_owners(
            torch.tensor([[0, 1]]), part, keys, plan.offsets[1], num_nodes
        )
