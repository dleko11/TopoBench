"""Halo-partitioned SCN on ``torch.distributed``: one process per worker.

Each process lifts its own core plus one-hop halo and keeps the cells it owns.
At setup it asks the owners of its ghost faces for them by vertex-tuple key
(one all-to-all per boundary matrix); owners answer in request order, so the
same index lists drive every later exchange:

* ``expand`` pulls ghost face values from their owners (all-to-all);
* ``fold`` pushes partial sums for ghost faces to their owners (all-to-all,
  then ``index_add`` at the owner).

Both use the autograd-aware collectives in ``torch.distributed.nn``, so the
backward of an expand is a fold and vice versa, and ``GraphNorm``'s
whole-graph statistics come from differentiable all-reduces. With parameter
gradients summed across processes, training computes the full-graph gradient.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import torch
import torch.distributed as dist
import torch.distributed.nn.functional as dist_fn

from topobench.distributed.halo import (
    WorkerShard,
    build_shard,
    drop_one_faces,
    encode_simplices,
    lift_shard_features,
)
from topobench.distributed.scn import (
    RANKS,
    SCNNodeClassifier,
    hodge_operators,
    scn_forward,
)


@dataclass
class TrafficLog:
    """Bytes this process sends, per operation.

    Attributes
    ----------
    sent : dict[str, int]
        Bytes sent, keyed by operation.
    calls : dict[str, int]
        Number of calls, keyed by operation.
    """

    sent: dict = field(default_factory=dict)
    calls: dict = field(default_factory=dict)

    def record(self, operation: str, num_bytes: int) -> None:
        """Add one exchange.

        Parameters
        ----------
        operation : str
            Operation name.
        num_bytes : int
            Bytes sent by this process.
        """
        self.sent[operation] = self.sent.get(operation, 0) + num_bytes
        self.calls[operation] = self.calls.get(operation, 0) + 1

    def reset(self) -> None:
        """Clear all records."""
        self.sent.clear()
        self.calls.clear()


def _all_to_all_counts(counts: torch.Tensor) -> torch.Tensor:
    """Tell every process how many items this process will send it.

    Parameters
    ----------
    counts : torch.Tensor
        Items this process sends to each process.

    Returns
    -------
    torch.Tensor
        Items this process receives from each process.
    """
    received = torch.empty_like(counts)
    dist.all_to_all_single(received, counts)
    return received


class TorchDistBoundaryOperator:
    r"""Boundary matrix ``B_k`` split across processes by coface ownership.

    Same interface as ``BoundaryOperator`` (``up``, ``down``, ``up_diagonal``)
    plus ``fold``, acting on this process's owned rows only.

    Parameters
    ----------
    matrix : torch.Tensor
        Sparse ``(n_owned_faces + n_ghosts, n_owned_cofaces)`` local block.
    num_owned_faces : int
        Owned face rows at the top of ``matrix``.
    send_index : torch.Tensor
        Owned face rows other processes asked for, grouped by requester.
    send_counts : list[int]
        Rows each process asked this one for.
    recv_counts : list[int]
        Ghost rows this process holds per owner, in ghost-buffer order.
    traffic : TrafficLog
        Communication log.
    name : str
        Prefix for log records.
    """

    def __init__(
        self,
        matrix: torch.Tensor,
        num_owned_faces: int,
        send_index: torch.Tensor,
        send_counts: list[int],
        recv_counts: list[int],
        traffic: TrafficLog,
        name: str,
    ) -> None:
        self.matrix = matrix.coalesce()
        self.matrix_t = self.matrix.transpose(0, 1).coalesce()
        self.num_owned_faces = num_owned_faces
        self.send_index = send_index
        self.send_counts = send_counts
        self.recv_counts = recv_counts
        self.traffic = traffic
        self.name = name
        self._diagonal = None

    def _pull(self, x: torch.Tensor) -> torch.Tensor:
        outgoing = x[self.send_index]
        incoming = x.new_empty((sum(self.recv_counts), x.size(1)))
        self.traffic.record(
            f"{self.name}.expand", outgoing.numel() * outgoing.element_size()
        )
        return dist_fn.all_to_all_single(
            incoming, outgoing, self.recv_counts, self.send_counts
        )

    def expand(self, x: torch.Tensor) -> torch.Tensor:
        r"""Compute :math:`B^\top x` for owned cofaces (pulls ghost faces).

        Parameters
        ----------
        x : torch.Tensor
            Owned face signal.

        Returns
        -------
        torch.Tensor
            Owned coface signal.
        """
        buffer = torch.cat([x, self._pull(x)])
        return torch.sparse.mm(self.matrix_t, buffer)

    def fold(
        self, y: torch.Tensor, matrix: torch.Tensor | None = None
    ) -> torch.Tensor:
        r"""Compute :math:`B y` for owned faces (pushes ghost partial sums).

        Parameters
        ----------
        y : torch.Tensor
            Owned coface signal.
        matrix : torch.Tensor, optional
            Local block to use instead of ``B``'s.

        Returns
        -------
        torch.Tensor
            Owned face signal.
        """
        partial = torch.sparse.mm(self.matrix if matrix is None else matrix, y)
        owned = partial[: self.num_owned_faces]
        outgoing = partial[self.num_owned_faces :].contiguous()
        incoming = y.new_empty((sum(self.send_counts), y.size(1)))
        self.traffic.record(
            f"{self.name}.fold", outgoing.numel() * outgoing.element_size()
        )
        received = dist_fn.all_to_all_single(
            incoming, outgoing, self.send_counts, self.recv_counts
        )
        return owned.index_add(0, self.send_index, received)

    def up(self, x: torch.Tensor) -> torch.Tensor:
        r"""Apply :math:`BB^\top` to an owned face signal.

        Parameters
        ----------
        x : torch.Tensor
            Owned face signal.

        Returns
        -------
        torch.Tensor
            Owned face signal.
        """
        return self.fold(self.expand(x))

    def down(self, x: torch.Tensor) -> torch.Tensor:
        r"""Apply :math:`B^\top B` to an owned coface signal.

        Parameters
        ----------
        x : torch.Tensor
            Owned coface signal.

        Returns
        -------
        torch.Tensor
            Owned coface signal.
        """
        return self.expand(self.fold(x))

    def up_diagonal(self) -> torch.Tensor:
        r"""Return the diagonal of :math:`BB^\top` on owned faces (cached).

        Returns
        -------
        torch.Tensor
            Owned face vector.
        """
        if self._diagonal is None:
            squared = torch.sparse_coo_tensor(
                self.matrix.indices(),
                self.matrix.values().square(),
                self.matrix.shape,
            ).coalesce()
            ones = self.matrix.values().new_ones((self.matrix.size(1), 1))
            with torch.no_grad():
                self._diagonal = self.fold(ones, matrix=squared).squeeze(1)
        return self._diagonal


@dataclass
class LocalPlan:
    """This process's piece of the partitioned complex.

    Attributes
    ----------
    shard : WorkerShard
        Lifted and owned cells.
    global_counts : list[int]
        Cells per rank across all processes.
    boundary : dict[int, TorchDistBoundaryOperator]
        Distributed boundary matrices for ranks 1 and 2.
    traffic : TrafficLog
        Communication log shared by the operators.
    timings : dict[str, float]
        Setup wall times in seconds.
    """

    shard: WorkerShard
    global_counts: list[int]
    boundary: dict
    traffic: TrafficLog
    timings: dict


def build_local_plan(
    edges: torch.Tensor,
    part: torch.Tensor,
    num_nodes: int,
    device: torch.device,
    dtype: torch.dtype,
    complex_dim: int = 2,
) -> LocalPlan:
    """Lift this process's piece and wire up its exchanges.

    Parameters
    ----------
    edges : torch.Tensor
        Canonical global edges (``src < dst``).
    part : torch.Tensor
        Process id of every node.
    num_nodes : int
        Number of nodes.
    device : torch.device
        Device for local matrices and collectives.
    dtype : torch.dtype
        Dtype of the local matrices.
    complex_dim : int, optional
        Maximum simplex rank.

    Returns
    -------
    LocalPlan
        Local cells and distributed boundary operators.

    Raises
    ------
    RuntimeError
        If an owner does not have a requested face (insufficient halo).
    """
    me, world = dist.get_rank(), dist.get_world_size()
    timings = {}
    start = time.perf_counter()
    shard = build_shard(me, edges, part, complex_dim)
    timings["lift_seconds"] = time.perf_counter() - start

    start = time.perf_counter()
    keys = [encode_simplices(cells, num_nodes) for cells in shard.owned]
    counts = torch.tensor(
        [cells.size(0) for cells in shard.owned], device=device
    )
    dist.all_reduce(counts)
    traffic = TrafficLog()
    boundary = {}
    for rank in range(1, complex_dim + 1):
        faces, columns = drop_one_faces(shard.owned[rank])
        owners = part[faces[:, 0]]
        face_keys = encode_simplices(faces, num_nodes)
        is_ghost = owners != me
        own_keys = keys[rank - 1]
        rows = torch.empty(faces.size(0), dtype=torch.long)
        position = torch.searchsorted(own_keys, face_keys[~is_ghost])
        if position.numel() and not torch.equal(
            own_keys[position.clamp(max=own_keys.numel() - 1)],
            face_keys[~is_ghost],
        ):
            raise RuntimeError("An owned face is missing locally.")
        rows[~is_ghost] = position

        # Unique ghost faces sorted by (owner, key): the ghost-buffer order.
        if bool(is_ghost.any()):
            requests, inverse = torch.unique(
                torch.stack([owners[is_ghost], face_keys[is_ghost]], dim=1),
                dim=0,
                return_inverse=True,
            )
        else:
            requests = torch.empty((0, 2), dtype=torch.long)
            inverse = torch.empty(0, dtype=torch.long)
        rows[is_ghost] = own_keys.numel() + inverse
        recv_counts = torch.bincount(requests[:, 0], minlength=world)
        send_counts = _all_to_all_counts(recv_counts.to(device)).cpu()
        asked = torch.empty(
            int(send_counts.sum()), dtype=torch.long, device=device
        )
        dist.all_to_all_single(
            asked,
            requests[:, 1].contiguous().to(device),
            send_counts.tolist(),
            recv_counts.tolist(),
        )
        asked = asked.cpu()
        send_index = torch.searchsorted(own_keys, asked)
        if asked.numel() and not torch.equal(
            own_keys[send_index.clamp(max=max(own_keys.numel() - 1, 0))], asked
        ):
            raise RuntimeError(
                "A requested face is missing at its owner; the halo does not "
                "cover every face of the owned cells."
            )
        matrix = torch.sparse_coo_tensor(
            torch.stack([rows, columns]),
            torch.ones(rows.numel(), dtype=dtype),
            size=(
                own_keys.numel() + requests.size(0),
                shard.owned[rank].size(0),
            ),
        ).to(device)
        boundary[rank] = TorchDistBoundaryOperator(
            matrix=matrix,
            num_owned_faces=own_keys.numel(),
            send_index=send_index.to(device),
            send_counts=send_counts.tolist(),
            recv_counts=recv_counts.tolist(),
            traffic=traffic,
            name=f"B{rank}",
        )
    timings["exchange_setup_seconds"] = time.perf_counter() - start
    return LocalPlan(
        shard=shard,
        global_counts=counts.cpu().tolist(),
        boundary=boundary,
        traffic=traffic,
        timings=timings,
    )


class DistributedSCN:
    """SCN forward over this process's owned cells.

    Parameters
    ----------
    model : SCNNodeClassifier
        Model replica (identical on every process).
    plan : LocalPlan
        This process's piece of the complex.
    x_0 : torch.Tensor
        Node features; only core and halo rows are read.
    num_nodes : int
        Number of nodes.
    device : torch.device
        Compute device.
    dtype : torch.dtype
        Compute dtype.
    """

    def __init__(
        self,
        model: SCNNodeClassifier,
        plan: LocalPlan,
        x_0: torch.Tensor,
        num_nodes: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        self.model = model
        self.plan = plan
        self.features = [
            f.to(device=device, dtype=dtype)
            for f in lift_shard_features(plan.shard, x_0, num_nodes)
        ]
        local_counts = [f.size(0) for f in self.features]
        self.operators = hodge_operators(
            plan.boundary, local_counts, dtype, device
        )
        self.global_counts = plan.global_counts

    def _graph_norm(self, norm, x: torch.Tensor, rank: int) -> torch.Tensor:
        count = self.global_counts[rank]
        total = dist_fn.all_reduce(x.sum(dim=0))
        self.plan.traffic.record(
            "graphnorm.allreduce", 2 * x.size(1) * x.element_size()
        )
        centred = x - total / count * norm.mean_scale
        variance = dist_fn.all_reduce((centred * centred).sum(dim=0)) / count
        return norm.weight * centred / (variance + norm.eps).sqrt() + norm.bias

    def __call__(self) -> dict:
        """Run the forward pass.

        Returns
        -------
        dict
            Stage outputs for owned rows; ``logits`` are for core nodes.
        """
        return scn_forward(
            self.model,
            self.features,
            self.plan.boundary,
            self.operators,
            self._graph_norm,
        )


def all_reduce_gradients(model: torch.nn.Module) -> None:
    """Sum parameter gradients across processes.

    Parameters
    ----------
    model : torch.nn.Module
        Model replica.
    """
    for parameter in model.parameters():
        if parameter.grad is not None:
            dist.all_reduce(parameter.grad)


def gather_rows(values: torch.Tensor) -> torch.Tensor:
    """Concatenate a variable-length tensor from every process.

    Parameters
    ----------
    values : torch.Tensor
        Local rows.

    Returns
    -------
    torch.Tensor
        Rows of all processes in process order.
    """
    world = dist.get_world_size()
    size = torch.tensor([values.size(0)], device=values.device)
    sizes = [torch.zeros_like(size) for _ in range(world)]
    dist.all_gather(sizes, size)
    sizes = [int(s) for s in sizes]
    padded = values.new_zeros((max(sizes), *values.shape[1:]))
    padded[: values.size(0)] = values
    chunks = [torch.zeros_like(padded) for _ in range(world)]
    dist.all_gather(chunks, padded)
    return torch.cat(
        [chunk[:n] for chunk, n in zip(chunks, sizes, strict=True)]
    )


__all__ = [
    "RANKS",
    "DistributedSCN",
    "LocalPlan",
    "TorchDistBoundaryOperator",
    "TrafficLog",
    "all_reduce_gradients",
    "build_local_plan",
    "gather_rows",
]
