"""Simulated communication for owner-partitioned simplicial signals.

Signals are owner-major concatenations of every worker's owned rows. Worker
``p`` computes only with its own slice and a ghost buffer; values cross
workers only through ``expand`` (pull ghost face values from their owners),
``fold`` (push partial sums for ghost faces to their owners) and all-reduces.
Every crossing is recorded in a :class:`CommLog`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch_geometric.nn.norm import GraphNorm

from topobench.distributed.halo import IncidencePlan


@dataclass
class CommLog:
    """Accumulate rows, values and bytes moved between workers.

    Attributes
    ----------
    phase : str
        Label attached to new records, e.g. ``"setup"`` or ``"forward"``.
    records : dict
        Totals keyed by ``(phase, operation)``.
    """

    phase: str = "forward"
    records: dict = field(default_factory=dict)

    def record(
        self, operation: str, rows: int, width: int, itemsize: int, pairs: int
    ) -> None:
        """Add one communication step.

        Parameters
        ----------
        operation : str
            Operation name.
        rows : int
            Vectors moved.
        width : int
            Length of each vector.
        itemsize : int
            Bytes per value.
        pairs : int
            Worker pairs that exchange data in this step.
        """
        entry = self.records.setdefault(
            (self.phase, operation),
            {"calls": 0, "rows": 0, "values": 0, "bytes": 0, "pairs": 0},
        )
        entry["calls"] += 1
        entry["rows"] += rows
        entry["values"] += rows * width
        entry["bytes"] += rows * width * itemsize
        entry["pairs"] += pairs

    def totals(self, phase: str) -> dict[str, int]:
        """Sum all records of a phase.

        Parameters
        ----------
        phase : str
            Phase label.

        Returns
        -------
        dict[str, int]
            Summed calls, rows, values, bytes and pairs.
        """
        out = {"calls": 0, "rows": 0, "values": 0, "bytes": 0, "pairs": 0}
        for (record_phase, _), entry in self.records.items():
            if record_phase == phase:
                for key in out:
                    out[key] += entry[key]
        return out


class DistributedBoundaryOperator:
    r"""Drop-in replacement for ``BoundaryOperator`` across workers.

    Provides ``up`` (:math:`BB^\top x`), ``down`` (:math:`B^\top B x`) and
    ``up_diagonal`` with the same semantics, so ``UnsignedHodgeOperator`` and
    the SCN layers run unchanged on top of it.

    Parameters
    ----------
    plan : IncidencePlan
        Per-worker pieces of the boundary matrix.
    comm : CommLog
        Communication log.
    name : str
        Prefix for log records.
    dtype : torch.dtype
        Dtype of ``up_diagonal``.
    device : torch.device
        Device of the local matrices.
    """

    def __init__(
        self,
        plan: IncidencePlan,
        comm: CommLog,
        name: str,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        self.plan = plan
        self.comm = comm
        self.name = name
        self.dtype = dtype
        self.device = device
        self.num_parts = len(plan.local)
        self.num_faces = int(plan.face_offsets[-1])
        self.ghosts = [ghost.to(device) for ghost in plan.ghosts]
        self.local = [
            m.to(device=device, dtype=dtype).coalesce() for m in plan.local
        ]
        self.local_t = [m.transpose(0, 1).coalesce() for m in self.local]
        self._diagonal = None

    def _slice(self, offsets: torch.Tensor, worker: int) -> slice:
        return slice(int(offsets[worker]), int(offsets[worker + 1]))

    def expand(self, x: torch.Tensor) -> torch.Tensor:
        r"""Compute :math:`B^\top x` (faces to cofaces).

        Each worker pulls its ghost face values from their owners.

        Parameters
        ----------
        x : torch.Tensor
            Owner-major face signal.

        Returns
        -------
        torch.Tensor
            Owner-major coface signal.
        """
        outputs, pulled = [], 0
        for worker in range(self.num_parts):
            ghost = self.ghosts[worker]
            buffer = torch.cat(
                [x[self._slice(self.plan.face_offsets, worker)], x[ghost]]
            )
            pulled += ghost.numel()
            outputs.append(torch.sparse.mm(self.local_t[worker], buffer))
        self.comm.record(
            f"{self.name}.expand",
            pulled,
            x.size(1),
            x.element_size(),
            self.plan.ghost_pairs,
        )
        return torch.cat(outputs)

    def fold(
        self, y: torch.Tensor, matrices: list[torch.Tensor] | None = None
    ) -> torch.Tensor:
        r"""Compute :math:`B y` (cofaces to faces).

        Each worker pushes partial sums for its ghost faces to their owners.

        Parameters
        ----------
        y : torch.Tensor
            Owner-major coface signal.
        matrices : list[torch.Tensor], optional
            Local matrices to use instead of ``B``'s pieces.

        Returns
        -------
        torch.Tensor
            Owner-major face signal.
        """
        matrices = self.local if matrices is None else matrices
        out = y.new_zeros((self.num_faces, y.size(1)))
        pushed = 0
        for worker in range(self.num_parts):
            faces = self._slice(self.plan.face_offsets, worker)
            partial = torch.sparse.mm(
                matrices[worker],
                y[self._slice(self.plan.coface_offsets, worker)],
            )
            num_owned = faces.stop - faces.start
            out[faces] += partial[:num_owned]
            ghost = self.ghosts[worker]
            out.index_add_(0, ghost, partial[num_owned:])
            pushed += ghost.numel()
        self.comm.record(
            f"{self.name}.fold",
            pushed,
            y.size(1),
            y.element_size(),
            self.plan.ghost_pairs,
        )
        return out

    def up(self, x: torch.Tensor) -> torch.Tensor:
        r"""Apply :math:`BB^\top` to a face signal.

        Parameters
        ----------
        x : torch.Tensor
            Owner-major face signal.

        Returns
        -------
        torch.Tensor
            Owner-major face signal.
        """
        return self.fold(self.expand(x))

    def down(self, x: torch.Tensor) -> torch.Tensor:
        r"""Apply :math:`B^\top B` to a coface signal.

        Parameters
        ----------
        x : torch.Tensor
            Owner-major coface signal.

        Returns
        -------
        torch.Tensor
            Owner-major coface signal.
        """
        return self.expand(self.fold(x))

    def up_diagonal(self) -> torch.Tensor:
        r"""Return the diagonal of :math:`BB^\top`, computed once and cached.

        Returns
        -------
        torch.Tensor
            Owner-major face vector.
        """
        if self._diagonal is None:
            squared = [
                torch.sparse_coo_tensor(
                    m.indices(), m.values().square(), m.shape
                ).coalesce()
                for m in self.local
            ]
            ones = torch.ones(
                (int(self.plan.coface_offsets[-1]), 1),
                dtype=self.dtype,
                device=self.device,
            )
            self._diagonal = self.fold(ones, matrices=squared).squeeze(1)
        return self._diagonal


def distributed_graph_norm(
    norm: GraphNorm,
    x: torch.Tensor,
    offsets: torch.Tensor,
    comm: CommLog,
    name: str,
) -> torch.Tensor:
    """Apply PyG ``GraphNorm`` over all workers' rows with two all-reduces.

    ``GraphNorm`` normalizes with the mean and variance over the whole graph,
    also at eval time, so it needs global statistics.

    Parameters
    ----------
    norm : GraphNorm
        Trained normalization module.
    x : torch.Tensor
        Owner-major signal.
    offsets : torch.Tensor
        Owner-major offsets, length ``P + 1``.
    comm : CommLog
        Communication log.
    name : str
        Prefix for log records.

    Returns
    -------
    torch.Tensor
        Normalized owner-major signal.
    """
    num_parts = offsets.numel() - 1
    blocks = [
        x[int(offsets[worker]) : int(offsets[worker + 1])]
        for worker in range(num_parts)
    ]
    count = x.size(0)
    total = torch.stack([block.sum(dim=0) for block in blocks]).sum(dim=0)
    comm.record(
        f"{name}.allreduce_mean",
        num_parts,
        x.size(1) + 1,
        x.element_size(),
        num_parts,
    )
    mean = total / count
    centred = [block - mean * norm.mean_scale for block in blocks]
    squares = torch.stack(
        [(block * block).sum(dim=0) for block in centred]
    ).sum(dim=0)
    comm.record(
        f"{name}.allreduce_var",
        num_parts,
        x.size(1),
        x.element_size(),
        num_parts,
    )
    std = (squares / count + norm.eps).sqrt()
    return torch.cat(
        [norm.weight * block / std + norm.bias for block in centred]
    )
