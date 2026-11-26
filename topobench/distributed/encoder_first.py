"""Encoder-first SCN input encoding: lifted input features never exist.

TopoBench lifts node features to every cell with ``ProjectionSum`` and then
encodes each rank with GraphNorm, Linear, ReLU and dropout. ``ProjectionSum``
gives a rank-k cell ``x_k(sigma) = k! * sum_{v in sigma} x_0(v)``, and
GraphNorm followed by Linear is one affine map ``x @ M + c``. Hence

    encoder_k(sigma) = ReLU(k! * sum_{v in sigma} (x_0(v) @ M_k) + c_k),

so each rank only needs the per-node vectors ``x_0 @ M_k`` (hidden width),
summed over a cell's vertices. GraphNorm's statistics are over the rank's
cells, and both are recovered from node features without lifting them:

    sum_sigma x_k(sigma)   = k!   * sum_v c_k(v) x_0(v)
    sum_sigma x_k(sigma)^2 = k!^2 * (sum_v c_k(v) x_0(v)^2
                                     + 2 sum_{u<v} p_k(u, v) x_0(u) x_0(v))

where ``c_k(v)`` counts the rank-k cells containing ``v`` and ``p_k(u, v)``
the rank-k cells containing both ``u`` and ``v`` (only edges can be such
pairs in a clique complex). The variance after centering with GraphNorm's
learnable scale ``s`` is ``E[x^2] - 2 s mean^2 + s^2 mean^2``. The function
and its gradients equal those of the lifted pipeline; only rounding differs.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import torch

from topobench.distributed.halo import HaloPlan
from topobench.distributed.operators import (
    CommLog,
    DistributedBoundaryOperator,
)
from topobench.distributed.scn import (
    RANKS,
    SCNNodeClassifier,
    hodge_operators,
    scn_forward,
)


def _pair_counts(
    cells: torch.Tensor, num_nodes: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Count, for every vertex pair, the cells that contain both.

    Parameters
    ----------
    cells : torch.Tensor
        Cells as sorted vertex tuples.
    num_nodes : int
        Number of nodes.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        First vertex, second vertex and cell count of each distinct pair.
    """
    width = cells.size(1)
    if width < 2 or cells.size(0) == 0:
        empty = cells.new_empty(0)
        return empty, empty, empty
    pairs = torch.cat(
        [
            cells[:, i] * num_nodes + cells[:, j]
            for i in range(width)
            for j in range(i + 1, width)
        ]
    )
    keys, counts = torch.unique(pairs, return_counts=True)
    return keys // num_nodes, keys % num_nodes, counts


class EncoderFirst:
    """Evaluate the SCN input encoders from node features only.

    Parameters
    ----------
    model : SCNNodeClassifier
        Model whose ``feature_encoder`` is evaluated.
    x_0 : torch.Tensor
        Node features on the compute device, indexed by the node ids used in
        ``cells``.
    cells : list[torch.Tensor]
        Per rank, the cells whose encodings are wanted, as sorted vertex
        tuples on the compute device (rank 0: ``(n, 1)`` node ids).
    dtype : torch.dtype
        Compute dtype.
    reduce : callable, optional
        Sums a tensor across processes (identity on one process).
    chunk_size : int, optional
        Vertex pairs processed at a time for the second moment.
    """

    def __init__(
        self,
        model: SCNNodeClassifier,
        x_0: torch.Tensor,
        cells: list[torch.Tensor],
        dtype: torch.dtype,
        reduce: Callable[[torch.Tensor], torch.Tensor] | None = None,
        chunk_size: int = 8192,
    ) -> None:
        self.model = model
        self.x_0 = x_0.to(dtype)
        self.cells = cells
        self.dtype = dtype
        reduce = reduce or (lambda t: t)
        self.moments = [
            self._moments(rank, cells[rank], reduce, chunk_size)
            for rank in RANKS
        ]

    def _moments(self, rank, cells, reduce, chunk_size):
        """Column mean and second moment of the lifted rank-k features.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Mean and mean of squares, float64, on the compute device.
        """
        x = self.x_0.double()
        num_nodes = x.size(0)
        degree = torch.bincount(cells.flatten(), minlength=num_nodes).double()
        first = (degree.unsqueeze(1) * x).sum(dim=0)
        second = (degree.unsqueeze(1) * x * x).sum(dim=0)
        left, right, counts = _pair_counts(cells, num_nodes)
        for start in range(0, left.numel(), chunk_size):
            stop = start + chunk_size
            second += 2 * (
                counts[start:stop].double().unsqueeze(1)
                * x[left[start:stop]]
                * x[right[start:stop]]
            ).sum(dim=0)
        totals = reduce(torch.stack([first, second]))
        count = reduce(
            torch.tensor([float(cells.size(0))], device=x.device)
        ).double()
        count = count.clamp(min=1)
        scale = math.factorial(rank)
        return scale * totals[0] / count, scale**2 * totals[1] / count

    def encode(self, rank: int) -> torch.Tensor:
        """Encoder output of every cell of one rank.

        Parameters
        ----------
        rank : int
            Simplex rank.

        Returns
        -------
        torch.Tensor
            Hidden features in the order of ``cells[rank]``.
        """
        encoder = getattr(self.model.feature_encoder, f"encoder_{rank}")
        norm, linear = encoder.BN, encoder.linear
        mean, second = self.moments[rank]
        s = norm.mean_scale.double()
        variance = second - 2 * s * mean * mean + s * s * mean * mean
        std = (variance + norm.eps).sqrt().to(self.dtype)
        mean = mean.to(self.dtype)
        gain = norm.weight / std
        matrix = gain.unsqueeze(1) * linear.weight.t()
        shift = (norm.bias - gain * norm.mean_scale * mean) @ linear.weight.t()
        shift = shift + linear.bias
        per_node = self.x_0 @ matrix
        cells = self.cells[rank]
        summed = per_node[cells].sum(dim=1) * math.factorial(rank)
        return encoder.dropout(encoder.relu(summed + shift))


class EncoderFirstSCN:
    """SCN on one device over an owner-partitioned complex, encoder-first.

    Parameters
    ----------
    model : SCNNodeClassifier
        Model, already on ``device``.
    plan : HaloPlan
        Owner-partitioned complex.
    x_0 : torch.Tensor
        Node features (any device).
    device : torch.device
        Compute device.
    dtype : torch.dtype
        Compute dtype.
    """

    def __init__(
        self,
        model: SCNNodeClassifier,
        plan: HaloPlan,
        x_0: torch.Tensor,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        self.model = model
        self.plan = plan
        cells = [plan.cells(rank).to(device) for rank in RANKS]
        self.encoder = EncoderFirst(model, x_0.to(device), cells, dtype)
        self.boundary = {
            rank: DistributedBoundaryOperator(
                plan.incidences[rank], CommLog(), f"B{rank}", dtype, device
            )
            for rank in (1, 2)
        }
        self.operators = hodge_operators(
            self.boundary, [plan.num_cells(r) for r in RANKS], dtype, device
        )

    def __call__(self) -> dict:
        """Run the forward pass.

        Returns
        -------
        dict
            Stage outputs in owner-major order; ``logits`` for all nodes.
        """
        return scn_forward(
            self.model,
            None,
            self.boundary,
            self.operators,
            None,
            encode=self.encoder.encode,
        )
