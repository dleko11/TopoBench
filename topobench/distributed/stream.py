"""One-GPU exact SCN: parts streamed through the GPU in succession.

The lifted input features of every part stay in host memory. For each part,
the GPU loads its rows, applies the input encoder, and keeps only the hidden
output; the backward pass reloads the part to form the weight gradient.
GraphNorm followed by Linear is one affine map,

    Linear(GraphNorm(x)) = x @ M + c,  M = diag(w / std) W^T,
    c = (b - w * s * mean / std) W^T + bias,

so only ``x @ M`` touches the wide features. GraphNorm's whole-graph column
statistics are data statistics: the mean and the second moment are streamed
once in float64, and the variance after centering with the learnable scale
``s`` follows in closed form, ``E[x^2] - 2 s mean^2 + s^2 mean^2``. The
function and its gradients are GraphNorm's.

Everything after the encoder (the SCN layers, the shared-face exchange and
the readout) runs on the GPU for all parts at once, as in the in-process
simulation; on graphs whose hidden states dominate, those stages would be
streamed too.
"""

from __future__ import annotations

import torch

from topobench.distributed.halo import HaloPlan, lift_input_features
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


class StreamedMatmul(torch.autograd.Function):
    """``x[start:end] @ M`` with ``x`` in host memory and no input gradient."""

    @staticmethod
    def forward(ctx, matrix, source, start, end):
        """Load the rows to the matrix's device and multiply.

        Parameters
        ----------
        ctx : Any
            Autograd context.
        matrix : torch.Tensor
            ``(features, hidden)`` matrix on the compute device.
        source : torch.Tensor
            Host tensor of input features (not differentiated).
        start, end : int
            Row range.

        Returns
        -------
        torch.Tensor
            ``(end - start, hidden)`` product on the compute device.
        """
        ctx.source, ctx.start, ctx.end = source, start, end
        rows = source[start:end].to(
            matrix.device, matrix.dtype, non_blocking=True
        )
        return rows @ matrix

    @staticmethod
    def backward(ctx, grad_output):
        """Reload the rows and return the matrix gradient only.

        Parameters
        ----------
        ctx : Any
            Autograd context.
        grad_output : torch.Tensor
            Gradient of the product.

        Returns
        -------
        tuple
            Gradient for ``matrix``; ``None`` for the other inputs.
        """
        rows = ctx.source[ctx.start : ctx.end].to(
            grad_output.device, grad_output.dtype, non_blocking=True
        )
        return rows.t() @ grad_output, None, None, None


class StreamedSCN:
    """SCN over owner-partitioned cells on one device, encoder streamed by part.

    Parameters
    ----------
    model : SCNNodeClassifier
        Model, already on ``device``.
    plan : HaloPlan
        Owner-partitioned complex (each part lifted in succession).
    x_0 : torch.Tensor
        Node features on the host.
    device : torch.device
        Compute device.
    dtype : torch.dtype
        Compute dtype.
    pin_memory : bool, optional
        Pin the host features for faster transfers.
    """

    def __init__(
        self,
        model: SCNNodeClassifier,
        plan: HaloPlan,
        x_0: torch.Tensor,
        device: torch.device,
        dtype: torch.dtype,
        pin_memory: bool = True,
    ) -> None:
        self.model = model
        self.plan = plan
        self.device = device
        self.dtype = dtype
        features = lift_input_features(plan, x_0)
        if pin_memory and device.type == "cuda":
            features = [f.pin_memory() for f in features]
        self.features = features
        self.moments = [self._moments(rank) for rank in RANKS]
        self.comm = CommLog()
        self.boundary = {
            rank: DistributedBoundaryOperator(
                plan.incidences[rank], self.comm, f"B{rank}", dtype, device
            )
            for rank in (1, 2)
        }
        self.operators = hodge_operators(
            self.boundary, [plan.num_cells(r) for r in RANKS], dtype, device
        )

    def _parts(self, rank: int):
        offsets = self.plan.offsets[rank]
        for part in range(offsets.numel() - 1):
            start, end = int(offsets[part]), int(offsets[part + 1])
            if end > start:
                yield start, end

    def _moments(self, rank: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Stream the column mean and second moment of one rank (float64).

        Parameters
        ----------
        rank : int
            Simplex rank.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Column mean and mean of squares on the compute device.
        """
        source = self.features[rank]
        total = torch.zeros(
            source.size(1), dtype=torch.float64, device=self.device
        )
        squares = torch.zeros_like(total)
        for start, end in self._parts(rank):
            rows = source[start:end].to(self.device, torch.float64)
            total += rows.sum(dim=0)
            squares += (rows * rows).sum(dim=0)
        count = max(source.size(0), 1)
        return total / count, squares / count

    def encode(self, rank: int) -> torch.Tensor:
        """Encoder output for one rank, streaming its parts through the device.

        Parameters
        ----------
        rank : int
            Simplex rank.

        Returns
        -------
        torch.Tensor
            Hidden features of every owned cell of that rank, owner-major.
        """
        encoder = getattr(self.model.feature_encoder, f"encoder_{rank}")
        norm, linear = encoder.BN, encoder.linear
        mean, second = self.moments[rank]
        scale = norm.mean_scale.double()
        variance = (
            second - 2 * scale * mean * mean + scale * scale * mean * mean
        )
        std = (variance + norm.eps).sqrt().to(self.dtype)
        mean = mean.to(self.dtype)
        gain = norm.weight / std
        matrix = gain.unsqueeze(1) * linear.weight.t()
        shift = (norm.bias - gain * norm.mean_scale * mean) @ linear.weight.t()
        shift = shift + linear.bias
        outputs = [
            StreamedMatmul.apply(matrix, self.features[rank], start, end)
            for start, end in self._parts(rank)
        ]
        hidden = torch.cat(outputs) + shift
        return encoder.dropout(encoder.relu(hidden))

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
            encode=self.encode,
        )
