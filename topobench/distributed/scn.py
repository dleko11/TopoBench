"""SCN node classifier: centralized reference and owner-partitioned forward.

The distributed forward reuses the trained modules and TopoBench's
``UnsignedHodgeOperator`` and ``SCN2MatrixFreeLayer`` unchanged. Only the
boundary operators (and ``GraphNorm``'s statistics) are replaced by their
communicating versions.
"""

from __future__ import annotations

import torch
from torch_geometric.data import Data

from topobench.distributed.halo import HaloPlan, encode_simplices
from topobench.distributed.operators import (
    CommLog,
    DistributedBoundaryOperator,
    distributed_graph_norm,
)
from topobench.nn.backbones.simplicial.incidence_operators import (
    UnsignedHodgeOperator,
)
from topobench.nn.backbones.simplicial.scn import SCN2MatrixFree
from topobench.nn.encoders.all_cell_encoder import AllCellFeatureEncoder
from topobench.nn.readouts.propagate_signal_down import PropagateSignalDown
from topobench.nn.wrappers.simplicial.scn_wrapper import SCNWrapper
from topobench.transforms.liftings.graph2simplicial.clique_lifting import (
    SimplicialCliqueLiftingSelective,
)

RANKS = (0, 1, 2)


class SCNNodeClassifier(torch.nn.Module):
    """SCN with the submodules and forward of ``TBModel`` (configs/model/simplicial/scn.yaml).

    Submodule names match ``TBModel``, so a TopoBench SCN checkpoint's
    ``feature_encoder.*``, ``backbone.*`` and ``readout.*`` weights load.

    Parameters
    ----------
    in_channels : int
        Input feature dimension, shared by all ranks under ``ProjectionSum``.
    hidden_channels : int
        Hidden dimension.
    num_classes : int
        Number of classes.
    n_layers : int
        Number of SCN layers.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        num_classes: int,
        n_layers: int,
    ) -> None:
        super().__init__()
        self.feature_encoder = AllCellFeatureEncoder(
            in_channels=[in_channels] * len(RANKS),
            out_channels=hidden_channels,
            proj_dropout=0.0,
            selected_dimensions=list(RANKS),
        )
        self.backbone = SCNWrapper(
            SCN2MatrixFree(
                hidden_channels,
                hidden_channels,
                hidden_channels,
                n_layers=n_layers,
            ),
            out_channels=hidden_channels,
            num_cell_dimensions=len(RANKS),
        )
        self.readout = PropagateSignalDown(
            readout_name="PropagateSignalDown",
            num_cell_dimensions=len(RANKS),
            hidden_dim=hidden_channels,
            out_channels=num_classes,
            task_level="node",
            pooling_type="sum",
        )

    def forward(self, batch: Data) -> dict:
        """Run the same three steps as ``TBModel.forward``.

        Parameters
        ----------
        batch : Data
            Lifted batch; the encoder overwrites its ``x_i`` in place.

        Returns
        -------
        dict
            Model output with ``logits``.
        """
        model_out = self.feature_encoder(batch)
        model_out = self.backbone(model_out)
        return self.readout(model_out=model_out, batch=batch)


def build_reference_batch(
    x_0: torch.Tensor, edge_index: torch.Tensor, y: torch.Tensor
) -> Data:
    """Lift the whole graph with TopoBench's ``clique_selective`` transform.

    Parameters
    ----------
    x_0 : torch.Tensor
        Node features.
    edge_index : torch.Tensor
        Undirected edge index.
    y : torch.Tensor
        Node labels.

    Returns
    -------
    Data
        Lifted full-graph batch with ``batch_i`` vectors.
    """
    lifting = SimplicialCliqueLiftingSelective(
        complex_dim=2,
        signed=False,
        feature_lifting="ProjectionSum",
        neighborhoods=["incidence-1", "incidence-2"],
    )
    batch = lifting(Data(x=x_0, edge_index=edge_index, y=y))
    for rank in RANKS:
        batch[f"batch_{rank}"] = torch.zeros(
            batch[f"x_{rank}"].size(0), dtype=torch.long
        )
    return batch


def cast_batch(batch: Data, dtype: torch.dtype) -> Data:
    """Cast features and incidences of a lifted batch.

    Parameters
    ----------
    batch : Data
        Lifted batch.
    dtype : torch.dtype
        Target floating dtype.

    Returns
    -------
    Data
        The same batch, modified in place.
    """
    for rank in RANKS:
        batch[f"x_{rank}"] = batch[f"x_{rank}"].to(dtype)
    for rank in (1, 2):
        batch[f"incidence_{rank}"] = batch[f"incidence_{rank}"].to(dtype)
    return batch


def reference_cells(batch: Data) -> list[torch.Tensor]:
    """Recover each reference row's vertex tuple from the incidences.

    Parameters
    ----------
    batch : Data
        Lifted full-graph batch.

    Returns
    -------
    list[torch.Tensor]
        Sorted vertex tuples per rank, in the reference row order.
    """
    cells = [torch.arange(batch.x_0.size(0)).view(-1, 1)]
    for rank in (1, 2):
        incidence = batch[f"incidence_{rank}"].coalesce()
        rows, cols = incidence.indices()
        num_cells = incidence.size(1)
        if not torch.equal(
            torch.bincount(cols, minlength=num_cells),
            torch.full((num_cells,), rank + 1),
        ):
            raise ValueError(f"Rank-{rank} cells must have {rank + 1} faces.")
        order = torch.argsort(cols * incidence.size(0) + rows)
        faces = rows[order].view(num_cells, rank + 1)
        vertices = cells[rank - 1][faces].view(num_cells, -1)
        # Each vertex appears in ``rank`` of the ``rank + 1`` faces.
        cells.append(vertices.sort(dim=1).values[:, ::rank])
    return cells


def reference_stages(
    model: SCNNodeClassifier, batch: Data, clone: bool = True
) -> dict:
    """Run the real forward on the full lifted graph and keep intermediates.

    Parameters
    ----------
    model : SCNNodeClassifier
        Model in eval mode.
    batch : Data
        Lifted full-graph batch.
    clone : bool, optional
        Work on a copy. With ``False`` the encoder overwrites ``batch.x_i``,
        which frees the wide input features as it goes.

    Returns
    -------
    dict
        Stage name to tensor in reference row order.
    """
    if clone:
        batch = batch.clone()
    layer_outputs = []
    hooks = [
        layer.register_forward_hook(
            lambda _module, _inputs, output: layer_outputs.append(output)
        )
        for layer in model.backbone.backbone.layers
    ]
    try:
        with torch.no_grad():
            encoded = model.feature_encoder(batch)
            stages = {
                f"encoder/x_{r}": encoded[f"x_{r}"].clone() for r in RANKS
            }
            model_out = model.backbone(encoded)
            for index, output in enumerate(layer_outputs):
                for rank in RANKS:
                    stages[f"layer{index}/x_{rank}"] = output[rank]
            for rank in RANKS:
                stages[f"backbone/x_{rank}"] = model_out[f"x_{rank}"].clone()
            model_out = model.readout(model_out=model_out, batch=encoded)
    finally:
        for hook in hooks:
            hook.remove()
    stages["readout/x_1"] = model_out["x_1"]
    stages["readout/x_0"] = model_out["x_0"]
    stages["logits"] = model_out["logits"]
    return stages


def distributed_stages(
    model: SCNNodeClassifier,
    plan: HaloPlan,
    features: list[torch.Tensor],
    comm: CommLog,
    grad: bool = False,
) -> dict:
    """Run the same model across workers, exchanging only ghost values.

    Parameters
    ----------
    model : SCNNodeClassifier
        Model in eval mode.
    plan : HaloPlan
        Owner-partitioned complex.
    features : list[torch.Tensor]
        Owner-major input features per rank.
    comm : CommLog
        Communication log; operator setup is logged under ``"setup"``.
    grad : bool, optional
        Record autograd. Every exchange is differentiable, and the backward
        of an expand is a fold and vice versa.

    Returns
    -------
    dict
        Stage name to tensor in owner-major row order.
    """
    dtype, device = features[0].dtype, features[0].device
    comm.phase = "setup"
    boundary = {
        rank: DistributedBoundaryOperator(
            plan.incidences[rank], comm, f"B{rank}", dtype, device
        )
        for rank in (1, 2)
    }
    operators = hodge_operators(
        boundary, [plan.num_cells(rank) for rank in RANKS], dtype, device
    )
    comm.phase = "forward"

    def graph_norm(norm, x, rank):
        return distributed_graph_norm(
            norm, x, plan.offsets[rank], comm, f"encoder{rank}.graphnorm"
        )

    with torch.set_grad_enabled(grad):
        return scn_forward(model, features, boundary, operators, graph_norm)


def hodge_operators(
    boundary: dict, counts: list[int], dtype: torch.dtype, device
) -> tuple[UnsignedHodgeOperator, ...]:
    """Build SCN's normalized Hodge operators as ``SCN2MatrixFree`` does.

    Parameters
    ----------
    boundary : dict
        Boundary operators for ranks 1 and 2 (any object with ``up``,
        ``down`` and ``up_diagonal``).
    counts : list[int]
        Number of rows per rank the operators act on.
    dtype : torch.dtype
        Signal dtype.
    device : torch.device
        Signal device.

    Returns
    -------
    tuple[UnsignedHodgeOperator, ...]
        Operators for ranks 0, 1 and 2.
    """
    return (
        UnsignedHodgeOperator(
            lower=None,
            upper=boundary[1],
            num_simplices=counts[0],
            dtype=dtype,
            device=device,
            normalize=True,
        ),
        UnsignedHodgeOperator(
            lower=boundary[1],
            upper=boundary[2],
            num_simplices=counts[1],
            dtype=dtype,
            device=device,
            normalize=True,
        ),
        UnsignedHodgeOperator(
            lower=boundary[2],
            upper=None,
            num_simplices=counts[2],
            dtype=dtype,
            device=device,
            normalize=True,
        ),
    )


def scn_forward(
    model: SCNNodeClassifier,
    features: list[torch.Tensor],
    boundary: dict,
    operators: tuple,
    graph_norm,
    encode=None,
) -> dict:
    """Run SCN's encoder, layers, wrapper and readout on partitioned signals.

    Shared by the in-process simulation, the ``torch.distributed`` runtime
    and the one-GPU streamed run; they differ only in their boundary
    operators and in how the input encoder is evaluated.

    Parameters
    ----------
    model : SCNNodeClassifier
        Model whose trained modules are reused.
    features : list[torch.Tensor] or None
        Input features per rank (unused when ``encode`` is given).
    boundary : dict
        Boundary operators for ranks 1 and 2, providing ``fold``.
    operators : tuple
        Hodge operators for ranks 0, 1 and 2.
    graph_norm : callable or None
        ``graph_norm(norm_module, x, rank)`` with whole-graph statistics.
    encode : callable, optional
        ``encode(rank)`` returning the encoder output for that rank, in
        place of GraphNorm, Linear, ReLU and dropout on ``features``.

    Returns
    -------
    dict
        Stage name to tensor.
    """
    stages = {}
    encoded = []
    for rank in RANKS:
        if encode is not None:
            hidden = encode(rank)
        else:
            encoder = getattr(model.feature_encoder, f"encoder_{rank}")
            hidden = graph_norm(encoder.BN, features[rank], rank)
            hidden = encoder.dropout(encoder.relu(encoder.linear(hidden)))
        encoded.append(hidden)
        stages[f"encoder/x_{rank}"] = hidden

    wrapper = model.backbone
    x = list(encoded)
    for index, layer in enumerate(wrapper.backbone.layers):
        x = list(layer(*x, operators))
        for rank in RANKS:
            stages[f"layer{index}/x_{rank}"] = x[rank]
    if wrapper.residual_connections:
        x = [
            getattr(wrapper, f"ln_{rank}")(x[rank] + encoded[rank])
            for rank in RANKS
        ]
    for rank in RANKS:
        stages[f"backbone/x_{rank}"] = x[rank]

    readout = model.readout
    for rank in readout.dimensions:
        conv = getattr(readout, f"agg_conv_{rank}")
        if conv.att or conv.aggr_norm or conv.update_func is not None:
            raise NotImplementedError("Only plain sum convolutions.")
        aggregated = boundary[rank].fold(torch.mm(x[rank], conv.weight))
        aggregated = getattr(readout, f"ln_{rank}")(aggregated)
        x[rank - 1] = getattr(readout, f"projector_{rank}")(
            torch.cat([aggregated, x[rank - 1]], dim=1)
        )
    stages["readout/x_1"] = x[1]
    stages["readout/x_0"] = x[0]
    stages["logits"] = readout.linear(x[0])
    return stages


def align_to_reference(
    plan: HaloPlan, reference: list[torch.Tensor]
) -> list[torch.Tensor]:
    """Map owner-major rows to reference rows and check the cell sets agree.

    Parameters
    ----------
    plan : HaloPlan
        Owner-partitioned complex.
    reference : list[torch.Tensor]
        Reference vertex tuples per rank.

    Returns
    -------
    list[torch.Tensor]
        For each rank, the reference row of every owner-major row.

    Raises
    ------
    AssertionError
        If the owned cells are not exactly the reference cells, each once.
    """
    alignment = []
    for rank in RANKS:
        owned = encode_simplices(plan.cells(rank), plan.num_nodes)
        target = encode_simplices(reference[rank], plan.num_nodes)
        assert owned.numel() == target.numel(), (
            f"rank {rank}: {owned.numel()} owned cells vs "
            f"{target.numel()} reference cells"
        )
        order = torch.argsort(target)
        position = torch.searchsorted(target[order], owned).clamp(
            max=max(target.numel() - 1, 0)
        )
        assert torch.equal(target[order][position], owned), (
            f"rank {rank}: owned cells missing from the reference complex"
        )
        rows = order[position]
        assert rows.unique().numel() == rows.numel(), (
            f"rank {rank}: a reference cell is owned more than once"
        )
        alignment.append(rows)
    return alignment


def compare_stages(
    distributed: dict, reference: dict, alignment: list[torch.Tensor]
) -> dict[str, float]:
    """Return the max absolute difference per stage after alignment.

    Parameters
    ----------
    distributed : dict
        Owner-major stage outputs.
    reference : dict
        Reference stage outputs.
    alignment : list[torch.Tensor]
        Reference row of each owner-major row, per rank.

    Returns
    -------
    dict[str, float]
        Max absolute difference per stage.
    """
    diffs = {}
    for name, value in distributed.items():
        rank = 0 if name == "logits" else int(name.rsplit("_", 1)[1])
        expected = reference[name][alignment[rank]]
        diffs[name] = (
            float((value - expected).abs().max()) if value.numel() else 0.0
        )
    return diffs
