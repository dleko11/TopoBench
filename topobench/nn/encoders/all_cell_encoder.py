"""Class to apply BaseEncoder to the features of higher order structures."""

import torch
import torch_geometric
from torch_geometric.nn.norm import GraphNorm

from topobench.nn.encoders.base import AbstractFeatureEncoder


class AllCellFeatureEncoder(AbstractFeatureEncoder):
    r"""Encoder class to apply BaseEncoder.

    The BaseEncoder is applied to the features of higher order
    structures. The class creates a BaseEncoder for each dimension specified in
    selected_dimensions. Then during the forward pass, the BaseEncoders are
    applied to the features of the corresponding dimensions.

    Parameters
    ----------
    in_channels : list[int]
        Input dimensions for the features.
    out_channels : list[int]
        Output dimensions for the features.
    proj_dropout : float, optional
        Dropout for the BaseEncoders (default: 0).
    selected_dimensions : list[int], optional
        List of indexes to apply the BaseEncoders to (default: None).
    encoder_first : bool, optional
        Compute the rank-1 and rank-2 encodings of a simplicial complex from
        node features and the unsigned incidences, without the lifted
        ``x_1``/``x_2`` (use with ``feature_lifting=null``). Exact for the
        ``ProjectionSum`` feature lifting (default: False).
    **kwargs : dict, optional
        Additional keyword arguments.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        proj_dropout=0,
        selected_dimensions=None,
        encoder_first=False,
        **kwargs,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.encoder_first = bool(encoder_first)
        self.dimensions = (
            selected_dimensions
            if (
                selected_dimensions is not None
            )  # and len(selected_dimensions) <= len(self.in_channels))
            else range(len(self.in_channels))
        )
        for i in self.dimensions:
            setattr(
                self,
                f"encoder_{i}",
                BaseEncoder(
                    self.in_channels[i],
                    self.out_channels,
                    dropout=proj_dropout,
                ),
            )

    def __repr__(self):
        return f"{self.__class__.__name__}(in_channels={self.in_channels}, out_channels={self.out_channels}, dimensions={self.dimensions})"

    def forward(
        self, data: torch_geometric.data.Data
    ) -> torch_geometric.data.Data:
        r"""Forward pass.

        The method applies the BaseEncoders to the features of the selected_dimensions.

        Parameters
        ----------
        data : torch_geometric.data.Data
            Input data object which should contain x_{i} features for each i in the selected_dimensions.

        Returns
        -------
        torch_geometric.data.Data
            Output data object with updated x_{i} features.
        """
        if not hasattr(data, "x_0"):
            data.x_0 = data.x

        if self.encoder_first:
            return self._encode_from_nodes(data)

        for i in self.dimensions:
            if hasattr(data, f"x_{i}") and hasattr(self, f"encoder_{i}"):
                batch = getattr(data, f"batch_{i}")
                data[f"x_{i}"] = getattr(self, f"encoder_{i}")(
                    data[f"x_{i}"], batch
                )
        return data

    def _encode_from_nodes(
        self, data: torch_geometric.data.Data
    ) -> torch_geometric.data.Data:
        r"""Encode every rank from node features only (encoder-first).

        ``ProjectionSum`` lifts ``x_1 = |B_1|^T x_0`` and
        ``x_2 = |B_2|^T x_1``, and GraphNorm followed by Linear is affine,
        so each rank's pre-activation is the same incidence chain applied
        to per-node hidden vectors, plus a per-graph constant. GraphNorm's
        per-graph statistics of the lifted features are recovered from node
        features, per-node cell counts and per-edge cell counts.

        Parameters
        ----------
        data : torch_geometric.data.Data
            Batch with ``x_0``, ``batch_0`` and unsigned incidences.

        Returns
        -------
        torch_geometric.data.Data
            The batch with encoded ``x_i`` for every selected rank.
        """
        x_0 = data.x_0
        batch_0 = data.batch_0
        num_nodes = x_0.size(0)
        num_graphs = int(batch_0.max()) + 1 if batch_0.numel() else 1
        top = max(self.dimensions)
        if top > 2:
            raise NotImplementedError("encoder_first supports ranks up to 2.")

        incidences = {}
        for rank in range(1, top + 1):
            matrix = data[f"incidence_{rank}"].coalesce()
            incidences[rank] = torch.sparse_coo_tensor(
                matrix.indices(),
                matrix.values().abs().to(x_0.dtype),
                matrix.shape,
            ).coalesce()

        if top >= 1:
            rows, cols = incidences[1].indices()
            num_edges = incidences[1].size(1)
            order = torch.argsort(cols * num_nodes + rows)
            ends = rows[order].view(num_edges, 2)
            edge_graph = batch_0[ends[:, 0]]

        for rank in self.dimensions:
            encoder = getattr(self, f"encoder_{rank}")
            if rank == 0:
                data["x_0"] = encoder(x_0, batch_0)
                continue
            if rank == 1:
                node_weight = torch.bincount(
                    ends.flatten(), minlength=num_nodes
                ).to(torch.float64)
                pair_weight = torch.ones(
                    num_edges, dtype=torch.float64, device=x_0.device
                )
                cell_graph = edge_graph
                factor = 1.0
            else:
                rows2, cols2 = incidences[2].indices()
                num_cells = incidences[2].size(1)
                pair_weight = torch.bincount(rows2, minlength=num_edges).to(
                    torch.float64
                )
                node_weight = torch.zeros(
                    num_nodes, dtype=torch.float64, device=x_0.device
                )
                node_weight.index_add_(0, ends[:, 0], pair_weight)
                node_weight.index_add_(0, ends[:, 1], pair_weight)
                node_weight = node_weight / 2  # triangles containing v
                order2 = torch.argsort(cols2 * num_edges + rows2)
                first_edge = rows2[order2].view(num_cells, 3)[:, 0]
                cell_graph = edge_graph[first_edge]
                factor = 2.0
            data[f"x_{rank}"] = self._encode_rank(
                encoder,
                x_0,
                batch_0,
                num_graphs,
                ends,
                edge_graph,
                node_weight,
                pair_weight,
                cell_graph,
                factor,
                incidences,
                rank,
            )
        return data

    @staticmethod
    def _encode_rank(
        encoder,
        x_0,
        batch_0,
        num_graphs,
        ends,
        edge_graph,
        node_weight,
        pair_weight,
        cell_graph,
        factor,
        incidences,
        rank,
        chunk_size=8192,
    ):
        r"""Encode one rank from node features (see ``_encode_from_nodes``).

        Parameters
        ----------
        encoder : BaseEncoder
            The rank's encoder.
        x_0 : torch.Tensor
            Node features.
        batch_0 : torch.Tensor
            Graph id of every node.
        num_graphs : int
            Number of graphs in the batch.
        ends : torch.Tensor
            Endpoints of every edge.
        edge_graph : torch.Tensor
            Graph id of every edge.
        node_weight : torch.Tensor
            Rank-k cells containing each node.
        pair_weight : torch.Tensor
            Rank-k cells containing each edge's endpoints.
        cell_graph : torch.Tensor
            Graph id of every rank-k cell.
        factor : float
            ``k!``, the ProjectionSum scale of a rank-k cell.
        incidences : dict
            Unsigned incidences by rank.
        rank : int
            Rank to encode.
        chunk_size : int, optional
            Edges processed at a time for the pair term.

        Returns
        -------
        torch.Tensor
            Encoded rank-k features.
        """
        norm, linear = encoder.BN, encoder.linear
        x = x_0.to(torch.float64)
        width = x.size(1)
        first = torch.zeros(
            num_graphs, width, dtype=torch.float64, device=x.device
        )
        second = torch.zeros_like(first)
        first.index_add_(0, batch_0, node_weight.unsqueeze(1) * x)
        second.index_add_(0, batch_0, node_weight.unsqueeze(1) * x * x)
        for start in range(0, ends.size(0), chunk_size):
            stop = start + chunk_size
            pair = ends[start:stop]
            second.index_add_(
                0,
                edge_graph[start:stop],
                2
                * pair_weight[start:stop].unsqueeze(1)
                * x[pair[:, 0]]
                * x[pair[:, 1]],
            )
        counts = torch.bincount(cell_graph, minlength=num_graphs)
        counts = counts.clamp(min=1).to(torch.float64).unsqueeze(1)
        mean = factor * first / counts
        mean_square = factor**2 * second / counts
        scale = norm.mean_scale.to(torch.float64)
        variance = (
            mean_square - 2 * scale * mean * mean + scale * scale * mean * mean
        )
        std = (variance + norm.eps).sqrt().to(x_0.dtype)
        mean = mean.to(x_0.dtype)
        gain = norm.weight / std  # (graphs, features)
        per_node = (x_0 * gain[batch_0]) @ linear.weight.t()
        hidden = torch.sparse.mm(incidences[1].t(), per_node)
        if rank == 2:
            hidden = torch.sparse.mm(incidences[2].t(), hidden)
        shift = (norm.bias - gain * norm.mean_scale * mean) @ linear.weight.t()
        hidden = hidden + shift[cell_graph] + linear.bias
        return encoder.dropout(encoder.relu(hidden))


class BaseEncoder(torch.nn.Module):
    r"""Base encoder class used by AllCellFeatureEncoder.

    This class uses two linear layers with GraphNorm, Relu activation function, and dropout between the two layers.

    Parameters
    ----------
    in_channels : int
        Dimension of input features.
    out_channels : int
        Dimensions of output features.
    dropout : float, optional
        Percentage of channels to discard between the two linear layers (default: 0).
    """

    def __init__(self, in_channels, out_channels, dropout=0):
        super().__init__()
        self.BN = GraphNorm(in_channels)
        self.linear = torch.nn.Linear(in_channels, out_channels)
        self.relu = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(dropout)

    def __repr__(self):
        return f"{self.__class__.__name__}(in_channels={self.linear.in_features}, out_channels={self.linear.out_features})"

    def forward(self, x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        r"""Forward pass of the encoder.

        It applies two linear layers with GraphNorm, Relu activation function, and dropout between the two layers.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of dimensions [N, in_channels].
        batch : torch.Tensor
            The batch vector which assigns each element to a specific example.

        Returns
        -------
        torch.Tensor
            Output tensor of shape [N, out_channels].
        """
        x = self.BN(x, batch=batch) if batch.shape[0] > 0 else self.BN(x)
        x = self.linear(x)
        x = self.dropout(self.relu(x))
        return x
