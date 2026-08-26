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
    lift_encoded_features : bool, optional
        Encode node features first, then project them to higher ranks using
        incidence matrices instead of encoding pre-lifted features.
    **kwargs : dict, optional
        Additional keyword arguments.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        proj_dropout=0,
        selected_dimensions=None,
        lift_encoded_features=False,
        **kwargs,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.lift_encoded_features = bool(lift_encoded_features)
        self.dimensions = (
            selected_dimensions
            if (
                selected_dimensions is not None
            )  # and len(selected_dimensions) <= len(self.in_channels))
            else range(len(self.in_channels))
        )
        if self.lift_encoded_features and 0 not in self.dimensions:
            raise ValueError(
                "Encoded feature lifting requires rank 0 in "
                "selected_dimensions."
            )
        for i in self.dimensions:
            if self.lift_encoded_features and i > 0:
                continue
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

        for i in self.dimensions:
            if self.lift_encoded_features and i > 0:
                self._project_to_rank(data, i)
                continue
            if hasattr(data, f"x_{i}") and hasattr(self, f"encoder_{i}"):
                batch = getattr(data, f"batch_{i}")
                data[f"x_{i}"] = getattr(self, f"encoder_{i}")(
                    data[f"x_{i}"], batch
                )
        return data

    @staticmethod
    def _project_to_rank(data: torch_geometric.data.Data, rank: int) -> None:
        r"""Project encoded features and batch assignments to one rank.

        Parameters
        ----------
        data : torch_geometric.data.Data
            Data containing the lower-rank features and incidence matrix.
        rank : int
            Target rank.
        """
        incidence_key = f"incidence_{rank}"
        lower_features_key = f"x_{rank - 1}"
        if not hasattr(data, incidence_key):
            raise ValueError(
                f"Encoded feature lifting requires {incidence_key}."
            )
        if not hasattr(data, lower_features_key):
            raise ValueError(
                f"Encoded feature lifting requires {lower_features_key}."
            )

        incidence = data[incidence_key]
        lower_features = data[lower_features_key]
        if incidence.layout == torch.strided:
            absolute_incidence = incidence.abs().to(dtype=lower_features.dtype)
            transpose = absolute_incidence.transpose(0, 1)
            higher_features = torch.mm(transpose, lower_features)
            lower_indices, higher_indices = torch.nonzero(
                absolute_incidence, as_tuple=True
            )
        else:
            incidence = incidence.to_sparse_coo().coalesce()
            absolute_incidence = torch.sparse_coo_tensor(
                incidence.indices(),
                incidence.values().abs().to(dtype=lower_features.dtype),
                incidence.size(),
                dtype=lower_features.dtype,
                device=incidence.device,
            ).coalesce()
            transpose = absolute_incidence.transpose(0, 1).coalesce()
            higher_features = torch.sparse.mm(transpose, lower_features)
            lower_indices, higher_indices = absolute_incidence.indices()
        data[f"x_{rank}"] = higher_features

        lower_batch = data.get(f"batch_{rank - 1}")
        if lower_batch is None:
            lower_batch = torch.zeros(
                lower_features.size(0),
                dtype=torch.long,
                device=lower_features.device,
            )
        higher_batch = torch.zeros(
            incidence.size(1),
            dtype=torch.long,
            device=lower_features.device,
        )
        higher_batch.scatter_(0, higher_indices, lower_batch[lower_indices])
        data[f"batch_{rank}"] = higher_batch


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
