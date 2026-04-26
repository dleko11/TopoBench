from typing import Final

from torch_geometric.nn.conv import MessagePassing, ClusterGCNConv
from torch_geometric.nn.models.basic_gnn import BasicGNN


class ClusterGCN(BasicGNN):
    r"""Cluster-GCN model wrapper using :class:`~torch_geometric.nn.conv.ClusterGCNConv`."""

    supports_edge_weight: Final[bool] = False
    supports_edge_attr: Final[bool] = False
    supports_norm_batch: Final[bool]

    def init_conv(self, in_channels: int, out_channels: int, **kwargs) -> MessagePassing:
        return ClusterGCNConv(in_channels, out_channels, **kwargs)