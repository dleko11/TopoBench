"""Loaders for Open Graph Benchmark node property datasets."""

import torch
from ogb.nodeproppred import PygNodePropPredDataset
from omegaconf import DictConfig
from torch_geometric.data import Dataset

from topobench.data.loaders.base import AbstractLoader


class OGBNDatasetLoader(AbstractLoader):
    """Load OGBN datasets.

    Parameters
    ----------
    parameters : DictConfig
        Configuration parameters containing the dataset directory and
        OGB dataset key, for example ``ogbn-products``.
    """

    def __init__(self, parameters: DictConfig) -> None:
        super().__init__(parameters)

    def load_dataset(self) -> Dataset:
        """Load an OGBN dataset.

        Returns
        -------
        Dataset
            The loaded OGBN dataset.
        """
        dataset = PygNodePropPredDataset(
            name=self.parameters.data_key,
            root=self.root_data_dir,
        )
        dataset._data.x = dataset._data.x.to(torch.float)
        dataset._data.y = dataset._data.y.squeeze(1)
        dataset.split_idx = dataset.get_idx_split()
        return dataset
