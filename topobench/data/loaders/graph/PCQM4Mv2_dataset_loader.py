"""Loaders for PCQM4Mv2 datasets."""

from omegaconf import DictConfig
from torch_geometric.data import Dataset
from torch_geometric.datasets import PCQM4Mv2

from topobench.data.loaders.base import AbstractLoader


class PCQM4Mv2DatasetLoader(AbstractLoader):
    """
    """

    def __init__(self, parameters: DictConfig) -> None:
        super().__init__(parameters)

    def load_dataset(self) -> Dataset:
        """Load PCQM4Mv2 dataset.

        Returns
        -------
        Dataset
            The loaded WebKB dataset (single-graph, node-classification).

        Raises
        ------
        RuntimeError
            If dataset loading fails.
        """
        dataset = PCQM4Mv2(
            root=str(self.root_data_dir),
        )
        return dataset
