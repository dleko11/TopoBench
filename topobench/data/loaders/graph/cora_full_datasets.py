"""Loaders for CoraFull datasets."""

from omegaconf import DictConfig
from torch_geometric.data import Dataset
from torch_geometric.datasets import CoraFull

from topobench.data.loaders.base import AbstractLoader


class CoraFullDatasetLoader(AbstractLoader):
    """Load CoraFull datasets.

    Parameters
    ----------
    parameters : DictConfig
        Configuration parameters containing the dataset directory.
    """

    def __init__(self, parameters: DictConfig) -> None:
        super().__init__(parameters)

    def load_dataset(self) -> Dataset:
        """Load the CoraFull dataset.

        Returns
        -------
        Dataset
            The loaded CoraFull dataset.
        """
        return CoraFull(root=str(self.root_data_dir))
