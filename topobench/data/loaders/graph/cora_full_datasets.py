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
        Configuration parameters containing:
            - data_dir: Root directory for data
            - data_name: Name of the dataset
            - data_type: Type of the dataset (e.g., "cocitation")
    """

    def __init__(self, parameters: DictConfig) -> None:
        super().__init__(parameters)

    def load_dataset(self) -> Dataset:
        """Load CoraFull dataset.

        Returns
        -------
        Dataset
            The loaded CoraFull dataset.

        Raises
        ------
        RuntimeError
            If dataset loading fails.
        """

        dataset = CoraFull(
            root=str(self.root_data_dir),
        )
        
        return dataset
