"""Loaders for Coauthor datasets."""

from omegaconf import DictConfig
from torch_geometric.data import Dataset
from torch_geometric.datasets import Coauthor

from topobench.data.loaders.base import AbstractLoader


class CoauthorDatasetLoader(AbstractLoader):
    """Load Coauthor datasets.

    Parameters
    ----------
    parameters : DictConfig
        Configuration parameters containing the dataset directory and
        ``data_name`` such as ``CS`` or ``Physics``.
    """

    def __init__(self, parameters: DictConfig) -> None:
        super().__init__(parameters)

    def load_dataset(self) -> Dataset:
        """Load a Coauthor dataset.

        Returns
        -------
        Dataset
            The loaded Coauthor dataset.
        """
        return Coauthor(
            root=str(self.root_data_dir),
            name=self.parameters.data_name,
        )
