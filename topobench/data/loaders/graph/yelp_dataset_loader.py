"""Loaders for Yelp dataset."""

from omegaconf import DictConfig
from torch_geometric.data import Dataset
from torch_geometric.datasets import Yelp

from topobench.data.loaders.base import AbstractLoader


class YelpDatasetLoader(AbstractLoader):
    """Load the Yelp dataset.

    Parameters
    ----------
    parameters : DictConfig
        Configuration parameters containing:
            - data_dir: Root directory for data
            - data_name: (optional) for consistency; not used by PyG's Yelp
    """

    def __init__(self, parameters: DictConfig) -> None:
        super().__init__(parameters)

    def load_dataset(self) -> Dataset:
        """Load Yelp dataset.

        Returns
        -------
        Dataset
            The loaded Yelp dataset.

        Raises
        ------
        RuntimeError
            If dataset loading fails.
        """
        try:
            dataset = Yelp(root=str(self.root_data_dir))
            return dataset
        except Exception as e:
            raise RuntimeError(f"Failed to load Yelp dataset: {e}") from e