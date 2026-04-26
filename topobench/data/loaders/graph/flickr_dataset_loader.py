"""Loaders for Flickr dataset."""

from omegaconf import DictConfig
from torch_geometric.data import Dataset
from torch_geometric.datasets import Flickr

from topobench.data.loaders.base import AbstractLoader


class FlickrDatasetLoader(AbstractLoader):
    """Load the Flickr dataset.

    Parameters
    ----------
    parameters : DictConfig
        Configuration parameters containing:
            - data_dir: Root directory for data
            - data_name: (optional) for consistency; not used by PyG's Flickr
    """

    def __init__(self, parameters: DictConfig) -> None:
        super().__init__(parameters)

    def load_dataset(self) -> Dataset:
        """Load Flickr dataset.

        Returns
        -------
        Dataset
            The loaded Flickr dataset.

        Raises
        ------
        RuntimeError
            If dataset loading fails.
        """
        try:
            dataset = Flickr(root=str(self.root_data_dir))
            return dataset
        except Exception as e:
            raise RuntimeError(f"Failed to load Flickr dataset: {e}") from e
