"""Loaders for Reddit datasets."""

from omegaconf import DictConfig
from torch_geometric.datasets import Reddit

from topobench.data.loaders.base import AbstractLoader


class RedditDatasetLoader(AbstractLoader):
    """Load the Reddit graph dataset.

    Parameters
    ----------
    parameters : DictConfig
        Configuration parameters containing the dataset directory and name.
    """

    def __init__(self, parameters: DictConfig) -> None:
        super().__init__(parameters)

    def load_dataset(self) -> Reddit:
        """Load the Reddit dataset.

        Returns
        -------
        Reddit
            The loaded Reddit dataset.
        """
        dataset = Reddit(
            root=str(self.root_data_dir / self.parameters.data_name)
        )
        data = dataset[0]
        dataset.split_idx = {
            "train": data.train_mask.nonzero(as_tuple=True)[0],
            "valid": data.val_mask.nonzero(as_tuple=True)[0],
            "test": data.test_mask.nonzero(as_tuple=True)[0],
        }
        self.data_dir = self.get_data_dir()
        return dataset
