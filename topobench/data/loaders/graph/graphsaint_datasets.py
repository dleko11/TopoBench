"""Load Flickr and Yelp with their published GraphSAINT node splits."""

from torch_geometric.data import InMemoryDataset
from torch_geometric.datasets import Flickr, Yelp

from topobench.data.loaders.base import AbstractLoader


class GraphSAINTDatasetLoader(AbstractLoader):
    """Load the PyG Flickr or Yelp graph and expose its fixed node indices."""

    def load_dataset(self) -> InMemoryDataset:
        """Load the graph and retain its published train/validation/test masks."""
        dataset_class = {"Flickr": Flickr, "Yelp": Yelp}[self.parameters.data_name]
        dataset = dataset_class(
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
