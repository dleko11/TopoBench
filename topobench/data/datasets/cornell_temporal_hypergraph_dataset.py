"""Loaders for the Cornell Temporal Hypergraph dataset."""

import torch
from torch_geometric.data import InMemoryDataset
from torch_geometric.datasets import (
    CornellTemporalHyperGraphDataset as PyGCornellTemporalHyperGraphDataset,
)


class CornellTemporalHypergraphDataset(InMemoryDataset):
    """Wrapper for the PyTorch Geometric CornellTemporalHyperGraphDataset.

    This class handles loading the train, validation, and test splits of the
    Cornell Temporal Hypergraph dataset and combines them into a single dataset
    object. It also stores the indices for each split.

    Parameters
    ----------
    root : str
        Root directory where the dataset should be saved.
    name : str
        The name of the dataset.
    setting : str, optional
        "transductive" or "inductive" setting, by default "transductive".
    """

    def __init__(
        self,
        root: str,
        name: str,
        setting: str = "transductive",
        *args,
        **kwargs,
    ) -> None:
        self.setting = setting
        self.name = name
        super().__init__(root=root, *args, **kwargs)

        train_dataset = PyGCornellTemporalHyperGraphDataset(
            root=self.root, name=self.name, split="train", setting=self.setting
        )
        val_dataset = PyGCornellTemporalHyperGraphDataset(
            root=self.root, name=self.name, split="val", setting=self.setting
        )
        test_dataset = PyGCornellTemporalHyperGraphDataset(
            root=self.root, name=self.name, split="test", setting=self.setting
        )

        data_list = [data for data in train_dataset] + \
                    [data for data in val_dataset] + \
                    [data for data in test_dataset]

        for data in data_list:
            if data.y is None:
                data.y = torch.zeros(1, dtype=torch.long)

        self.data, self.slices = self.collate(data_list)

        self.train_indices = torch.arange(len(train_dataset))
        self.val_indices = torch.arange(
            len(train_dataset), len(train_dataset) + len(val_dataset)
        )
        self.test_indices = torch.arange(
            len(train_dataset) + len(val_dataset), len(self)
        )

    def download(self) -> None:
        """Downloads the dataset."""
        PyGCornellTemporalHyperGraphDataset(root=self.root, name=self.name, split="train", setting=self.setting)
        PyGCornellTemporalHyperGraphDataset(root=self.root, name=self.name, split="val", setting=self.setting)
        PyGCornellTemporalHyperGraphDataset(root=self.root, name=self.name, split="test", setting=self.setting)

    def process(self) -> None:
        """Processes the dataset."""
        # Processing is handled by the PyG dataset, so this can be left empty.

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return []
