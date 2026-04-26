
"""Loaders for Protein-Protein interactions datasets (PPI)."""

import os
from pathlib import Path

import numpy as np
from omegaconf import DictConfig
from torch_geometric.data import Dataset
from torch_geometric.datasets import PPI

from topobench.data.loaders.base import AbstractLoader


class PPIDatasetLoader(AbstractLoader):
    """Loader for the inductive PPI dataset."""

    def __init__(self, parameters: DictConfig) -> None:
        """Initializes the loader."""
        super().__init__(parameters)
        self.datasets: list[Dataset] = []

    def load_dataset(self) -> Dataset:
        """Load the molecule dataset with predefined splits.

        Returns
        -------
        Dataset
            The combined dataset with predefined splits.
        """

        self._load_splits()
        split_idx = self._prepare_split_idx()
        combined_dataset = self._combine_splits()
        combined_dataset.split_idx = split_idx
        return combined_dataset

    def _load_splits(self) -> None:
        """Load the dataset splits for the specified dataset."""
        for split in ["train", "val", "test"]:
            # The `subset=True` argument was removed as it's not a valid
            # parameter for the PPI dataset class.
            self.datasets.append(
                PPI(
                    root=str(self.root_data_dir),
                    split=split,
                )
            )

    def _prepare_split_idx(self) -> dict[str, np.ndarray]:
        """Prepare the split indices for the dataset.

        Returns
        -------
        dict[str, np.ndarray]
            A dictionary mapping split names to index arrays.
        """
        # Calculate the number of graphs in each split
        num_train = len(self.datasets[0])
        num_val = len(self.datasets[1])
        num_test = len(self.datasets[2])

        # Create indices for each split based on their position in the combined dataset
        split_idx = {"train": np.arange(num_train)}
        split_idx["valid"] = np.arange(num_train, num_train + num_val)
        split_idx["test"] = np.arange(num_train + num_val, num_train + num_val + num_test)
        
        return split_idx

    def _combine_splits(self) -> Dataset:
        """Combine the dataset splits into a single dataset.

        Returns
        -------
        Dataset
            The combined dataset containing all splits.
        """
        return self.datasets[0] + self.datasets[1] + self.datasets[2]

    def get_data_dir(self) -> Path:
        """Get the data directory.

        Returns
        -------
        Path
            The path to the dataset directory.
        """
        # This assumes self.parameters.data_name is 'PPI'
        return os.fspath(Path(self.root_data_dir) / self.parameters.data_name)
