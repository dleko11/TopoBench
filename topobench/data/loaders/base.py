"""Abstract Loader class."""

import hashlib
import json
import os
from abc import ABC, abstractmethod
from pathlib import Path

import torch
import torch_geometric
from filelock import FileLock
from omegaconf import DictConfig


class AbstractLoader(ABC):
    """Abstract class that provides an interface to load data.

    Parameters
    ----------
    parameters : DictConfig
        Configuration parameters.
    """

    def __init__(self, parameters: DictConfig):
        self.parameters = parameters
        self.root_data_dir = Path(parameters["data_dir"])

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(parameters={self.parameters})"

    def get_data_dir(self) -> Path:
        """Get the data directory.

        Returns
        -------
        Path
            The path to the dataset directory.
        """
        return os.path.join(self.root_data_dir, self.parameters.data_name)

    def get_load_lock_path(self) -> Path:
        """Return the lock path used while loading a raw dataset.

        Raw dataset loading can download, extract, and process files under the
        dataset root. Serializing this step prevents first-run races when many
        jobs start against the same dataset cache.

        Returns
        -------
        Path
            Path to the dataset load lock file.
        """
        dataset_id = self.parameters.get("data_key", None)
        if dataset_id is None:
            dataset_id = self.parameters.get("data_name", None)

        lock_payload = {
            "loader": f"{self.__class__.__module__}.{self.__class__.__name__}",
            "data_dir": str(self.root_data_dir),
            "dataset_id": dataset_id,
        }
        lock_id = hashlib.sha1(
            json.dumps(lock_payload, sort_keys=True).encode()
        ).hexdigest()[:12]
        return self.root_data_dir / f".dataset_load_{lock_id}.lock"

    @abstractmethod
    def load_dataset(
        self,
    ) -> torch_geometric.data.Dataset | torch.utils.data.Dataset:
        """Load data into a dataset.

        Raises
        ------
        NotImplementedError
            If the method is not implemented.

        Returns
        -------
        Union[torch_geometric.data.Dataset, torch.utils.data.Dataset]
            The loaded dataset, which could be a PyG or PyTorch dataset.
        """
        raise NotImplementedError

    def load(self, **kwargs) -> tuple[torch_geometric.data.Data, str]:
        """Load data.

        Parameters
        ----------
        **kwargs : dict
            Additional keyword arguments.

        Returns
        -------
        tuple[torch_geometric.data.Data, str]
            Tuple containing the loaded data and the data directory.
        """
        lock_path = self.get_load_lock_path()
        lock_path.parent.mkdir(parents=True, exist_ok=True)

        with FileLock(lock_path, timeout=-1):
            dataset = self.load_dataset(**kwargs)
            data_dir = self.get_data_dir()

        return dataset, data_dir
