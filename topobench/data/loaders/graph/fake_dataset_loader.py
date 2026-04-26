"""Loaders for synthetic FakeDataset graphs."""

from omegaconf import DictConfig
from torch_geometric.data import Dataset
from torch_geometric.datasets import FakeDataset

from topobench.data.loaders.base import AbstractLoader


class FakeDatasetLoader(AbstractLoader):
    """Loader for torch_geometric FakeDataset.

    Parameters
    ----------
    parameters : DictConfig
        Configuration with keys such as ``num_graphs``, ``avg_num_nodes``,
        ``avg_degree``, ``num_channels``, ``edge_dim``, ``num_classes``,
        ``task``, and ``is_undirected``.
    """

    def __init__(self, parameters: DictConfig) -> None:
        super().__init__(parameters)

    def load_dataset(self) -> Dataset:
        """Load and return a FakeDataset instance.

        Returns
        -------
        Dataset
            Instantiated :class:`torch_geometric.datasets.FakeDataset`.

        Raises
        ------
        RuntimeError
            If dataset creation fails.
        """
        try:
            dataset = FakeDataset(
                # root=str(self.root_data_dir),
                num_graphs=self.parameters.get("num_graphs", 1),
                avg_num_nodes=self.parameters.get("avg_num_nodes", 1000),
                avg_degree=self.parameters.get("avg_degree", 10.0),
                num_channels=self.parameters.get("num_channels", 64),
                edge_dim=self.parameters.get("edge_dim", 0),
                num_classes=self.parameters.get("num_classes", 10),
                task=self.parameters.get("task", "auto"),
                is_undirected=self.parameters.get("is_undirected", True),
            )
        except Exception as e:
            raise RuntimeError("Failed to load FakeOnDiskDataset") from e

        return dataset
