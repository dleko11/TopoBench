"""Loaders for WebKB datasets."""

from omegaconf import DictConfig
from torch_geometric.data import Dataset
from torch_geometric.datasets import WebKB
from torch_geometric.utils import coalesce, remove_self_loops

from topobench.data.loaders.base import AbstractLoader


class WebKBCleanDatasetLoader(AbstractLoader):
    """Load WebKB datasets (Cornell, Texas, Wisconsin).
    
    (Docstrings unchanged)
    """

    def __init__(self, parameters: DictConfig) -> None:
        super().__init__(parameters)

    def load_dataset(self) -> Dataset:
        """Load WebKB dataset and sanitize it by removing self-loops."""
        dataset = WebKB(
            root=str(self.root_data_dir),
            name=self.parameters.data_name,
        )

        data = dataset.data

        cleaned_edge_index, _ = remove_self_loops(data.edge_index)
        data.edge_index = coalesce(cleaned_edge_index, num_nodes=data.num_nodes)

        return dataset
