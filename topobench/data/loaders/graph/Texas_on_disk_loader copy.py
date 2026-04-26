from omegaconf import DictConfig
from torch_geometric.data import Dataset

from topobench.data.datasets import TransductiveOnDiskDataset
from topobench.data.loaders.base import AbstractLoader


class TexasOnDiskDatasetLoader(AbstractLoader):
    """
    This is a generic dataloader for all partitioned datasets saved
    on an open-source platform.
    
    To implement for a different dataset: change url.
    """
    def __init__(self, parameters: DictConfig) -> None:
        super().__init__(parameters)

    def load_dataset(self) -> Dataset:
        try:
            dataset = TransductiveOnDiskDataset(
                root=str(self.root_data_dir),
                name=self.parameters.get("data_name", None),
                backend=self.parameters.get("backend", "sqlite"),
                url="https://github.com/dleko11/TopoBench/blob/main/for_download/WebKB/Texas/processed.zip?raw=1",
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to load OnDiskDataset from {self.root_data_dir}"
            ) from e
        return dataset
