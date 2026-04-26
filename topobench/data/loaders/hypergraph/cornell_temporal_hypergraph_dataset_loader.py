"""Loaders for Citation Hypergraph dataset."""

from omegaconf import DictConfig

from topobench.data.datasets import CornellTemporalHypergraphDataset
from topobench.data.loaders.base import AbstractLoader


class CornellTemporalHypergraphDatasetLoader(AbstractLoader):
    """Load Cornell Temporal Hypergraph dataset with configurable parameters.

    Parameters
    ----------
    parameters : DictConfig
        Configuration parameters containing:
            - data_dir: Root directory for data
            - data_name: Name of the dataset
            - setting: 'transductive' or 'inductive'
    """

    def __init__(self, parameters: DictConfig) -> None:
        super().__init__(parameters)

    def load_dataset(self) -> CornellTemporalHypergraphDataset:
        """Load the Cornell Temporal Hypergraph dataset.

        Returns
        -------
        CornellTemporalHypergraphDataset
            The loaded Cornell Temporal Hypergraph dataset with the appropriate `data_dir`.
        """
        return self._initialize_dataset()

    def _initialize_dataset(self) -> CornellTemporalHypergraphDataset:
        """Initialize the Cornell Temporal Hypergraph dataset.

        Returns
        -------
        CornellTemporalHypergraphDataset
            The initialized dataset instance.
        """
        return CornellTemporalHypergraphDataset(
            root=str(self.root_data_dir),
            name=self.parameters.data_name,
            setting=self.parameters.get("setting", "transductive"),
        )
