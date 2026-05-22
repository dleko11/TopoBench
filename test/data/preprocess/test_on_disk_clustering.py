"""Unit and integration tests for the on-disk Cluster-GCN partitioning and dataloading pipeline."""

import os
import os.path as osp
import tempfile
from unittest.mock import MagicMock, patch
import pytest
import torch
from torch_geometric.data import Data, InMemoryDataset

from topobench.data.preprocessor.preprocessor import PreProcessor
from topobench.dataloader import ClusterGCNDataModule


class SyntheticGraphDataset(InMemoryDataset):
    """A minimal synthetic dataset for testing on-disk partitioning.

    Parameters
    ----------
    root : str
        Root directory where the dataset is stored.
    data : torch_geometric.data.Data
        The graph data object to store in the dataset.
    """

    def __init__(self, root, data):
        super().__init__(root)
        self.data = data
        self._data = data
        self.slices = None
        self.split_idx = {"train": torch.tensor([0, 1]), "val": torch.tensor([2]), "test": torch.tensor([3])}


class TestOnDiskClusteringPipeline:
    """Test the entire on-disk Cluster-GCN partitioning and streaming datamodule."""

    def test_pipeline_end_to_end(self):
        """Verify that partitioning constructs correct handle metadata and datamodule streams batches successfully."""
        num_nodes = 8
        # Create a simple line/cycle-like graph
        edge_index = torch.tensor([
            [0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 0],
            [1, 0, 2, 1, 3, 2, 4, 3, 5, 4, 6, 5, 7, 6, 0, 7]
        ], dtype=torch.long)

        x = torch.randn(num_nodes, 4)
        y = torch.randint(0, 2, (num_nodes,))
        edge_attr = torch.randn(edge_index.size(1), 2)

        # Splitting masks
        train_mask = torch.tensor([True, True, False, False, False, False, False, False])
        val_mask = torch.tensor([False, False, True, True, False, False, False, False])
        test_mask = torch.tensor([False, False, False, False, True, True, True, True])

        graph_data = Data(
            x=x,
            edge_index=edge_index,
            y=y,
            edge_attr=edge_attr,
            train_mask=train_mask,
            val_mask=val_mask,
            test_mask=test_mask,
            num_nodes=num_nodes
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            dataset = SyntheticGraphDataset(tmpdir, graph_data)

            # Mock load_dataset_splits to return fake split datasets to avoid complex split logic
            mock_train = MagicMock()
            mock_train.data_lst = [graph_data]

            with patch.object(PreProcessor, "load_dataset_splits", return_value=(mock_train, mock_train, mock_train)):
                preprocessor = PreProcessor(dataset, tmpdir, transforms_config=None)

                split_params = {"learning_setting": "transductive", "type": "random"}
                cluster_params = {
                    "num_parts": 2,
                    "recursive": False,
                    "keep_inter_cluster_edges": False,
                    "sparse_format": "csr"
                }
                stream_params = {
                    "precompute_split_parts": True,
                    "stream": {"q_val": 1.0}
                }

                # 1. Run partition
                handle = preprocessor.pack_global_partition(
                    split_params=split_params,
                    cluster_params=cluster_params,
                    stream_params=stream_params,
                    dtype_policy="preserve"
                )

                # 2. Check handle outputs and saved files
                assert isinstance(handle, dict)
                assert handle["num_parts"] == 2
                assert handle["sparse_format"] == "csr"
                assert handle["has_x"] is True
                assert handle["has_y"] is True
                assert handle["has_edge_attr"] is True

                # Check paths
                for key, path in handle["paths"].items():
                    assert osp.exists(path), f"File {key} at {path} does not exist!"

                # 3. Instantiate and run ClusterGCNDataModule
                datamodule = ClusterGCNDataModule(
                    data_handle=handle,
                    q=1,
                    num_workers=0,
                    with_edge_attr=True
                )

                datamodule.setup()

                # Get train loader and check a batch
                train_loader = datamodule.train_dataloader()
                batches = list(train_loader)
                assert len(batches) > 0

                # Validate batch contents
                batch = batches[0]
                assert hasattr(batch, "edge_index")
                assert hasattr(batch, "x")
                assert hasattr(batch, "y")
                assert hasattr(batch, "edge_attr")
                assert hasattr(batch, "supervised_mask")
                assert hasattr(batch, "global_nid")
                assert hasattr(batch, "num_nodes")
