"""Unit and integration tests for the on-disk Cluster-GCN partitioning and dataloading pipeline."""

import os.path as osp
import tempfile
from unittest.mock import MagicMock, patch

import numpy as np
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

    def test_random_splits_share_only_structural_partition(self, tmp_path):
        """Different data seeds retain distinct masks without duplicating graph data."""
        num_nodes = 8
        edge_index = torch.tensor(
            [
                [0, 1, 2, 3, 4, 5, 6, 7],
                [1, 2, 3, 4, 5, 6, 7, 0],
            ],
            dtype=torch.long,
        )
        original = Data(
            x=torch.randn(num_nodes, 3),
            y=torch.arange(num_nodes) % 2,
            edge_index=edge_index,
            train_mask=torch.tensor(
                [True, True, False, False, False, False, False, False]
            ),
            val_mask=torch.tensor(
                [False, False, True, True, False, False, False, False]
            ),
            test_mask=torch.tensor(
                [False, False, False, False, True, True, True, True]
            ),
            num_nodes=num_nodes,
        )
        dataset = SyntheticGraphDataset(str(tmp_path), original)
        preprocessor_a = PreProcessor(
            dataset, str(tmp_path), transforms_config=None
        )
        preprocessor_b = PreProcessor(
            dataset, str(tmp_path), transforms_config=None
        )

        split_a = original.clone()
        split_a.train_mask = torch.tensor([0, 2, 4])
        split_a.val_mask = torch.tensor([1, 3])
        split_a.test_mask = torch.tensor([5, 6, 7])
        split_b = original.clone()
        split_b.train_mask = torch.tensor([1, 3, 5])
        split_b.val_mask = torch.tensor([0, 2])
        split_b.test_mask = torch.tensor([4, 6, 7])

        dataset_a = MagicMock()
        dataset_a.data_lst = [split_a]
        dataset_b = MagicMock()
        dataset_b.data_lst = [split_b]
        cluster_params = {
            "num_parts": 2,
            "recursive": False,
            "keep_inter_cluster_edges": False,
            "sparse_format": "csr",
        }

        with patch.object(
            PreProcessor,
            "load_dataset_splits",
            side_effect=[
                (dataset_a, None, None),
                (dataset_b, None, None),
            ],
        ):
            handle_a = preprocessor_a.pack_global_partition(
                split_params={"data_seed": 0, "standardize": False},
                cluster_params=cluster_params,
                stream_params={},
            )
            handle_b = preprocessor_b.pack_global_partition(
                split_params={"data_seed": 1, "standardize": False},
                cluster_params=cluster_params,
                stream_params={},
            )

        assert handle_a["config_hash"] == handle_b["config_hash"]
        assert handle_a["processed_dir"] == handle_b["processed_dir"]
        assert handle_a["split_hash"] != handle_b["split_hash"]
        assert handle_a["paths"]["X_perm"] == handle_b["paths"]["X_perm"]
        assert (
            handle_a["paths"]["train_mask_perm"]
            != handle_b["paths"]["train_mask_perm"]
        )

        node_perm = np.load(handle_a["paths"]["perm_to_global"])
        expected_train_a = np.zeros(num_nodes, dtype=bool)
        expected_train_a[[0, 2, 4]] = True
        stored_train_a = np.load(handle_a["paths"]["train_mask_perm"])
        assert np.array_equal(stored_train_a, expected_train_a[node_perm])

        datamodule = ClusterGCNDataModule(
            data_handle=handle_a,
            q=1,
            q_val=1,
            num_workers=0,
            cache_num_workers=0,
            cleanup_val_cache=True,
        )
        datamodule.setup("fit")
        cache_path = datamodule._val_cache_path
        assert cache_path is not None and osp.isdir(cache_path)
        expected_cache_base = osp.join(handle_a["processed_dir"], "val_cache")
        assert osp.commonpath([cache_path, expected_cache_base]) == (
            expected_cache_base
        )
        cleanup_path = datamodule._val_cache_cleanup_path
        datamodule.cleanup_validation_cache()
        assert not osp.exists(cache_path)
        assert cleanup_path is not None and not osp.exists(cleanup_path)
