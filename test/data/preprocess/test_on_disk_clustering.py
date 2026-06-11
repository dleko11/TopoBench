"""Unit and integration tests for the on-disk Cluster-GCN partitioning and dataloading pipeline."""

import os
import os.path as osp
import tempfile
from unittest.mock import MagicMock, patch

import numpy as np
import torch
from torch_geometric.data import Data, InMemoryDataset

from topobench.data.preprocessor.preprocessor import PreProcessor
from topobench.dataloader import ClusterGCNDataModule
from topobench.dataloader.dataload_cluster import (
    BlockCSRBatchCollator,
    _HandleAdapter,
)


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
        self.split_idx = {
            "train": torch.tensor([0, 1]),
            "val": torch.tensor([2]),
            "test": torch.tensor([3]),
        }


def _write_handcrafted_cluster_memmaps(root: str) -> dict:
    """Write a tiny two-cluster CSR bundle with cross-cluster edges."""
    processed_dir = osp.join(root, "processed")
    mm_dir = osp.join(processed_dir, "perm_memmap")
    os.makedirs(mm_dir, exist_ok=True)

    arrays = {
        "partptr.npy": np.array([0, 2, 4], dtype=np.int64),
        "indptr.npy": np.array([0, 2, 3, 5, 6], dtype=np.int64),
        "indices.npy": np.array([1, 2, 0, 3, 1, 2], dtype=np.int64),
        "X_perm.npy": np.arange(8, dtype=np.float32).reshape(4, 2),
        "y_perm.npy": np.array([0, 1, 0, 1], dtype=np.int64),
        "edge_attr_perm.npy": np.array(
            [[10], [12], [20], [33], [31], [42]], dtype=np.float32
        ),
        "train_mask_perm.npy": np.array([True, True, True, True]),
        "val_mask_perm.npy": np.array([True, True, True, True]),
        "test_mask_perm.npy": np.array([True, True, True, True]),
        "parts_with_train.npy": np.array([0, 1], dtype=np.int64),
        "parts_with_val.npy": np.array([0, 1], dtype=np.int64),
        "parts_with_test.npy": np.array([0, 1], dtype=np.int64),
    }
    for name, array in arrays.items():
        np.save(osp.join(mm_dir, name), array)

    return {
        "root": root,
        "processed_dir": processed_dir,
        "memmap_dir": mm_dir,
        "num_parts": 2,
        "sparse_format": "csr",
        "has_x": True,
        "has_y": True,
        "has_edge_attr": True,
        "paths": {
            "parts_with_train": osp.join(mm_dir, "parts_with_train.npy"),
            "parts_with_val": osp.join(mm_dir, "parts_with_val.npy"),
            "parts_with_test": osp.join(mm_dir, "parts_with_test.npy"),
        },
    }


def _collate(handle: dict, reconstruct_cross_cluster_edges: bool, parts):
    return BlockCSRBatchCollator(
        _HandleAdapter(handle),
        with_edge_attr=True,
        reconstruct_cross_cluster_edges=reconstruct_cross_cluster_edges,
    )(parts)


class TestOnDiskClusteringPipeline:
    """Test the entire on-disk Cluster-GCN partitioning and streaming datamodule."""

    def test_pipeline_end_to_end(self):
        """Verify that partitioning constructs correct handle metadata and datamodule streams batches successfully."""
        num_nodes = 8
        # Create a simple line/cycle-like graph
        edge_index = torch.tensor(
            [
                [0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 0],
                [1, 0, 2, 1, 3, 2, 4, 3, 5, 4, 6, 5, 7, 6, 0, 7],
            ],
            dtype=torch.long,
        )

        x = torch.randn(num_nodes, 4)
        y = torch.randint(0, 2, (num_nodes,))
        edge_attr = torch.randn(edge_index.size(1), 2)

        # Splitting masks
        train_mask = torch.tensor(
            [True, True, False, False, False, False, False, False]
        )
        val_mask = torch.tensor(
            [False, False, True, True, False, False, False, False]
        )
        test_mask = torch.tensor(
            [False, False, False, False, True, True, True, True]
        )

        graph_data = Data(
            x=x,
            edge_index=edge_index,
            y=y,
            edge_attr=edge_attr,
            train_mask=train_mask,
            val_mask=val_mask,
            test_mask=test_mask,
            num_nodes=num_nodes,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            dataset = SyntheticGraphDataset(tmpdir, graph_data)

            # Mock load_dataset_splits to return fake split datasets to avoid complex split logic
            mock_train = MagicMock()
            mock_train.data_lst = [graph_data]

            with patch.object(
                PreProcessor,
                "load_dataset_splits",
                return_value=(mock_train, mock_train, mock_train),
            ):
                preprocessor = PreProcessor(
                    dataset, tmpdir, transforms_config=None
                )

                split_params = {
                    "learning_setting": "transductive",
                    "type": "random",
                }
                cluster_params = {
                    "num_parts": 2,
                    "recursive": False,
                    "keep_inter_cluster_edges": False,
                    "sparse_format": "csr",
                }
                stream_params = {
                    "precompute_split_parts": True,
                    "stream": {"q_val": 1.0},
                }

                # 1. Run partition
                handle = preprocessor.pack_global_partition(
                    split_params=split_params,
                    cluster_params=cluster_params,
                    stream_params=stream_params,
                    dtype_policy="preserve",
                )

                # 2. Check handle outputs and saved files
                assert isinstance(handle, dict)
                assert handle["num_parts"] == 2
                assert handle["sparse_format"] == "csr"
                assert handle["has_x"] is True
                assert handle["has_y"] is True
                assert handle["has_edge_attr"] is True

                handle_reconstructed = preprocessor.pack_global_partition(
                    split_params=split_params,
                    cluster_params=cluster_params,
                    stream_params={
                        **stream_params,
                        "reconstruct_cross_cluster_edges": True,
                    },
                    dtype_policy="preserve",
                )
                handle_partition_local = preprocessor.pack_global_partition(
                    split_params=split_params,
                    cluster_params=cluster_params,
                    stream_params={
                        **stream_params,
                        "reconstruct_cross_cluster_edges": False,
                    },
                    dtype_policy="preserve",
                )
                assert (
                    handle_reconstructed["config_hash"]
                    == handle["config_hash"]
                )
                assert (
                    handle_partition_local["config_hash"]
                    == handle["config_hash"]
                )

                # Check paths
                for key, path in handle["paths"].items():
                    assert osp.exists(path), (
                        f"File {key} at {path} does not exist!"
                    )

                # 3. Instantiate and run ClusterGCNDataModule
                datamodule = ClusterGCNDataModule(
                    data_handle=handle, q=1, num_workers=0, with_edge_attr=True
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

    def test_collator_can_drop_cross_cluster_edges_inside_sampled_union(self):
        """Verify the streaming edge policy for multi-cluster batches."""
        with tempfile.TemporaryDirectory() as tmpdir:
            handle = _write_handcrafted_cluster_memmaps(tmpdir)

            reconstructed = _collate(
                handle, reconstruct_cross_cluster_edges=True, parts=[0, 1]
            )
            assert reconstructed.edge_index.tolist() == [
                [0, 0, 1, 2, 2, 3],
                [1, 2, 0, 3, 1, 2],
            ]
            assert reconstructed.edge_attr.view(-1).tolist() == [
                10,
                12,
                20,
                33,
                31,
                42,
            ]

            partition_local = _collate(
                handle, reconstruct_cross_cluster_edges=False, parts=[0, 1]
            )
            assert partition_local.edge_index.tolist() == [
                [0, 1, 2, 3],
                [1, 0, 3, 2],
            ]
            assert partition_local.edge_attr.view(-1).tolist() == [
                10,
                20,
                33,
                42,
            ]

    def test_collator_edge_policy_is_identical_for_single_cluster_batches(
        self,
    ):
        """A single sampled cluster has no cross-cluster reconstruction."""
        with tempfile.TemporaryDirectory() as tmpdir:
            handle = _write_handcrafted_cluster_memmaps(tmpdir)

            reconstructed = _collate(
                handle, reconstruct_cross_cluster_edges=True, parts=[0]
            )
            partition_local = _collate(
                handle, reconstruct_cross_cluster_edges=False, parts=[0]
            )

            assert torch.equal(
                reconstructed.edge_index, partition_local.edge_index
            )
            assert torch.equal(
                reconstructed.edge_attr, partition_local.edge_attr
            )

    def test_validation_cache_keys_include_cross_cluster_edge_policy(self):
        """Validation caches must not be shared across edge policies."""
        with tempfile.TemporaryDirectory() as tmpdir:
            handle = _write_handcrafted_cluster_memmaps(tmpdir)
            cache_dir = osp.join(tmpdir, "val_cache")

            reconstructed = ClusterGCNDataModule(
                data_handle=handle,
                q=2,
                q_val=2,
                cache_val=True,
                val_cache_dir=cache_dir,
                with_edge_attr=True,
                reconstruct_cross_cluster_edges=True,
            )
            reconstructed.setup("fit")

            partition_local = ClusterGCNDataModule(
                data_handle=handle,
                q=2,
                q_val=2,
                cache_val=True,
                val_cache_dir=cache_dir,
                with_edge_attr=True,
                reconstruct_cross_cluster_edges=False,
            )
            partition_local.setup("fit")

            cache_dirs = [
                name
                for name in os.listdir(cache_dir)
                if name.startswith("val_")
                and osp.isdir(osp.join(cache_dir, name))
            ]
            assert len(cache_dirs) == 2
