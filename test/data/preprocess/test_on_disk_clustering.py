"""Unit and integration tests for the on-disk Cluster-GCN partitioning and dataloading pipeline."""

import os
import os.path as osp
import tempfile
from unittest.mock import MagicMock, patch

import numpy as np
import torch
from torch.utils.data import RandomSampler, SequentialSampler
from torch_geometric.data import Data, InMemoryDataset

from topobench.data.preprocessor.preprocessor import PreProcessor
from topobench.data.utils.cluster_utils import ClusterOnDisk
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
    """Write a two-cluster CSR bundle containing cross-cluster edges."""
    processed_dir = osp.join(root, "processed")
    memmap_dir = osp.join(processed_dir, "perm_memmap")
    os.makedirs(memmap_dir, exist_ok=True)

    arrays = {
        "partptr.npy": np.array([0, 2, 4], dtype=np.int64),
        "indptr.npy": np.array([0, 2, 3, 5, 6], dtype=np.int64),
        "indices.npy": np.array([1, 2, 0, 3, 1, 2], dtype=np.int64),
        "X_perm.npy": np.arange(8, dtype=np.float32).reshape(4, 2),
        "y_perm.npy": np.array([0, 1, 0, 1], dtype=np.int64),
        "edge_attr_perm.npy": np.array(
            [[10], [12], [20], [33], [31], [42]], dtype=np.float32
        ),
        "train_mask_perm.npy": np.ones(4, dtype=bool),
        "val_mask_perm.npy": np.ones(4, dtype=bool),
        "test_mask_perm.npy": np.ones(4, dtype=bool),
        "parts_with_train.npy": np.array([0, 1], dtype=np.int64),
        "parts_with_val.npy": np.array([0, 1], dtype=np.int64),
        "parts_with_test.npy": np.array([0, 1], dtype=np.int64),
    }
    for name, array in arrays.items():
        np.save(osp.join(memmap_dir, name), array)

    return {
        "root": root,
        "processed_dir": processed_dir,
        "memmap_dir": memmap_dir,
        "num_parts": 2,
        "sparse_format": "csr",
        "has_x": True,
        "has_y": True,
        "has_edge_attr": True,
        "paths": {
            "parts_with_train": osp.join(memmap_dir, "parts_with_train.npy"),
            "parts_with_val": osp.join(memmap_dir, "parts_with_val.npy"),
            "parts_with_test": osp.join(memmap_dir, "parts_with_test.npy"),
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

    def test_random_partitioning_is_seeded_and_balanced(self, tmp_path):
        """Random partitions are reproducible and differ across seeds."""
        num_nodes = 12
        source = torch.arange(num_nodes)
        graph = Data(
            x=torch.randn(num_nodes, 2),
            edge_index=torch.stack([source, source.roll(-1)]),
            num_nodes=num_nodes,
        )

        partitions = []
        for index, seed in enumerate((7, 7, 8)):
            dataset = ClusterOnDisk(
                str(tmp_path / f"random_{index}"),
                graph_getter=lambda: graph,
                num_parts=5,
                partition_method="random",
                partition_seed=seed,
            )
            partitions.append(dataset.partition)
            assert dataset.partition_method == "random"
            assert dataset.partition_seed == seed

        assert torch.equal(partitions[0].node_perm, partitions[1].node_perm)
        assert not torch.equal(
            partitions[0].node_perm,
            partitions[2].node_perm,
        )
        part_sizes = partitions[0].partptr.diff()
        assert int(part_sizes.max() - part_sizes.min()) <= 1

    def test_cross_cluster_edge_policy(self, tmp_path):
        """Cross-cluster edges can be preserved or removed per batch."""
        handle = _write_handcrafted_cluster_memmaps(str(tmp_path))

        preserved = _collate(handle, True, [0, 1])
        removed = _collate(handle, False, [0, 1])

        assert preserved.edge_index.tolist() == [
            [0, 0, 1, 2, 2, 3],
            [1, 2, 0, 3, 1, 2],
        ]
        assert preserved.edge_attr.view(-1).tolist() == [
            10,
            12,
            20,
            33,
            31,
            42,
        ]
        assert removed.edge_index.tolist() == [
            [0, 1, 2, 3],
            [1, 0, 3, 2],
        ]
        assert removed.edge_attr.view(-1).tolist() == [10, 20, 33, 42]

    def test_train_shuffle_selects_expected_sampler(self, tmp_path):
        """The training loader exposes fixed and reshuffled cluster groups."""
        handle = _write_handcrafted_cluster_memmaps(str(tmp_path))

        fixed = ClusterGCNDataModule(
            data_handle=handle,
            q=1,
            train_shuffle=False,
            cache_val=False,
        )
        reshuffled = ClusterGCNDataModule(
            data_handle=handle,
            q=1,
            train_shuffle=True,
            cache_val=False,
        )

        fixed_sampler = fixed.train_dataloader().sampler
        reshuffled_sampler = reshuffled.train_dataloader().sampler

        assert isinstance(fixed_sampler, SequentialSampler)
        assert isinstance(reshuffled_sampler, RandomSampler)
        assert list(fixed_sampler) == list(fixed_sampler)
        generator_state = reshuffled_sampler.generator.get_state().clone()
        list(reshuffled_sampler)
        assert not torch.equal(
            generator_state,
            reshuffled_sampler.generator.get_state(),
        )

    def test_validation_cache_separates_edge_policies(self, tmp_path):
        """Validation batches are not reused across different edge policies."""
        handle = _write_handcrafted_cluster_memmaps(str(tmp_path))
        cache_dir = str(tmp_path / "val_cache")

        for reconstruct in (True, False):
            datamodule = ClusterGCNDataModule(
                data_handle=handle,
                q=2,
                q_val=2,
                cache_val=True,
                val_cache_dir=cache_dir,
                reconstruct_cross_cluster_edges=reconstruct,
            )
            datamodule.setup("fit")

        cache_directories = [
            path
            for path in os.listdir(cache_dir)
            if path.startswith("val_") and osp.isdir(osp.join(cache_dir, path))
        ]
        assert len(cache_directories) == 2
