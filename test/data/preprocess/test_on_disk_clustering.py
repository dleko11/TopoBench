"""Unit and integration tests for the on-disk Cluster-GCN partitioning and dataloading pipeline."""

import os.path as osp
import tempfile
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from omegaconf import DictConfig
from torch_geometric.data import Data, InMemoryDataset

from topobench.data.loaders.graph.reddit_dataset_loader import RedditDatasetLoader
from topobench.data.loaders.graph.graphsaint_datasets import GraphSAINTDatasetLoader
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

    def test_fixed_splits_survive_partitioning_and_streaming(self, tmp_path):
        """Preserve fixed node membership across seeds and cached partitions."""
        num_nodes = 8
        nodes = torch.arange(num_nodes)
        edges = torch.stack([nodes, nodes.roll(-1)])
        data = Data(
            x=torch.arange(num_nodes * 3).reshape(num_nodes, 3).float(),
            y=nodes % 2,
            edge_index=torch.cat([edges, edges.flip(0)], dim=1),
            num_nodes=num_nodes,
        )
        dataset = SyntheticGraphDataset(str(tmp_path / "raw"), data)
        indices = {
            "train": torch.tensor([0, 4]),
            "valid": torch.tensor([2]),
            "test": torch.tensor([1, 3, 5, 6, 7]),
        }
        dataset.split_idx = indices
        cluster_params = {
            "num_parts": 2,
            "recursive": False,
            "keep_inter_cluster_edges": False,
            "sparse_format": "csr",
        }
        handles = []
        for seed in [0, 4]:
            preprocessor = PreProcessor(
                dataset, str(tmp_path / "partition"), transforms_config=None
            )
            handles.append(preprocessor.pack_global_partition(
                split_params=DictConfig({
                    "learning_setting": "transductive",
                    "split_type": "fixed",
                    "data_seed": seed,
                    "standardize": False,
                }),
                cluster_params=cluster_params,
                stream_params={},
            ))
        assert handles[0]["split_hash"] == handles[1]["split_hash"]
        assert handles[0]["paths"] == handles[1]["paths"]

        handle = handles[0]
        permutation = np.load(handle["paths"]["perm_to_global"])
        original_ids = torch.from_numpy(permutation)
        datamodule = ClusterGCNDataModule(
            data_handle=handle, q=1, num_workers=0, cache_val=False
        )
        for split, key in [("train", "train"), ("val", "valid"), ("test", "test")]:
            expected = torch.zeros(num_nodes, dtype=torch.bool)
            expected[indices[key]] = True
            stored = np.load(handle["paths"][f"{split}_mask_perm"])
            assert np.array_equal(stored, expected.numpy()[permutation])
            observed = []
            for batch in datamodule.inference_dataloader(
                split=split, q=1, cover_parts="all"
            ):
                node_ids = original_ids[batch.global_nid]
                assert torch.equal(batch.x, data.x[node_ids])
                assert torch.equal(batch.y, data.y[node_ids])
                assert torch.equal(
                    batch.supervised_mask, expected[node_ids]
                )
                observed.extend(node_ids[batch.supervised_mask].tolist())
            assert sorted(observed) == sorted(indices[key].tolist())

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


@pytest.mark.parametrize("learning_setting", ["inductive", "transductive"])
@pytest.mark.parametrize("split_type", ["fixed", "random"])
@pytest.mark.parametrize("cache_val", [False, True])
def test_node_learning_setting_controls_topology(
    tmp_path, learning_setting, split_type, cache_val
):
    """Withheld nodes and their edges never enter inductive training lifting."""
    nodes = torch.arange(8)
    edge_index = torch.combinations(nodes).t()
    edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
    data = Data(
        x=nodes[:, None].float(), y=nodes % 2, edge_index=edge_index,
        edge_attr=(edge_index[0] * 8 + edge_index[1])[:, None].float(),
        num_nodes=8,
    )
    dataset = SyntheticGraphDataset(str(tmp_path / "raw"), data)
    dataset.split_idx = {
        "train": torch.tensor([0, 2, 4, 6]),
        "valid": torch.tensor([1]),
        "test": torch.tensor([3, 5, 7]),
    }
    preprocessor = PreProcessor(dataset, str(tmp_path / "partitions"))
    handle = preprocessor.pack_global_partition(
        split_params=DictConfig({
            "learning_setting": learning_setting, "split_type": split_type,
            "data_seed": 0, "train_prop": 0.5, "standardize": False,
            "data_split_dir": str(tmp_path / "splits"),
        }),
        cluster_params={"num_parts": 2, "sparse_format": "csr"},
        stream_params={},
    )
    permutation = torch.from_numpy(np.load(handle["paths"]["perm_to_global"]))
    train_mask = torch.from_numpy(np.load(handle["paths"]["train_mask_perm"]))
    train_nodes = set(permutation[train_mask].tolist())
    seen_by_lifting = []

    def inspect_lifting_input(batch):
        seen_by_lifting.append(set(batch.x[:, 0].long().tolist()))
        return batch

    datamodule = ClusterGCNDataModule(
        data_handle=handle, q=2, q_val=2, q_test=2,
        cache_val=cache_val, with_edge_attr=True,
        post_batch_transform=inspect_lifting_input,
    )
    batch = next(iter(datamodule.train_dataloader()))
    expected_nodes = train_nodes if learning_setting == "inductive" else set(range(8))
    assert seen_by_lifting == [expected_nodes]
    assert set(batch.original_nid.tolist()) == expected_nodes
    assert torch.equal(batch.x[:, 0].long(), batch.original_nid)
    assert torch.equal(batch.y, data.y[batch.original_nid])
    assert set(batch.original_nid[batch.supervised_mask].tolist()) == train_nodes
    observed_edges = batch.original_nid[batch.edge_index]
    assert set(map(tuple, observed_edges.t().tolist())) == {
        (a, b) for a in expected_nodes for b in expected_nodes if a != b
    }
    assert torch.equal(
        batch.edge_attr[:, 0], (observed_edges[0] * 8 + observed_edges[1]).float()
    )
    assert ("train_handle" in handle) == (learning_setting == "inductive")

    datamodule.setup("validate")
    for split, loader in [
        ("val", datamodule.val_dataloader()),
        ("test", datamodule.test_dataloader()),
    ]:
        batch = next(iter(loader))
        expected_mask = torch.from_numpy(np.load(handle["paths"][f"{split}_mask_perm"]))
        expected_supervised = set(permutation[expected_mask].tolist())
        assert set(batch.original_nid[batch.supervised_mask].tolist()) == expected_supervised
        assert torch.equal(batch.y, data.y[batch.original_nid])
        assert torch.equal(batch.x[:, 0].long(), batch.original_nid)
        eval_edges = batch.original_nid[batch.edge_index]
        assert set(map(tuple, eval_edges.t().tolist())) == {
            (a, b) for a in batch.original_nid.tolist()
            for b in batch.original_nid.tolist() if a != b
        }
    full_eval_batch = next(iter(datamodule.inference_dataloader(
        split="test", q=2, cover_parts="all"
    )))
    assert set(full_eval_batch.original_nid.tolist()) == set(range(8))
    assert seen_by_lifting[-1] == set(range(8))


def test_inductive_partition_cache_tracks_training_membership(tmp_path):
    """Changed training membership rebuilds only the induced graph partition."""
    nodes = torch.arange(8)
    edges = torch.combinations(nodes).t()
    data = Data(
        x=nodes[:, None].float(), y=nodes % 2,
        edge_index=torch.cat([edges, edges.flip(0)], dim=1), num_nodes=8,
    )
    handles = []
    for train, valid, test in [
        ([0, 2, 4, 6], [1], [3, 5, 7]),
        ([0, 1, 4, 6], [2], [3, 5, 7]),
        ([0, 2, 4, 6], [3], [1, 5, 7]),
    ]:
        dataset = SyntheticGraphDataset(str(tmp_path / "raw"), data.clone())
        dataset.split_idx = {
            "train": torch.tensor(train), "valid": torch.tensor(valid),
            "test": torch.tensor(test),
        }
        preprocessor = PreProcessor(dataset, str(tmp_path / "partitions"))
        handles.append(preprocessor.pack_global_partition(
            split_params=DictConfig({
                "learning_setting": "inductive", "split_type": "fixed",
                "standardize": False,
            }),
            cluster_params={"num_parts": 2, "sparse_format": "csr"},
            stream_params={},
        ))
    first, changed_train, changed_eval = handles
    assert first["config_hash"] == changed_train["config_hash"] == changed_eval["config_hash"]
    assert first["train_handle"]["config_hash"] != changed_train["train_handle"]["config_hash"]
    assert first["train_handle"]["paths"] == changed_eval["train_handle"]["paths"]
    assert first["split_hash"] != changed_eval["split_hash"]
    for handle in handles:
        training = handle["train_handle"]
        assert training["config_hash"] != handle["config_hash"]
        dm = ClusterGCNDataModule(data_handle=handle, q=2, cache_val=False)
        batch = next(iter(dm.train_dataloader()))
        assert torch.equal(batch.x[:, 0].long(), batch.original_nid)


@pytest.mark.parametrize("name,loader_class,module", [
    ("Reddit", RedditDatasetLoader, "reddit_dataset_loader"),
    ("Flickr", GraphSAINTDatasetLoader, "graphsaint_datasets"),
    ("Yelp", GraphSAINTDatasetLoader, "graphsaint_datasets"),
])
def test_dataset_exposes_official_node_indices(tmp_path, name, loader_class, module):
    """Preserve all three PyG masks as indices accepted by fixed node splits."""
    data = Data(
        train_mask=torch.tensor([True, False, False, True]),
        val_mask=torch.tensor([False, True, False, False]),
        test_mask=torch.tensor([False, False, True, False]),
    )
    dataset = MagicMock()
    dataset.__getitem__.return_value = data
    loader = loader_class(DictConfig({
        "data_dir": str(tmp_path), "data_name": name,
    }))
    with patch(
        f"topobench.data.loaders.graph.{module}.{name}",
        return_value=dataset,
    ):
        loaded = loader.load_dataset()
    assert loaded is dataset
    assert torch.equal(loaded.split_idx["train"], torch.tensor([0, 3]))
    assert torch.equal(loaded.split_idx["valid"], torch.tensor([1]))
    assert torch.equal(loaded.split_idx["test"], torch.tensor([2]))
