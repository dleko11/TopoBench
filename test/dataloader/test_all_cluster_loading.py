"""Regression tests for all-cluster streaming semantics."""

from unittest.mock import MagicMock, patch

import numpy as np
import torch
from torch_geometric.data import Data, InMemoryDataset

from topobench.data.preprocessor.preprocessor import PreProcessor
from topobench.dataloader import ClusterGCNDataModule


class _SyntheticGraphDataset(InMemoryDataset):
    """Minimal in-memory dataset used to build an on-disk partition."""

    def __init__(self, root: str, data: Data) -> None:
        super().__init__(root)
        self.data = data
        self._data = data
        self.slices = None


def _build_partition(tmp_path) -> tuple[dict, torch.Tensor]:
    """Build a two-part cycle graph and return its streaming handle."""
    num_nodes = 8
    edge_index = torch.tensor(
        [
            [0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 0],
            [1, 0, 2, 1, 3, 2, 4, 3, 5, 4, 6, 5, 7, 6, 0, 7],
        ],
        dtype=torch.long,
    )
    graph = Data(
        x=torch.randn(num_nodes, 4),
        edge_index=edge_index,
        y=torch.randint(0, 2, (num_nodes,)),
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
    dataset = _SyntheticGraphDataset(str(tmp_path), graph)
    split_dataset = MagicMock(data_lst=[graph])

    with patch.object(
        PreProcessor,
        "load_dataset_splits",
        return_value=(split_dataset, split_dataset, split_dataset),
    ):
        preprocessor = PreProcessor(dataset, str(tmp_path), None)
        handle = preprocessor.pack_global_partition(
            split_params={
                "learning_setting": "transductive",
                "type": "random",
            },
            cluster_params={
                "num_parts": 2,
                "recursive": False,
                "keep_inter_cluster_edges": False,
                "sparse_format": "csr",
            },
            stream_params={},
            dtype_policy="preserve",
        )

    # Pretend only one partition contains active nodes for each split. These
    # sidecars remain useful metadata, but must not filter default loaders.
    for split in ("train", "val", "test"):
        np.save(
            handle["paths"][f"parts_with_{split}"],
            np.asarray([0], dtype=np.int64),
        )

    return handle, edge_index


def _covered_node_ids(loader) -> torch.Tensor:
    """Return sorted global node IDs observed in a loader iteration."""
    return torch.cat([batch.global_nid for batch in loader]).sort().values


def test_default_loaders_cover_all_clusters(tmp_path):
    """Training and default evaluation include supervision-empty clusters."""
    handle, edge_index = _build_partition(tmp_path)
    datamodule = ClusterGCNDataModule(
        data_handle=handle,
        q=1,
        num_workers=0,
        cache_val=False,
    )
    expected_nodes = torch.arange(8)

    assert torch.equal(
        _covered_node_ids(datamodule.train_dataloader()),
        expected_nodes,
    )
    assert torch.equal(
        _covered_node_ids(datamodule.val_dataloader()),
        expected_nodes,
    )
    assert torch.equal(
        _covered_node_ids(datamodule.test_dataloader()),
        expected_nodes,
    )

    # Selecting all parts in one batch restores all cross-partition edges.
    full_batch = next(
        iter(
            datamodule.inference_dataloader(
                split="train",
                q=handle["num_parts"],
                cover_parts="all_parts",
            )
        )
    )
    assert full_batch.edge_index.shape[1] == edge_index.shape[1]

    cached_datamodule = ClusterGCNDataModule(
        data_handle=handle,
        q=1,
        num_workers=0,
        cache_val=True,
        val_cache_dir=str(tmp_path / "val_cache"),
    )
    cached_datamodule.setup("fit")
    assert torch.equal(
        _covered_node_ids(cached_datamodule.val_dataloader()),
        expected_nodes,
    )


def test_train_shuffle_changes_by_epoch_and_replays_from_seed(tmp_path):
    """All-cluster recombination changes by epoch and remains reproducible."""
    handle, _ = _build_partition(tmp_path)
    datamodule = ClusterGCNDataModule(
        data_handle=handle,
        q=1,
        num_workers=0,
        cache_val=False,
        seed=42,
    )

    loader = datamodule.train_dataloader()
    epoch_orders = [
        tuple(int(batch.global_nid[0]) for batch in loader) for _ in range(4)
    ]
    replay_loader = datamodule.train_dataloader()
    replay_orders = [
        tuple(int(batch.global_nid[0]) for batch in replay_loader)
        for _ in range(4)
    ]

    assert len(set(epoch_orders)) > 1
    assert replay_orders == epoch_orders


def test_split_only_evaluation_remains_available(tmp_path):
    """The legacy split-only evaluation mode remains an explicit option."""
    handle, _ = _build_partition(tmp_path)
    datamodule = ClusterGCNDataModule(
        data_handle=handle,
        q=1,
        num_workers=0,
        cache_val=False,
        eval_cover_strategy="split_parts",
    )

    assert _covered_node_ids(datamodule.val_dataloader()).numel() < 8
