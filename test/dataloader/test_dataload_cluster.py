"""Tests for cluster dataloader validation ordering."""

from unittest.mock import sentinel

import numpy as np
import pytest
import torch
from torch_geometric.data import Data

from topobench.dataloader.dataload_cluster import ClusterGCNDataModule


@pytest.mark.parametrize("val_shuffle", [False, True])
def test_validation_loader_respects_shuffle_flag(
    tmp_path, mocker, val_shuffle
):
    """Forward the configured validation shuffle value to the loader."""
    datamodule = ClusterGCNDataModule(
        data_handle={
            "processed_dir": str(tmp_path),
            "num_parts": 4,
            "sparse_format": "csr",
        },
        q=2,
        val_shuffle=val_shuffle,
        cache_val=False,
    )
    build_loader = mocker.patch.object(
        datamodule,
        "_build_loader",
        return_value=sentinel.validation_loader,
    )

    loader = datamodule.val_dataloader()

    assert loader is sentinel.validation_loader
    build_loader.assert_called_once()
    loader_args = build_loader.call_args.kwargs
    assert loader_args["split"] == "val"
    assert loader_args["shuffle"] is False
    observed_parts = loader_args["part_ids"]
    expected_parts = np.arange(4, dtype=np.int64)
    if val_shuffle:
        generator = torch.Generator().manual_seed(42)
        expected_parts = expected_parts[
            torch.randperm(4, generator=generator).numpy()
        ]
    assert np.array_equal(observed_parts, expected_parts)


@pytest.mark.parametrize("val_shuffle", [False, True])
def test_validation_cache_respects_shuffle_flag(tmp_path, mocker, val_shuffle):
    """Use deterministic validation ordering while building the cache."""
    seed = 7
    datamodule = ClusterGCNDataModule(
        data_handle={
            "processed_dir": str(tmp_path),
            "num_parts": 6,
            "sparse_format": "csr",
        },
        q=2,
        q_val=2,
        val_shuffle=val_shuffle,
        seed=seed,
        cache_val=True,
        val_cache_dir=str(tmp_path / "cache"),
    )
    mocker.patch.object(
        datamodule,
        "_part_ids_for_split",
        return_value=np.arange(6, dtype=np.int64),
    )
    observed_batches = []

    def collate(parts):
        observed_batches.append(list(parts))
        return Data(x=torch.ones(1, 1))

    mocker.patch(
        "topobench.dataloader.dataload_cluster.BlockCSRBatchCollator",
        return_value=collate,
    )

    datamodule.setup("validate")

    expected_parts = np.arange(6, dtype=np.int64)
    if val_shuffle:
        generator = torch.Generator().manual_seed(seed)
        expected_parts = expected_parts[
            torch.randperm(6, generator=generator).numpy()
        ]
    expected_batches = [
        expected_parts[i : i + 2].tolist() for i in range(0, 6, 2)
    ]
    assert observed_batches == expected_batches
