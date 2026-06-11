"""Unit and integration tests for the on-disk Cluster-GCN partitioning and dataloading pipeline."""

import os.path as osp
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import hydra
import numpy as np
import pytest
import torch
from hydra.core.global_hydra import GlobalHydra
from torch_geometric.data import Data, InMemoryDataset

from topobench.data.preprocessor.preprocessor import PreProcessor
from topobench.dataloader import ClusterGCNDataModule
from topobench.run import _average_ensemble_logits_by_global_nid


def _write_cluster_memmap_handle(
    root: Path,
    *,
    parts_with_test: list[int],
) -> dict:
    """Create a deterministic tiny cluster memmap handle for loader tests."""
    processed_dir = root / "processed"
    mm_dir = processed_dir / "perm_memmap"
    mm_dir.mkdir(parents=True)

    partptr = np.array([0, 2, 4, 6, 8], dtype=np.int64)
    np.save(mm_dir / "partptr.npy", partptr)
    np.save(mm_dir / "indptr.npy", np.zeros(9, dtype=np.int64))
    np.save(mm_dir / "indices.npy", np.empty(0, dtype=np.int64))
    np.save(
        mm_dir / "X_perm.npy",
        np.arange(16, dtype=np.float32).reshape(8, 2),
    )
    np.save(mm_dir / "y_perm.npy", np.arange(8, dtype=np.int64) % 2)

    train_mask = np.zeros(8, dtype=bool)
    val_mask = np.zeros(8, dtype=bool)
    test_mask = np.zeros(8, dtype=bool)
    train_mask[0:2] = True
    val_mask[4:6] = True
    for part in parts_with_test:
        test_mask[partptr[part] : partptr[part + 1]] = True

    np.save(mm_dir / "train_mask_perm.npy", train_mask)
    np.save(mm_dir / "val_mask_perm.npy", val_mask)
    np.save(mm_dir / "test_mask_perm.npy", test_mask)
    np.save(mm_dir / "parts_with_train.npy", np.array([0], dtype=np.int64))
    np.save(mm_dir / "parts_with_val.npy", np.array([2], dtype=np.int64))
    np.save(
        mm_dir / "parts_with_test.npy",
        np.array(parts_with_test, dtype=np.int64),
    )

    return {
        "processed_dir": str(processed_dir),
        "num_parts": 4,
        "sparse_format": "csr",
        "paths": {
            "partptr": str(mm_dir / "partptr.npy"),
            "indptr": str(mm_dir / "indptr.npy"),
            "indices": str(mm_dir / "indices.npy"),
            "X_perm": str(mm_dir / "X_perm.npy"),
            "y_perm": str(mm_dir / "y_perm.npy"),
            "train_mask_perm": str(mm_dir / "train_mask_perm.npy"),
            "val_mask_perm": str(mm_dir / "val_mask_perm.npy"),
            "test_mask_perm": str(mm_dir / "test_mask_perm.npy"),
            "parts_with_train": str(mm_dir / "parts_with_train.npy"),
            "parts_with_val": str(mm_dir / "parts_with_val.npy"),
            "parts_with_test": str(mm_dir / "parts_with_test.npy"),
        },
    }


def _global_nid_batches(loader) -> list[list[int]]:
    """Return global_nid lists for all batches in a loader."""
    return [batch.global_nid.tolist() for batch in loader]


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

    def test_test_dataloader_keeps_split_coverage(self, tmp_path):
        """Verify current test_dataloader remains split-aware."""
        handle = _write_cluster_memmap_handle(
            tmp_path,
            parts_with_test=[1, 3],
        )
        datamodule = ClusterGCNDataModule(
            data_handle=handle,
            q=1,
            q_test=1,
            num_workers=0,
        )

        current_batches = _global_nid_batches(datamodule.test_dataloader())
        inference_batches = _global_nid_batches(
            datamodule.inference_dataloader(split="test"),
        )

        assert current_batches == [[2, 3], [6, 7]]
        assert inference_batches == current_batches

    def test_full_graph_inference_loader_covers_all_parts(self, tmp_path):
        """Verify full-graph inference uses every partition part."""
        handle = _write_cluster_memmap_handle(
            tmp_path,
            parts_with_test=[1, 3],
        )
        datamodule = ClusterGCNDataModule(
            data_handle=handle,
            q=1,
            q_test=1,
            num_workers=0,
        )

        batches = list(
            datamodule.inference_dataloader(
                split="test",
                q=datamodule.num_parts,
                shuffle=False,
                cover_parts="all",
            )
        )

        assert len(batches) == 1
        assert batches[0].global_nid.tolist() == list(range(8))
        assert batches[0].test_mask.sum().item() == 4

    def test_ensemble_loaders_shuffle_same_test_parts(self, tmp_path):
        """Verify ensemble loaders change grouping order but keep coverage."""
        handle = _write_cluster_memmap_handle(
            tmp_path,
            parts_with_test=[1, 3],
        )
        datamodule = ClusterGCNDataModule(
            data_handle=handle,
            q=1,
            q_test=1,
            num_workers=0,
        )

        first = _global_nid_batches(
            datamodule.inference_dataloader(
                split="test",
                q=1,
                shuffle=True,
                seed=1,
            )
        )
        second = _global_nid_batches(
            datamodule.inference_dataloader(
                split="test",
                q=1,
                shuffle=True,
                seed=2,
            )
        )

        assert first != second
        assert sorted(first) == sorted(second) == [[2, 3], [6, 7]]


class TestTestInferenceEnsembleAggregation:
    """Tests for averaging ensemble predictions by global node id."""

    def test_logits_are_averaged_by_global_nid(self):
        """Verify logits are averaged while labels stay aligned."""
        avg_logits, labels, global_nids = (
            _average_ensemble_logits_by_global_nid(
                logit_chunks=[
                    torch.tensor([[1.0, 3.0], [10.0, 0.0]]),
                    torch.tensor([[3.0, 5.0], [14.0, 2.0]]),
                ],
                label_chunks=[
                    torch.tensor([0, 1]),
                    torch.tensor([0, 1]),
                ],
                nid_chunks=[
                    torch.tensor([10, 20]),
                    torch.tensor([10, 20]),
                ],
                expected_runs=2,
            )
        )

        assert global_nids.tolist() == [10, 20]
        assert labels.tolist() == [0, 1]
        assert torch.allclose(
            avg_logits,
            torch.tensor([[2.0, 4.0], [12.0, 1.0]]),
        )

    def test_missing_coverage_raises(self):
        """Verify missing ensemble coverage is rejected."""
        with pytest.raises(ValueError, match="coverage mismatch"):
            _average_ensemble_logits_by_global_nid(
                logit_chunks=[
                    torch.tensor([[1.0], [2.0]]),
                    torch.tensor([[3.0]]),
                ],
                label_chunks=[
                    torch.tensor([0, 1]),
                    torch.tensor([0]),
                ],
                nid_chunks=[
                    torch.tensor([10, 20]),
                    torch.tensor([10]),
                ],
                expected_runs=2,
            )

    def test_inconsistent_labels_raise(self):
        """Verify duplicate predictions for a node must agree on labels."""
        with pytest.raises(ValueError, match="Inconsistent labels"):
            _average_ensemble_logits_by_global_nid(
                logit_chunks=[
                    torch.tensor([[1.0]]),
                    torch.tensor([[2.0]]),
                ],
                label_chunks=[
                    torch.tensor([0]),
                    torch.tensor([1]),
                ],
                nid_chunks=[
                    torch.tensor([10]),
                    torch.tensor([10]),
                ],
                expected_runs=2,
            )


class TestTestInferenceConfig:
    """Hydra composition checks for test inference config."""

    def setup_method(self):
        """Clear Hydra before each composition test."""
        GlobalHydra.instance().clear()

    def teardown_method(self):
        """Clear Hydra after each composition test."""
        GlobalHydra.instance().clear()

    def test_default_test_inference_config_composes(self):
        """Verify the default protocol config composes."""
        with hydra.initialize_config_dir(
            version_base="1.3",
            config_dir=str(Path.cwd() / "configs"),
        ):
            cfg = hydra.compose(config_name="run.yaml")

        assert list(cfg.test_inference.protocols) == ["batched"]
        assert cfg.test_inference.ensemble_runs == 10
        assert cfg.test_inference.ensemble_seed == cfg.seed

    def test_protocol_override_composes(self):
        """Verify the full ablation protocol override composes."""
        with hydra.initialize_config_dir(
            version_base="1.3",
            config_dir=str(Path.cwd() / "configs"),
        ):
            cfg = hydra.compose(
                config_name="run.yaml",
                overrides=[
                    "test_inference.protocols=[batched,full_graph,ensemble]"
                ],
            )

        assert list(cfg.test_inference.protocols) == [
            "batched",
            "full_graph",
            "ensemble",
        ]
