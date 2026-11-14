"""Single-graph inductive splits must restrict topology before lifting."""

from pathlib import Path

import hydra
import numpy as np
import pytest
import torch
from omegaconf import DictConfig
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.loader import ClusterData

from topobench.data.loaders import GraphSAINTDatasetLoader
from topobench.data.preprocessor import PreProcessor
from topobench.run import run
from topobench.dataloader import ClusterGCNDataModule, TBDataloader
from topobench.transforms.liftings.graph2simplicial import SimplicialCliqueLifting


@pytest.fixture
def fixed_partition(monkeypatch):
    """Test data alignment against known partitions without requiring METIS."""
    def partition(self, edge_index, num_nodes):
        return torch.arange(num_nodes) % self.num_parts

    monkeypatch.setattr(ClusterData, "_metis", partition)


@pytest.fixture
def node_dataset():
    nodes = torch.arange(8)
    edges = torch.combinations(nodes).t()
    dataset = InMemoryDataset()
    dataset._data = Data(
        x=nodes[:, None].float(), y=nodes % 2, num_nodes=8,
        edge_index=torch.cat([edges, edges.flip(0)], dim=1),
    )
    dataset.slices = None
    dataset.split_idx = {
        "train": torch.tensor([0, 2, 4, 6]),
        "valid": torch.tensor([1]), "test": torch.tensor([3, 5, 7]),
    }
    return dataset


def test_full_graph_lifts_training_nodes_only_and_reuses_cache(
    tmp_path, monkeypatch, node_dataset
):
    observed = []
    original = SimplicialCliqueLifting.lift_topology

    def record_topology(self, data):
        observed.append(set(data.original_nid.tolist()))
        return original(self, data)

    monkeypatch.setattr(SimplicialCliqueLifting, "lift_topology", record_topology)
    params = DictConfig({"learning_setting": "inductive", "split_type": "fixed"})
    transform = DictConfig({"clique": {
        "transform_name": "SimplicialCliqueLifting", "complex_dim": 2,
    }})
    preprocessor = PreProcessor(node_dataset, str(tmp_path))
    train, valid, test = preprocessor.load_inductive_node_splits(params, transform)
    assert observed == [{0, 2, 4, 6}, set(range(8))]
    training, evaluation = train.data_lst[0], valid.data_lst[0]
    assert training.incidence_2.shape[1] == 4
    assert evaluation.incidence_2.shape[1] == 56
    assert training.train_mask.all()
    assert training.supervised_mask.all()
    assert evaluation.original_nid[evaluation.supervised_mask].tolist() == [1]
    testing = test.data_lst[0]
    assert testing.original_nid[testing.supervised_mask].tolist() == [3, 5, 7]
    assert evaluation.x.data_ptr() == testing.x.data_ptr()
    assert not training.val_mask.any()
    assert not training.test_mask.any()
    assert torch.equal(training.y, node_dataset[0].y[training.original_nid])
    loader = TBDataloader(train, valid, test)
    assert next(iter(loader.train_dataloader())).num_nodes == 4
    assert next(iter(loader.val_dataloader())).num_nodes == 8

    PreProcessor(node_dataset, str(tmp_path)).load_inductive_node_splits(params, transform)
    assert len(observed) == 2
    node_dataset.split_idx["train"] = torch.tensor([0, 2, 4])
    PreProcessor(node_dataset, str(tmp_path)).load_inductive_node_splits(params, transform)
    assert observed[2:] == [{0, 2, 4}, set(range(8))]


@pytest.mark.parametrize("partitioned", [False, True])
def test_multilabel_targets_keep_node_alignment(
    tmp_path, node_dataset, partitioned, fixed_partition
):
    node_dataset._data.y = torch.stack([
        torch.arange(8) % 2, (torch.arange(8) // 2) % 2,
        (torch.arange(8) // 4) % 2,
    ], dim=1)
    params = DictConfig({"learning_setting": "inductive", "split_type": "fixed"})
    preprocessor = PreProcessor(node_dataset, str(tmp_path))
    if partitioned:
        handle = preprocessor.pack_global_partition(
            params, {"num_parts": 2, "sparse_format": "csr"}, {},
        )
        stored = np.load(handle["paths"]["y_perm"])
        assert stored.shape == (8, 3)
        loader = ClusterGCNDataModule(
            data_handle=handle, q=2, q_val=2, q_test=2, cache_val=False,
        )
    else:
        loader = TBDataloader(*preprocessor.load_inductive_node_splits(params))
    for split in ("train", "val", "test"):
        batches = (
            loader.inference_dataloader(split=split, cover_parts="all")
            if partitioned else getattr(loader, f"{split}_dataloader")()
        )
        batch = next(iter(batches))
        assert batch.y.shape == (batch.num_nodes, 3)
        assert torch.equal(batch.y, node_dataset[0].y[batch.original_nid])
        if split == "train":
            assert set(batch.original_nid.tolist()) == {0, 2, 4, 6}
        else:
            assert set(batch.original_nid.tolist()) == set(range(8))


def test_full_graph_rejects_overlapping_training_nodes(tmp_path, node_dataset):
    node_dataset.split_idx["valid"] = torch.tensor([0])
    preprocessor = PreProcessor(node_dataset, str(tmp_path))
    with pytest.raises(ValueError, match="overlap"):
        preprocessor.load_inductive_node_splits(DictConfig({
            "learning_setting": "inductive", "split_type": "fixed",
        }))


@pytest.mark.parametrize("dataset", ["flickr", "yelp"])
@pytest.mark.parametrize("partitioned", [False, True])
@pytest.mark.parametrize("model", ["graph/gcn", "simplicial/scn"])
def test_inductive_training_and_checkpoint_evaluation(
    tmp_path, monkeypatch, node_dataset, fixed_partition, dataset, partitioned, model
):
    """Exercise both pipelines through training, validation, and checkpoint testing."""
    if dataset == "yelp":
        node_dataset._data.y = torch.stack([
            torch.arange(8) % 2, (torch.arange(8) // 2) % 2,
            (torch.arange(8) // 4) % 2,
        ], dim=1)
    # AbstractLoader supplies locking and paths, only the dataset download is replaced.
    monkeypatch.setattr(GraphSAINTDatasetLoader, "load_dataset", lambda self: node_dataset)
    repo = Path(__file__).resolve().parents[2]
    suffix = "_for_partitioning" if partitioned else ""
    overrides = [
        f"dataset=graph/{dataset}{suffix}", f"model={model}",
        "dataset.parameters.num_features=1", "dataset.parameters.num_nodes=8",
        f"dataset.parameters.num_classes={3 if dataset == 'yelp' else 2}",
        "trainer=cpu", "trainer.max_epochs=1", "trainer.min_epochs=1",
        "+trainer.enable_progress_bar=false", "+trainer.enable_model_summary=false",
        "model.feature_encoder.out_channels=8", "logger=[]",
        "callbacks=model_checkpoint", "test=true",
        f"paths.data_dir={tmp_path / 'data'}", f"paths.output_dir={tmp_path / 'output'}",
        f"paths.log_dir={tmp_path / 'logs'}",
    ]
    if dataset == "flickr":
        overrides.append("++dataset.parameters.metrics=[accuracy]")
    if partitioned:
        overrides += ["dataset.loader.parameters.cluster.num_parts=2",
                      "dataset.loader.parameters.stream.q=2"]
    with hydra.initialize_config_dir(config_dir=str(repo / "configs"), version_base="1.3"):
        cfg = hydra.compose(config_name="run.yaml", overrides=overrides)
    metrics, objects = run(cfg)
    expected = "f1_micro" if dataset == "yelp" else "accuracy"
    assert torch.isfinite(metrics[f"val/{expected}"])
    assert torch.isfinite(metrics["best_monitored_score"])
    datamodule = objects["datamodule"]
    train = next(iter(datamodule.train_dataloader()))
    valid = next(iter(datamodule.val_dataloader()))
    test = next(iter(datamodule.test_dataloader()))
    assert set(train.original_nid.tolist()) == {0, 2, 4, 6}
    assert valid.original_nid[valid.supervised_mask].tolist() == [1]
    assert set(test.original_nid[test.supervised_mask].tolist()) == {3, 5, 7}
