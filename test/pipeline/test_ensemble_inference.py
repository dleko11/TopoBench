"""Regression tests for ensemble device placement and aggregation."""

import random
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch
from lightning import LightningModule
from lightning.pytorch.callbacks import ModelCheckpoint
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from torch_geometric.data import Data

import topobench.run as pipeline
from topobench.dataloader import ClusterGCNDataModule
from topobench.evaluator.evaluator import TBEvaluator
from topobench.loss.dataset import DatasetLoss


@pytest.mark.parametrize("ensemble_runs", [1, 3, 10])
@pytest.mark.parametrize("label_shape", [(), (2,)])
def test_aggregation_matches_grouped_reference(ensemble_runs, label_shape):
    generator = torch.Generator().manual_seed(17)
    node_ids = torch.arange(17) * 7 + 3
    node_labels = torch.randint(0, 4, (17, *label_shape), generator=generator)
    if label_shape:
        node_labels = node_labels.float()
    node_order = torch.arange(17).repeat(ensemble_runs)
    node_order = node_order[
        torch.randperm(len(node_order), generator=generator)
    ]
    nids = node_ids[node_order]
    labels = node_labels[node_order]
    logits = torch.randn(len(nids), 4, generator=generator)

    averaged, actual_labels, actual_ids = (
        pipeline._average_ensemble_logits_by_global_nid(
            logit_chunks=list(logits.split(7)),
            label_chunks=list(labels.split(7)),
            nid_chunks=list(nids.split(7)),
            expected_runs=ensemble_runs,
        )
    )

    reference = torch.stack([logits[nids == nid].mean(0) for nid in node_ids])
    torch.testing.assert_close(averaged, reference)
    assert torch.equal(actual_labels, node_labels)
    assert torch.equal(actual_ids, node_ids)
    assert actual_labels.dtype == node_labels.dtype


@pytest.mark.parametrize("vector_labels", [False, True])
def test_inconsistent_labels_are_rejected(vector_labels):
    labels = torch.tensor([1, 0, 0, 0])
    if vector_labels:
        labels = torch.stack([torch.zeros_like(labels), labels], dim=1).float()
    with pytest.raises(ValueError, match="Inconsistent labels.*global_nid=9"):
        pipeline._average_ensemble_logits_by_global_nid(
            logit_chunks=[torch.ones(4, 2)],
            label_chunks=[labels],
            nid_chunks=[torch.tensor([9, 2, 9, 2])],
            expected_runs=2,
        )


@pytest.mark.parametrize("nids", [[3, 9, 3], [3, 9, 3, 9, 9]])
def test_missing_or_extra_predictions_are_rejected(nids):
    with pytest.raises(ValueError, match="Ensemble coverage mismatch"):
        pipeline._average_ensemble_logits_by_global_nid(
            logit_chunks=[torch.ones(len(nids), 2)],
            label_chunks=[torch.zeros(len(nids), dtype=torch.long)],
            nid_chunks=[torch.tensor(nids)],
            expected_runs=2,
        )


@pytest.mark.parametrize("empty_batch", [False, True])
def test_empty_predictions_are_rejected(empty_batch):
    with pytest.raises(
        ValueError, match="no predictions|no supervised test nodes"
    ):
        pipeline._average_ensemble_logits_by_global_nid(
            logit_chunks=[torch.empty(0, 2)] if empty_batch else [],
            label_chunks=[torch.empty(0, dtype=torch.long)]
            if empty_batch
            else [],
            nid_chunks=[torch.empty(0, dtype=torch.long)]
            if empty_batch
            else [],
            expected_runs=2,
        )


@pytest.mark.parametrize(
    "device", [torch.device("cpu"), torch.device("cuda:2")]
)
def test_run_passes_trainer_device_after_fit(monkeypatch, device):
    cfg = OmegaConf.create(
        {
            "seed": 0,
            "train": True,
            "test": True,
            "dataset": {
                "loader": {"_target_": "loader", "parameters": {}},
                "parameters": {"task_level": "node"},
                "split_params": {},
            },
            "model": {"_target_": "model"},
            "trainer": {"_target_": "trainer"},
            "evaluator": {},
            "optimizer": {},
            "loss": {},
        }
    )
    loader = Mock()
    loader.load.return_value = (object(), "/unused")
    model = SimpleNamespace(device=torch.device("cpu"))
    trainer = Mock()
    trainer.strategy.root_device = device
    trainer.callback_metrics = {}
    preprocessor = Mock()
    preprocessor.load_dataset_splits.return_value = ([], [], [])
    monkeypatch.setattr(
        pipeline.hydra.utils,
        "instantiate",
        Mock(side_effect=[loader, model, trainer]),
    )
    monkeypatch.setattr(
        pipeline, "PreProcessor", Mock(return_value=preprocessor)
    )
    monkeypatch.setattr(pipeline, "TBDataloader", Mock())
    monkeypatch.setattr(pipeline, "instantiate_loggers", Mock(return_value=[]))
    monkeypatch.setattr(
        pipeline, "instantiate_callbacks", Mock(return_value=[])
    )
    monkeypatch.setattr(pipeline, "set_current_phase_tracker", Mock())
    rerun = Mock()
    monkeypatch.setattr(pipeline, "rerun_best_model_checkpoint", rerun)

    pipeline.run.__wrapped__(cfg)

    trainer.fit.assert_called_once()
    assert rerun.call_args.kwargs["device"] == device


class EnsembleModel(LightningModule):
    """Small classifier with real metrics and observable device placement."""

    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.eye(2))
        self.evaluator = TBEvaluator(
            "classification", num_classes=2, metrics=["accuracy"]
        )
        self.loss = DatasetLoss(
            {"task": "classification", "loss_type": "cross_entropy"}
        )
        self.validation_finished = False
        self.forward_devices = []

    def forward(self, batch):
        assert torch.is_inference_mode_enabled()
        assert not self.training
        assert batch.x.device == self.weight.device
        self.forward_devices.append(self.weight.device)
        return {"logits": batch.x @ self.weight, "labels": batch.y}

    def validation_step(self, batch, batch_idx):
        assert batch.x.device == self.weight.device

    def on_validation_end(self):
        self.validation_finished = True


@pytest.mark.parametrize(
    "accelerator",
    [
        "cpu",
        pytest.param(
            "gpu",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="CUDA unavailable"
            ),
        ),
    ],
)
def test_checkpoint_ensemble_restores_device_after_validation(
    tmp_path, monkeypatch, caplog, accelerator
):
    device = torch.device("cuda:0" if accelerator == "gpu" else "cpu")
    model = EnsembleModel()
    checkpoint = tmp_path / "best.ckpt"
    torch.save({"state_dict": model.state_dict()}, checkpoint)
    model.weight.data.zero_()
    callback = ModelCheckpoint()
    callback.best_model_path = str(checkpoint)
    cfg = OmegaConf.create(
        {
            "seed": 0,
            "trainer": {
                "_target_": "lightning.pytorch.Trainer",
                "accelerator": accelerator,
                "devices": 1,
                "default_root_dir": str(tmp_path),
                "enable_checkpointing": False,
                "enable_model_summary": False,
                "enable_progress_bar": False,
            },
            "test_inference": {
                "protocols": ["ensemble"],
                "ensemble_runs": 2,
                "ensemble_seed": 7,
            },
        }
    )
    batches = [
        Data(
            x=torch.tensor([[3.0, 0.0], [0.0, 3.0], [0.0, 0.0]]),
            y=torch.tensor([0, 1, -1]),
            test_mask=torch.tensor([True, True, False]),
            global_nid=torch.tensor([9, 1, 999]),
        ),
        Data(
            x=torch.tensor([[1.0, 4.0], [4.0, 1.0]]),
            y=torch.tensor([1, 0]),
            test_mask=torch.tensor([True, True]),
            global_nid=torch.tensor([1, 9]),
        ),
    ]
    datamodule = Mock(spec=ClusterGCNDataModule)
    datamodule.val_dataloader.return_value = DataLoader(
        [batches[0]], batch_size=None
    )
    datamodule.inference_dataloader.side_effect = [
        [batch.clone()] for batch in batches
    ]
    moves_after_validation = []
    original_to = model.to

    def record_move(target):
        if model.validation_finished:
            moves_after_validation.append(target)
        return original_to(target)

    monkeypatch.setattr(model, "to", record_move)
    logger = Mock()
    caplog.set_level("INFO", logger="topobench.run")

    pipeline.rerun_best_model_checkpoint(
        checkpoint_model=model,
        cfg=cfg,
        datamodule=datamodule,
        device=device,
        callbacks=[callback],
        logger=[logger],
    )

    assert moves_after_validation == [device]
    assert model.forward_devices == [device, device]
    assert checkpoint.exists()
    assert [
        call.kwargs["seed"]
        for call in datamodule.inference_dataloader.call_args_list
    ] == [7, 8]
    metrics = logger.log_metrics.call_args.args[0]
    assert metrics["test_inference/ensemble/accuracy"] == 1.0
    expected_loss = torch.nn.functional.cross_entropy(
        torch.tensor([[0.5, 3.5], [3.5, 0.5]]), torch.tensor([1, 0])
    )
    assert metrics["test_inference/ensemble/loss"] == pytest.approx(
        expected_loss.item()
    )
    for message in [
        "Completed ensemble pass 1/2",
        "Completed ensemble pass 2/2",
        "Aggregating ensemble predictions",
        "Computing ensemble test metrics",
        "Completed ensemble test inference",
    ]:
        assert message in caplog.text


@pytest.fixture
def resumable_ensemble(tmp_path):
    cfg = OmegaConf.create({
        "seed": 23,
        "dataset": {"split_params": {"data_seed": 0}},
        "test_inference": {
            "ensemble_runs": 3,
            "ensemble_seed": 7,
            "ensemble_shuffle": True,
            "ensemble_checkpoint_dir": str(tmp_path / "passes"),
        },
    })
    calls = []
    failure = {"enabled": False}

    def inference_dataloader(*, split, shuffle, seed, cover_parts):
        assert (split, shuffle, cover_parts) == ("test", True, "split")
        calls.append(seed)

        def batches():
            for index, nid in enumerate((9, 1)):
                if failure["enabled"] and seed == 8 and index == 1:
                    raise RuntimeError("simulated interruption during a pass")
                noise = random.random() + np.random.random() + torch.rand(()).item()
                x = torch.tensor([[3.0, noise] if index == 0 else [noise, 3.0]])
                yield Data(
                    x=x,
                    y=torch.tensor([index]),
                    test_mask=torch.tensor([True]),
                    global_nid=torch.tensor([nid]),
                )

        return batches()

    datamodule = SimpleNamespace(
        handle={"config_hash": 123, "split_hash": "split0", "num_parts": 4},
        q_test=2,
        transform_config=None,
        inference_dataloader=inference_dataloader,
    )
    return cfg, datamodule, calls, failure


@pytest.mark.parametrize(
    "device",
    [
        torch.device("cpu"),
        pytest.param(
            torch.device("cuda:0"),
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="CUDA unavailable"
            ),
        ),
    ],
)
def test_ensemble_resume_matches_uninterrupted(
    resumable_ensemble, monkeypatch, device
):
    cfg, datamodule, calls, failure = resumable_ensemble
    compute = Mock(wraps=pipeline._compute_ensemble_metrics)
    monkeypatch.setattr(pipeline, "_compute_ensemble_metrics", compute)
    uninterrupted_cfg = OmegaConf.create(OmegaConf.to_container(cfg))
    uninterrupted_cfg.test_inference.ensemble_checkpoint_dir = None
    pipeline.L.seed_everything(cfg.seed, workers=True)
    reference = pipeline._run_ensemble_test_inference(
        checkpoint_model=EnsembleModel(), cfg=uninterrupted_cfg,
        datamodule=datamodule, device=device,
    )
    expected_logits = compute.call_args.kwargs["logits"].clone()

    pipeline.L.seed_everything(cfg.seed, workers=True)
    failure["enabled"] = True
    with pytest.raises(RuntimeError, match="simulated interruption"):
        pipeline._run_ensemble_test_inference(
            checkpoint_model=EnsembleModel(), cfg=cfg,
            datamodule=datamodule, device=device,
        )
    directory = pipeline.Path(cfg.test_inference.ensemble_checkpoint_dir)
    assert [path.name for path in directory.glob("*.pt")] == ["pass_000.pt"]
    (directory / "pass_001.pt.tmp").write_bytes(b"interrupted write")

    calls.clear()
    failure["enabled"] = False
    pipeline.L.seed_everything(999, workers=True)
    actual = pipeline._run_ensemble_test_inference(
        checkpoint_model=EnsembleModel(), cfg=cfg,
        datamodule=datamodule, device=device,
    )
    assert calls == [8, 9]
    assert len(list(directory.glob("*.pt"))) == 3
    assert not list(directory.glob("*.tmp"))
    torch.testing.assert_close(
        compute.call_args.kwargs["logits"], expected_logits, rtol=0, atol=0
    )
    for key in reference:
        torch.testing.assert_close(
            torch.as_tensor(actual[key]), torch.as_tensor(reference[key]),
            rtol=0, atol=0,
        )

    calls.clear()
    complete = pipeline._run_ensemble_test_inference(
        checkpoint_model=EnsembleModel(), cfg=cfg,
        datamodule=datamodule, device=device,
    )
    assert calls == []
    torch.testing.assert_close(
        torch.as_tensor(complete["loss"]), torch.as_tensor(reference["loss"]),
        rtol=0, atol=0,
    )


@pytest.mark.parametrize(
    "change", ["weights", "seed", "shuffle", "q", "partition", "transforms"]
)
def test_ensemble_resume_rejects_mismatched_identity(resumable_ensemble, change):
    cfg, datamodule, calls, _ = resumable_ensemble
    model = EnsembleModel()
    pipeline._run_ensemble_test_inference(
        checkpoint_model=model, cfg=cfg,
        datamodule=datamodule, device=torch.device("cpu"),
    )
    calls.clear()
    if change == "weights":
        model.weight.data.add_(1)
    elif change == "seed":
        cfg.test_inference.ensemble_seed += 1
    elif change == "shuffle":
        cfg.test_inference.ensemble_shuffle = False
    elif change == "q":
        datamodule.q_test += 1
    elif change == "partition":
        datamodule.handle["split_hash"] = "different"
    elif change == "transforms":
        datamodule.transform_config = {"lifting": "different"}
    with pytest.raises(ValueError, match="Ensemble checkpoint mismatch"):
        pipeline._run_ensemble_test_inference(
            checkpoint_model=model, cfg=cfg,
            datamodule=datamodule, device=torch.device("cpu"),
        )
    assert calls == []
