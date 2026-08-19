"""Main entry point for training and testing models."""

import random
from pathlib import Path
from typing import Any

import hydra
import lightning as L
import numpy as np
import rootutils
import torch
from lightning import Callback, LightningModule, Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import Logger
from lightning.pytorch.loggers.wandb import WandbLogger
from omegaconf import DictConfig, OmegaConf

from topobench.data.preprocessor import OnDiskPreProcessor, PreProcessor
from topobench.dataloader import TBDataloader
from topobench.utils import (
    PhaseResourceTracker,
    RankedLogger,
    extras,
    get_metric_value,
    instantiate_callbacks,
    instantiate_loggers,
    log_hyperparameters,
    set_current_phase_tracker,
    task_wrapper,
)
from topobench.utils.config_resolvers import register_all_resolvers
from topobench.utils.phase_tracking import track_phase

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
# ------------------------------------------------------------------------------------ #
# the setup_root above is equivalent to:
# - adding project root dir to PYTHONPATH
#       (so you don't need to force user to install project as a package)
#       (necessary before importing any local modules e.g. `from src import utils`)
# - setting up PROJECT_ROOT environment variable
#       (which is used as a base for paths in "configs/paths/default.yaml")
#       (this way all filepaths are the same no matter where you run the code)
# - loading environment variables from ".env" in root dir
#
# you can remove it if you:
# 1. either install project as a package or move entry files to project root dir
# 2. set `root_dir` to "." in "configs/paths/default.yaml"
#
# more info: https://github.com/ashleve/rootutils
# ------------------------------------------------------------------------------------ #


# Register custom resolvers before Hydra initialization
register_all_resolvers()


def initialize_hydra() -> DictConfig:
    """Initialize Hydra when main is not an option (e.g. tests).

    Returns
    -------
    DictConfig
        A DictConfig object containing the config tree.
    """
    hydra.initialize(
        version_base="1.3", config_path="../configs", job_name="run"
    )
    cfg = hydra.compose(config_name="run.yaml")
    return cfg


torch.set_num_threads(1)
log = RankedLogger(__name__, rank_zero_only=True)

_TEST_INFERENCE_PROTOCOLS = {"batched", "full_graph", "ensemble"}


@task_wrapper
def run(cfg: DictConfig) -> tuple[dict[str, Any], dict[str, Any]]:
    """Train the model.

    Can additionally evaluate on a testset, using best weights obtained during training.

    This method is wrapped in optional @task_wrapper decorator, that controls
    the behavior during failure. Useful for multiruns, saving info about the
    crash, etc.

    Parameters
    ----------
    cfg : DictConfig
        Configuration composed by Hydra.

    Returns
    -------
    tuple[dict[str, Any], dict[str, Any]]
        A tuple with metrics and dict with all instantiated objects.
    """
    # Set seed for random number generators in pytorch, numpy and python.random
    L.seed_everything(cfg.seed, workers=True)
    # Seed for torch
    torch.manual_seed(cfg.seed)
    # Seed for numpy
    np.random.seed(cfg.seed)
    # Seed for python random
    random.seed(cfg.seed)

    if cfg.get("deterministic", False):
        # Enable cudnn deterministic algorithms for reproducibility
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)
        log.info(
            "Enabled cudnn.deterministic and torch.use_deterministic_algorithms"
        )

    log.info("Instantiating loggers...")
    logger: list[Logger] = instantiate_loggers(cfg.get("logger"))
    phase_tracker = PhaseResourceTracker(logger)
    phase_tracker.initialize()
    set_current_phase_tracker(phase_tracker)

    with phase_tracker.track("dataset_load"):
        log.info(f"Instantiating loader <{cfg.dataset.loader._target_}>")
        dataset_loader = hydra.utils.instantiate(cfg.dataset.loader)
        dataset, dataset_dir = dataset_loader.load()

    raw_transform_config = cfg.get("transforms", None)

    memory_type = cfg.dataset.loader.parameters.get("memory_type", "in_memory")

    if memory_type == "on_disk_cluster":
        # Loads a graph in memory, performs partitioning
        from topobench.data.utils import build_cluster_transform, make_hash
        from topobench.dataloader import ClusterGCNDataModule

        with phase_tracker.track("full_graph_preprocessing"):
            log.info("Instantiating preprocessor...")
            preprocessor = PreProcessor(dataset, dataset_dir, None)
            build_cluster_transform(raw_transform_config)

        with phase_tracker.track("partition_build"):
            handle = preprocessor.pack_global_partition(
                split_params=cfg.dataset.get("split_params", {}),
                cluster_params=cfg.dataset.loader.parameters.get(
                    "cluster", {}
                ),
                stream_params=cfg.dataset.loader.parameters.get("stream", {}),
                dtype_policy=cfg.dataset.loader.parameters.get(
                    "dtype_policy", "preserve"
                ),
                pack_db=True,
                pack_memmaps=True,
            )

        with phase_tracker.track("datamodule_init"):
            transform_cfg_container = (
                OmegaConf.to_container(raw_transform_config, resolve=True)
                if raw_transform_config is not None
                else None
            )
            stream_cfg = cfg.dataset.loader.parameters.get("stream", {})
            q = int(stream_cfg.get("q", 1))
            q_val = stream_cfg.get("q_val", None)
            if q_val is not None:
                resolved_q_val = int(q_val)
            else:
                val_batches = stream_cfg.get("val_batches", 5)
                if val_batches is None:
                    resolved_q_val = q
                else:
                    val_batches = int(val_batches)
                    num_parts = int(handle.get("num_parts"))
                    resolved_q_val = max(
                        q, (num_parts + val_batches - 1) // val_batches
                    )
            eval_cover_strategy = cfg.get("eval", {}).get(
                "cover_strategy", "all_parts"
            )
            val_cache_fingerprint = make_hash(
                {
                    "partition_hash": handle.get("config_hash", None),
                    "transform": transform_cfg_container,
                    "q_val": resolved_q_val,
                    "with_edge_attr": stream_cfg.get("with_edge_attr", False),
                    "reconstruct_cross_cluster_edges": stream_cfg.get(
                        "reconstruct_cross_cluster_edges", True
                    ),
                    "seed": cfg.get("seed", 42),
                    "eval_cover_strategy": eval_cover_strategy,
                }
            )

            # Build streaming loaders
            datamodule = ClusterGCNDataModule(
                data_handle=handle,
                q=q,
                q_test=stream_cfg.get("q_test", None),
                q_val=stream_cfg.get("q_val", None),
                val_batches=stream_cfg.get("val_batches", 5),
                test_batches=stream_cfg.get("test_batches", None),
                num_workers=stream_cfg.get("num_workers", 0),
                cache_num_workers=stream_cfg.get("cache_num_workers", None),
                pin_memory=stream_cfg.get("pin_memory", False),
                with_edge_attr=stream_cfg.get("with_edge_attr", False),
                reconstruct_cross_cluster_edges=stream_cfg.get(
                    "reconstruct_cross_cluster_edges", True
                ),
                train_shuffle=stream_cfg.get("train_shuffle", True),
                eval_cover_strategy=eval_cover_strategy,
                seed=cfg.get("seed", 42),
                transform_config=transform_cfg_container,
                cache_val=True,
                val_cache_fingerprint=val_cache_fingerprint,
            )
    else:
        with phase_tracker.track("full_graph_preprocessing"):
            log.info("Instantiating preprocessor...")
            transform_config = (
                hydra.utils.instantiate(raw_transform_config)
                if raw_transform_config is not None
                else None
            )
            # TB standard in-memory pipeline and on-disk inductive pipeline
            preprocessor_cls = (
                OnDiskPreProcessor
                if memory_type == "on_disk"
                else PreProcessor
            )
            preprocessor = preprocessor_cls(
                dataset,
                dataset_dir,
                transform_config,
            )
            dataset_train, dataset_val, dataset_test = (
                preprocessor.load_dataset_splits(cfg.dataset.split_params)
            )

        with phase_tracker.track("datamodule_init"):
            log.info("Instantiating datamodule...")
            if cfg.dataset.parameters.task_level in ["node", "graph"]:
                datamodule = TBDataloader(
                    dataset_train=dataset_train,
                    dataset_val=dataset_val,
                    dataset_test=dataset_test,
                    **cfg.dataset.get("dataloader_params", {}),
                )
            else:
                raise ValueError("Invalid task_level")

    # Model for us is Network + logic: inputs backbone, readout, losses
    with phase_tracker.track("model_init"):
        log.info(f"Instantiating model <{cfg.model._target_}>")
        model: LightningModule = hydra.utils.instantiate(
            cfg.model,
            evaluator=cfg.evaluator,
            optimizer=cfg.optimizer,
            loss=cfg.loss,
        )

    with phase_tracker.track("callbacks_init"):
        log.info("Instantiating callbacks...")
        callbacks: list[Callback] = instantiate_callbacks(cfg.get("callbacks"))

    # Log to wandb preprocessor time
    if logger:
        preprocessing_time = getattr(preprocessor, "preprocessing_time", 0.0)
        for log_temp in logger:
            if isinstance(log_temp, L.pytorch.loggers.wandb.WandbLogger):
                log_temp.log_metrics(
                    {
                        "preprocessor_time": preprocessing_time,
                    }
                )

    with phase_tracker.track("trainer_init"):
        log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
        trainer: Trainer = hydra.utils.instantiate(
            cfg.trainer,
            callbacks=callbacks,
            logger=logger,
            num_sanity_val_steps=0,
            log_every_n_steps=1,  # Log metrics every step (Lightning requires >=1)
        )

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "callbacks": callbacks,
        "logger": logger,
        "trainer": trainer,
    }

    if logger:
        log.info("Logging hyperparameters!")
        log_hyperparameters(object_dict)

    if cfg.get("train"):
        log.info("Starting training!")
        trainer.fit(
            model=model, datamodule=datamodule, ckpt_path=cfg.get("ckpt_path")
        )
        # Log the best model checkpoint path into wandb
        for logger_elem in logger:
            if isinstance(logger_elem, WandbLogger) and hasattr(
                logger_elem, "experiment"
            ):
                logger_elem.experiment.log(
                    {"checkpoint": trainer.checkpoint_callback.best_model_path}
                )
                logger_elem.experiment.log(
                    {
                        "best_monitored_score": trainer.checkpoint_callback.best_model_score
                    }
                )

    train_metrics = trainer.callback_metrics
    if cfg.get("test"):
        log.info("Starting testing!")

        rerun_best_model_checkpoint(
            checkpoint_model=model,
            cfg=cfg,
            datamodule=datamodule,
            device=model.device,
            callbacks=callbacks,
            logger=logger,
        )

    # Merge train and test metrics
    metric_dict = {**train_metrics}

    return metric_dict, object_dict


def _resolve_test_inference_protocols(cfg: DictConfig) -> list[str]:
    """Return configured test inference protocols.

    Parameters
    ----------
    cfg : DictConfig
        Composed Hydra configuration.

    Returns
    -------
    list[str]
        Normalized protocol names.
    """
    inference_cfg = cfg.get("test_inference", {}) or {}
    protocols = inference_cfg.get("protocols", ["batched"])
    if isinstance(protocols, str):
        protocols = [protocols]

    resolved = [str(protocol).lower() for protocol in protocols]
    if not resolved:
        raise ValueError("test_inference.protocols must not be empty.")

    invalid = sorted(set(resolved) - _TEST_INFERENCE_PROTOCOLS)
    if invalid:
        raise ValueError(
            "Unsupported test inference protocol(s): "
            f"{invalid}. Expected one of {sorted(_TEST_INFERENCE_PROTOCOLS)}."
        )

    return resolved


def _validate_test_inference_protocols(
    protocols: list[str],
    datamodule: Any,
) -> bool:
    """Validate protocols against the datamodule type.

    Parameters
    ----------
    protocols : list[str]
        Configured test inference protocol names.
    datamodule : Any
        Datamodule used for the current run.

    Returns
    -------
    bool
        Whether the datamodule is a ClusterGCNDataModule.
    """
    from topobench.dataloader import ClusterGCNDataModule

    is_cluster = isinstance(datamodule, ClusterGCNDataModule)
    if not is_cluster and any(protocol != "batched" for protocol in protocols):
        raise ValueError(
            "test_inference protocols 'full_graph' and 'ensemble' require "
            "ClusterGCNDataModule/on_disk_cluster. Non-cluster dataloaders "
            "support only 'batched'."
        )
    return is_cluster


def _metric_suffix(key: str) -> str:
    """Strip a Lightning metric namespace from a metric key.

    Parameters
    ----------
    key : str
        Metric key, optionally namespaced by a slash.

    Returns
    -------
    str
        Metric key without the leading namespace.
    """
    return key.split("/", 1)[1] if "/" in key else key


def _metric_log_value(value: Any) -> Any:
    """Convert scalar tensors to plain values for external loggers.

    Parameters
    ----------
    value : Any
        Metric value to normalize.

    Returns
    -------
    Any
        Plain scalar for scalar tensors, otherwise the original value.
    """
    if torch.is_tensor(value):
        value = value.detach().cpu()
        if value.numel() == 1:
            return value.item()
    return value


def _log_prefixed_metrics(
    *,
    prefix: str,
    metrics: dict[str, Any],
    logger: list[Logger],
) -> None:
    """Log metrics under a prefix to configured external loggers.

    Parameters
    ----------
    prefix : str
        Metric namespace prefix.
    metrics : dict[str, Any]
        Unprefixed metric values.
    logger : list[Logger]
        Configured Lightning loggers.
    """
    logged = {
        f"{prefix}/{key}": _metric_log_value(value)
        for key, value in metrics.items()
    }
    log.info(logged)
    for lgr in logger:
        if hasattr(lgr, "log_metrics"):
            lgr.log_metrics(logged)


def _run_lightning_test_protocol(
    *,
    checkpoint_trainer: Trainer,
    checkpoint_model: LightningModule,
    dataloader: Any,
    phase: str,
) -> dict[str, Any]:
    """Run a standard Lightning test protocol.

    Parameters
    ----------
    checkpoint_trainer : Trainer
        Trainer used for checkpoint evaluation.
    checkpoint_model : LightningModule
        Model loaded with the best checkpoint weights.
    dataloader : Any
        Test dataloader for the protocol.
    phase : str
        Phase-tracking name.

    Returns
    -------
    dict[str, Any]
        Unprefixed test metrics.
    """
    with track_phase(phase):
        results = checkpoint_trainer.test(
            model=checkpoint_model,
            dataloaders=dataloader,
        )
    if not results:
        return {}
    return {_metric_suffix(k): v for k, v in results[0].items()}


def _average_ensemble_logits_by_global_nid(
    *,
    logit_chunks: list[torch.Tensor],
    label_chunks: list[torch.Tensor],
    nid_chunks: list[torch.Tensor],
    expected_runs: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Average repeated logits per global node id and validate coverage.

    Parameters
    ----------
    logit_chunks : list[torch.Tensor]
        Per-batch supervised logits from ensemble passes.
    label_chunks : list[torch.Tensor]
        Per-batch supervised labels from ensemble passes.
    nid_chunks : list[torch.Tensor]
        Per-batch global node identifiers from ensemble passes.
    expected_runs : int
        Required number of predictions per test node.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        Averaged logits, aligned labels, and sorted global node identifiers.
    """
    if expected_runs <= 0:
        raise ValueError(
            f"expected_runs must be positive, got {expected_runs}."
        )
    if not logit_chunks:
        raise ValueError("Ensemble inference produced no predictions.")

    logits = torch.cat(logit_chunks, dim=0)
    labels = torch.cat(label_chunks, dim=0)
    global_nids = torch.cat(nid_chunks, dim=0).to(torch.long)
    if global_nids.numel() == 0:
        raise ValueError("Ensemble inference found no supervised test nodes.")

    unique_nids, inverse = torch.unique(
        global_nids,
        sorted=True,
        return_inverse=True,
    )
    counts = torch.bincount(inverse, minlength=unique_nids.numel())
    bad_counts = counts != expected_runs
    if bad_counts.any():
        bad_nids = unique_nids[bad_counts][:10].tolist()
        raise ValueError(
            "Ensemble coverage mismatch: each test node must appear exactly "
            f"{expected_runs} times. Example mismatched global_nid values: "
            f"{bad_nids}."
        )

    logit_sums = torch.zeros(
        (unique_nids.numel(), *logits.shape[1:]),
        dtype=logits.dtype,
    )
    logit_sums.index_add_(0, inverse, logits)
    count_shape = (counts.shape[0],) + (1,) * (logits.dim() - 1)
    avg_logits = logit_sums / counts.view(count_shape).to(logits.dtype)

    avg_labels = []
    for idx, nid in enumerate(unique_nids):
        node_labels = labels[inverse == idx]
        first = node_labels[0]
        if not torch.equal(node_labels, first.expand_as(node_labels)):
            raise ValueError(
                "Inconsistent labels encountered for repeated ensemble "
                f"predictions at global_nid={int(nid)}."
            )
        avg_labels.append(first)

    return avg_logits, torch.stack(avg_labels, dim=0), unique_nids


def _dataset_loss_module(checkpoint_model: LightningModule) -> Any:
    """Return the dataset loss object used for ensemble loss.

    Parameters
    ----------
    checkpoint_model : LightningModule
        Model containing the configured loss module.

    Returns
    -------
    Any
        Dataset loss object with ``forward_criterion``.
    """
    loss_module = getattr(checkpoint_model, "loss", None)
    if hasattr(loss_module, "forward_criterion"):
        return loss_module

    for candidate in getattr(loss_module, "losses", []):
        if hasattr(candidate, "forward_criterion"):
            return candidate

    raise ValueError(
        "Ensemble loss computation requires a dataset loss with "
        "forward_criterion."
    )


def _compute_ensemble_metrics(
    *,
    checkpoint_model: LightningModule,
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> dict[str, Any]:
    """Compute evaluator metrics and dataset loss for averaged logits.

    Parameters
    ----------
    checkpoint_model : LightningModule
        Model providing evaluator and loss objects.
    logits : torch.Tensor
        Averaged logits.
    labels : torch.Tensor
        Labels aligned with averaged logits.

    Returns
    -------
    dict[str, Any]
        Evaluator metrics plus dataset loss.
    """
    checkpoint_model.evaluator.reset()
    checkpoint_model.evaluator.update({"logits": logits, "labels": labels})
    metrics = dict(checkpoint_model.evaluator.compute())
    checkpoint_model.evaluator.reset()

    dataset_loss = _dataset_loss_module(checkpoint_model)
    metrics["loss"] = dataset_loss.forward_criterion(
        logits,
        labels,
    ).detach()
    return metrics


def _run_ensemble_test_inference(
    *,
    checkpoint_model: LightningModule,
    cfg: DictConfig,
    datamodule: Any,
    device: torch.device,
) -> dict[str, Any]:
    """Run shuffled batched inference repeatedly and average logits.

    Parameters
    ----------
    checkpoint_model : LightningModule
        Model loaded with the best checkpoint weights.
    cfg : DictConfig
        Composed Hydra configuration.
    datamodule : Any
        Cluster datamodule providing inference dataloaders.
    device : torch.device
        Device used for model inference.

    Returns
    -------
    dict[str, Any]
        Metrics computed from averaged ensemble logits.
    """
    inference_cfg = cfg.get("test_inference", {}) or {}
    average = str(inference_cfg.get("ensemble_average", "logits")).lower()
    if average != "logits":
        raise ValueError(
            "Only test_inference.ensemble_average=logits is supported."
        )

    ensemble_runs = int(inference_cfg.get("ensemble_runs", 10))
    if ensemble_runs <= 0:
        raise ValueError(
            "test_inference.ensemble_runs must be positive, "
            f"got {ensemble_runs}."
        )
    ensemble_shuffle = bool(inference_cfg.get("ensemble_shuffle", True))
    ensemble_seed = int(
        inference_cfg.get("ensemble_seed", cfg.get("seed", 42))
    )

    logit_chunks: list[torch.Tensor] = []
    label_chunks: list[torch.Tensor] = []
    nid_chunks: list[torch.Tensor] = []
    was_training = checkpoint_model.training
    checkpoint_model.eval()

    try:
        with torch.inference_mode():
            for run_idx in range(ensemble_runs):
                loader = datamodule.inference_dataloader(
                    split="test",
                    shuffle=ensemble_shuffle,
                    seed=ensemble_seed + run_idx,
                    cover_parts="split",
                )
                for batch in loader:
                    batch = batch.to(device)
                    batch["model_state"] = "Test"
                    checkpoint_model.state_str = "Test"
                    model_out = checkpoint_model.forward(batch)

                    mask = batch.test_mask.to(torch.bool)
                    logits = model_out["logits"]
                    labels = model_out["labels"]
                    if logits.size(0) != mask.numel():
                        raise ValueError(
                            "Ensemble inference expects node-level logits "
                            "aligned with batch.test_mask."
                        )
                    if labels.size(0) != mask.numel():
                        raise ValueError(
                            "Ensemble inference expects node-level labels "
                            "aligned with batch.test_mask."
                        )
                    if not hasattr(batch, "global_nid"):
                        raise ValueError(
                            "Ensemble inference requires batch.global_nid."
                        )

                    logit_chunks.append(logits[mask].detach().cpu())
                    label_chunks.append(labels[mask].detach().cpu())
                    nid_chunks.append(batch.global_nid[mask].detach().cpu())
    finally:
        if was_training:
            checkpoint_model.train()

    avg_logits, avg_labels, _ = _average_ensemble_logits_by_global_nid(
        logit_chunks=logit_chunks,
        label_chunks=label_chunks,
        nid_chunks=nid_chunks,
        expected_runs=ensemble_runs,
    )
    return _compute_ensemble_metrics(
        checkpoint_model=checkpoint_model,
        logits=avg_logits,
        labels=avg_labels,
    )


def rerun_best_model_checkpoint(
    checkpoint_model: LightningModule,
    cfg: DictConfig,
    datamodule: LightningModule,
    device: torch.device,
    callbacks: list[Callback],
    logger: list[Logger],
) -> None:
    """Rerun the best model checkpoint on validation and test datasets.

    This function iterates through the callbacks to locate the `ModelCheckpoint`, loads the
    best model weights, and runs validation plus configured test inference
    protocols. Metrics are logged with `val_best_rerun/`, `test_inference/*/`,
    and the backward-compatible `test_best_rerun/` batched-test alias.

    Parameters
    ----------
    checkpoint_model : LightningModule
        The model instance to load weights into.
    cfg : DictConfig
        Configuration composed by Hydra.
    datamodule : LightningModule
        The data module providing `val_dataloader` and `test_dataloader`.
    device : torch.device
        The target device (CPU/GPU) for the model.
    callbacks : list[Callback]
        A list of callbacks to search for the `ModelCheckpoint`.
    logger : list[Logger]
        A list of loggers (e.g., WandbLogger) to record the re-run metrics.
    """
    model_path: Path | None = None
    for callback in callbacks:
        if isinstance(callback, ModelCheckpoint):
            log.info(
                f"Loading best model from checkpoint at {callback.best_model_path}"
            )
            with track_phase("checkpoint_load"):
                model_path = Path(callback.best_model_path)
                ckpt = torch.load(
                    model_path, map_location="cpu", weights_only=False
                )

                checkpoint_model.load_state_dict(
                    ckpt["state_dict"],
                    strict=True,
                )
                checkpoint_model.to(device)
            break  # there is only one checkpoint callback

    # New trainer to log final metrics on validation set
    # Because wandb displays validation metrics from the final, not the best epoch.
    checkpoint_trainer: Trainer = hydra.utils.instantiate(
        cfg.trainer,
        num_sanity_val_steps=0,
        enable_progress_bar=cfg.trainer.get("enable_progress_bar", True),
        logger=False,
    )

    log.info("Re-testing best model checkpoint on validation set!")
    val_loader = datamodule.val_dataloader()
    # TODO: Fix the issue with the on_validation_epoch_start hook as it is strictly attached to the training procedure.
    checkpoint_model.on_validation_epoch_start = lambda: None
    with track_phase("val_best_rerun"):
        results = checkpoint_trainer.validate(
            model=checkpoint_model,
            dataloaders=val_loader,
        )
    if results:
        logged = {}
        for k, v in results[0].items():
            suffix = k.split("/", 1)[1] if "/" in k else k
            logged[f"val_best_rerun/{suffix}"] = v
        log.info(logged)
        for lgr in logger:
            if isinstance(lgr, WandbLogger):
                lgr.log_metrics(logged)

    protocols = _resolve_test_inference_protocols(cfg)
    is_cluster_datamodule = _validate_test_inference_protocols(
        protocols,
        datamodule,
    )

    log.info(
        "Re-testing best model checkpoint on test set with protocols: "
        f"{protocols}"
    )
    for protocol in protocols:
        if protocol == "batched":
            metrics = _run_lightning_test_protocol(
                checkpoint_trainer=checkpoint_trainer,
                checkpoint_model=checkpoint_model,
                dataloader=datamodule.test_dataloader(),
                phase="test_inference_batched",
            )
        elif protocol == "full_graph":
            metrics = _run_lightning_test_protocol(
                checkpoint_trainer=checkpoint_trainer,
                checkpoint_model=checkpoint_model,
                dataloader=datamodule.inference_dataloader(
                    split="test",
                    q=datamodule.num_parts,
                    shuffle=False,
                    cover_parts="all",
                ),
                phase="test_inference_full_graph",
            )
        elif protocol == "ensemble":
            with track_phase("test_inference_ensemble"):
                metrics = _run_ensemble_test_inference(
                    checkpoint_model=checkpoint_model,
                    cfg=cfg,
                    datamodule=datamodule,
                    device=device,
                )
        else:  # pragma: no cover - validated before dispatch.
            raise ValueError(f"Unsupported test inference protocol {protocol}")

        _log_prefixed_metrics(
            prefix=f"test_inference/{protocol}",
            metrics=metrics,
            logger=logger,
        )
        if protocol == "batched":
            _log_prefixed_metrics(
                prefix="test_best_rerun",
                metrics=metrics,
                logger=logger,
            )

    if not is_cluster_datamodule:
        log.info("Completed default batched test inference for datamodule.")

    if (
        cfg.get("delete_checkpoint_after_test", False)
        and model_path
        and model_path.exists()
    ):
        log.info(f"Cleaning up: Deleting checkpoint at {model_path}")
        try:
            model_path.unlink()
        except Exception as e:
            log.warning(
                f"Failed to delete checkpoint at {model_path}. Error: {e}"
            )


def count_number_of_parameters(
    model: torch.nn.Module, only_trainable: bool = True
) -> int:
    """Count the number of trainable params.

    If all params, specify only_trainable = False.

    Ref:
        - https://discuss.pytorch.org/t/how-do-i-check-the-number-of-parameters-of-a-model/4325/9?u=brando_miranda
        - https://stackoverflow.com/questions/49201236/check-the-total-number-of-parameters-in-a-pytorch-model/62764464#62764464

    Parameters
    ----------
    model : torch.nn.Module
        The model.
    only_trainable : bool, optional
        If True, only count trainable parameters (default: True).

    Returns
    -------
    int
        The number of parameters.
    """
    if only_trainable:
        num_params: int = sum(
            p.numel() for p in model.parameters() if p.requires_grad
        )
    else:  # counts trainable and none-traibale
        num_params: int = sum(p.numel() for p in model.parameters() if p)
    assert num_params > 0, f"Err: {num_params=}"
    return int(num_params)


@hydra.main(
    version_base="1.3", config_path="../configs", config_name="run.yaml"
)
def main(cfg: DictConfig) -> float | None:
    """Main entry point for training.

    Parameters
    ----------
    cfg : DictConfig
        Configuration composed by Hydra.

    Returns
    -------
    float | None
        Optional[float] with optimized metric value.
    """
    # apply extra utilities
    # (e.g. ask for tags if none are provided in cfg, print cfg tree, etc.)
    extras(cfg)

    # train the model
    metric_dict, _ = run(cfg)

    # safely retrieve metric value for hydra-based hyperparameter optimization
    metric_value = get_metric_value(
        metric_dict=metric_dict, metric_name=cfg.get("optimized_metric")
    )

    # return optimized metric
    return metric_value


if __name__ == "__main__":
    main()
