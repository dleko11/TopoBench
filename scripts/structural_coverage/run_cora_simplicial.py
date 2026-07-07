"""Run the Cora simplicial structural coverage experiment."""

from __future__ import annotations

import random
import sys
import time
from pathlib import Path
from typing import Any

import hydra
import lightning as L
import numpy as np
import rootutils
import torch
from omegaconf import DictConfig, OmegaConf, open_dict

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from scripts.structural_coverage.coverage import (  # noqa: E402
    StructuralCoverageCallback,
    copy_csv_logger_metrics,
    write_csv_rows,
)
from topobench.data.preprocessor import PreProcessor  # noqa: E402
from topobench.data.utils import (  # noqa: E402
    build_cluster_transform,
    make_hash,
)
from topobench.dataloader import ClusterGCNDataModule  # noqa: E402
from topobench.utils import (  # noqa: E402
    RankedLogger,
    extras,
    instantiate_callbacks,
    instantiate_loggers,
    log_hyperparameters,
)
from topobench.utils.config_resolvers import (  # noqa: E402
    register_all_resolvers,
)

register_all_resolvers()

log = RankedLogger(__name__, rank_zero_only=True)

DEFAULT_OVERRIDES = [
    "dataset=graph/cocitation_cora_for_partitioning",
    "model=simplicial/scn",
    "transforms=liftings/graph2simplicial_default",
    "trainer=cpu",
    "logger=csv",
    "test=false",
    "trainer.max_epochs=50",
    "trainer.min_epochs=1",
    "trainer.check_val_every_n_epoch=1",
    "extras.print_config=false",
    "extras.enforce_tags=false",
    "+coverage.results_root=scripts/structural_coverage/results",
    "+coverage.save_batch_events=false",
    "+coverage.audit_induced_edges=true",
    "+coverage.audit_max_batches=10",
    "+coverage.require_equal_batches=false",
    "+coverage.structure_family=auto",
]


def _override_value(argv: list[str], key: str) -> str | None:
    """Return the last Hydra override value for a plain ``key=value`` item."""
    prefix = f"{key}="
    value = None
    for item in argv:
        if item.startswith(prefix):
            value = item[len(prefix) :]
    return value


def _has_override(argv: list[str], key: str) -> bool:
    """Return whether argv contains a Hydra override for a key."""
    prefixes = (f"{key}=", f"+{key}=", f"++{key}=")
    return any(item.startswith(prefixes) for item in argv)


def _profile_overrides(argv: list[str]) -> list[str]:
    """Return default overrides that are valid for the selected profile."""
    transforms = _override_value(argv, "transforms")
    if transforms is None:
        transforms = "liftings/graph2simplicial_default"

    if (
        "graph2simplicial" in transforms
        and not _has_override(
            argv,
            "transforms.graph2simplicial_lifting.complex_dim",
        )
    ):
        return ["transforms.graph2simplicial_lifting.complex_dim=2"]
    return []


def compose_config(argv: list[str] | None = None) -> DictConfig:
    """Compose the experiment config with Cora defaults and CLI overrides."""
    argv = list(sys.argv[1:] if argv is None else argv)
    config_dir = Path(__file__).resolve().parents[2] / "configs"
    with hydra.initialize_config_dir(
        version_base="1.3",
        config_dir=str(config_dir),
        job_name="structural_coverage_cora_simplicial",
    ):
        return hydra.compose(
            config_name="run.yaml",
            overrides=DEFAULT_OVERRIDES + _profile_overrides(argv) + argv,
        )


def _seed_everything(cfg: DictConfig) -> None:
    L.seed_everything(cfg.seed, workers=True)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)

    if cfg.get("deterministic", False):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)


def _sanitize_run_label(value: Any) -> str:
    """Return a filesystem-friendly run label fragment."""
    text = str(value or "unknown")
    return "".join(ch if ch.isalnum() else "_" for ch in text).strip("_")


def infer_structure_family(cfg: DictConfig) -> str:
    """Infer coverage structure family from model/transform config."""
    configured = cfg.get("coverage", {}).get("structure_family", "auto")
    if configured != "auto":
        return str(configured)

    model_domain = cfg.get("model", {}).get("model_domain", "")
    if OmegaConf.select(cfg, "transforms.graph2cell_lifting") is not None:
        return "cell_cycle"
    if OmegaConf.select(cfg, "transforms.graph2hypergraph_lifting") is not None:
        return "hypergraph_khop"
    if OmegaConf.select(cfg, "transforms.graph2simplicial_lifting") is not None:
        return "simplicial_clique"
    if model_domain == "cell":
        return "cell_cycle"
    if model_domain == "hypergraph":
        return "hypergraph_khop"
    return "simplicial_clique"


def structure_params_for_family(
    cfg: DictConfig,
    structure_family: str,
) -> dict[str, Any]:
    """Extract family parameters from the composed transform config."""
    if structure_family == "cell_cycle":
        return {
            "max_cell_length": OmegaConf.select(
                cfg,
                "transforms.graph2cell_lifting.max_cell_length",
                default=None,
            )
        }
    if structure_family == "hypergraph_khop":
        return {
            "k_value": OmegaConf.select(
                cfg,
                "transforms.graph2hypergraph_lifting.k_value",
                default=1,
            )
        }
    return {}


def _result_dir(cfg: DictConfig) -> Path:
    q = int(cfg.dataset.loader.parameters.stream.get("q", 1))
    seed = int(cfg.get("seed", 42))
    family = infer_structure_family(cfg)
    model_name = cfg.get("model", {}).get("model_name", "model")
    label = _sanitize_run_label(f"{family}_{model_name}")
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_id = f"cora_{label}_q{q}_seed{seed}_{timestamp}"
    return Path(cfg.coverage.results_root).resolve() / run_id


def _resolve_q_val(
    *,
    stream_cfg: DictConfig,
    q: int,
    num_parts: int,
) -> int:
    q_val = stream_cfg.get("q_val", None)
    if q_val is not None:
        return int(q_val)

    val_batches = stream_cfg.get("val_batches", 5)
    if val_batches is None:
        return q
    val_batches = int(val_batches)
    return max(q, (num_parts + val_batches - 1) // val_batches)


def build_experiment_objects(
    cfg: DictConfig,
    results_dir: Path,
) -> dict[str, Any]:
    """Instantiate data, model, callbacks, loggers and trainer."""
    logger = instantiate_loggers(cfg.get("logger"))

    log.info(f"Instantiating loader <{cfg.dataset.loader._target_}>")
    dataset_loader = hydra.utils.instantiate(cfg.dataset.loader)
    dataset, dataset_dir = dataset_loader.load()

    raw_transform_config = cfg.get("transforms", None)
    memory_type = cfg.dataset.loader.parameters.get("memory_type", "in_memory")
    if memory_type != "on_disk_cluster":
        raise ValueError(
            "Structural coverage experiment requires "
            "dataset.loader.parameters.memory_type=on_disk_cluster."
        )

    build_cluster_transform(raw_transform_config)
    preprocessor = PreProcessor(dataset, dataset_dir, None)
    handle = preprocessor.pack_global_partition(
        split_params=cfg.dataset.get("split_params", {}),
        cluster_params=cfg.dataset.loader.parameters.get("cluster", {}),
        stream_params=cfg.dataset.loader.parameters.get("stream", {}),
        dtype_policy=cfg.dataset.loader.parameters.get(
            "dtype_policy", "preserve"
        ),
        pack_db=True,
        pack_memmaps=True,
    )

    transform_cfg_container = (
        OmegaConf.to_container(raw_transform_config, resolve=True)
        if raw_transform_config is not None
        else None
    )
    stream_cfg = cfg.dataset.loader.parameters.get("stream", {})
    q = int(stream_cfg.get("q", 1))
    q_val = _resolve_q_val(
        stream_cfg=stream_cfg,
        q=q,
        num_parts=int(handle["num_parts"]),
    )
    val_cache_fingerprint = make_hash(
        {
            "partition_hash": handle.get("config_hash", None),
            "transform": transform_cfg_container,
            "q_val": q_val,
            "with_edge_attr": stream_cfg.get("with_edge_attr", False),
            "seed": cfg.get("seed", 42),
            "eval_cover_strategy": cfg.get("eval", {}).get(
                "cover_strategy", "all_parts"
            ),
        }
    )

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
        eval_cover_strategy=cfg.get("eval", {}).get(
            "cover_strategy", "all_parts"
        ),
        seed=cfg.get("seed", 42),
        transform_config=transform_cfg_container,
        cache_val=True,
        val_cache_fingerprint=val_cache_fingerprint,
    )

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model = hydra.utils.instantiate(
        cfg.model,
        evaluator=cfg.evaluator,
        optimizer=cfg.optimizer,
        loss=cfg.loss,
    )

    callbacks = instantiate_callbacks(cfg.get("callbacks"))
    max_epochs = int(cfg.trainer.get("max_epochs", 1))
    coverage_cfg = cfg.get("coverage", {})
    structure_family = infer_structure_family(cfg)
    coverage_callback = StructuralCoverageCallback(
        handle=handle,
        q=q,
        results_dir=results_dir,
        max_epochs=int(coverage_cfg.get("max_theory_epochs", max_epochs)),
        structure_family=structure_family,
        structure_params=structure_params_for_family(cfg, structure_family),
        cfg_snapshot=OmegaConf.to_container(cfg, resolve=False),
        save_batch_events=coverage_cfg.get("save_batch_events", False),
        audit_induced_edges=coverage_cfg.get("audit_induced_edges", True),
        audit_max_batches=coverage_cfg.get("audit_max_batches", 10),
        require_equal_batches=coverage_cfg.get("require_equal_batches", True),
    )
    callbacks.append(coverage_callback)

    trainer = hydra.utils.instantiate(
        cfg.trainer,
        callbacks=callbacks,
        logger=logger,
        num_sanity_val_steps=0,
        log_every_n_steps=1,
    )

    return {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "callbacks": callbacks,
        "logger": logger,
        "trainer": trainer,
        "coverage_callback": coverage_callback,
    }


def run(cfg: DictConfig) -> Path:
    """Run training-integrated structural coverage tracking."""
    _seed_everything(cfg)
    results_dir = _result_dir(cfg)
    results_dir.mkdir(parents=True, exist_ok=True)
    lightning_dir = results_dir / "lightning"
    lightning_dir.mkdir(parents=True, exist_ok=True)

    with open_dict(cfg):
        cfg.paths.output_dir = str(lightning_dir)
        cfg.paths.work_dir = str(Path.cwd())

    extras(cfg)
    objects = build_experiment_objects(cfg, results_dir)

    if objects["logger"]:
        log_hyperparameters(objects)

    trainer = objects["trainer"]
    model = objects["model"]
    datamodule = objects["datamodule"]

    log.info(f"Writing structural coverage artifacts to {results_dir}")
    if cfg.get("train", True):
        trainer.fit(
            model=model,
            datamodule=datamodule,
            ckpt_path=cfg.get("ckpt_path"),
        )

    if cfg.get("test", False):
        ckpt_path = None
        checkpoint_callback = getattr(trainer, "checkpoint_callback", None)
        if checkpoint_callback is not None:
            ckpt_path = checkpoint_callback.best_model_path or None
        trainer.test(model=model, datamodule=datamodule, ckpt_path=ckpt_path)

    metrics_copy = copy_csv_logger_metrics(
        loggers=objects["logger"],
        destination=results_dir / "metrics.csv",
    )
    if metrics_copy is not None:
        log.info(f"Copied CSV metrics to {metrics_copy}")
    else:
        write_csv_rows(
            results_dir / "metrics.csv",
            [
                {
                    "status": "unavailable",
                    "message": (
                        "Lightning CSV metrics file was not produced; this "
                        "can happen in fast_dev_run."
                    ),
                }
            ],
            ["status", "message"],
        )

    return results_dir


def main(argv: list[str] | None = None) -> None:
    """CLI entrypoint."""
    cfg = compose_config(argv)
    results_dir = run(cfg)
    print(f"Structural coverage results: {results_dir}")


if __name__ == "__main__":
    main()
