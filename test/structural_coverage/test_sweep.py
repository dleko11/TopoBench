"""Tests for the restartable structural-coverage sweep."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import torch
from omegaconf import DictConfig
from torch_geometric.data import Data

from scripts.structural_coverage.run import compose_config
from scripts.structural_coverage.sweep import (
    DEFAULT_PROFILES,
    DEFAULT_Q_VALUES,
    DEFAULT_SEEDS,
    REQUIRED_PLOTTING_FILES,
    SweepTask,
    build_tasks,
    command_for_task,
    export_task,
    parse_args,
    task_directory,
    task_is_complete,
    write_manifest,
)
from topobench.data.preprocessor.preprocessor import PreProcessor
from topobench.dataloader import DataloadDataset


def test_default_sweep_has_240_unique_tasks():
    tasks = build_tasks(DEFAULT_PROFILES, DEFAULT_Q_VALUES, DEFAULT_SEEDS)

    assert len(tasks) == 240
    assert len({task.key for task in tasks}) == 240
    assert tasks[0].key == "simplicial__q01__seed00"
    assert tasks[-1].key == "cell_simple_coverage__q32__seed09"


def test_configured_transductive_graph_replaces_source_masks():
    """The split-pipeline graph is authoritative for partition caches."""
    source = Data(
        x=torch.randn(4, 2),
        train_mask=torch.tensor([True, False, False, False]),
    )
    configured = source.clone()
    configured.train_mask = torch.tensor([True, True, True, False])

    preprocessor = object.__new__(PreProcessor)
    preprocessor.dataset = MagicMock(data=source)
    preprocessor.load_dataset_splits = MagicMock(
        return_value=(DataloadDataset([configured]), None, None)
    )
    split_params = DictConfig({"learning_setting": "transductive"})

    result = preprocessor.load_configured_transductive_graph(split_params)

    assert result is configured
    assert int(result.train_mask.sum()) == 3
    assert int(preprocessor.dataset.data.train_mask.sum()) == 1


def test_cell_simple_command_composes_expected_config(tmp_path):
    args = parse_args(
        [
            "run",
            "--trainer",
            "cpu",
            "--results-root",
            str(tmp_path / "results"),
            "--plotting-root",
            str(tmp_path / "plotting"),
        ]
    )
    task = SweepTask(
        index=0,
        profile="cell_simple_coverage",
        q=16,
        seed=7,
    )
    command = command_for_task(task=task, args=args, project_root=tmp_path)
    cfg = compose_config(command[3:])

    assert cfg.seed == 7
    assert cfg.trainer.max_epochs == 200
    assert cfg.trainer.min_epochs == 200
    assert cfg.callbacks.early_stopping is None
    assert cfg.dataset.loader.parameters.cluster.num_parts == 64
    assert cfg.dataset.loader.parameters.stream.q == 16
    assert cfg.dataset.loader.parameters.stream.q_val == 64
    assert cfg.dataset.split_params.train_prop == 0.5
    assert cfg.coverage.structure_family == "cell_simple_cycles"
    assert cfg.coverage.max_support_nodes == 8
    assert Path(cfg.coverage.run_dir).name == "seed07"


def test_export_contract_and_manifest(tmp_path):
    task = SweepTask(index=0, profile="simplicial", q=1, seed=0)
    results_root = tmp_path / "results"
    plotting_root = tmp_path / "plotting"
    run_dir = task_directory(results_root, task)
    run_dir.mkdir(parents=True)
    for name in REQUIRED_PLOTTING_FILES:
        (run_dir / name).write_text(
            "{}\n" if name.endswith(".json") else "x\n"
        )
    status = {"task": task.key, "status": "success", "exit_code": 0}
    from scripts.structural_coverage.sweep import atomic_write_json

    atomic_write_json(run_dir / "sweep_status.json", status)
    destination = export_task(
        task=task,
        results_root=results_root,
        plotting_root=plotting_root,
        status=status,
    )

    assert task_is_complete(task, results_root, plotting_root)
    assert all(
        (destination / name).is_file() for name in REQUIRED_PLOTTING_FILES
    )
    manifest, counts = write_manifest([task], results_root, plotting_root)
    assert manifest.is_file()
    assert counts == {"success": 1}
