"""Check split selection and experiment isolation without launching training."""

import os
import shlex
import subprocess
from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir


REPO = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("launcher", ["final_partitioning.sh", "optuna.sh"])
@pytest.mark.parametrize("split_type", ["", "fixed", "random"])
@pytest.mark.parametrize("dataset", ["ogbn_products", "reddit"])
@pytest.mark.parametrize("learning_setting", ["", "inductive", "transductive"])
def test_launcher_selects_split_and_separates_runs(
    tmp_path, monkeypatch, launcher, split_type, dataset, learning_setting
):
    """Official-split jobs cannot collide with old random-split job names."""
    monkeypatch.setenv("PROJECT_ROOT", str(REPO))
    env = {
        **os.environ,
        "DRY_RUN": "true",
        "RESUME": "true",
        "SELECTED_GPUS": "0",
        "JOBS_PER_GPU_OVERRIDE": "1",
        "MAX_CONCURRENT_RUNS": "1",
        "DATASET_FILTER": dataset,
        "MODEL_FILTER": "edgnn",
        "DATA_SEEDS_OVERRIDE": "0,4",
        "SPLIT_TYPE_OVERRIDE": split_type,
        "LEARNING_SETTING_OVERRIDE": learning_setting,
        "TRAIN_PROP_OVERRIDE": "",
        "CACHE_VAL": "false",
        "VAL_SHUFFLE": "true",
        "MAX_EPOCHS": "2",
        "TEST": "false",
        "LOGGER": "wandb",
        "WANDB_PROJECT_PREFIX": "split_test",
        "WANDB_PROJECT_NAME": "",
        "WANDB_PROJECT_SUFFIX": "",
        "STUDY_PREFIX": "split_test",
        "RUN_NAME_PREFIX": "split_test",
        "LOG_GROUP": "split_test",
        "OPTUNA_N_JOBS": "1",
        "N_TRIALS": "1",
    }
    result = subprocess.run(
        ["bash", str(REPO / "scripts/partitioning" / launcher)],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    commands = [
        shlex.split(line.removeprefix("[DRY_RUN] "))
        for line in result.stdout.splitlines()
        if line.startswith("[DRY_RUN] ")
    ]
    assert len(commands) == (2 if launcher == "final_partitioning.sh" else 1)
    for command in commands:
        overrides = [token for token in command[3:] if token != "--multirun"]
        with initialize_config_dir(
            config_dir=str(REPO / "configs"), version_base="1.3"
        ):
            cfg = compose(
                config_name="run.yaml",
                overrides=overrides,
                return_hydra_config=True,
            )
        expected_split = split_type or "fixed"
        expected_setting = learning_setting or (
            "inductive" if dataset == "reddit" else "transductive"
        )
        assert cfg.dataset.split_params.split_type == expected_split
        assert cfg.dataset.split_params.learning_setting == expected_setting
        assert cfg.model.model_name == "edgnn"
        assert cfg.trainer.max_epochs == 2
        assert cfg.test is False
        assert cfg.dataset.loader.parameters.stream.cache_val is False
        assert cfg.dataset.loader.parameters.stream.val_shuffle is True
        suffix = f"_split_{expected_split}_{expected_setting}"
        assert cfg.logger.wandb.project.endswith(suffix)
        # Optuna's trial suffix contains an unresolved Hydra job interpolation.
        name = next(
            token
            for token in overrides
            if token.startswith("+logger.wandb.name=")
        )
        assert suffix in name
        if launcher == "optuna.sh":
            assert cfg.hydra.sweeper.study_name.endswith(suffix)
    expected_log = (
        "optuna_sweep" if launcher == "optuna.sh" else "split_test"
    ) + (f"_split_{split_type}" if split_type else "")
    expected_log += f"_{learning_setting}" if learning_setting else ""
    assert (tmp_path / "logs" / expected_log).is_dir()


@pytest.mark.parametrize("launcher", ["final_partitioning.sh", "optuna.sh"])
def test_launcher_rejects_invalid_split_before_creating_logs(
    tmp_path, launcher
):
    """A misspelled split selection fails before jobs or log folders appear."""
    result = subprocess.run(
        ["bash", str(REPO / "scripts/partitioning" / launcher)],
        cwd=tmp_path,
        env={**os.environ, "SPLIT_TYPE_OVERRIDE": "official"},
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "unsupported SPLIT_TYPE_OVERRIDE" in result.stderr
    assert not list(tmp_path.iterdir())


def test_fixed_split_rejects_train_proportion_override(tmp_path):
    """The official split must not be confused with a random 8% split."""
    result = subprocess.run(
        ["bash", str(REPO / "scripts/partitioning/final_partitioning.sh")],
        cwd=tmp_path,
        env={
            **os.environ,
            "SPLIT_TYPE_OVERRIDE": "fixed",
            "TRAIN_PROP_OVERRIDE": "0.08",
        },
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert (
        "TRAIN_PROP_OVERRIDE cannot be used with fixed splits" in result.stderr
    )
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("launcher", ["final_partitioning.sh", "optuna.sh"])
def test_launcher_rejects_unknown_learning_setting(tmp_path, launcher):
    """Fail before creating logs when the pipeline selector is invalid."""
    result = subprocess.run(
        ["bash", str(REPO / "scripts/partitioning" / launcher)],
        cwd=tmp_path,
        env={**os.environ, "LEARNING_SETTING_OVERRIDE": "inductve"},
        capture_output=True, text=True,
    )
    assert result.returncode != 0
    assert "unsupported LEARNING_SETTING_OVERRIDE" in result.stderr
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("launcher,dataset", [
    ("optuna.sh", "reddit"),
    ("optuna.sh", "flickr"),
    ("optuna.sh", "yelp"),
    ("optuna_full_ds.sh", "flickr"),
    ("optuna_full_ds.sh", "yelp"),
])
def test_inductive_optuna_configurations(tmp_path, launcher, dataset):
    """All seven model commands use the official split and correct objective."""
    result = subprocess.run(
        ["bash", str(REPO / "scripts/partitioning" / launcher)],
        cwd=tmp_path,
        env={
            **os.environ, "PROJECT_ROOT": str(REPO),
            "DRY_RUN": "true", "RESUME": "false", "SELECTED_GPUS": "0",
            "JOBS_PER_GPU_OVERRIDE": "1", "DATASET_FILTER": dataset,
            "MODEL_FILTER": "", "SPLIT_TYPE_OVERRIDE": "",
            "LEARNING_SETTING_OVERRIDE": "", "LOGGER": "wandb",
            "CACHE_VAL": "false", "VAL_SHUFFLE": "true",
            "N_TRIALS": "50", "OPTUNA_N_JOBS": "5",
            "OPTUNA_N_STARTUP_TRIALS": "5",
        },
        capture_output=True, text=True, check=True,
    )
    commands = [shlex.split(line.removeprefix("[DRY_RUN] "))
                for line in result.stdout.splitlines()
                if line.startswith("[DRY_RUN] ")]
    assert len(commands) == 7
    for command in commands:
        overrides = [arg for arg in command[3:] if arg != "--multirun"]
        with initialize_config_dir(config_dir=str(REPO / "configs"), version_base="1.3"):
            cfg = compose(config_name="run.yaml", overrides=overrides,
                          return_hydra_config=True)
        assert cfg.dataset.split_params.learning_setting == "inductive"
        assert cfg.dataset.split_params.split_type == "fixed"
        assert cfg.hydra.sweeper.study_name.endswith("_split_fixed_inductive")
        assert cfg.logger.wandb.project.endswith("_split_fixed_inductive")
        assert cfg.logger.wandb.group == cfg.hydra.sweeper.study_name
        assert cfg.optimized_metric == "best_monitored_score"
        assert cfg.hydra.sweeper.n_trials == 50
        assert cfg.hydra.sweeper.n_jobs == 5
        assert cfg.hydra.launcher.n_jobs == 5
        assert cfg.hydra.sweeper.sampler.n_startup_trials == 5
        expected_metric = "f1_micro" if dataset == "yelp" else "accuracy"
        assert cfg.callbacks.model_checkpoint.monitor == f"val/{expected_metric}"
        assert cfg.dataset.parameters.monitor_metric == expected_metric
        assert cfg.test is False
        if dataset == "yelp":
            assert cfg.dataset.parameters.task == "multilabel classification"
            assert cfg.dataset.parameters.loss_type == "BCE"
            assert list(cfg.evaluator.metrics) == ["f1_micro"]
        if launcher == "optuna.sh":
            assert cfg.dataset.loader.parameters.memory_type == "on_disk_cluster"
            assert cfg.dataset.loader.parameters.stream.cache_val is False
            assert cfg.dataset.loader.parameters.stream.val_shuffle is True
            assert "dataset.loader.parameters.cluster.num_parts" in cfg.hydra.sweeper.params
        else:
            assert "cluster" not in cfg.dataset.loader.parameters
            assert not any("cluster." in key or "stream." in key
                           for key in cfg.hydra.sweeper.params)
