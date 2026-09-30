"""Check split selection and experiment isolation without launching training."""

import os
import shlex
import subprocess
from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir


REPO = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("launcher", ["final_partitioning.sh", "optuna.sh"])
@pytest.mark.parametrize("split_type", ["", "fixed"])
def test_launcher_selects_split_and_separates_runs(
    tmp_path, monkeypatch, launcher, split_type
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
        "DATASET_FILTER": "ogbn_products",
        "MODEL_FILTER": "edgnn",
        "DATA_SEEDS_OVERRIDE": "0,4",
        "SPLIT_TYPE_OVERRIDE": split_type,
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
        assert cfg.dataset.split_params.split_type == (split_type or "random")
        assert cfg.model.model_name == "edgnn"
        assert cfg.trainer.max_epochs == 2
        assert cfg.test is False
        assert cfg.dataset.loader.parameters.stream.cache_val is False
        assert cfg.dataset.loader.parameters.stream.val_shuffle is True
        suffix = "_split_fixed"
        assert cfg.logger.wandb.project.endswith(suffix) == bool(split_type)
        # Optuna's trial suffix contains an unresolved Hydra job interpolation.
        name = next(
            token
            for token in overrides
            if token.startswith("+logger.wandb.name=")
        )
        assert (suffix in name) == bool(split_type)
        if launcher == "optuna.sh":
            assert cfg.hydra.sweeper.study_name.endswith(suffix) == bool(
                split_type
            )
    expected_log = (
        "optuna_sweep" if launcher == "optuna.sh" else "split_test"
    ) + ("_split_fixed" if split_type else "")
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
