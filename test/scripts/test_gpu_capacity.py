"""Regression checks for GPU capacity experiment planning and results."""

import json
import os
import shlex
import subprocess
import sys
from threading import Barrier, Lock
from types import SimpleNamespace

import matplotlib.pyplot as plt
import pandas as pd
import pytest
import torch
from hydra import compose, initialize_config_dir

from scripts.partitioning import gpu_capacity
from scripts.partitioning.gpu_capacity import (
    LAUNCHER,
    REPO,
    build_grid,
    complete_result,
    run_environment,
)
from scripts.partitioning.plot_gpu_capacity import load_results, plot_results
from topobench.callbacks.gpu_memory import GPUMemoryBenchmarkCallback


def test_grid_has_36_unique_configurations():
    jobs = build_grid(
        [1, 2, 4, 8, 16], [32, 64, 128, 256, 512], [0], ["cwn", "sccnn"]
    )
    assert len(jobs) == len({job["id"] for job in jobs}) == 36
    assert len([j for j in jobs if j["depth"] == 4 and j["width"] == 128]) == 4


def test_hardware_check_uses_cuda_capacity_for_resume(monkeypatch):
    outputs = iter(
        [
            "0, GPU-0, NVIDIA A100-SXM4-80GB, 81920\n",
            "",
            "79.249755859375\n",
        ]
    )
    calls = []

    def check_output(command, **kwargs):
        calls.append((command, kwargs))
        return next(outputs)

    monkeypatch.setattr(gpu_capacity.subprocess, "check_output", check_output)
    devices = gpu_capacity.validate_gpus(["0"])
    assert devices["0"]["gpu_total_gib"] == 79.249755859375
    assert calls[-1][1]["env"]["CUDA_VISIBLE_DEVICES"] == "0"


@pytest.mark.parametrize("model", ["cwn", "sccnn"])
@pytest.mark.parametrize("mode", ["full", "partitioned"])
def test_launcher_composes_training_only_config(tmp_path, model, mode):
    job = {
        "id": "test",
        "model": model,
        "mode": mode,
        "seed": 0,
        "width": 256,
        "depth": 8,
    }
    env = {
        **os.environ,
        **run_environment(job, "0", tmp_path, "test_capacity", "test"),
    }
    env["DRY_RUN"] = "true"
    env["WANDB_PROJECT_NAME"] = "test_capacity_combined"
    result = subprocess.run(
        ["bash", str(LAUNCHER)],
        cwd=tmp_path,
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )
    command = next(
        line.removeprefix("[DRY_RUN] ")
        for line in result.stdout.splitlines()
        if line.startswith("[DRY_RUN]")
    )
    tokens = shlex.split(command)
    assert "CUDA_VISIBLE_DEVICES=0" in tokens
    overrides = tokens[tokens.index("topobench") + 1 :]
    with initialize_config_dir(
        config_dir=str(REPO / "configs"), version_base="1.3"
    ):
        cfg = compose(config_name="run.yaml", overrides=overrides)
    assert cfg.model.backbone.n_layers == 8
    assert cfg.logger.wandb.project == "test_capacity_combined"
    assert cfg.model.feature_encoder.out_channels == 256
    if model == "cwn":
        assert cfg.model.backbone.hid_channels == 256
    else:
        assert list(cfg.model.backbone.hidden_channels_all) == [256] * 3
    assert cfg.trainer.max_epochs == cfg.trainer.min_epochs == 3
    assert cfg.trainer.limit_val_batches == 0
    assert cfg.trainer.precision == "32-true"
    assert cfg.train and not cfg.test
    assert "early_stopping" not in cfg.callbacks
    assert "phase_tracking" not in cfg.callbacks
    assert cfg.callbacks.model_checkpoint.save_top_k == 0
    assert cfg.callbacks.model_checkpoint.monitor is None
    assert cfg.callbacks.gpu_memory.result_path == str(
        tmp_path / "measurements/test.json"
    )
    if mode == "partitioned":
        assert cfg.dataset.loader.parameters.cluster.num_parts == 64
        assert cfg.dataset.loader.parameters.stream.q == 8


@pytest.mark.parametrize(
    "measurement,returncode,status",
    [
        ({}, 0, "error"),
        ({"status": "running"}, 0, "error"),
        ({"status": "cuda_oom"}, 1, "cuda_oom"),
        ({"status": "success", "completed_epochs": 2}, 0, "error"),
        (
            {
                "status": "success",
                "completed_epochs": 3,
                "optimizer_steps": 3,
                "peak_allocated_gib": 1,
                "peak_reserved_gib": 2,
            },
            0,
            "success",
        ),
        (
            {
                "status": "success",
                "completed_epochs": 3,
                "optimizer_steps": 3,
                "peak_allocated_gib": 1,
                "peak_reserved_gib": 2,
            },
            1,
            "error",
        ),
    ],
)
def test_result_classification(measurement, returncode, status):
    assert complete_result({}, measurement, returncode)["status"] == status


@pytest.mark.parametrize(
    "exception,status",
    [
        (
            torch.cuda.OutOfMemoryError(
                "CUDA out of memory. Tried to allocate 1 GiB"
            ),
            "cuda_oom",
        ),
        (
            RuntimeError(
                "CUDA error: an illegal memory access was encountered"
            ),
            "error",
        ),
        (RuntimeError("DefaultCPUAllocator: not enough memory"), "error"),
        (torch.cuda.OutOfMemoryError("CPU allocation failed"), "error"),
    ],
)
def test_callback_preserves_failure_type(tmp_path, exception, status):
    path = tmp_path / "result.json"
    callback = GPUMemoryBenchmarkCallback(str(path))
    trainer = SimpleNamespace(global_step=1, loggers=[])
    callback.on_exception(trainer, None, exception)
    result = json.loads(path.read_text())
    assert result["status"] == status
    assert result["error_type"] == type(exception).__name__


def test_callback_counts_complete_epochs_and_preserves_peak(
    tmp_path, monkeypatch
):
    path = tmp_path / "result.json"
    callback = GPUMemoryBenchmarkCallback(str(path))
    ended = []
    callback.tracker = SimpleNamespace(
        cuda_phase_peaks=lambda phase: {
            "tracking/resource/cuda_peak_allocated_mb": 1024,
            "tracking/resource/cuda_peak_reserved_mb": 2048,
        },
        end_phase=lambda phase: ended.append(phase),
    )
    trainer = SimpleNamespace(
        global_step=3,
        max_epochs=3,
        loggers=[],
        strategy=SimpleNamespace(root_device="cuda:0"),
    )
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device: None)
    for _ in range(3):
        callback.on_train_epoch_end(trainer, None)
    callback.on_fit_end(trainer, None)
    result = json.loads(path.read_text())
    assert result["status"] == "success"
    assert result["completed_epochs"] == 3
    assert result["peak_allocated_gib"] == 1
    assert result["peak_reserved_gib"] == 2
    assert ended == [callback.phase]


def test_plot_keeps_oom_out_of_measured_curve(tmp_path):
    jobs = build_grid([1, 4, 8], [32, 128, 256], [0], ["cwn", "sccnn"])
    rows = [
        {
            **job,
            "gpu_name": "test GPU",
            "gpu_total_gib": 80,
            "status": "cuda_oom"
            if job["depth"] == 8 and job["mode"] == "full"
            else "success",
            "completed_epochs": 3,
            "peak_reserved_gib": 20,
            "peak_allocated_gib": 15,
        }
        for job in jobs
    ]
    fig = plot_results(pd.DataFrame(rows), tmp_path)
    full_curve = fig.axes[0].lines[1]  # First line is the OOM marker.
    assert pd.isna(full_curve.get_ydata()[-1])
    assert all(
        (tmp_path / f"gpu_capacity_reserved.{suffix}").stat().st_size > 0
        for suffix in ("pdf", "png", "svg")
    )
    assert "CUDA OOM" in (tmp_path / "gpu_capacity_reserved.svg").read_text()
    plt.close(fig)


def test_analysis_refuses_missing_runs(tmp_path):
    (tmp_path / "plan.json").write_text(
        json.dumps({"jobs": [{"id": "missing"}]})
    )
    with pytest.raises(ValueError, match="unfinished"):
        load_results(tmp_path)


def test_launcher_propagates_training_failure(tmp_path):
    job = {
        "id": "failure",
        "model": "cwn",
        "mode": "full",
        "seed": 0,
        "width": 32,
        "depth": 1,
    }
    env = {
        **os.environ,
        **run_environment(job, "0", tmp_path, "test_capacity", "failure"),
    }
    binary_dir = tmp_path / "bin"
    binary_dir.mkdir()
    python = binary_dir / "python"
    python.write_text("#!/bin/sh\nexit 17\n")
    python.chmod(0o755)
    env["PATH"] = str(binary_dir) + os.pathsep + env["PATH"]
    result = subprocess.run(
        ["bash", str(LAUNCHER)],
        cwd=tmp_path,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode != 0
    assert "exit code: 17" in result.stdout


def test_scheduler_uses_both_gpus_and_resumes_terminal_results(
    tmp_path, monkeypatch
):
    argv = [
        "gpu_capacity",
        "--gpus",
        "0,1",
        "--depths",
        "4",
        "--widths",
        "128",
        "--output-dir",
        str(tmp_path),
    ]
    monkeypatch.setattr(sys, "argv", argv)
    hardware = {
        gpu: {"gpu_name": "test GPU", "gpu_total_gib": 80, "gpu_uuid": gpu}
        for gpu in ("0", "1")
    }
    monkeypatch.setattr(gpu_capacity, "validate_gpus", lambda gpus: hardware)
    barrier = Barrier(2)
    lock = Lock()
    active = set()
    calls = []

    def fake_run(command, *, env, **kwargs):
        gpu = env["SELECTED_GPUS"]
        with lock:
            assert gpu not in active
            active.add(gpu)
            calls.append(env["WANDB_RUN_ID"])
        barrier.wait(timeout=5)
        result = {
            "status": "success",
            "completed_epochs": 3,
            "optimizer_steps": 3,
            "peak_allocated_gib": 1,
            "peak_reserved_gib": 2,
        }
        if (
            env["MODEL_FILTER"] == "sccnn"
            and env["FULL_GRAPH_BASELINE"] == "true"
        ):
            result["status"] = "cuda_oom"
        gpu_capacity.write_json(
            gpu_capacity.Path(env["GPU_MEMORY_RESULT_PATH"]), result
        )
        with lock:
            active.remove(gpu)
        return SimpleNamespace(returncode=int(result["status"] == "cuda_oom"))

    monkeypatch.setattr(gpu_capacity.subprocess, "run", fake_run)
    gpu_capacity.main()
    assert len(calls) == len(set(calls)) == 4
    frame = load_results(tmp_path)
    assert frame.status.value_counts().to_dict() == {
        "success": 3,
        "cuda_oom": 1,
    }
    gpu_capacity.main()
    assert len(calls) == 4
