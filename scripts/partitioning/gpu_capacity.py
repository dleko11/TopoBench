#!/usr/bin/env python3
"""Run the Cora Full CWN/SCCNN GPU memory sweep, one process per GPU.

Example: uv run --no-sync python -m scripts.partitioning.gpu_capacity \
    --gpus 0,1 --output-dir outputs/gpu_capacity_cora
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shlex
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from queue import Empty, Queue
from threading import Event
from uuid import uuid4

REPO = Path(__file__).resolve().parents[2]
LAUNCHER = REPO / "scripts/partitioning/final_partitioning.sh"


def build_grid(depths, widths, seeds, models):
    """Deduplicate the common (depth=4, width=128) configuration."""
    capacities = sorted({(d, 128) for d in depths} | {(4, w) for w in widths})
    return [
        {
            "id": f"{model}_{mode}_l{depth}_h{width}_seed{seed}",
            "model": model,
            "mode": mode,
            "depth": depth,
            "width": width,
            "seed": seed,
        }
        for depth, width in capacities
        for mode in ("full", "partitioned")
        for model in models
        for seed in seeds
    ]


def run_environment(job, gpu, output_dir, project_prefix, attempt):
    """Use the existing launcher for models, lifting, splits, and logging."""
    return {
        "SELECTED_GPUS": str(gpu),
        "JOBS_PER_GPU_OVERRIDE": "1",
        "MAX_CONCURRENT_RUNS": "1",
        "DATASET_FILTER": "cora_full",
        "MODEL_FILTER": job["model"],
        "DATA_SEEDS_OVERRIDE": str(job["seed"]),
        "HIDDEN_CHANNELS_OVERRIDE": str(job["width"]),
        "N_LAYERS_OVERRIDE": str(job["depth"]),
        "FULL_GRAPH_BASELINE": str(job["mode"] == "full").lower(),
        "PARTITION_GRID_OVERRIDE": "64:8"
        if job["mode"] == "partitioned"
        else "",
        "PARTITION_CACHE_NAMESPACE": "",
        "FORCE_RELOAD_PREPROCESSING": "false",
        "GPU_MEMORY_BENCHMARK": "true",
        "GPU_MEMORY_RESULT_PATH": str(
            output_dir / "measurements" / f"{job['id']}.json"
        ),
        "MAX_EPOCHS": "3",
        "MIN_EPOCHS": "3",
        "TRAIN": "true",
        "TEST": "false",
        "TRAINER": "gpu",
        "LOGGER": "wandb",
        "STREAM_NUM_WORKERS": "1",
        "CACHE_NUM_WORKERS": "1",
        "CACHE_VAL": "false",
        "TEST_INFERENCE_PROTOCOLS": "[batched]",
        "MAX_ATTEMPTS": "1",
        "RESUME": "true",
        "DRY_RUN": "false",
        "KEEP_SUCCESS_LOGS": "true",
        "WANDB_PROJECT_PREFIX": project_prefix,
        "WANDB_PROJECT_SUFFIX": "",
        "WANDB_RUN_GROUP": output_dir.name,
        "WANDB_RUN_ID": attempt,
        "WANDB_RESUME": "never",
        "WANDB_MODE": "online",
        "RUN_NAME_PREFIX": f"capacity_{job['mode']}_{attempt}",
        "LOG_GROUP": f"gpu_capacity/{output_dir.name}/{job['id']}/{attempt}",
        "MPLBACKEND": "Agg",
        "PYTHONUNBUFFERED": "1",
        "PATH": f"{Path(sys.executable).parent}{os.pathsep}{os.environ['PATH']}",
    }


def complete_result(job, measurement, returncode):
    """A missing measurement or unrelated failure must never become OOM."""
    result = {**job, **measurement, "launcher_returncode": returncode}
    if result.get("status") == "cuda_oom":
        return result
    if (
        returncode == 0
        and result.get("status") == "success"
        and result.get("completed_epochs") == 3
        and result.get("optimizer_steps", 0) > 0
        and result.get("peak_allocated_gib", 0) > 0
        and result.get("peak_reserved_gib", 0) > 0
    ):
        return result
    result["status"] = "error"
    result.setdefault(
        "error",
        "Run did not produce complete GPU measurements; inspect its logs.",
    )
    return result


def write_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def validate_gpus(gpus):
    """Require the selected devices to exist and have no compute processes."""
    output = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,name,memory.total",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    devices = {
        row[0].strip(): {
            "gpu_uuid": row[1].strip(),
            "gpu_name": row[2].strip(),
            "gpu_total_gib": float(row[3]) / 1024,
        }
        for row in csv.reader(output.splitlines())
    }
    missing = set(gpus) - devices.keys()
    if missing:
        raise ValueError(f"Selected GPUs do not exist: {sorted(missing)}")
    hardware = {
        (devices[gpu]["gpu_name"], devices[gpu]["gpu_total_gib"])
        for gpu in gpus
    }
    if len(hardware) != 1:
        raise ValueError("Use GPUs with the same model and memory capacity.")
    active = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid",
            "--format=csv,noheader",
        ],
        text=True,
    ).splitlines()
    if any(devices[gpu]["gpu_uuid"] in active for gpu in gpus):
        raise ValueError(
            "Selected GPUs already have compute processes. Use idle GPUs for memory measurements."
        )
    # CUDA-usable memory can differ from NVML's physical capacity (e.g. ECC).
    # Match the callback's definition when checking hardware on resume.
    for gpu in gpus:
        total = subprocess.check_output(
            [
                sys.executable,
                "-c",
                "import torch; print(torch.cuda.mem_get_info(0)[1] / 1024**3)",
            ],
            env={
                **os.environ,
                "CUDA_VISIBLE_DEVICES": gpu,
                "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
            },
            text=True,
        )
        devices[gpu]["gpu_total_gib"] = float(total.strip())
    return devices


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--gpus", default="0", help="Comma-separated GPU indices"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/gpu_capacity_cora")
    )
    parser.add_argument("--project-prefix", default="gpu_capacity")
    parser.add_argument(
        "--depths", nargs="+", type=int, default=[1, 2, 4, 8, 16]
    )
    parser.add_argument(
        "--widths", nargs="+", type=int, default=[32, 64, 128, 256, 512]
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument(
        "--models",
        nargs="+",
        choices=["cwn", "sccnn"],
        default=["cwn", "sccnn"],
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the plan without creating files or starting runs",
    )
    args = parser.parse_args()
    gpus = args.gpus.split(",")
    if len(set(gpus)) != len(gpus) or not all(g.isdigit() for g in gpus):
        parser.error("--gpus requires distinct nonnegative indices")
    if any(v <= 0 for v in args.depths + args.widths) or any(
        s < 0 for s in args.seeds
    ):
        parser.error(
            "Depths and widths must be positive; seeds must be nonnegative"
        )
    jobs = build_grid(
        args.depths,
        args.widths,
        sorted(set(args.seeds)),
        list(dict.fromkeys(args.models)),
    )
    output_dir = args.output_dir.resolve()
    print(
        f"{len(jobs)} configurations, 3 training epochs each, one process per GPU ({','.join(gpus)}).",
        flush=True,
    )
    if args.dry_run:
        for index, job in enumerate(jobs):
            env = run_environment(
                job,
                gpus[index % len(gpus)],
                output_dir,
                args.project_prefix,
                "dryrun",
            )
            print(
                shlex.join(
                    [
                        "env",
                        *(f"{k}={v}" for k, v in env.items() if k != "PATH"),
                        "bash",
                        str(LAUNCHER),
                    ]
                )
            )
        return

    plan = {
        "dataset": "cora_full",
        "epochs": 3,
        "K": 64,
        "q": 8,
        "precision": "32-true",
        "project_prefix": args.project_prefix,
        "project_name": os.environ.get("WANDB_PROJECT_NAME"),
        "jobs": jobs,
    }
    plan_path = output_dir / "plan.json"
    if plan_path.exists() and json.loads(plan_path.read_text()) != plan:
        parser.error(
            "Output directory contains a different sweep. Choose a new output directory."
        )
    devices = validate_gpus(gpus)
    output_dir.mkdir(parents=True, exist_ok=True)
    for directory in ("runs", "measurements", "logs"):
        (output_dir / directory).mkdir(exist_ok=True)
    write_json(plan_path, plan)
    pending = Queue()
    stopped = Event()
    for job in jobs:
        path = output_dir / "runs" / f"{job['id']}.json"
        if path.exists():
            previous = json.loads(path.read_text())
            device = devices[gpus[0]]
            if previous.get("status") in {"success", "cuda_oom"}:
                if (
                    previous.get("gpu_name") != device["gpu_name"]
                    or abs(previous["gpu_total_gib"] - device["gpu_total_gib"])
                    > 0.01
                ):
                    parser.error(
                        "GPU hardware changed. Start a new output directory."
                    )
                continue
        pending.put(job)
    print(
        f"{pending.qsize()} pending; completed and CUDA-OOM configurations are retained.",
        flush=True,
    )

    def worker(gpu):
        while not stopped.is_set():
            try:
                job = pending.get_nowait()
            except Empty:
                return
            attempt = uuid4().hex[:8]
            env = run_environment(
                job, gpu, output_dir, args.project_prefix, attempt
            )
            result_path = output_dir / "runs" / f"{job['id']}.json"
            measurement_path = Path(env["GPU_MEMORY_RESULT_PATH"])
            # A retried process must not inherit its previous measurements.
            measurement_path.unlink(missing_ok=True)
            record = {
                **job,
                **devices[gpu],
                "gpu": gpu,
                "attempt": attempt,
                "status": "running",
            }
            write_json(result_path, record)
            print(f"GPU {gpu}: {job['id']}", flush=True)
            log_path = output_dir / "logs" / f"{job['id']}_{attempt}.log"
            with log_path.open("w") as log:
                process = subprocess.run(
                    ["bash", str(LAUNCHER)],
                    cwd=REPO,
                    env={**os.environ, **env},
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=False,
                )
            measurement = (
                json.loads(measurement_path.read_text())
                if measurement_path.exists()
                else {}
            )
            result = complete_result(record, measurement, process.returncode)
            result["log_path"] = str(log_path)
            write_json(result_path, result)
            print(f"GPU {gpu}: {job['id']} -> {result['status']}", flush=True)
            if result["status"] == "error":
                stopped.set()
                raise RuntimeError(
                    f"Unexpected failure in {job['id']}; inspect {log_path}. Other GPU workers finish their current run."
                )

    with ThreadPoolExecutor(max_workers=len(gpus)) as executor:
        futures = [executor.submit(worker, gpu) for gpu in gpus]
        for future in futures:
            future.result()
    print(f"Sweep complete. Results: {output_dir}")


if __name__ == "__main__":
    main()
