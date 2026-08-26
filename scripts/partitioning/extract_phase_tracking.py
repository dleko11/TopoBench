#!/usr/bin/env python3
"""Extract phase timing and memory metrics from W&B runs."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import wandb

DEFAULT_ENTITY = "topobench-scalability"
DEFAULT_OUTPUT_DIR = Path("outputs/phase_tracking")


@dataclass(frozen=True)
class ProjectSpec:
    dataset: str
    mode: str
    project: str


DEFAULT_PROJECTS = (
    ProjectSpec("cora_full", "full", "final_cora_full_full"),
    ProjectSpec(
        "cora_full",
        "partitioning",
        "memory_cora_full_partitioning",
    ),
    ProjectSpec("amazon_ratings", "full", "final_amazon_ratings_full"),
    ProjectSpec(
        "amazon_ratings",
        "partitioning",
        "memory_amazon_ratings_partitioning",
    ),
    ProjectSpec("questions", "full", "final_questions_full"),
    ProjectSpec(
        "questions",
        "partitioning",
        "memory_questions_partitioning",
    ),
)

REQUIRED_HISTORY_KEYS = (
    "tracking/phase_id",
    "tracking/is_start",
    "tracking/is_end",
    "tracking/duration_sec",
    "tracking/epoch",
    "tracking/global_step",
)

RESOURCE_HISTORY_KEYS = (
    "tracking/resource/rss_mb",
    "tracking/resource/tree_rss_mb",
    "tracking/resource/cuda_allocated_mb",
    "tracking/resource/cuda_reserved_mb",
    "tracking/resource/cuda_peak_allocated_mb",
    "tracking/resource/cuda_peak_reserved_mb",
)

RUN_FIELDS = (
    "dataset",
    "mode",
    "project",
    "run_id",
    "run_name",
    "run_state",
    "run_url",
    "created_at",
    "heartbeat_at",
    "runtime_sec",
    "model",
    "seed",
    "data_seed",
    "model_params_total",
    "model_params_trainable",
    "final_epoch",
    "final_global_step",
    "gpu",
    "gpu_count_visible",
    "gpu_memory_total_mb",
    "trainer_devices",
    "cpu_count",
    "host_memory_total_mb",
    "python",
    "git_commit",
    "stream_q",
    "stream_q_val",
    "stream_q_test",
    "stream_num_workers",
    "cache_num_workers",
    "num_parts",
    "test_protocols",
    "phase_tracking_available",
)

PHASE_FIELDS = RUN_FIELDS + (
    "phase",
    "phase_id",
    "start_event_count",
    "end_event_count",
    "events_complete",
    "first_epoch",
    "last_epoch",
    "first_global_step",
    "last_global_step",
    "duration_total_sec",
    "duration_mean_sec",
    "duration_median_sec",
    "duration_min_sec",
    "duration_max_sec",
    "rss_boundary_max_mb",
    "tree_rss_boundary_max_mb",
    "cuda_allocated_end_max_mb",
    "cuda_reserved_end_max_mb",
    "cuda_peak_allocated_max_mb",
    "cuda_peak_reserved_max_mb",
)

PROJECT_STATUS_FIELDS = (
    "dataset",
    "mode",
    "project",
    "total_runs",
    "finished",
    "running",
    "failed",
    "crashed",
    "killed",
    "other",
    "phase_tracking_available",
    "phase_metrics_extracted",
    "incomplete_phase_rows",
)


def _nested(mapping: dict[str, Any], *keys: str) -> Any:
    value: Any = mapping
    for key in keys:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def _bytes_to_mb(value: Any) -> float | None:
    if value in (None, ""):
        return None
    return float(value) / 1024**2


def _json_cell(value: Any) -> str:
    if value is None:
        return ""
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _finite_values(rows: list[dict[str, Any]], key: str) -> list[float]:
    values = []
    for row in rows:
        value = row.get(key)
        if isinstance(value, int | float):
            values.append(float(value))
    return values


def _nonnegative_int_values(rows: list[dict[str, Any]], key: str) -> list[int]:
    values = []
    for row in rows:
        value = row.get(key)
        if isinstance(value, int | float) and value >= 0:
            values.append(int(value))
    return values


def _max_or_none(values: list[float]) -> float | None:
    return max(values) if values else None


def _min_or_none(values: list[int]) -> int | None:
    return min(values) if values else None


def _max_int_or_none(values: list[int]) -> int | None:
    return max(values) if values else None


def _phase_id_map(summary: dict[str, Any]) -> dict[int, str]:
    raw_map = summary.get("tracking/phase_id_map")
    if isinstance(raw_map, str):
        raw_map = json.loads(raw_map)
    if not isinstance(raw_map, dict):
        return {}
    return {int(phase_id): str(phase) for phase, phase_id in raw_map.items()}


def _run_metadata(spec: ProjectSpec, run: Any) -> dict[str, Any]:
    config = dict(run.config)
    summary = dict(run.summary)
    metadata = run.metadata or {}
    dataset = config.get("dataset") or {}
    loader_parameters = _nested(dataset, "loader", "parameters") or {}
    stream = loader_parameters.get("stream") or {}
    cluster = loader_parameters.get("cluster") or {}
    model = config.get("model") or {}
    trainer = config.get("trainer") or {}
    memory = metadata.get("memory") or {}
    gpu_nvidia = metadata.get("gpu_nvidia") or []
    first_gpu = gpu_nvidia[0] if gpu_nvidia else {}
    split_params = dataset.get("split_params") or {}
    test_inference = config.get("test_inference") or {}

    return {
        "dataset": spec.dataset,
        "mode": spec.mode,
        "project": spec.project,
        "run_id": run.id,
        "run_name": run.name,
        "run_state": run.state,
        "run_url": run.url,
        "created_at": run.created_at,
        "heartbeat_at": run.heartbeat_at,
        "runtime_sec": summary.get("_runtime"),
        "model": model.get("model_name"),
        "seed": config.get("seed"),
        "data_seed": split_params.get("data_seed"),
        "model_params_total": config.get("model/params/total"),
        "model_params_trainable": config.get("model/params/trainable"),
        "final_epoch": summary.get("epoch"),
        "final_global_step": summary.get("trainer/global_step"),
        "gpu": metadata.get("gpu"),
        "gpu_count_visible": metadata.get("gpu_count"),
        "gpu_memory_total_mb": _bytes_to_mb(first_gpu.get("memoryTotal")),
        "trainer_devices": _json_cell(trainer.get("devices")),
        "cpu_count": metadata.get("cpu_count"),
        "host_memory_total_mb": _bytes_to_mb(memory.get("total")),
        "python": metadata.get("python"),
        "git_commit": _nested(metadata, "git", "commit"),
        "stream_q": stream.get("q"),
        "stream_q_val": stream.get("q_val"),
        "stream_q_test": stream.get("q_test"),
        "stream_num_workers": stream.get("num_workers"),
        "cache_num_workers": stream.get("cache_num_workers"),
        "num_parts": cluster.get("num_parts"),
        "test_protocols": _json_cell(test_inference.get("protocols")),
        "phase_tracking_available": bool(_phase_id_map(summary)),
    }


def _extract_run_phases(
    run_metadata: dict[str, Any],
    run: Any,
) -> list[dict[str, Any]]:
    summary = dict(run.summary)
    phase_names = _phase_id_map(summary)
    if not phase_names:
        raise ValueError(
            f"Run {run.path} has no tracking/phase_id_map in its summary."
        )

    summary_keys = set(summary)
    history_keys = list(REQUIRED_HISTORY_KEYS)
    history_keys.extend(
        key for key in RESOURCE_HISTORY_KEYS if key in summary_keys
    )
    history = list(run.scan_history(keys=history_keys, page_size=1000))
    phase_rows: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in history:
        phase_id = row.get("tracking/phase_id")
        if isinstance(phase_id, int | float):
            phase_rows[int(phase_id)].append(row)

    if not phase_rows:
        raise ValueError(f"Run {run.path} has no phase history rows.")

    extracted = []
    for phase_id, rows in phase_rows.items():
        end_rows = [row for row in rows if row.get("tracking/is_end") == 1]
        start_count = sum(row.get("tracking/is_start") == 1 for row in rows)
        durations = _finite_values(end_rows, "tracking/duration_sec")
        epochs = _nonnegative_int_values(end_rows, "tracking/epoch")
        global_steps = _nonnegative_int_values(
            end_rows, "tracking/global_step"
        )
        row = dict(run_metadata)
        row.update(
            {
                "phase": phase_names.get(phase_id, f"unknown_{phase_id}"),
                "phase_id": phase_id,
                "start_event_count": start_count,
                "end_event_count": len(end_rows),
                "events_complete": start_count == len(end_rows),
                "first_epoch": _min_or_none(epochs),
                "last_epoch": _max_int_or_none(epochs),
                "first_global_step": _min_or_none(global_steps),
                "last_global_step": _max_int_or_none(global_steps),
                "duration_total_sec": sum(durations),
                "duration_mean_sec": (
                    statistics.mean(durations) if durations else None
                ),
                "duration_median_sec": (
                    statistics.median(durations) if durations else None
                ),
                "duration_min_sec": min(durations) if durations else None,
                "duration_max_sec": max(durations) if durations else None,
                "rss_boundary_max_mb": _max_or_none(
                    _finite_values(rows, "tracking/resource/rss_mb")
                ),
                "tree_rss_boundary_max_mb": _max_or_none(
                    _finite_values(rows, "tracking/resource/tree_rss_mb")
                ),
                "cuda_allocated_end_max_mb": _max_or_none(
                    _finite_values(
                        end_rows,
                        "tracking/resource/cuda_allocated_mb",
                    )
                ),
                "cuda_reserved_end_max_mb": _max_or_none(
                    _finite_values(
                        end_rows,
                        "tracking/resource/cuda_reserved_mb",
                    )
                ),
                "cuda_peak_allocated_max_mb": _max_or_none(
                    _finite_values(
                        end_rows,
                        "tracking/resource/cuda_peak_allocated_mb",
                    )
                ),
                "cuda_peak_reserved_max_mb": _max_or_none(
                    _finite_values(
                        end_rows,
                        "tracking/resource/cuda_peak_reserved_mb",
                    )
                ),
            }
        )
        extracted.append(row)

    return sorted(extracted, key=lambda row: (row["phase_id"], row["phase"]))


def _parse_project_spec(value: str) -> ProjectSpec:
    parts = value.split(":", maxsplit=2)
    if len(parts) != 3 or not all(parts):
        raise argparse.ArgumentTypeError(
            "Project must have the form DATASET:MODE:PROJECT."
        )
    return ProjectSpec(*parts)


def _write_csv(
    path: Path,
    rows: list[dict[str, Any]],
    fieldnames: tuple[str, ...],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _status_rows(
    specs_and_runs: list[tuple[ProjectSpec, list[Any]]],
    extracted_counts: Counter[str],
    incomplete_counts: Counter[str],
) -> list[dict[str, Any]]:
    rows = []
    known_states = {"finished", "running", "failed", "crashed", "killed"}
    for spec, runs in specs_and_runs:
        states = Counter(run.state for run in runs)
        rows.append(
            {
                "dataset": spec.dataset,
                "mode": spec.mode,
                "project": spec.project,
                "total_runs": len(runs),
                "finished": states["finished"],
                "running": states["running"],
                "failed": states["failed"],
                "crashed": states["crashed"],
                "killed": states["killed"],
                "other": sum(
                    count
                    for state, count in states.items()
                    if state not in known_states
                ),
                "phase_tracking_available": sum(
                    bool(_phase_id_map(dict(run.summary))) for run in runs
                ),
                "phase_metrics_extracted": extracted_counts[spec.project],
                "incomplete_phase_rows": incomplete_counts[spec.project],
            }
        )
    return rows


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--entity", default=DEFAULT_ENTITY)
    parser.add_argument(
        "--project",
        action="append",
        type=_parse_project_spec,
        help=(
            "Project as DATASET:MODE:PROJECT. Repeat to replace the six "
            "default projects."
        ),
    )
    parser.add_argument(
        "--states",
        default="finished",
        help="Comma-separated run states whose phase history is extracted.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--timeout", type=int, default=120)
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    if args.workers < 1:
        raise ValueError("--workers must be at least 1.")

    specs = tuple(args.project) if args.project else DEFAULT_PROJECTS
    included_states = {
        state.strip() for state in args.states.split(",") if state.strip()
    }
    if not included_states:
        raise ValueError("--states must contain at least one run state.")

    api = wandb.Api(timeout=args.timeout)
    specs_and_runs = []
    run_rows = []
    extraction_tasks = []
    for spec in specs:
        runs = list(api.runs(f"{args.entity}/{spec.project}", per_page=100))
        specs_and_runs.append((spec, runs))
        for run in runs:
            metadata = _run_metadata(spec, run)
            run_rows.append(metadata)
            if run.state in included_states:
                extraction_tasks.append((metadata, run))

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        phase_groups = list(
            executor.map(
                lambda task: _extract_run_phases(*task),
                extraction_tasks,
            )
        )

    phase_rows = [row for group in phase_groups for row in group]
    run_rows.sort(
        key=lambda row: (
            row["dataset"],
            row["mode"],
            row["model"] or "",
            row["seed"] if row["seed"] is not None else -1,
            row["run_id"],
        )
    )
    phase_rows.sort(
        key=lambda row: (
            row["dataset"],
            row["mode"],
            row["model"] or "",
            row["seed"] if row["seed"] is not None else -1,
            row["phase_id"],
        )
    )

    extracted_counts = Counter(
        metadata["project"] for metadata, _ in extraction_tasks
    )
    incomplete_counts = Counter(
        row["project"] for row in phase_rows if not row["events_complete"]
    )
    status_rows = _status_rows(
        specs_and_runs,
        extracted_counts,
        incomplete_counts,
    )

    output_dir = args.output_dir
    _write_csv(output_dir / "runs.csv", run_rows, RUN_FIELDS)
    _write_csv(output_dir / "phase_metrics.csv", phase_rows, PHASE_FIELDS)
    _write_csv(
        output_dir / "project_status.csv",
        status_rows,
        PROJECT_STATUS_FIELDS,
    )

    timestamp = datetime.now(UTC).isoformat()
    print(f"Extracted at: {timestamp}")
    print(f"Runs listed: {len(run_rows)}")
    print(f"Runs with phase metrics: {len(extraction_tasks)}")
    print(f"Phase rows written: {len(phase_rows)}")
    print(f"Incomplete phase rows: {sum(incomplete_counts.values())}")
    for filename in ("runs.csv", "phase_metrics.csv", "project_status.csv"):
        print(output_dir / filename)


if __name__ == "__main__":
    main()
