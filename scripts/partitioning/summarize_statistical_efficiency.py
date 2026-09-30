#!/usr/bin/env python3
"""Summarize validation selection and test performance from W&B runs."""

from __future__ import annotations

import argparse
import csv
import math
import statistics
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import wandb

from scripts.partitioning.extract_phase_tracking import (
    ProjectSpec,
    _phase_id_map,
    _run_metadata,
)

DEFAULT_ENTITY = "topobench-scalability"
DEFAULT_OUTPUT_DIR = Path("outputs/statistical_efficiency")
DEFAULT_PROJECTS = (
    ProjectSpec("cora_full", "full", "table1_rerun_cora_full_full"),
    ProjectSpec(
        "cora_full",
        "partitioning",
        "table1_rerun_cora_full_partitioning",
    ),
    ProjectSpec(
        "amazon_ratings",
        "full",
        "table1_rerun_amazon_ratings_full",
    ),
    ProjectSpec(
        "amazon_ratings",
        "partitioning",
        "table1_rerun_amazon_ratings_partitioning",
    ),
    ProjectSpec("questions", "full", "table1_rerun_questions_full"),
    ProjectSpec(
        "questions",
        "partitioning",
        "table1_rerun_questions_partitioning",
    ),
)
DATASET_METRICS = {
    "cora_full": "accuracy",
    "amazon_ratings": "accuracy",
    "questions": "auroc",
}
DATASET_ORDER = ("cora_full", "amazon_ratings", "questions")
MODEL_ORDER = ("gcn", "edgnn", "unignn", "cwn", "topotune", "scn", "sccnn")
DATASET_NAMES = {
    "cora_full": "Cora Full",
    "amazon_ratings": "Amazon Ratings",
    "questions": "Questions",
}
MODEL_NAMES = {
    "gcn": "GCN",
    "edgnn": "EDHNN",
    "unignn": "UniGNN",
    "cwn": "CWN",
    "topotune": "Cell TopoTune",
    "scn": "SCN",
    "sccnn": "SCCNN",
}
MEASURES = (
    "first_validation_epoch",
    "first_validation_global_step",
    "first_validation_metric",
    "selected_epoch",
    "selected_global_step",
    "selected_validation_metric",
    "checkpoint_validation_metric",
    "test_metric",
    "final_validation_epoch",
    "final_validation_global_step",
    "final_validation_metric",
)


def _parse_project_spec(value: str) -> ProjectSpec:
    parts = value.split(":", maxsplit=2)
    if len(parts) != 3 or not all(parts):
        raise argparse.ArgumentTypeError(
            "Project must have the form DATASET:MODE:PROJECT."
        )
    return ProjectSpec(*parts)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--entity", default=DEFAULT_ENTITY)
    parser.add_argument(
        "--project",
        action="append",
        type=_parse_project_spec,
        help=(
            "Project as DATASET:MODE:PROJECT. Repeat to replace the six "
            "default Table 1 projects."
        ),
    )
    parser.add_argument(
        "--expected-seeds",
        default="0,1,2,3,4",
        help="Comma-separated seeds required for every dataset/model/mode.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--timeout", type=int, default=120)
    return parser


def _finite_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _finite_int(value: Any) -> int | None:
    number = _finite_float(value)
    if number is None:
        return None
    return int(number)


def _primary_metric(dataset: str) -> str:
    try:
        return DATASET_METRICS[dataset]
    except KeyError as error:
        raise ValueError(f"No primary metric configured for {dataset!r}.") from error


def _test_key(mode: str, metric: str) -> tuple[str, str]:
    if mode == "full":
        return f"test_inference/batched/{metric}", "full_graph"
    if mode == "partitioning":
        return f"test_inference/ensemble/{metric}", "ensemble"
    raise ValueError(f"Unsupported mode {mode!r}.")


def _monitor_mode(run: Any) -> str:
    callbacks = run.config.get("callbacks") or {}
    checkpoint = callbacks.get("model_checkpoint") or {}
    mode = checkpoint.get("mode")
    if mode not in {"min", "max"}:
        raise ValueError(f"Run {run.path} has unsupported monitor mode {mode!r}.")
    return str(mode)


def _validation_steps(run: Any) -> dict[int, int]:
    summary = dict(run.summary)
    phase_ids = _phase_id_map(summary)
    try:
        validation_phase_id = next(
            phase_id
            for phase_id, name in phase_ids.items()
            if name == "validation_epoch"
        )
    except StopIteration as error:
        raise ValueError(
            f"Run {run.path} has no validation_epoch phase identifier."
        ) from error

    steps: dict[int, int] = {}
    history = run.scan_history(
        keys=[
            "tracking/phase_id",
            "tracking/is_end",
            "tracking/epoch",
            "tracking/global_step",
        ],
        page_size=1000,
    )
    for row in history:
        phase_id = _finite_int(row.get("tracking/phase_id"))
        is_end = _finite_int(row.get("tracking/is_end"))
        epoch = _finite_int(row.get("tracking/epoch"))
        global_step = _finite_int(row.get("tracking/global_step"))
        if (
            phase_id == validation_phase_id
            and is_end == 1
            and epoch is not None
            and epoch >= 0
            and global_step is not None
            and global_step >= 0
        ):
            steps[epoch] = global_step
    if not steps:
        raise ValueError(f"Run {run.path} has no validation phase-end markers.")
    return steps


def _validation_history(run: Any, metric: str) -> list[dict[str, Any]]:
    key = f"val/{metric}"
    rows = []
    for row in run.scan_history(
        keys=["epoch", key, "_runtime"],
        page_size=1000,
    ):
        epoch = _finite_int(row.get("epoch"))
        value = _finite_float(row.get(key))
        if epoch is None or epoch < 0 or value is None:
            continue
        rows.append(
            {
                "epoch_zero_based": epoch,
                "epoch": epoch + 1,
                "validation_metric": value,
                "runtime_sec": _finite_float(row.get("_runtime")),
            }
        )
    rows.sort(key=lambda row: row["epoch_zero_based"])
    if not rows:
        raise ValueError(f"Run {run.path} has no finite {key!r} history.")
    return rows


def _selected_index(rows: list[dict[str, Any]], mode: str) -> int:
    values = [float(row["validation_metric"]) for row in rows]
    best = min(values) if mode == "min" else max(values)
    return values.index(best)


def _summary_metric(summary: dict[str, Any], key: str, run: Any) -> float:
    value = _finite_float(summary.get(key))
    if value is None:
        raise ValueError(f"Run {run.path} has no finite summary metric {key!r}.")
    return value


def _extract_run(
    task: tuple[ProjectSpec, dict[str, Any], Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    spec, metadata, run = task
    metric = _primary_metric(spec.dataset)
    summary = dict(run.summary)
    validation_rows = _validation_history(run, metric)
    validation_steps = _validation_steps(run)
    selected_index = _selected_index(validation_rows, _monitor_mode(run))
    selected = validation_rows[selected_index]
    selected_epoch_zero_based = int(selected["epoch_zero_based"])
    if selected_epoch_zero_based not in validation_steps:
        raise ValueError(
            f"Run {run.path} has no phase step for selected epoch "
            f"{selected_epoch_zero_based}."
        )

    selected_value = float(selected["validation_metric"])
    best_monitored_score = _summary_metric(
        summary,
        "best_monitored_score",
        run,
    )
    if not math.isclose(
        selected_value,
        best_monitored_score,
        rel_tol=1e-5,
        abs_tol=1e-7,
    ):
        raise ValueError(
            f"Run {run.path} history best {selected_value} does not match "
            f"best_monitored_score {best_monitored_score}."
        )

    test_key, test_protocol = _test_key(spec.mode, metric)
    run_row = {
        "dataset": spec.dataset,
        "mode": spec.mode,
        "project": spec.project,
        "run_id": run.id,
        "run_name": run.name,
        "run_state": run.state,
        "model": metadata["model"],
        "seed": metadata["seed"],
        "metric": metric,
        "num_parts": metadata["num_parts"],
        "q": metadata["stream_q"],
        "validation_observations": len(validation_rows),
        "first_validation_epoch": validation_rows[0]["epoch"],
        "first_validation_global_step": validation_steps[
            int(validation_rows[0]["epoch_zero_based"])
        ],
        "first_validation_metric": validation_rows[0]["validation_metric"],
        "selected_epoch_zero_based": selected_epoch_zero_based,
        "selected_epoch": selected["epoch"],
        "selected_global_step": validation_steps[selected_epoch_zero_based],
        "selected_validation_metric": selected_value,
        "checkpoint_validation_metric": _summary_metric(
            summary,
            f"val_best_rerun/{metric}",
            run,
        ),
        "test_metric": _summary_metric(summary, test_key, run),
        "test_protocol": test_protocol,
        "final_validation_epoch": validation_rows[-1]["epoch"],
        "final_validation_global_step": validation_steps[
            int(validation_rows[-1]["epoch_zero_based"])
        ],
        "final_validation_metric": validation_rows[-1]["validation_metric"],
        "best_monitored_score": best_monitored_score,
    }

    trajectory_rows = []
    for index, row in enumerate(validation_rows, start=1):
        epoch_zero_based = int(row["epoch_zero_based"])
        if epoch_zero_based not in validation_steps:
            raise ValueError(
                f"Run {run.path} has no phase step for validation epoch "
                f"{epoch_zero_based}."
            )
        trajectory_rows.append(
            {
                "dataset": spec.dataset,
                "mode": spec.mode,
                "project": spec.project,
                "run_id": run.id,
                "run_name": run.name,
                "model": metadata["model"],
                "seed": metadata["seed"],
                "metric": metric,
                "num_parts": metadata["num_parts"],
                "q": metadata["stream_q"],
                "validation_index": index,
                "epoch_zero_based": epoch_zero_based,
                "epoch": row["epoch"],
                "global_step": validation_steps[epoch_zero_based],
                "validation_metric": row["validation_metric"],
                "is_selected": int(index - 1 == selected_index),
                "runtime_sec": row["runtime_sec"],
            }
        )
    return run_row, trajectory_rows


def _mean_std(values: list[float]) -> tuple[float, float]:
    if not values:
        raise ValueError("Cannot aggregate an empty value list.")
    mean = statistics.fmean(values)
    std = statistics.stdev(values) if len(values) > 1 else 0.0
    return mean, std


def _aggregate(run_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in run_rows:
        key = (str(row["dataset"]), str(row["model"]), str(row["mode"]))
        grouped.setdefault(key, []).append(row)

    records = []
    for (dataset, model, mode), rows in grouped.items():
        rows.sort(key=lambda row: int(row["seed"]))
        record: dict[str, Any] = {
            "dataset": dataset,
            "model": model,
            "mode": mode,
            "metric": rows[0]["metric"],
            "n": len(rows),
            "seeds": ",".join(str(row["seed"]) for row in rows),
            "num_parts": rows[0]["num_parts"],
            "q": rows[0]["q"],
            "test_protocol": rows[0]["test_protocol"],
        }
        for measure in MEASURES:
            values = [float(row[measure]) for row in rows]
            mean, std = _mean_std(values)
            record[f"{measure}_mean"] = mean
            record[f"{measure}_std"] = std
        records.append(record)
    return records


def _pair_aggregates(
    aggregate_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    indexed = {
        (row["dataset"], row["model"], row["mode"]): row
        for row in aggregate_rows
    }
    records = []
    for dataset in DATASET_ORDER:
        for model in MODEL_ORDER:
            full = indexed[(dataset, model, "full")]
            partitioning = indexed[(dataset, model, "partitioning")]
            record = {
                "dataset": dataset,
                "model": model,
                "metric": full["metric"],
                "n_full": full["n"],
                "n_partitioning": partitioning["n"],
                "num_parts": partitioning["num_parts"],
                "q": partitioning["q"],
            }
            for prefix, source in (("full", full), ("partitioning", partitioning)):
                for measure in MEASURES:
                    record[f"{prefix}_{measure}_mean"] = source[
                        f"{measure}_mean"
                    ]
                    record[f"{prefix}_{measure}_std"] = source[
                        f"{measure}_std"
                    ]
            records.append(record)
    return records


def _validate_coverage(
    run_rows: list[dict[str, Any]],
    expected_seeds: set[int],
) -> None:
    grouped: dict[tuple[str, str, str], set[int]] = {}
    for row in run_rows:
        key = (str(row["dataset"]), str(row["model"]), str(row["mode"]))
        grouped.setdefault(key, set()).add(int(row["seed"]))

    expected_groups = {
        (dataset, model, mode)
        for dataset in DATASET_ORDER
        for model in MODEL_ORDER
        for mode in ("full", "partitioning")
    }
    if set(grouped) != expected_groups:
        missing = sorted(expected_groups - set(grouped))
        extra = sorted(set(grouped) - expected_groups)
        raise ValueError(f"Unexpected group coverage; missing={missing}, extra={extra}.")
    for key, seeds in grouped.items():
        if seeds != expected_seeds:
            raise ValueError(
                f"Unexpected seeds for {key}: {sorted(seeds)}; expected "
                f"{sorted(expected_seeds)}."
            )


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty CSV {path}.")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _format_stat(
    row: dict[str, Any],
    prefix: str,
    measure: str,
    decimals: int,
    scale: float = 1.0,
) -> str:
    mean = float(row[f"{prefix}_{measure}_mean"]) * scale
    std = float(row[f"{prefix}_{measure}_std"]) * scale
    return f"{mean:.{decimals}f} $\\pm$ {std:.{decimals}f}"


def _latex_table(pair_rows: list[dict[str, Any]]) -> str:
    lines = [
        r"\begin{table*}[!htbp]",
        r"\color{blue}",
        r"\centering",
        (
            r"\caption{Statistical-efficiency summary for full-graph (FG) "
            r"and partitioned (P) training. $E^{*}$ is the epoch selected "
            r"by the primary validation metric, and $U^{*}$ is the "
            r"number of optimizer updates completed at that checkpoint. "
            r"Validation and test results are percentages. Each entry is the "
            r"mean $\pm$ sample standard deviation across five seeds. The test "
            r"metric is accuracy for Cora Full and Amazon Ratings and AUROC "
            r"for Questions; FG uses one full-domain test pass, whereas P uses "
            r"the ten-pass ensemble protocol.}"
        ),
        r"\label{tab:tdl_statistical_efficiency}",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{2.4pt}",
        r"\renewcommand{\arraystretch}{1.04}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{llrr*{4}{r}*{4}{r}}",
        r"\toprule",
        (
            r"& & & & \multicolumn{4}{c}{\textbf{FG}} & "
            r"\multicolumn{4}{c}{\textbf{P}} \\"
        ),
        r"\cmidrule(lr){5-8}\cmidrule(lr){9-12}",
        (
            r"\textbf{Dataset} & \textbf{Model} & $\boldsymbol{K}$ & "
            r"$\boldsymbol{q}$ & $\boldsymbol{E^{*}}$ & "
            r"$\boldsymbol{U^{*}}$ & \textbf{Val.} & \textbf{Test} & "
            r"$\boldsymbol{E^{*}}$ & $\boldsymbol{U^{*}}$ & "
            r"\textbf{Val.} & \textbf{Test} \\"
        ),
        r"\midrule",
    ]
    for dataset_index, dataset in enumerate(DATASET_ORDER):
        dataset_rows = [row for row in pair_rows if row["dataset"] == dataset]
        dataset_title = DATASET_NAMES[dataset].replace(" ", r"\\")
        for model_index, row in enumerate(dataset_rows):
            dataset_cell = (
                rf"\multirow{{7}}{{*}}{{\makecell[l]{{{dataset_title}}}}}"
                if model_index == 0
                else ""
            )
            cells = [
                dataset_cell,
                MODEL_NAMES[str(row["model"])],
                str(int(float(row["num_parts"]))),
                str(int(float(row["q"]))),
                _format_stat(row, "full", "selected_epoch", 1),
                _format_stat(row, "full", "selected_global_step", 1),
                _format_stat(row, "full", "selected_validation_metric", 2, 100),
                _format_stat(row, "full", "test_metric", 2, 100),
                _format_stat(row, "partitioning", "selected_epoch", 1),
                _format_stat(row, "partitioning", "selected_global_step", 1),
                _format_stat(
                    row,
                    "partitioning",
                    "selected_validation_metric",
                    2,
                    100,
                ),
                _format_stat(row, "partitioning", "test_metric", 2, 100),
            ]
            lines.append(" & ".join(cells) + r" \\")
        if dataset_index < len(DATASET_ORDER) - 1:
            lines.append(r"\midrule")
    lines.extend(
        (
            r"\bottomrule",
            r"\end{tabular}%",
            r"}",
            r"\end{table*}",
        )
    )
    return "\n".join(lines) + "\n"


def _select_latest_finished_runs(
    api: Any,
    entity: str,
    specs: tuple[ProjectSpec, ...],
) -> list[tuple[ProjectSpec, dict[str, Any], Any]]:
    selected: dict[tuple[str, str, str, int], tuple[ProjectSpec, dict[str, Any], Any]] = {}
    for spec in specs:
        for run in api.runs(f"{entity}/{spec.project}", per_page=100):
            if run.state != "finished":
                continue
            metadata = _run_metadata(spec, run)
            model = metadata["model"]
            seed = metadata["seed"]
            if model not in MODEL_ORDER or seed is None:
                continue
            key = (spec.dataset, str(model), spec.mode, int(seed))
            current = selected.get(key)
            if current is None or str(run.created_at) >= str(current[2].created_at):
                selected[key] = (spec, metadata, run)
    return list(selected.values())


def main() -> None:
    args = _build_parser().parse_args()
    if args.workers < 1:
        raise ValueError("--workers must be at least 1.")
    expected_seeds = {
        int(seed.strip())
        for seed in args.expected_seeds.split(",")
        if seed.strip()
    }
    if not expected_seeds:
        raise ValueError("--expected-seeds must contain at least one seed.")

    specs = tuple(args.project) if args.project else DEFAULT_PROJECTS
    api = wandb.Api(timeout=args.timeout)
    tasks = _select_latest_finished_runs(api, args.entity, specs)
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        extracted = list(executor.map(_extract_run, tasks))

    run_rows = [run_row for run_row, _ in extracted]
    trajectory_rows = [row for _, rows in extracted for row in rows]
    _validate_coverage(run_rows, expected_seeds)
    dataset_rank = {value: index for index, value in enumerate(DATASET_ORDER)}
    model_rank = {value: index for index, value in enumerate(MODEL_ORDER)}
    mode_rank = {"full": 0, "partitioning": 1}
    run_rows.sort(
        key=lambda row: (
            dataset_rank[str(row["dataset"])],
            model_rank[str(row["model"])],
            mode_rank[str(row["mode"])],
            int(row["seed"]),
        )
    )
    trajectory_rows.sort(
        key=lambda row: (
            dataset_rank[str(row["dataset"])],
            model_rank[str(row["model"])],
            mode_rank[str(row["mode"])],
            int(row["seed"]),
            int(row["validation_index"]),
        )
    )
    aggregate_rows = _aggregate(run_rows)
    aggregate_rows.sort(
        key=lambda row: (
            dataset_rank[str(row["dataset"])],
            model_rank[str(row["model"])],
            mode_rank[str(row["mode"])],
        )
    )
    pair_rows = _pair_aggregates(aggregate_rows)

    output_dir = args.output_dir
    _write_csv(output_dir / "per_run.csv", run_rows)
    _write_csv(output_dir / "validation_trajectories.csv", trajectory_rows)
    _write_csv(output_dir / "aggregate.csv", aggregate_rows)
    _write_csv(output_dir / "paired_summary.csv", pair_rows)
    table_path = output_dir / "statistical_efficiency_table.tex"
    table_path.write_text(_latex_table(pair_rows), encoding="utf-8")

    print(f"Selected finished runs: {len(run_rows)}")
    print(f"Validation observations: {len(trajectory_rows)}")
    print(f"Dataset/model pairs: {len(pair_rows)}")
    for filename in (
        "per_run.csv",
        "validation_trajectories.csv",
        "aggregate.csv",
        "paired_summary.csv",
        "statistical_efficiency_table.tex",
    ):
        print(output_dir / filename)


if __name__ == "__main__":
    main()
