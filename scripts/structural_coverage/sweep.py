"""Run the restartable Cora structural-coverage cluster sweep.

The default matrix contains four structure profiles, six partition batch
sizes, and ten random seeds (240 runs). Complete training artifacts remain in
``results`` while compact, transferable CSV/JSON artifacts are copied to
``results_for_plotting`` after each successful run.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

DEFAULT_Q_VALUES = (1, 2, 4, 8, 16, 32)
DEFAULT_SEEDS = tuple(range(10))
DEFAULT_PROFILES = (
    "simplicial",
    "hypergraph",
    "cell_basis",
    "cell_simple_coverage",
)
REQUIRED_PLOTTING_FILES = (
    "empirical_coverage.csv",
    "theory_curves.csv",
    "span_histogram.csv",
    "metrics.csv",
    "run_metadata.json",
)


@dataclass(frozen=True)
class Profile:
    """Hydra configuration for one structural family."""

    name: str
    model: str
    transforms: str
    structure_family: str
    overrides: tuple[str, ...] = ()


@dataclass(frozen=True)
class SweepTask:
    """One profile/q/seed experiment in the sweep."""

    index: int
    profile: str
    q: int
    seed: int

    @property
    def key(self) -> str:
        return f"{self.profile}__q{self.q:02d}__seed{self.seed:02d}"


PROFILES = {
    "simplicial": Profile(
        name="simplicial",
        model="simplicial/scn",
        transforms="liftings/graph2simplicial_default",
        structure_family="simplicial_clique",
        overrides=("transforms.graph2simplicial_lifting.complex_dim=2",),
    ),
    "hypergraph": Profile(
        name="hypergraph",
        model="hypergraph/unignn",
        transforms="liftings/graph2hypergraph_default",
        structure_family="hypergraph_khop",
    ),
    "cell_basis": Profile(
        name="cell_basis",
        model="cell/cwn",
        transforms="liftings/graph2cell_default",
        structure_family="cell_cycle",
    ),
    "cell_simple_coverage": Profile(
        name="cell_simple_coverage",
        model="cell/cwn",
        transforms="liftings/graph2cell_default",
        structure_family="cell_simple_cycles",
    ),
}


def parse_int_values(value: str) -> tuple[int, ...]:
    """Parse a comma/space-separated collection of unique integers."""
    tokens = value.replace(",", " ").split()
    values = tuple(dict.fromkeys(int(token) for token in tokens))
    if not values:
        raise argparse.ArgumentTypeError("at least one integer is required")
    return values


def parse_profile_values(value: str) -> tuple[str, ...]:
    """Parse and validate profile names."""
    names = tuple(dict.fromkeys(value.replace(",", " ").split()))
    unknown = sorted(set(names) - set(PROFILES))
    if unknown:
        raise argparse.ArgumentTypeError(
            f"unknown profiles {unknown}; choose from {sorted(PROFILES)}"
        )
    if not names:
        raise argparse.ArgumentTypeError("at least one profile is required")
    return names


def build_tasks(
    profiles: tuple[str, ...],
    q_values: tuple[int, ...],
    seeds: tuple[int, ...],
) -> list[SweepTask]:
    """Build a stable profile-major task matrix."""
    tasks: list[SweepTask] = []
    for profile in profiles:
        for q in q_values:
            for seed in seeds:
                tasks.append(
                    SweepTask(
                        index=len(tasks),
                        profile=profile,
                        q=int(q),
                        seed=int(seed),
                    )
                )
    return tasks


def task_directory(root: Path, task: SweepTask) -> Path:
    """Return the deterministic directory for a task."""
    return root / task.profile / f"q{task.q:02d}" / f"seed{task.seed:02d}"


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    """Atomically write formatted JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def read_json(path: Path) -> dict[str, Any] | None:
    """Read a JSON object, returning ``None`` for missing/invalid files."""
    try:
        payload = json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None
    return payload if isinstance(payload, dict) else None


def task_is_complete(
    task: SweepTask,
    results_root: Path,
    plotting_root: Path,
    expected_signature: str | None = None,
) -> bool:
    """Return whether full and portable artifacts are complete."""
    run_dir = task_directory(results_root, task)
    plot_dir = task_directory(plotting_root, task)
    status = read_json(run_dir / "sweep_status.json")
    if status is None or status.get("status") != "success":
        return False
    if (
        expected_signature is not None
        and status.get("configuration_signature") != expected_signature
    ):
        return False
    if not all((run_dir / name).is_file() for name in REQUIRED_PLOTTING_FILES):
        return False
    if not all(
        (plot_dir / name).is_file() for name in REQUIRED_PLOTTING_FILES
    ):
        return False
    portable_status = read_json(plot_dir / "status.json")
    return (
        portable_status is not None
        and portable_status.get("status") == "success"
    )


def git_commit(project_root: Path) -> str | None:
    """Return the current Git commit when available."""
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=project_root,
        text=True,
        capture_output=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def command_for_task(
    *,
    task: SweepTask,
    args: argparse.Namespace,
    project_root: Path,
) -> list[str]:
    """Build the structural-coverage command for one task."""
    profile = PROFILES[task.profile]
    run_dir = task_directory(Path(args.results_root), task).resolve()
    cache_root = (Path(args.results_root) / "_coverage_cache").resolve()
    command = [
        sys.executable,
        "-m",
        "scripts.structural_coverage.run",
        f"dataset={args.dataset}",
        f"model={profile.model}",
        f"transforms={profile.transforms}",
        f"trainer={args.trainer}",
        "logger=csv",
        "test=false",
        f"seed={task.seed}",
        f"dataset.split_params.data_seed={args.data_split_seed}",
        f"dataset.split_params.train_prop={args.train_prop}",
        f"dataset.loader.parameters.cluster.num_parts={args.num_parts}",
        f"dataset.loader.parameters.stream.q={task.q}",
        f"++dataset.loader.parameters.stream.q_val={args.eval_q}",
        f"++dataset.loader.parameters.stream.q_test={args.eval_q}",
        f"trainer.max_epochs={args.max_epochs}",
        f"trainer.min_epochs={args.max_epochs}",
        "trainer.check_val_every_n_epoch=1",
        "callbacks.early_stopping=null",
        "+trainer.enable_progress_bar=false",
        "extras.print_config=false",
        "extras.enforce_tags=false",
        f"coverage.structure_family={profile.structure_family}",
        "coverage.save_batch_events=false",
        "coverage.audit_induced_edges=true",
        f"coverage.audit_max_batches={args.audit_max_batches}",
        "coverage.require_equal_batches=false",
        f"coverage.results_root={Path(args.results_root).resolve()}",
        f"coverage.run_dir={run_dir}",
        f"coverage.cache_root={cache_root}",
    ]
    command.extend(profile.overrides)
    if task.profile == "hypergraph":
        command.append(
            f"transforms.graph2hypergraph_lifting.k_value={args.hypergraph_k}"
        )
    if task.profile == "cell_simple_coverage":
        command.append(f"+coverage.max_support_nodes={args.max_support_nodes}")
    return command


def command_signature(command: list[str]) -> str:
    """Hash the effective Hydra overrides used for restart compatibility."""
    payload = json.dumps(command[3:], separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def export_task(
    *,
    task: SweepTask,
    results_root: Path,
    plotting_root: Path,
    status: dict[str, Any],
) -> Path:
    """Copy the portable artifact contract for a completed task."""
    source = task_directory(results_root, task)
    missing = [
        name
        for name in REQUIRED_PLOTTING_FILES
        if not (source / name).is_file()
    ]
    if missing:
        raise FileNotFoundError(
            f"cannot export {task.key}; missing artifacts: {missing}"
        )

    destination = task_directory(plotting_root, task)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(
            prefix=f".{destination.name}.",
            dir=destination.parent,
        )
    )
    try:
        for name in REQUIRED_PLOTTING_FILES:
            shutil.copy2(source / name, temporary / name)
        portable_status = {
            **status,
            "status": "success",
            "source_result_dir": str(source.resolve()),
            "exported_at_unix": time.time(),
        }
        atomic_write_json(temporary / "status.json", portable_status)
        if destination.exists():
            shutil.rmtree(destination)
        os.replace(temporary, destination)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return destination


class SweepRunner:
    """Concurrent, restartable executor for structural-coverage tasks."""

    def __init__(self, args: argparse.Namespace, project_root: Path) -> None:
        self.args = args
        self.project_root = project_root
        self.results_root = Path(args.results_root).resolve()
        self.plotting_root = Path(args.plotting_root).resolve()
        self.commit = git_commit(project_root)
        self.gpus = tuple(
            item.strip() for item in str(args.gpus).split(",") if item.strip()
        )
        if args.trainer == "gpu" and not self.gpus:
            raise ValueError("at least one GPU must be supplied with --gpus")
        self.print_lock = threading.Lock()

    def gpu_for_task(self, task: SweepTask) -> str | None:
        """Assign a physical GPU round-robin while allowing oversubscription."""
        if self.args.trainer != "gpu":
            return None
        return self.gpus[task.index % len(self.gpus)]

    def log(self, message: str) -> None:
        with self.print_lock:
            print(message, flush=True)

    def run_one(self, task: SweepTask) -> dict[str, Any]:
        """Execute and export one task."""
        run_dir = task_directory(self.results_root, task)
        plot_dir = task_directory(self.plotting_root, task)
        command = command_for_task(
            task=task,
            args=self.args,
            project_root=self.project_root,
        )
        configuration_signature = command_signature(command)
        if not self.args.force and task_is_complete(
            task,
            self.results_root,
            self.plotting_root,
            expected_signature=configuration_signature,
        ):
            self.log(f"[skip] {task.key}")
            return {"task": task.key, "status": "skipped"}

        existing_status = read_json(run_dir / "sweep_status.json")
        full_artifacts_exist = all(
            (run_dir / name).is_file() for name in REQUIRED_PLOTTING_FILES
        )
        if (
            not self.args.force
            and existing_status is not None
            and existing_status.get("exit_code") == 0
            and existing_status.get("configuration_signature")
            == configuration_signature
            and full_artifacts_exist
        ):
            try:
                existing_status["status"] = "success"
                export_task(
                    task=task,
                    results_root=self.results_root,
                    plotting_root=self.plotting_root,
                    status=existing_status,
                )
                atomic_write_json(
                    run_dir / "sweep_status.json", existing_status
                )
                self.log(f"[re-exported] {task.key}")
                return {"task": task.key, "status": "skipped"}
            except BaseException as error:
                self.log(f"[export_failed] {task.key}: {error!r}")
                return {
                    "task": task.key,
                    "status": "export_failed",
                    "export_error": repr(error),
                }

        run_dir.mkdir(parents=True, exist_ok=True)
        gpu = self.gpu_for_task(task)
        started = time.time()
        base_status: dict[str, Any] = {
            **asdict(task),
            "task": task.key,
            "status": "running",
            "started_at_unix": started,
            "gpu": gpu,
            "command": command,
            "command_shell": shlex.join(command),
            "configuration_signature": configuration_signature,
            "git_commit": self.commit,
            "project_root": str(self.project_root),
            "result_dir": str(run_dir),
            "plotting_dir": str(plot_dir),
        }
        atomic_write_json(run_dir / "sweep_status.json", base_status)
        self.log(f"[start gpu={gpu or 'cpu'}] {task.key}")

        environment = os.environ.copy()
        environment.update(
            {
                "HYDRA_FULL_ERROR": "1",
                "OMP_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
                "OPENBLAS_NUM_THREADS": "1",
                "VECLIB_MAXIMUM_THREADS": "1",
                "NUMEXPR_NUM_THREADS": "1",
            }
        )
        if gpu is not None:
            environment["CUDA_VISIBLE_DEVICES"] = gpu

        stdout_path = run_dir / "sweep_stdout.log"
        stderr_path = run_dir / "sweep_stderr.log"
        with stdout_path.open("w") as stdout, stderr_path.open("w") as stderr:
            completed = subprocess.run(
                command,
                cwd=self.project_root,
                env=environment,
                stdout=stdout,
                stderr=stderr,
                check=False,
            )

        final_status = {
            **base_status,
            "status": "success" if completed.returncode == 0 else "failed",
            "exit_code": completed.returncode,
            "finished_at_unix": time.time(),
            "duration_seconds": time.time() - started,
        }
        if completed.returncode == 0:
            try:
                export_task(
                    task=task,
                    results_root=self.results_root,
                    plotting_root=self.plotting_root,
                    status=final_status,
                )
            except BaseException as error:
                final_status["status"] = "export_failed"
                final_status["export_error"] = repr(error)

        atomic_write_json(run_dir / "sweep_status.json", final_status)
        self.log(
            f"[{final_status['status']}] {task.key} "
            f"({final_status['duration_seconds']:.1f}s)"
        )
        return final_status


def status_row(
    task: SweepTask,
    results_root: Path,
    plotting_root: Path,
) -> dict[str, Any]:
    """Build one manifest row from on-disk task state."""
    run_dir = task_directory(results_root, task)
    status = read_json(run_dir / "sweep_status.json") or {}
    complete = task_is_complete(task, results_root, plotting_root)
    reported_status = status.get("status", "pending")
    if not complete and reported_status == "success":
        reported_status = "portable_missing"
    return {
        "index": task.index,
        "task": task.key,
        "profile": task.profile,
        "q": task.q,
        "seed": task.seed,
        "status": "success" if complete else reported_status,
        "exit_code": status.get("exit_code", ""),
        "duration_seconds": status.get("duration_seconds", ""),
        "gpu": status.get("gpu", ""),
        "result_dir": str(run_dir),
        "plotting_dir": str(task_directory(plotting_root, task)),
    }


def write_manifest(
    tasks: list[SweepTask],
    results_root: Path,
    plotting_root: Path,
) -> tuple[Path, dict[str, int]]:
    """Rebuild the portable sweep manifest and return status counts."""
    rows = [status_row(task, results_root, plotting_root) for task in tasks]
    counts: dict[str, int] = {}
    for row in rows:
        status = str(row["status"])
        counts[status] = counts.get(status, 0) + 1

    plotting_root.mkdir(parents=True, exist_ok=True)
    manifest = plotting_root / "manifest.csv"
    temporary = manifest.with_name(f".{manifest.name}.{os.getpid()}.tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, manifest)
    atomic_write_json(
        plotting_root / "manifest_summary.json",
        {
            "total": len(tasks),
            "counts": counts,
            "updated_at_unix": time.time(),
        },
    )
    return manifest, counts


def export_existing(
    tasks: list[SweepTask],
    results_root: Path,
    plotting_root: Path,
) -> tuple[int, list[str]]:
    """Export portable artifacts from successful existing run directories."""
    exported = 0
    failures: list[str] = []
    for task in tasks:
        run_dir = task_directory(results_root, task)
        status = read_json(run_dir / "sweep_status.json")
        if status is None or status.get("exit_code") != 0:
            continue
        try:
            status["status"] = "success"
            export_task(
                task=task,
                results_root=results_root,
                plotting_root=plotting_root,
                status=status,
            )
            atomic_write_json(run_dir / "sweep_status.json", status)
            exported += 1
        except FileNotFoundError:
            continue
        except BaseException as error:
            failures.append(f"{task.key}: {error!r}")
    return exported, failures


def add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--profiles",
        type=parse_profile_values,
        default=DEFAULT_PROFILES,
        help="Comma-separated profiles (default: all four).",
    )
    parser.add_argument(
        "--q-values",
        type=parse_int_values,
        default=DEFAULT_Q_VALUES,
        help="Comma-separated train partition batch sizes.",
    )
    parser.add_argument(
        "--seeds",
        type=parse_int_values,
        default=DEFAULT_SEEDS,
        help="Comma-separated training/loader seeds.",
    )
    parser.add_argument(
        "--results-root",
        default="scripts/structural_coverage/results/cora_np64_sweep",
    )
    parser.add_argument(
        "--plotting-root",
        default=(
            "scripts/structural_coverage/results_for_plotting/cora_np64_sweep"
        ),
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the sweep command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="action", required=True)

    run_parser = subparsers.add_parser("run", help="Run missing sweep tasks.")
    add_common_arguments(run_parser)
    run_parser.add_argument("--workers", type=int, default=10)
    run_parser.add_argument(
        "--gpus",
        default=os.environ.get("SELECTED_GPUS", "0"),
        help="Visible physical GPU IDs; workers are assigned round-robin.",
    )
    run_parser.add_argument("--trainer", choices=("gpu", "cpu"), default="gpu")
    run_parser.add_argument(
        "--dataset", default="graph/cocitation_cora_for_partitioning"
    )
    run_parser.add_argument("--num-parts", type=int, default=64)
    run_parser.add_argument("--eval-q", type=int, default=64)
    run_parser.add_argument("--max-epochs", type=int, default=200)
    run_parser.add_argument("--data-split-seed", type=int, default=0)
    run_parser.add_argument("--train-prop", type=float, default=0.5)
    run_parser.add_argument("--hypergraph-k", type=int, default=1)
    run_parser.add_argument("--max-support-nodes", type=int, default=8)
    run_parser.add_argument("--audit-max-batches", type=int, default=10)
    run_parser.add_argument("--force", action="store_true")
    run_parser.add_argument("--dry-run", action="store_true")

    for action in ("plan", "status", "export"):
        action_parser = subparsers.add_parser(action)
        add_common_arguments(action_parser)

    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    args = parse_args(argv)
    project_root = Path(__file__).resolve().parents[2]
    tasks = build_tasks(args.profiles, args.q_values, args.seeds)
    results_root = Path(args.results_root).resolve()
    plotting_root = Path(args.plotting_root).resolve()

    if args.action == "plan":
        print(f"Total tasks: {len(tasks)}")
        for task in tasks:
            print(f"{task.index:03d}\t{task.key}")
        return 0

    if args.action == "status":
        manifest, counts = write_manifest(tasks, results_root, plotting_root)
        print(f"Status counts: {counts}")
        print(f"Manifest: {manifest}")
        return 0

    if args.action == "export":
        exported, failures = export_existing(
            tasks, results_root, plotting_root
        )
        manifest, counts = write_manifest(tasks, results_root, plotting_root)
        print(f"Exported {exported} task(s); status counts: {counts}")
        print(f"Manifest: {manifest}")
        for failure in failures:
            print(f"[export_failed] {failure}", file=sys.stderr)
        return 1 if failures else 0

    if args.workers <= 0:
        raise SystemExit("--workers must be positive")
    runner = SweepRunner(args, project_root)
    if args.dry_run:
        for task in tasks:
            command = command_for_task(
                task=task,
                args=args,
                project_root=project_root,
            )
            print(
                f"CUDA_VISIBLE_DEVICES={runner.gpu_for_task(task)} "
                f"{shlex.join(command)}"
            )
        return 0

    outcomes: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(runner.run_one, task): task for task in tasks
        }
        for future in as_completed(futures):
            task = futures[future]
            try:
                outcomes.append(future.result())
            except BaseException as error:
                runner.log(f"[orchestrator_failed] {task.key}: {error!r}")
                outcomes.append(
                    {"task": task.key, "status": "orchestrator_failed"}
                )

    manifest, counts = write_manifest(tasks, results_root, plotting_root)
    print(f"Status counts: {counts}")
    print(f"Manifest: {manifest}")
    failed = [
        outcome
        for outcome in outcomes
        if outcome.get("status") not in ("success", "skipped")
    ]
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
