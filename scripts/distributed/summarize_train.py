"""Summarize full-graph vs halo-distributed SCN training runs.

python scripts/distributed/summarize_train.py logs/distributed/train
"""

from __future__ import annotations

import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path


def main() -> None:
    """Print a per-dataset table and write ``summary.json``."""
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "logs/distributed/train")
    runs = defaultdict(dict)
    for path in sorted(root.glob("*_seed*_float32.json")):
        report = json.loads(path.read_text())
        if report.get("mode") == "check":
            continue
        args = report["args"]
        config = f"{report['mode']}_P{report['world']}"
        runs[(args["dataset"], config)][args["data_seed"]] = report

    summary = {}
    for (dataset, config), by_seed in sorted(runs.items()):
        tests = [r["best"]["test"] for r in by_seed.values()]
        vals = [r["best"]["val"] for r in by_seed.values()]
        row = {
            "seeds": sorted(by_seed),
            "test_mean": statistics.mean(tests),
            "test_std": statistics.stdev(tests) if len(tests) > 1 else 0.0,
            "val_mean": statistics.mean(vals),
            "best_epoch_mean": statistics.mean(
                r["best"]["epoch"] for r in by_seed.values()
            ),
        }
        if (
            0 in by_seed
        ):  # seed 0 ran alone on the node: clean timing and memory
            ref = by_seed[0]
            procs = ref["processes"]
            row.update(
                epoch_seconds=ref["mean_epoch_seconds"],
                epochs_run=len(ref["history"]),
                peak_gpu_gb_max=max(p["train_peak_gpu_bytes"] for p in procs)
                / 1e9,
                peak_gpu_gb_each=[
                    round(p["train_peak_gpu_bytes"] / 1e9, 2) for p in procs
                ],
                cells_each=[p["cells"] for p in procs],
                setup_seconds_max=max(p["setup_seconds"] for p in procs),
                lift_seconds_max=max(
                    p["timings"]["lift_seconds"] for p in procs
                ),
                partition_seconds=ref.get("partition_seconds", 0.0),
                forward_mb_sent_per_call_max=max(
                    p["bytes_sent_forward_total"] for p in procs
                )
                / ref["num_forward_calls"]
                / 1e6,
            )
        summary[f"{dataset}/{config}"] = row

    for key, row in summary.items():
        line = (
            f"{key:28s} test {row['test_mean']:.4f} ± {row['test_std']:.4f} "
            f"(n={len(row['seeds'])})"
        )
        if "epoch_seconds" in row:
            line += (
                f" | peak GPU {row['peak_gpu_gb_max']:.2f} GB/proc"
                f" | {row['epoch_seconds']:.3f} s/epoch"
                f" | setup {row['setup_seconds_max']:.1f}s"
                f" | sent {row['forward_mb_sent_per_call_max']:.1f} MB/fwd"
            )
        print(line)
    (root / "summary.json").write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
