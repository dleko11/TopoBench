#!/usr/bin/env python3
"""Measure exact structure spans under a fixed graph partition."""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import os
import time
from collections import Counter
from itertools import chain
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
import torch
import wandb
from torch_geometric.data import Data
from torch_geometric.loader import ClusterData

from scripts.partitioning.count_clique_simplices import (
    DATASET_CONFIGS,
    REPO_ROOT,
    _load_dataset,
    build_simple_undirected_graph,
    count_triangles_from_edges,
)
from topobench.utils.config_resolvers import register_all_resolvers

FAMILIES = ("hypergraph", "cell", "simplicial")
FAMILY_DEFINITIONS = {
    "hypergraph": "unique closed 1-hop neighborhood hyperedges",
    "cell": "cycle-basis cells of length 2..{max_cell_length}",
    "simplicial": "2-simplices from clique triangles",
}
HISTOGRAM_FIELDS = (
    "dataset",
    "family",
    "definition",
    "num_parts",
    "num_nodes",
    "num_edges",
    "span",
    "count",
    "fraction",
    "structure_count",
    "count_time_sec",
)
CURVE_FIELDS = (
    "dataset",
    "family",
    "definition",
    "num_parts",
    "q",
    "q_over_k",
    "q_over_k_percent",
    "observable_count",
    "structure_count",
    "observable_fraction",
)
THRESHOLD_FIELDS = (
    "dataset",
    "family",
    "num_parts",
    "structure_count",
    "q_95",
    "q_95_over_k_percent",
    "q_99",
    "q_99_over_k_percent",
)


def topology_fingerprint(edge_index: torch.Tensor, num_nodes: int) -> str:
    """Return an order-sensitive fingerprint for a loaded graph topology."""
    array = edge_index.detach().cpu().contiguous().numpy()
    digest = hashlib.sha256()
    digest.update(np.asarray([num_nodes, array.shape[1]], dtype=np.int64))
    digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


def partition_labels_from_partition(
    partptr: np.ndarray,
    node_perm: np.ndarray,
) -> np.ndarray:
    """Map original node IDs to their cluster IDs."""
    if partptr.ndim != 1 or node_perm.ndim != 1:
        raise ValueError("partptr and node_perm must be one-dimensional.")
    if len(partptr) < 2 or int(partptr[-1]) != len(node_perm):
        raise ValueError("Partition pointers do not cover every node.")

    permuted_labels = np.repeat(
        np.arange(len(partptr) - 1, dtype=np.int32),
        np.diff(partptr),
    )
    labels = np.empty(len(node_perm), dtype=np.int32)
    labels[node_perm] = permuted_labels
    return labels


def load_or_build_partition_labels(
    *,
    dataset: str,
    edge_index: torch.Tensor,
    num_nodes: int,
    num_parts: int,
    output_dir: Path,
    force: bool,
) -> np.ndarray:
    """Load cached labels or build the benchmark's Cluster-GCN partition."""
    labels_path = output_dir / "partition_labels.npy"
    metadata_path = output_dir / "partition_metadata.json"
    fingerprint = topology_fingerprint(edge_index, num_nodes)
    expected = {
        "dataset": dataset,
        "num_nodes": num_nodes,
        "num_input_edges": int(edge_index.shape[1]),
        "num_parts": num_parts,
        "recursive": False,
        "topology_sha256": fingerprint,
    }

    if labels_path.exists() or metadata_path.exists():
        if force:
            pass
        elif not labels_path.exists() or not metadata_path.exists():
            raise RuntimeError(
                "The partition cache is incomplete. Rerun with "
                "--force-partition to replace it."
            )
        else:
            with metadata_path.open(encoding="utf-8") as handle:
                observed = json.load(handle)
            if observed != expected:
                raise RuntimeError(
                    "The partition cache does not match the loaded graph. "
                    "Rerun with --force-partition to replace it."
                )
            labels = np.load(labels_path)
            if labels.shape != (num_nodes,):
                raise RuntimeError("Cached partition labels have wrong shape.")
            return labels.astype(np.int32, copy=False)

    structural_data = Data(edge_index=edge_index, num_nodes=num_nodes)
    clustered = ClusterData(
        structural_data,
        num_parts=num_parts,
        recursive=False,
        keep_inter_cluster_edges=False,
        sparse_format="csr",
        save_dir=None,
        log=True,
    )
    partition = clustered.partition
    labels = partition_labels_from_partition(
        partition.partptr.detach().cpu().numpy(),
        partition.node_perm.detach().cpu().numpy(),
    )
    del clustered, partition, structural_data
    gc.collect()

    output_dir.mkdir(parents=True, exist_ok=True)
    labels_tmp = labels_path.with_suffix(".npy.tmp")
    with labels_tmp.open("wb") as handle:
        np.save(handle, labels)
    labels_tmp.replace(labels_path)
    metadata_tmp = metadata_path.with_suffix(".json.tmp")
    with metadata_tmp.open("w", encoding="utf-8") as handle:
        json.dump(expected, handle, indent=2, sort_keys=True)
        handle.write("\n")
    metadata_tmp.replace(metadata_path)
    return labels


def hypergraph_span_histogram(
    graph: nx.Graph,
    partition_labels: np.ndarray,
) -> Counter[int]:
    """Count spans of unique closed one-hop neighborhoods."""
    histogram: Counter[int] = Counter()
    seen: set[tuple[int, ...]] = set()
    for node in graph.nodes:
        support = tuple(sorted(chain((node,), graph.adj[node])))
        if support in seen:
            continue
        seen.add(support)
        span = len({int(partition_labels[index]) for index in support})
        histogram[span] += 1
    return histogram


def cell_span_histogram(
    graph: nx.Graph,
    partition_labels: np.ndarray,
    max_cell_length: int,
) -> Counter[int]:
    """Count spans of the cycle-basis cells used by the benchmark."""
    histogram: Counter[int] = Counter()
    for cycle in nx.cycle_basis(graph):
        if len(cycle) == 1 or len(cycle) > max_cell_length:
            continue
        span = len({int(partition_labels[node]) for node in cycle})
        histogram[span] += 1
    return histogram


def simplicial_span_histogram(
    graph: nx.Graph,
    partition_labels: np.ndarray,
) -> Counter[int]:
    """Count triangle spans without materializing triangle node tuples."""
    num_nodes = graph.number_of_nodes()
    total = count_triangles_from_edges(num_nodes, graph.edges())

    within_edges = [
        (u, v)
        for u, v in graph.edges()
        if partition_labels[u] == partition_labels[v]
    ]
    span_one = count_triangles_from_edges(num_nodes, within_edges)
    del within_edges
    gc.collect()

    cross_edges = [
        (u, v)
        for u, v in graph.edges()
        if partition_labels[u] != partition_labels[v]
    ]
    span_three = count_triangles_from_edges(num_nodes, cross_edges)
    del cross_edges
    gc.collect()

    span_two = total - span_one - span_three
    if span_two < 0:
        raise RuntimeError(
            "Triangle span decomposition produced a negative count."
        )
    return Counter(
        {
            span: count
            for span, count in (
                (1, span_one),
                (2, span_two),
                (3, span_three),
            )
            if count
        }
    )


def observability_rows(
    histogram_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Build the complete q curve and its 95% and 99% thresholds."""
    if not histogram_rows:
        raise ValueError(
            "A structure histogram must contain at least one row."
        )
    first = histogram_rows[0]
    num_parts = int(first["num_parts"])
    total = int(first["structure_count"])
    counts = np.zeros(num_parts + 1, dtype=np.int64)
    for row in histogram_rows:
        span = int(row["span"])
        if not 1 <= span <= num_parts:
            raise ValueError(
                f"Invalid structure span {span} for K={num_parts}."
            )
        counts[span] += int(row["count"])
    if int(counts.sum()) != total:
        raise ValueError("Histogram counts do not match structure_count.")

    cumulative = np.cumsum(counts)
    rows = []
    for q in range(1, num_parts + 1):
        observable = int(cumulative[q])
        rows.append(
            {
                "dataset": first["dataset"],
                "family": first["family"],
                "definition": first["definition"],
                "num_parts": num_parts,
                "q": q,
                "q_over_k": q / num_parts,
                "q_over_k_percent": 100 * q / num_parts,
                "observable_count": observable,
                "structure_count": total,
                "observable_fraction": observable / total,
            }
        )

    def threshold_q(target: float) -> int:
        return next(
            q
            for q in range(1, num_parts + 1)
            if cumulative[q] / total >= target
        )

    q_95 = threshold_q(0.95)
    q_99 = threshold_q(0.99)
    threshold = {
        "dataset": first["dataset"],
        "family": first["family"],
        "num_parts": num_parts,
        "structure_count": total,
        "q_95": q_95,
        "q_95_over_k_percent": 100 * q_95 / num_parts,
        "q_99": q_99,
        "q_99_over_k_percent": 100 * q_99 / num_parts,
    }
    return rows, threshold


def write_csv(
    rows: list[dict[str, Any]],
    path: Path,
    fieldnames: tuple[str, ...],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def read_csv(path: Path) -> list[dict[str, Any]]:
    """Read one CSV into dictionaries."""
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _histogram_rows(
    *,
    dataset: str,
    family: str,
    definition: str,
    num_parts: int,
    num_nodes: int,
    num_edges: int,
    histogram: Counter[int],
    count_time_sec: float,
) -> list[dict[str, Any]]:
    total = sum(histogram.values())
    if total == 0:
        raise RuntimeError(f"No {family} structures were found for {dataset}.")
    return [
        {
            "dataset": dataset,
            "family": family,
            "definition": definition,
            "num_parts": num_parts,
            "num_nodes": num_nodes,
            "num_edges": num_edges,
            "span": span,
            "count": count,
            "fraction": count / total,
            "structure_count": total,
            "count_time_sec": count_time_sec,
        }
        for span, count in sorted(histogram.items())
    ]


def _log_wandb(
    *,
    project: str,
    entity: str,
    rows: list[dict[str, Any]],
    histogram_path: Path,
) -> None:
    curve, threshold = observability_rows(rows)
    first = rows[0]
    run = wandb.init(
        entity=entity,
        project=project,
        name=(
            f"structural_observability_{first['dataset']}_"
            f"{first['family']}_k{first['num_parts']}"
        ),
        job_type="structure_span_count",
        config={
            "dataset": first["dataset"],
            "family": first["family"],
            "num_parts": first["num_parts"],
            "definition": first["definition"],
        },
        reinit=True,
    )
    run.summary.update(
        {
            "structure_count": first["structure_count"],
            "count_time_sec": first["count_time_sec"],
            **{
                key: value
                for key, value in threshold.items()
                if key.startswith("q_")
            },
        }
    )
    run.log(
        {
            "observable_fraction": wandb.Table(
                columns=["q", "q_over_k_percent", "observable_fraction"],
                data=[
                    [
                        row["q"],
                        row["q_over_k_percent"],
                        row["observable_fraction"],
                    ]
                    for row in curve
                ],
            )
        }
    )
    artifact = wandb.Artifact(
        name=(
            f"structural-observability-{first['dataset']}-"
            f"{first['family']}-k{first['num_parts']}"
        ),
        type="structural-observability",
        metadata={
            "dataset": first["dataset"],
            "family": first["family"],
            "num_parts": first["num_parts"],
            "structure_count": first["structure_count"],
        },
    )
    artifact.add_file(str(histogram_path), name=histogram_path.name)
    run.log_artifact(artifact, aliases=["latest"])
    run.finish()


def _parse_families(value: str) -> tuple[str, ...]:
    families = tuple(item.strip() for item in value.split(",") if item.strip())
    unknown = sorted(set(families) - set(FAMILIES))
    if unknown:
        raise argparse.ArgumentTypeError(
            f"Unknown structure families: {', '.join(unknown)}"
        )
    if not families:
        raise argparse.ArgumentTypeError("At least one family is required.")
    return families


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=DATASET_CONFIGS, required=True)
    parser.add_argument("--num-parts", type=int, required=True)
    parser.add_argument(
        "--families",
        type=_parse_families,
        default=FAMILIES,
        help="Comma-separated subset of hypergraph,cell,simplicial.",
    )
    parser.add_argument("--max-cell-length", type=int, default=9)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/structural_observability"),
    )
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--force-partition", action="store_true")
    parser.add_argument("--wandb-project")
    parser.add_argument("--wandb-entity", default="topobench-scalability")
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    if args.num_parts < 1:
        raise ValueError("--num-parts must be positive.")
    if args.max_cell_length < 2:
        raise ValueError("--max-cell-length must be at least 2.")

    os.environ.setdefault("PROJECT_ROOT", str(REPO_ROOT))
    register_all_resolvers()
    result_dir = args.output_dir / f"{args.dataset}_k{args.num_parts}"
    upload_key = hashlib.sha256(
        f"{args.wandb_entity}/{args.wandb_project}".encode()
    ).hexdigest()[:12]

    def output_path(family: str) -> Path:
        return result_dir / f"span_histogram_{family}.csv"

    def upload_marker(family: str) -> Path:
        return result_dir / f".wandb_{upload_key}_{family}.uploaded"

    pending = [
        family
        for family in args.families
        if not (args.resume and output_path(family).exists())
    ]
    if args.wandb_project:
        for family in args.families:
            path = output_path(family)
            if family in pending or upload_marker(family).exists():
                continue
            rows = read_csv(path)
            _log_wandb(
                project=args.wandb_project,
                entity=args.wandb_entity,
                rows=rows,
                histogram_path=path,
            )
            upload_marker(family).touch()
    if not pending:
        print(f"All requested results already exist in {result_dir}")
        return

    loaded = _load_dataset(args.dataset)
    num_nodes = int(loaded.num_nodes)
    if args.num_parts > num_nodes:
        raise ValueError("--num-parts cannot exceed the number of nodes.")
    edge_index = loaded.edge_index.detach().cpu()
    del loaded
    gc.collect()

    labels = load_or_build_partition_labels(
        dataset=args.dataset,
        edge_index=edge_index,
        num_nodes=num_nodes,
        num_parts=args.num_parts,
        output_dir=result_dir,
        force=args.force_partition,
    )
    graph = build_simple_undirected_graph(edge_index, num_nodes)
    num_edges = graph.number_of_edges()
    del edge_index
    gc.collect()

    for family in pending:
        started = time.perf_counter()
        if family == "hypergraph":
            histogram = hypergraph_span_histogram(graph, labels)
        elif family == "cell":
            histogram = cell_span_histogram(
                graph,
                labels,
                args.max_cell_length,
            )
        elif family == "simplicial":
            histogram = simplicial_span_histogram(graph, labels)
        else:
            raise ValueError(f"Unsupported structure family: {family}")
        elapsed = time.perf_counter() - started
        definition = FAMILY_DEFINITIONS[family].format(
            max_cell_length=args.max_cell_length
        )
        rows = _histogram_rows(
            dataset=args.dataset,
            family=family,
            definition=definition,
            num_parts=args.num_parts,
            num_nodes=num_nodes,
            num_edges=num_edges,
            histogram=histogram,
            count_time_sec=elapsed,
        )
        output = output_path(family)
        write_csv(rows, output, HISTOGRAM_FIELDS)
        summary = {
            "dataset": args.dataset,
            "family": family,
            "num_parts": args.num_parts,
            "structure_count": rows[0]["structure_count"],
            "count_time_sec": elapsed,
            "output": str(output),
        }
        print(json.dumps(summary, sort_keys=True), flush=True)
        if args.wandb_project:
            _log_wandb(
                project=args.wandb_project,
                entity=args.wandb_entity,
                rows=rows,
                histogram_path=output,
            )
            upload_marker(family).touch()
        del histogram, rows
        gc.collect()


if __name__ == "__main__":
    main()
