#!/usr/bin/env python3
"""Count full-graph higher-order structures without lifting features."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import time
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

import hydra
import igraph as ig
import networkx as nx
import numpy as np
import torch
import wandb

from topobench.utils.config_resolvers import register_all_resolvers

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = Path("outputs/full_graph_structure_counts.csv")
COUNT_TYPES = ("simplices", "cells")
COUNT_STAGE_INDEX = {
    "graph": 1,
    "simplices": 2,
    "cells": 3,
    "complete": 4,
}

DATASET_CONFIGS = {
    "amazon_ratings": "graph/amazon_ratings",
    "questions": "graph/questions",
    "cora_full": "graph/cocitation_cora_full",
    "coauthor_physics": "graph/coauthor_physics",
    "reddit": "graph/reddit",
    "ogbn_products": "graph/ogbn_products_for_partitioning",
}
DEFAULT_DATASETS = tuple(DATASET_CONFIGS)

OUTPUT_FIELDS = (
    "dataset",
    "num_nodes",
    "num_edges",
    "num_hyperedges",
    "num_2_cells",
    "num_2_simplices",
    "load_time_sec",
    "cell_count_time_sec",
    "simplex_count_time_sec",
    "max_cell_length",
    "hypergraph_definition",
    "cell_definition",
    "simplicial_definition",
)


def build_simple_undirected_graph(
    edge_index: torch.Tensor,
    num_nodes: int,
) -> nx.Graph:
    """Build the simple undirected graph used by benchmark liftings."""
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError("edge_index must have shape [2, num_edges].")

    edge_index_cpu = edge_index.detach().cpu()
    sources = edge_index_cpu[0].tolist()
    targets = edge_index_cpu[1].tolist()
    graph = nx.Graph()
    graph.add_nodes_from(range(num_nodes))
    graph.add_edges_from(zip(sources, targets, strict=True))
    return graph


def count_triangles_from_edges(
    num_nodes: int,
    edges: Iterable[tuple[int, int]],
) -> int:
    """Count graph triangles without returning their node tuples."""
    edge_list = edges if isinstance(edges, list) else list(edges)
    clique_graph = ig.Graph(
        n=num_nodes,
        edges=edge_list,
        directed=False,
    )
    clique_graph.simplify(multiple=True, loops=True)
    degrees = np.asarray(clique_graph.degree(), dtype=np.int64)
    connected_triples = sum(
        int(degree) * (int(degree) - 1) // 2 for degree in degrees
    )
    transitivity = float(clique_graph.transitivity_undirected(mode="zero"))
    return int(round(transitivity * connected_triples / 3))


def count_graph_structures(
    edge_index: torch.Tensor,
    num_nodes: int,
    max_cell_length: int = 9,
    count_types: tuple[str, ...] = COUNT_TYPES,
    checkpoint: Callable[[str, dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Count the structures used by the full-graph benchmark liftings."""
    unknown = set(count_types) - set(COUNT_TYPES)
    if unknown:
        raise ValueError(f"Unknown count types: {', '.join(sorted(unknown))}")
    if not count_types:
        raise ValueError("At least one count type is required.")

    graph = build_simple_undirected_graph(edge_index, num_nodes)
    result = {
        "num_nodes": num_nodes,
        "num_edges": graph.number_of_edges(),
        "num_hyperedges": num_nodes,
        "num_2_cells": None,
        "num_2_simplices": None,
        "cell_count_time_sec": None,
        "simplex_count_time_sec": None,
        "max_cell_length": max_cell_length,
        "hypergraph_definition": "one k=1 neighborhood per node",
        "cell_definition": (
            f"cycle-basis cells of length 2..{max_cell_length}"
        ),
        "simplicial_definition": "triangles from clique lifting",
    }
    if checkpoint is not None:
        checkpoint("graph", dict(result))

    if "simplices" in count_types:
        simplex_started = time.perf_counter()
        if "cells" not in count_types:
            edges = list(graph.edges())
            del graph
            gc.collect()
        else:
            edges = graph.edges()
        result["num_2_simplices"] = count_triangles_from_edges(
            num_nodes,
            edges,
        )
        result["simplex_count_time_sec"] = (
            time.perf_counter() - simplex_started
        )
        if checkpoint is not None:
            checkpoint("simplices", dict(result))

    if "cells" in count_types:
        cell_started = time.perf_counter()
        cycles = nx.cycle_basis(graph)
        result["num_2_cells"] = sum(
            1 for cycle in cycles if 1 < len(cycle) <= max_cell_length
        )
        result["cell_count_time_sec"] = time.perf_counter() - cell_started
        del cycles
        if checkpoint is not None:
            checkpoint("cells", dict(result))

    return result


def _parse_datasets(value: str) -> tuple[str, ...]:
    datasets = tuple(item.strip() for item in value.split(",") if item.strip())
    unknown = sorted(set(datasets) - set(DATASET_CONFIGS))
    if unknown:
        raise argparse.ArgumentTypeError(
            f"Unknown datasets: {', '.join(unknown)}"
        )
    if not datasets:
        raise argparse.ArgumentTypeError("At least one dataset is required.")
    return datasets


def _parse_count_types(value: str) -> tuple[str, ...]:
    requested = {item.strip() for item in value.split(",") if item.strip()}
    unknown = sorted(requested - set(COUNT_TYPES))
    if unknown:
        raise argparse.ArgumentTypeError(
            f"Unknown count types: {', '.join(unknown)}"
        )
    selected = tuple(item for item in COUNT_TYPES if item in requested)
    if not selected:
        raise argparse.ArgumentTypeError(
            "At least one count type is required."
        )
    return selected


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets",
        type=_parse_datasets,
        default=DEFAULT_DATASETS,
        help="Comma-separated dataset aliases.",
    )
    parser.add_argument("--max-cell-length", type=int, default=9)
    parser.add_argument(
        "--count-types",
        type=_parse_count_types,
        default=COUNT_TYPES,
        help="Comma-separated expensive counts: simplices,cells.",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--wandb-project")
    parser.add_argument("--wandb-entity", default="topobench-scalability")
    return parser


def _load_dataset(alias: str):
    config_dir = str(REPO_ROOT / "configs")
    with hydra.initialize_config_dir(
        version_base="1.3",
        config_dir=config_dir,
        job_name="count_full_graph_structures",
    ):
        cfg = hydra.compose(
            config_name="run",
            overrides=[
                f"dataset={DATASET_CONFIGS[alias]}",
                "model=graph/gcn",
            ],
        )
    loader = hydra.utils.instantiate(cfg.dataset.loader)
    dataset, _ = loader.load()
    return dataset[0]


def _write_csv(rows: list[dict[str, Any]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(output: Path) -> list[dict[str, Any]]:
    if not output.exists():
        return []
    with output.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _upsert_dataset_row(
    rows: list[dict[str, Any]], row: dict[str, Any]
) -> None:
    existing = next(
        (item for item in rows if item.get("dataset") == row["dataset"]),
        None,
    )
    updates = {
        key: value for key, value in row.items() if value not in (None, "")
    }
    if existing is None:
        rows.append(updates)
    else:
        existing.update(updates)


def main() -> None:
    args = _build_parser().parse_args()
    if args.max_cell_length < 2:
        raise ValueError("--max-cell-length must be at least 2.")

    os.environ.setdefault("PROJECT_ROOT", str(REPO_ROOT))
    register_all_resolvers()
    rows = _read_csv(args.output)
    for alias in args.datasets:
        count_label = "_".join(args.count_types)
        run_name = f"full_graph_structure_counts_{alias}"
        if args.count_types != COUNT_TYPES:
            run_name = f"{run_name}_{count_label}"
        run = None
        if args.wandb_project:
            run = wandb.init(
                entity=args.wandb_entity,
                project=args.wandb_project,
                name=run_name,
                job_type="structure_count",
                config={
                    "dataset": alias,
                    "count_types": list(args.count_types),
                    "max_cell_length": args.max_cell_length,
                },
                reinit=True,
            )

        try:
            load_started = time.perf_counter()
            data = _load_dataset(alias)
            load_time = time.perf_counter() - load_started
            if run is not None:
                run.summary.update(
                    {
                        "last_completed_stage": "dataset_load",
                        "load_time_sec": load_time,
                    }
                )
                run.log({"count/checkpoint": 0})

            def checkpoint(
                stage: str,
                result: dict[str, Any],
                dataset_alias: str = alias,
                dataset_load_time: float = load_time,
                current_run: Any = run,
            ) -> None:
                row = {
                    "dataset": dataset_alias,
                    **result,
                    "load_time_sec": dataset_load_time,
                }
                _upsert_dataset_row(rows, row)
                _write_csv(rows, args.output)
                print(
                    json.dumps(
                        {"stage": stage, **row},
                        sort_keys=True,
                    ),
                    flush=True,
                )
                if current_run is not None:
                    summary = {
                        key: value
                        for key, value in row.items()
                        if value is not None
                    }
                    summary["last_completed_stage"] = stage
                    current_run.summary.update(summary)
                    current_run.log(
                        {
                            "count/checkpoint": COUNT_STAGE_INDEX[stage],
                        }
                    )

            row = {
                "dataset": alias,
                **count_graph_structures(
                    data.edge_index,
                    int(data.num_nodes),
                    args.max_cell_length,
                    args.count_types,
                    checkpoint,
                ),
                "load_time_sec": load_time,
            }
            _upsert_dataset_row(rows, row)
            _write_csv(rows, args.output)
            print(json.dumps(row, sort_keys=True), flush=True)
            if run is not None:
                run.summary["last_completed_stage"] = "complete"
                run.log({"count/checkpoint": COUNT_STAGE_INDEX["complete"]})
        except Exception:
            if run is not None:
                run.finish(exit_code=1)
            raise
        else:
            if run is not None:
                run.finish()

    print(args.output)


if __name__ == "__main__":
    main()
