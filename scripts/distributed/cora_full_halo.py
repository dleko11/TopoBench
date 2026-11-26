"""Halo-lifted, owner-partitioned SCN inference on Cora Full vs the full graph.

For each partition, every simulated worker lifts its core plus one-hop halo,
keeps the cells it owns, and runs SCN exchanging only ghost values. The script
checks that the owned cells are exactly the global clique complex and that
every intermediate matches the full-graph forward, then records replication,
ghost counts, lifting time and communication.

Example
-------
python scripts/distributed/cora_full_halo.py --parts 1 2 4 8 16 64 \
    --random-parts 8 --feature-dim 256 --dtype float64
"""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import torch
from torch_geometric.datasets import CoraFull

from topobench.distributed.halo import (
    build_halo_plan,
    canonical_edges,
    cluster_batch_counts,
    lift_clique_complex,
    lift_input_features,
    partition_nodes,
    plan_statistics,
    vertex_weights,
)
from topobench.distributed.operators import CommLog
from topobench.distributed.scn import (
    RANKS,
    SCNNodeClassifier,
    align_to_reference,
    build_reference_batch,
    cast_batch,
    compare_stages,
    distributed_stages,
    reference_cells,
    reference_stages,
)

STAGE_GROUPS = ("encoder", "layer", "backbone", "readout", "logits")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns
    -------
    argparse.Namespace
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--data-root", default="datasets/graph/cocitation/CoraFull"
    )
    parser.add_argument(
        "--parts", type=int, nargs="+", default=[1, 2, 4, 8, 16, 64]
    )
    parser.add_argument("--random-parts", type=int, nargs="*", default=[8])
    parser.add_argument(
        "--node-weight",
        choices=["none", "triangles", "cells"],
        default="none",
        help="METIS vertex weights; 'none' balances node counts.",
    )
    parser.add_argument(
        "--feature-dim",
        type=int,
        default=256,
        help="Keep the most frequent word columns; 0 keeps all 8,710.",
    )
    parser.add_argument(
        "--dtype", choices=["float32", "float64"], default="float64"
    )
    parser.add_argument("--hidden", type=int, default=32)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--checkpoint", default=None, help="Optional TBModel SCN checkpoint."
    )
    parser.add_argument("--cluster-parts", type=int, default=64)
    parser.add_argument("--cluster-q", type=int, default=16)
    parser.add_argument("--cluster-batches", type=int, default=5)
    parser.add_argument("--out", default="logs/distributed")
    return parser.parse_args()


def load_graph(root: str, feature_dim: int):
    """Load Cora Full and select feature columns.

    Parameters
    ----------
    root : str
        Dataset root.
    feature_dim : int
        Number of most frequent columns to keep, or 0 for all.

    Returns
    -------
    tuple
        Node features, symmetric edge index, labels.
    """
    data = CoraFull(root=root)[0]
    x = data.x
    if feature_dim:
        frequency = (x > 0).sum(dim=0)
        columns = (
            torch.argsort(frequency, descending=True)[:feature_dim]
            .sort()
            .values
        )
        x = x[:, columns]
    edges = canonical_edges(data.edge_index, data.num_nodes)
    return x, torch.cat([edges, edges.flip(0)], dim=1), data.y


def load_checkpoint(model: SCNNodeClassifier, path: str) -> None:
    """Load the encoder, backbone and readout weights of a TBModel checkpoint.

    Parameters
    ----------
    model : SCNNodeClassifier
        Model to load into.
    path : str
        Lightning checkpoint path.
    """
    state = torch.load(path, map_location="cpu", weights_only=False)[
        "state_dict"
    ]
    prefixes = ("feature_encoder.", "backbone.", "readout.")
    state = {k: v for k, v in state.items() if k.startswith(prefixes)}
    model.load_state_dict(state, strict=True)


def summarize_diffs(diffs: dict[str, float]) -> dict[str, float]:
    """Reduce per-stage differences to one maximum per stage group.

    Parameters
    ----------
    diffs : dict[str, float]
        Max absolute difference per stage.

    Returns
    -------
    dict[str, float]
        Max absolute difference per group, plus the overall maximum.
    """
    out = {
        f"diff_{group}": max(
            v for k, v in diffs.items() if k.startswith(group)
        )
        for group in STAGE_GROUPS
    }
    out["diff_max"] = max(diffs.values())
    return out


def run_partition(
    method, num_parts, args, graph, model, reference, cells, dtype, weights
):
    """Build, check and measure one partition.

    Returns
    -------
    dict
        One results row.
    """
    x_0, edge_index, _ = graph
    num_nodes = x_0.size(0)
    part = partition_nodes(
        edge_index,
        num_nodes,
        num_parts,
        method,
        args.seed,
        node_weight=weights if method == "metis" else None,
    )

    start = time.perf_counter()
    plan = build_halo_plan(edge_index, num_nodes, part)
    row = {
        "method": method,
        "parts": num_parts,
        "build_seconds": time.perf_counter() - start,
    }
    row.update(plan_statistics(plan))

    alignment = align_to_reference(plan, cells)  # raises unless exact
    row["structure_exact"] = True

    features = [f.to(dtype) for f in lift_input_features(plan, x_0)]
    comm = CommLog()
    halo_rows = sum(shard.halo.numel() for shard in plan.shards)
    comm.phase = "setup"
    comm.record("halo_input_features", halo_rows, x_0.size(1), 4, 0)
    stages = distributed_stages(model, plan, features, comm)

    row.update(summarize_diffs(compare_stages(stages, reference, alignment)))
    predictions = stages["logits"].argmax(dim=1)
    expected = reference["logits"][alignment[0]].argmax(dim=1)
    row["predictions_identical"] = bool(torch.equal(predictions, expected))
    row["prediction_mismatches"] = int((predictions != expected).sum())

    forward = comm.totals("forward")
    embedding_bytes = (
        sum(plan.num_cells(r) for r in RANKS)
        * args.hidden
        * stages["logits"].element_size()
    )
    row["forward_rows_moved"] = forward["rows"]
    row["forward_bytes"] = forward["bytes"]
    row["forward_bytes_per_embedding_set"] = forward["bytes"] / embedding_bytes
    row["setup_bytes"] = comm.totals("setup")["bytes"]
    row["exchange_steps"] = sum(
        entry["calls"]
        for (phase, operation), entry in comm.records.items()
        if phase == "forward"
        and ("expand" in operation or "fold" in operation)
    )
    del features, stages, plan
    gc.collect()
    return row


def main() -> None:
    """Run the Cora Full sweep and write results."""
    args = parse_args()
    dtype = getattr(torch, args.dtype)
    torch.manual_seed(args.seed)

    graph = load_graph(args.data_root, args.feature_dim)
    x_0, edge_index, y = graph
    num_nodes = x_0.size(0)
    edges = canonical_edges(edge_index, num_nodes)
    print(
        f"Cora Full: {num_nodes} nodes, {edges.size(1)} edges, {x_0.size(1)} features"
    )

    _, global_lift_seconds = lift_clique_complex(
        edges, torch.arange(num_nodes), 2
    )
    batch = cast_batch(build_reference_batch(x_0, edge_index, y), dtype)
    cells = reference_cells(batch)
    print(
        f"Global clique complex: {[c.size(0) for c in cells]} cells, lifted in {global_lift_seconds:.2f}s"
    )

    model = SCNNodeClassifier(
        x_0.size(1), args.hidden, int(y.max()) + 1, args.n_layers
    )
    if args.checkpoint:
        load_checkpoint(model, args.checkpoint)
    model = model.to(dtype).eval()
    reference = reference_stages(model, batch, clone=False)
    del batch
    gc.collect()

    weights = (
        None
        if args.node_weight == "none"
        else vertex_weights(edge_index, num_nodes, args.node_weight)
    )
    runs = [("metis", p) for p in args.parts] + [
        ("random", p) for p in args.random_parts
    ]
    rows = []
    for method, num_parts in runs:
        row = run_partition(
            method,
            num_parts,
            args,
            graph,
            model,
            reference,
            cells,
            dtype,
            weights,
        )
        row["global_lift_seconds"] = global_lift_seconds
        row["node_weight"] = args.node_weight if method == "metis" else "none"
        rows.append(row)
        print(
            f"{method:6s} P={num_parts:<3d} max|diff|={row['diff_max']:.2e} "
            f"same_preds={row['predictions_identical']} "
            f"held max(v,e,t)={row['held_0_max_frac']:.3f},"
            f"{row['held_1_max_frac']:.3f},{row['held_2_max_frac']:.3f} "
            f"imb(t)={row['owned_2_imbalance']:.2f} "
            f"halo/core={row['halo_ratio_mean']:.2f} (max {row['halo_ratio_max']:.2f}) "
            f"lift dup(e,t)={row['lift_duplication_1']:.2f},{row['lift_duplication_2']:.2f} "
            f"ghosts(v,e)={row['ghosts_0_total']:.0f},{row['ghosts_1_total']:.0f} "
            f"comm/fwd={row['forward_bytes'] / 1e6:.1f}MB "
            f"({row['forward_bytes_per_embedding_set']:.2f}x embeddings) "
            f"lift max/global={row['lift_seconds_max'] / global_lift_seconds:.2f}"
        )

    cluster_part = partition_nodes(
        edge_index, num_nodes, args.cluster_parts, "metis"
    )
    batches = cluster_batch_counts(
        edge_index,
        num_nodes,
        cluster_part,
        args.cluster_q,
        args.cluster_batches,
        seed=args.seed,
    )
    cluster = {
        "parts": args.cluster_parts,
        "q": args.cluster_q,
        "batches": batches,
        "mean_cells": [
            sum(b[r] for b in batches) / len(batches) for r in RANKS
        ],
        "global_cells": [c.size(0) for c in cells],
    }
    print(
        f"Cluster-TNN batch ({args.cluster_q}/{args.cluster_parts} clusters): "
        f"mean cells {[round(v) for v in cluster['mean_cells']]} "
        f"of global {cluster['global_cells']}"
    )

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = (
        f"f{args.feature_dim or x_0.size(1)}_{args.dtype}_L{args.n_layers}"
        f"_w{args.node_weight}"
    )
    path = out_dir / f"cora_full_halo_{tag}.json"
    path.write_text(
        json.dumps(
            {"args": vars(args), "rows": rows, "cluster_tnn": cluster},
            indent=2,
        )
    )
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
