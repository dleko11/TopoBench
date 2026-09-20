"""Run frozen-reference support recovery for the four local paper datasets.

This is a structural diagnostic. It does not repeatedly execute a local lifting,
and the cellular curve denotes support availability, not exact local-basis identity.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
from matplotlib.ticker import PercentFormatter

from scripts.structural_coverage.plot_appendix_sweep_results_classic import (
    configure_style,
    style_axis,
)
from scripts.structural_coverage.recovery_core import (
    cycle_basis_references,
    neighbourhood_references,
    triangle_references,
)
from scripts.structural_coverage.support_recovery_multidataset import (
    analyze_q_sweep,
    analyze_recovery,
    labels_from_partition,
    validate_ordered_graph,
)


DATASETS = {
    "cora_full": {"K": 32, "q": 4, "nodes": 19793,
                  "counts": (19793, 24306, 48386),
                  "processed": "datasets/graph/cocitation/cora/processed/data_undirected.pt"},
    "amazon_ratings": {"K": 32, "q": 8, "nodes": 24492,
                       "counts": (24492, 63667, 110765),
                       "processed": "datasets/graph/heterophilic/amazon_ratings/processed/data.pt"},
    "questions": {"K": 500, "q": 50, "nodes": 48921,
                  "counts": (48921, 24405, 110209),
                  "processed": "datasets/graph/heterophilic/questions/processed/data.pt"},
    "coauthor_physics": {"K": 2000, "q": 20, "nodes": 34493,
                         "counts": (34493, 102152, 468550)},
}
Q_GRIDS = {
    "cora_full": [1, 2, 4, 8, 16, 32],
    "amazon_ratings": [1, 2, 4, 8, 16, 32],
    "questions": [1, 2, 5, 10, 20, 50, 100, 250, 500],
    "coauthor_physics": [1, 2, 4, 10, 20, 40, 100, 200, 500, 1000, 2000],
}
FAMILIES = ("hypergraph", "cellular", "simplicial")
COLORS = {"hypergraph": "#0072B2", "cellular": "#E69F00",
          "simplicial": "#CC79A7"}
LABELS = {"hypergraph": "Hyperedges", "cellular": "Cellular 2-cells",
          "simplicial": "Simplicial 2-cells"}
TITLES = {"cora_full": "Cora Full", "amazon_ratings": "Amazon Ratings",
          "questions": "Questions", "coauthor_physics": "Coauthor Physics"}


def edge_fingerprint(edge_index: np.ndarray) -> str:
    """Hash ordered directed edge entries using the saved-manifest convention."""
    return hashlib.sha256(np.asarray(edge_index, dtype=np.int64).tobytes()).hexdigest()


def load_graph(dataset: str, *, source_root: Path, coauthor_root: Path,
               cora_edge_index: Path):
    """Read the same processed graph used by the structural count scripts."""
    if dataset == "coauthor_physics":
        from torch_geometric.datasets import Coauthor

        data = Coauthor(root=str(coauthor_root), name="Physics")[0]
        return data.edge_index.detach().cpu().numpy(), int(data.num_nodes)
    if dataset == "cora_full":
        return np.load(cora_edge_index), DATASETS[dataset]["nodes"]
    path = source_root / DATASETS[dataset]["processed"]
    payload = torch.load(path, map_location="cpu", weights_only=False)
    data = payload[0]
    edges = data["edge_index"]
    return edges.detach().cpu().numpy(), int(data["x"].shape[0])


def prepare_coauthor_partition(
    edge_index: np.ndarray, *, num_nodes: int, output_dir: Path
) -> tuple[np.ndarray, dict]:
    """Create one deterministic local METIS assignment and fixed train split."""
    from torch_geometric.data import Data
    from torch_geometric.loader import ClusterData

    partition_dir = output_dir / "partition"
    partition_dir.mkdir(parents=True, exist_ok=True)
    partptr_path = partition_dir / "partptr.npy"
    permutation_path = partition_dir / "perm_to_global.npy"
    mask_path = partition_dir / "train_mask_perm.npy"
    manifest_path = partition_dir / "manifest.json"
    if all(path.exists() for path in (partptr_path, permutation_path,
                                      mask_path, manifest_path)):
        manifest = json.loads(manifest_path.read_text())
        validate_ordered_graph(edge_index, num_nodes=num_nodes, manifest=manifest)
        labels = labels_from_partition(
            partptr=np.load(partptr_path),
            perm_to_global=np.load(permutation_path),
            train_mask_perm=np.load(mask_path),
        )
        return labels, manifest

    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)
    partition = ClusterData(
        Data(edge_index=torch.as_tensor(edge_index), num_nodes=num_nodes),
        num_parts=2000, recursive=False, keep_inter_cluster_edges=False,
        sparse_format="csr", save_dir=None, log=False,
    ).partition
    partptr = partition.partptr.detach().cpu().numpy()
    permutation = partition.node_perm.detach().cpu().numpy()
    rng = np.random.RandomState(42)
    split = rng.permutation(num_nodes)
    train_mask = np.zeros(num_nodes, dtype=bool)
    train_mask[split[:int(num_nodes * 0.7)]] = True
    train_mask_perm = train_mask[permutation]
    labels = labels_from_partition(
        partptr=partptr, perm_to_global=permutation,
        train_mask_perm=train_mask_perm,
    )
    manifest = {
        "dataset": "coauthor_physics",
        "num_nodes": num_nodes,
        "num_edges_directed": int(edge_index.shape[1]),
        "edge_order_sha256": edge_fingerprint(edge_index),
        "num_parts": 2000,
        "active_cluster_count": 2000,
        "partition_source": "new_local_metis",
        "partition_seed_requested": 0,
        "split_seed": 0,
        "global_split_seed": 42,
        "train_prop": 0.7,
        "recursive": False,
        "keep_inter_cluster_edges": False,
        "sparse_format": "csr",
    }
    np.save(partptr_path, partptr)
    np.save(permutation_path, permutation)
    np.save(mask_path, train_mask_perm)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return labels, manifest


def load_saved_partition(
    dataset: str, *, partition_root: Path, edge_index: np.ndarray,
    num_nodes: int
) -> tuple[np.ndarray, dict]:
    """Use the earlier saved METIS assignment with an ordered-graph check."""
    K = DATASETS[dataset]["K"]
    directory = partition_root / f"k{K}" / dataset
    manifest = json.loads((directory / "manifest.json").read_text())
    validate_ordered_graph(edge_index, num_nodes=num_nodes, manifest=manifest)
    memmap = directory / "structural_handle" / "perm_memmap"
    labels = labels_from_partition(
        partptr=np.load(memmap / "partptr.npy"),
        perm_to_global=np.load(memmap / "perm_to_global.npy"),
        train_mask_perm=np.load(memmap / "train_mask_perm.npy"),
    )
    if len(np.unique(labels)) != K:
        raise ValueError("saved partition does not have expected K")
    return labels, manifest


def make_plot(result: dict, *, dataset: str):
    """Use the approved recovery-plot visual language for one dataset."""
    configure_style()
    horizon = result["epochs"]
    epochs = np.arange(horizon + 1)
    fig, ax = plt.subplots(figsize=(5.3, 3.5))
    marker_epochs = [0, 10, 25, 50, 100, 150, 200]
    marker_epochs = [epoch for epoch in marker_epochs if epoch <= horizon]
    for family in FAMILIES:
        ax.plot(
            epochs, result["families"][family]["coverage_mean"],
            color=COLORS[family], linestyle="--", linewidth=1.5,
            marker="o", markevery=marker_epochs, markersize=3.8,
            markerfacecolor="white", markeredgecolor=COLORS[family],
            markeredgewidth=1.0, label=LABELS[family],
        )
    ax.set_xlim(0, horizon)
    ax.set_ylim(-0.015, 1.025)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Full-graph references")
    style_axis(ax)
    ax.grid(axis="y", color="#DCE1E7", linewidth=0.55, linestyle=":")
    ax.legend(loc="upper right" if dataset == "coauthor_physics" else "lower right",
              frameon=False)
    ax.set_title(f"Cumulative support availability  |  "
                 f"K={result['K']}, q={result['q']}", fontsize=8.1, pad=7)
    fig.suptitle(TITLES[dataset], fontsize=10.0, y=0.98)
    repetitions = len(result["seeds"])
    fig.text(0.5, 0.025, f"{repetitions} reshufflings · fixed METIS partition",
             ha="center", fontsize=6.5)
    fig.subplots_adjust(left=0.13, right=0.98, bottom=0.20, top=0.80)
    return fig


def make_q_plot(sweep: dict[int, dict], *, dataset: str):
    """Plot epoch-end empirical recovery against mini-batch cluster count."""
    configure_style()
    q_values = sorted(sweep)
    first = sweep[q_values[0]]
    horizon = first["epochs"]
    fig, ax = plt.subplots(figsize=(5.3, 3.5))
    for family in FAMILIES:
        means = [sweep[q]["families"][family]["coverage_mean"][-1]
                 for q in q_values]
        spreads = [sweep[q]["families"][family]["coverage_sample_sd"][-1] or 0
                   for q in q_values]
        color = COLORS[family]
        ax.plot(
            q_values, means, color=color, linestyle="--", linewidth=1.5,
            marker="o", markersize=3.8, markerfacecolor="white",
            markeredgecolor=color, markeredgewidth=1.0,
            label=LABELS[family],
        )
        ax.fill_between(
            q_values,
            np.clip(np.asarray(means) - spreads, 0, 1),
            np.clip(np.asarray(means) + spreads, 0, 1),
            color=color, alpha=0.12, linewidth=0,
        )
    ax.set_xscale("log", base=2)
    ax.set_xlim(min(q_values), max(q_values))
    ax.set_xticks(q_values)
    ax.set_xticklabels([str(q) for q in q_values], rotation=35 if len(q_values) > 7 else 0)
    ax.set_ylim(-0.015, 1.025)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    ax.set_xlabel("Clusters per mini-batch, $q$")
    ax.set_ylabel("Full-graph references")
    style_axis(ax)
    ax.grid(axis="y", color="#DCE1E7", linewidth=0.55, linestyle=":")
    ax.legend(loc="lower right", frameon=False)
    ax.set_title(f"Cumulative support availability after {horizon} epochs"
                 f"  |  K={first['K']}", fontsize=8.1, pad=7)
    fig.suptitle(TITLES[dataset], fontsize=10.0, y=0.98)
    fig.text(0.5, 0.025, f"{len(first['seeds'])} reshufflings · fixed METIS partition",
             ha="center", fontsize=6.5)
    fig.subplots_adjust(left=0.13, right=0.98, bottom=0.22, top=0.80)
    return fig


def export_q_sweep(sweep: dict[int, dict], *, dataset: str, output_dir: Path,
                   input_manifest: dict):
    """Save one auditable epoch-end row per family and q value."""
    output_dir.mkdir(parents=True, exist_ok=True)
    q_values = sorted(sweep)
    first = sweep[q_values[0]]
    manifest = {
        "dataset": dataset, "K": first["K"], "q_values": q_values,
        "epochs": first["epochs"], "seeds": first["seeds"],
        "metric": "fraction of full-graph references whose complete support was seen by the final epoch",
        "cellular_note": "support availability, not exact local cycle-basis selection",
        "reference_counts": {
            family: first["families"][family]["reference_count"]
            for family in FAMILIES
        },
        "partition_manifest": input_manifest,
    }
    (output_dir / "q_recovery_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    with (output_dir / "q_recovery_source_data.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "dataset", "K", "q", "family", "epochs", "reference_count",
            "q_observable_count", "theoretical_expected_fraction",
            "empirical_mean", "empirical_sample_sd",
            *[f"seed_{seed}_count" for seed in first["seeds"]],
        ])
        for family in FAMILIES:
            total = first["families"][family]["reference_count"]
            for q in q_values:
                result = sweep[q]
                data = result["families"][family]
                if data["reference_count"] != total:
                    raise ValueError("full-graph reference denominator changed across q")
                writer.writerow([
                    dataset, result["K"], q, family, result["epochs"], total,
                    data["observable_count"], data["expected_coverage"][-1],
                    data["coverage_mean"][-1], data["coverage_sample_sd"][-1],
                    *[data["counts_by_seed"][seed][-1] for seed in first["seeds"]],
                ])
    figure = make_q_plot(sweep, dataset=dataset)
    for suffix in ("png", "pdf", "svg"):
        figure.savefig(output_dir / f"q_recovery.{suffix}", dpi=220)
    plt.close(figure)


def export_result(result: dict, *, dataset: str, output_dir: Path,
                  input_manifest: dict):
    """Write audit-friendly source data, metadata, and figure files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "dataset": dataset, "K": result["K"], "q": result["q"],
        "epochs": result["epochs"], "seeds": result["seeds"],
        "metric": "cumulative support availability over frozen full-graph references",
        "cellular_note": "not exact local cycle-basis selection",
        "reference_counts": {family: result["families"][family]["reference_count"]
                             for family in FAMILIES},
        "observable_counts": {family: result["families"][family]["observable_count"]
                              for family in FAMILIES},
        "partition_manifest": input_manifest,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    with (output_dir / "source_data.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["dataset", "K", "q", "family", "epoch",
                         "reference_count", "q_observable_count",
                         "expected_coverage", "empirical_mean", "empirical_sample_sd",
                         *[f"seed_{seed}_count" for seed in result["seeds"]]])
        for family in FAMILIES:
            data = result["families"][family]
            for epoch in range(result["epochs"] + 1):
                writer.writerow([
                    dataset, result["K"], result["q"], family, epoch,
                    data["reference_count"], data["observable_count"],
                    data["expected_coverage"][epoch],
                    data["coverage_mean"][epoch],
                    data["coverage_sample_sd"][epoch],
                    *[data["counts_by_seed"][seed][epoch] for seed in result["seeds"]],
                ])
    figure = make_plot(result, dataset=dataset)
    for suffix in ("png", "pdf", "svg"):
        figure.savefig(output_dir / f"support_recovery.{suffix}", dpi=220)
    plt.close(figure)


def run_one(dataset: str, *, source_root: Path, partition_root: Path,
            coauthor_root: Path, cora_edge_index: Path, output_root: Path, epochs: int,
            seeds: list[int]):
    """Run a validated, fixed-reference dataset configuration."""
    setting = DATASETS[dataset]
    edge_index, num_nodes = load_graph(
        dataset, source_root=source_root, coauthor_root=coauthor_root,
        cora_edge_index=cora_edge_index)
    if num_nodes != setting["nodes"]:
        raise ValueError(f"{dataset}: unexpected node count {num_nodes}")
    if dataset == "coauthor_physics":
        labels, input_manifest = prepare_coauthor_partition(
            edge_index, num_nodes=num_nodes, output_dir=output_root / dataset)
    else:
        labels, input_manifest = load_saved_partition(
            dataset, partition_root=partition_root,
            edge_index=edge_index, num_nodes=num_nodes)
    graph = nx.Graph()
    graph.add_nodes_from(range(num_nodes))
    graph.add_edges_from((int(u), int(v)) for u, v in edge_index.T)
    references = {
        "hypergraph": neighbourhood_references(graph),
        "cellular": cycle_basis_references(graph, max_length=9),
        "simplicial": triangle_references(graph),
    }
    actual_counts = tuple(len(references[family]) for family in FAMILIES)
    if actual_counts != setting["counts"]:
        raise ValueError(f"{dataset}: reference counts {actual_counts} "
                         f"differ from reported {setting['counts']}")
    result = analyze_recovery(
        references, labels=labels, K=setting["K"], q=setting["q"],
        seeds=seeds, epochs=epochs,
    )
    export_result(result, dataset=dataset, output_dir=output_root / dataset,
                  input_manifest=input_manifest)
    return {
        "dataset": dataset, "K": setting["K"], "q": setting["q"],
        "reference_counts": dict(zip(FAMILIES, actual_counts, strict=True)),
        "final_coverage": {family: result["families"][family]["coverage_mean"][-1]
                           for family in FAMILIES},
        "output_dir": str(output_root / dataset),
    }


def run_q_sweep_one(dataset: str, *, source_root: Path, partition_root: Path,
                    coauthor_root: Path, cora_edge_index: Path,
                    output_root: Path, epochs: int, seeds: list[int]):
    """Run the approved q grid on one fixed graph and partition."""
    setting = DATASETS[dataset]
    edge_index, num_nodes = load_graph(
        dataset, source_root=source_root, coauthor_root=coauthor_root,
        cora_edge_index=cora_edge_index)
    if num_nodes != setting["nodes"]:
        raise ValueError(f"{dataset}: unexpected node count {num_nodes}")
    if dataset == "coauthor_physics":
        labels, input_manifest = prepare_coauthor_partition(
            edge_index, num_nodes=num_nodes, output_dir=output_root / dataset)
    else:
        labels, input_manifest = load_saved_partition(
            dataset, partition_root=partition_root,
            edge_index=edge_index, num_nodes=num_nodes)
    graph = nx.Graph()
    graph.add_nodes_from(range(num_nodes))
    graph.add_edges_from((int(u), int(v)) for u, v in edge_index.T)
    references = {
        "hypergraph": neighbourhood_references(graph),
        "cellular": cycle_basis_references(graph, max_length=9),
        "simplicial": triangle_references(graph),
    }
    actual_counts = tuple(len(references[family]) for family in FAMILIES)
    if actual_counts != setting["counts"]:
        raise ValueError(f"{dataset}: reference counts {actual_counts} "
                         f"differ from reported {setting['counts']}")
    sweep = analyze_q_sweep(
        references, labels=labels, K=setting["K"],
        q_values=Q_GRIDS[dataset], seeds=seeds, epochs=epochs,
    )
    export_q_sweep(
        sweep, dataset=dataset, output_dir=output_root / dataset,
        input_manifest=input_manifest,
    )
    return {
        "dataset": dataset, "K": setting["K"],
        "q_values": Q_GRIDS[dataset],
        "reference_counts": dict(zip(FAMILIES, actual_counts, strict=True)),
        "output_dir": str(output_root / dataset),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", choices=tuple(DATASETS))
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--partition-root", type=Path, required=True)
    parser.add_argument("--coauthor-root", type=Path, required=True)
    parser.add_argument("--cora-edge-index", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--q-sweep", action="store_true",
                        help="plot epoch-end recovery against q")
    args = parser.parse_args()
    run = run_q_sweep_one if args.q_sweep else run_one
    result = run(
        args.dataset, source_root=args.source_root,
        partition_root=args.partition_root, coauthor_root=args.coauthor_root,
        cora_edge_index=args.cora_edge_index,
        output_root=args.output_root, epochs=args.epochs,
        seeds=list(range(args.seeds)),
    )
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
