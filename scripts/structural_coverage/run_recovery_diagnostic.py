"""Topology-only adapter for a saved native Cluster-GCN CSR partition.

This module deliberately does not load node features or invoke model training.
The grouped-batch ordering mirrors ``BlockCSRBatchCollator``: selected parts
are sorted by their partition offset, and each selected CSR row retains its
stored neighbor order. This ordering can affect ``networkx.cycle_basis``.
"""

import argparse
import hashlib
import importlib.metadata
import json
import re
import subprocess
import time
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

import networkx as nx
import numpy as np
import torch
import torch_geometric
from torch_geometric.data import Data

from scripts.structural_coverage.recovery_core import (
    ReferenceStructure,
    cluster_span,
    cycle_basis_references,
    cycle_identity,
    expected_coverage,
    generate_epoch_schedules,
    local_cycle_basis_ids,
    neighbourhood_references,
    recovery_entropy,
    summarize_repetitions,
    triangle_references,
)
from scripts.structural_coverage.recovery_io import (
    CANONICAL_FAMILY_DEFINITIONS,
    load_recovery_bundle,
    load_recovery_config,
    validate_recovery_config,
    write_recovery_bundle,
)


@dataclass(frozen=True)
class RecoveryBatch:
    """One induced batch with original-ID and local-ID coordinates."""

    global_nodes: tuple[int, ...]
    permuted_nodes: tuple[int, ...]
    edge_index: torch.Tensor

    def as_data(self) -> Data:
        """Provide only one dummy feature per node for topology lifting checks."""
        return Data(
            x=torch.ones((len(self.global_nodes), 1), dtype=torch.float32),
            edge_index=self.edge_index,
            num_nodes=len(self.global_nodes),
        )

    def local_graph(self) -> nx.Graph:
        """Recreate the graph passed to native liftings in batch-local IDs."""
        graph = nx.Graph()
        graph.add_nodes_from(range(len(self.global_nodes)))
        graph.add_edges_from(
            (int(source), int(target))
            for source, target in self.edge_index.T.tolist()
        )
        return graph

    def global_graph(self) -> nx.Graph:
        """Expose global supports; do not select a cycle basis on this graph.

        NetworkX basis selection can change when local IDs are relabeled.
        Use ``local_basis_ids_in_global_coordinates`` for cellular recovery.
        """
        graph = nx.Graph()
        graph.add_nodes_from(self.global_nodes)
        graph.add_edges_from(
            (self.global_nodes[int(source)], self.global_nodes[int(target)])
            for source, target in self.edge_index.T.tolist()
        )
        return graph


def local_basis_ids_in_global_coordinates(
    batch: RecoveryBatch,
    *,
    max_length: int = 9,
    local_graph: nx.Graph | None = None,
) -> set[tuple]:
    """Select the native local basis, then map its boundaries to global IDs."""
    graph = local_graph if local_graph is not None else batch.local_graph()
    return {
        cycle_identity(batch.global_nodes[local] for local in cycle)
        for cycle in nx.cycle_basis(graph)
        if len(cycle) != 1 and len(cycle) <= max_length
    }


class RecoveryGraphAdapter:
    """Read native permuted CSR topology without copying graph features."""

    def __init__(
        self,
        partptr: np.ndarray,
        indptr: np.ndarray,
        indices: np.ndarray,
        perm_to_global: np.ndarray,
    ) -> None:
        self.partptr = np.asarray(partptr, dtype=np.int64)
        self.indptr = np.asarray(indptr, dtype=np.int64)
        self.indices = np.asarray(indices, dtype=np.int64)
        self.perm_to_global = np.asarray(perm_to_global, dtype=np.int64)
        self.num_nodes = len(self.perm_to_global)
        self.K = len(self.partptr) - 1
        if self.K <= 0 or len(self.indptr) != self.num_nodes + 1:
            raise ValueError("invalid partition or CSR node count")
        if (
            self.partptr[0] != 0
            or self.partptr[-1] != self.num_nodes
            or np.any(np.diff(self.partptr) <= 0)
        ):
            raise ValueError("partition must have K nonempty clusters")
        if (
            self.indptr[0] != 0
            or self.indptr[-1] != len(self.indices)
            or np.any(np.diff(self.indptr) < 0)
            or np.any(self.indices < 0)
            or np.any(self.indices >= self.num_nodes)
        ):
            raise ValueError("invalid CSR adjacency")
        if len(np.unique(self.perm_to_global)) != self.num_nodes:
            raise ValueError("permutation must cover each graph node once")

    @classmethod
    def from_memmap_dir(cls, directory: str | Path) -> "RecoveryGraphAdapter":
        """Open saved native structural arrays read-only via NumPy memmaps."""
        directory = Path(directory)
        return cls(
            *(
                np.load(directory / f"{name}.npy", mmap_mode="r")
                for name in (
                    "partptr",
                    "indptr",
                    "indices",
                    "perm_to_global",
                )
            )
        )

    def csr_order_sha256(self) -> str:
        """Hash the saved per-row neighbor order, independent of edge sets."""
        return _array_sha256(self.indptr) + ":" + _array_sha256(self.indices)

    def batch_for_clusters(
        self, parts: list[int] | tuple[int, ...]
    ) -> RecoveryBatch:
        """Return the original graph induced on the chosen cluster union."""
        if not parts or len(set(parts)) != len(parts):
            raise ValueError("cluster group must be nonempty and distinct")
        if any(
            type(part) is not int or part < 0 or part >= self.K
            for part in parts
        ):
            raise ValueError("cluster ID outside saved partition")
        ordered_parts = sorted(parts, key=lambda part: int(self.partptr[part]))
        permuted_nodes = tuple(
            row
            for part in ordered_parts
            for row in range(
                int(self.partptr[part]), int(self.partptr[part + 1])
            )
        )
        local_of_permuted = {
            row: local for local, row in enumerate(permuted_nodes)
        }
        sources: list[int] = []
        targets: list[int] = []
        for local_source, row in enumerate(permuted_nodes):
            for position in range(
                int(self.indptr[row]), int(self.indptr[row + 1])
            ):
                local_target = local_of_permuted.get(
                    int(self.indices[position])
                )
                if local_target is not None:
                    sources.append(local_source)
                    targets.append(local_target)
        edge_index = torch.tensor([sources, targets], dtype=torch.long)
        return RecoveryBatch(
            tuple(int(self.perm_to_global[row]) for row in permuted_nodes),
            permuted_nodes,
            edge_index,
        )


def preflight_partition(
    adapter: RecoveryGraphAdapter,
    *,
    expected_nodes: int,
    K: int,
    train_mask: np.ndarray,
) -> dict[str, int]:
    """Check the complete, training-active partition used by the theory."""
    if adapter.num_nodes != expected_nodes:
        raise ValueError("node count differs from configured dataset")
    if adapter.K != K:
        raise ValueError("cluster count differs from configured K")
    mask = np.asarray(train_mask, dtype=bool)
    if mask.shape != (adapter.num_nodes,):
        raise ValueError("training mask must use permuted graph coordinates")
    active = sum(
        bool(
            np.any(
                mask[
                    int(adapter.partptr[part]) : int(adapter.partptr[part + 1])
                ]
            )
        )
        for part in range(adapter.K)
    )
    if active != adapter.K:
        raise ValueError(
            "inactive clusters violate the fixed-K theory configuration"
        )
    return {
        "num_nodes": adapter.num_nodes,
        "K": adapter.K,
        "active_clusters": active,
        "directed_csr_entries": len(adapter.indices),
    }


def _array_sha256(array: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(array).tobytes()).hexdigest()


def preflight_provenance(
    adapter: RecoveryGraphAdapter,
    *,
    edge_index: np.ndarray,
    partition_metadata: dict,
    manifest: dict,
    train_mask_perm: np.ndarray,
    expected_dataset: str,
    expected_cycle_cap: int,
) -> dict[str, int | str]:
    """Check saved partition and run metadata against the loaded graph.

    ``partition_metadata`` is a verified sidecar, not PyG's native
    ``cluster_meta.pt``. It must record the input-graph and CSR-order hashes.
    ``edge_index`` must retain the processed PyG tensor's original column
    order before partitioning. A set of edges is inadequate because
    cycle-basis selection can depend on insertion order.
    """
    edges = np.asarray(edge_index)
    if edges.shape[0] != 2 or edges.ndim != 2:
        raise ValueError("processed edge_index must have shape (2, E)")
    if not np.issubdtype(edges.dtype, np.integer):
        raise ValueError("processed edge_index must contain integer node IDs")
    edges = np.ascontiguousarray(edges, dtype=np.int64)
    if np.any(edges < 0) or np.any(edges >= adapter.num_nodes):
        raise ValueError("processed graph has edge outside node range")
    ordered_digest = _array_sha256(edges)
    if manifest.get("edge_order_sha256") != ordered_digest:
        raise ValueError("edge order hash differs from saved run manifest")
    topology_digest = hashlib.sha256()
    topology_digest.update(
        np.asarray(
            [adapter.num_nodes, edges.shape[1]], dtype=np.int64
        ).tobytes()
    )
    topology_digest.update(edges.tobytes())
    if (
        partition_metadata.get("topology_sha256")
        != topology_digest.hexdigest()
    ):
        raise ValueError("partition topology fingerprint differs from graph")
    if (
        partition_metadata.get("csr_order_sha256")
        != adapter.csr_order_sha256()
    ):
        raise ValueError(
            "verified partition sidecar is missing or disagrees with CSR order"
        )
    for key, expected in (
        ("num_nodes", adapter.num_nodes),
        ("num_parts", adapter.K),
        ("num_input_edges", edges.shape[1]),
    ):
        if partition_metadata.get(key) != expected:
            raise ValueError(f"partition {key} differs from processed graph")
    if manifest.get("dataset") != expected_dataset:
        raise ValueError("manifest dataset differs from expected dataset")
    for key, expected in (
        ("num_nodes", adapter.num_nodes),
        ("num_parts", adapter.K),
        ("num_edges_directed", edges.shape[1]),
        ("max_cell_length", expected_cycle_cap),
    ):
        if manifest.get(key) != expected:
            raise ValueError(
                f"manifest {key} differs from current configuration"
            )
    partition_digest = (
        _array_sha256(adapter.perm_to_global)
        + ":"
        + _array_sha256(adapter.partptr)
    )
    if manifest.get("partition_assignment_sha256") != partition_digest:
        raise ValueError(
            "partition assignment fingerprint differs from manifest"
        )
    if set(adapter.perm_to_global.tolist()) != set(range(adapter.num_nodes)):
        raise ValueError(
            "Cora Full provenance requires contiguous original node IDs"
        )
    csr_sources = np.repeat(
        np.arange(adapter.num_nodes, dtype=np.int64),
        np.diff(adapter.indptr),
    )
    csr_edges = Counter(
        zip(
            adapter.perm_to_global[csr_sources].tolist(),
            adapter.perm_to_global[adapter.indices].tolist(),
            strict=True,
        )
    )
    processed_edges = Counter(
        zip(edges[0].tolist(), edges[1].tolist(), strict=True)
    )
    if csr_edges != processed_edges:
        raise ValueError("CSR graph edges differ from the processed graph")
    permuted_mask = np.asarray(train_mask_perm, dtype=bool)
    original_mask = np.empty_like(permuted_mask)
    original_mask[adapter.perm_to_global] = permuted_mask
    if manifest.get("train_mask_sha256") != _array_sha256(original_mask):
        raise ValueError("training split fingerprint differs from manifest")
    summary = preflight_partition(
        adapter,
        expected_nodes=adapter.num_nodes,
        K=adapter.K,
        train_mask=permuted_mask,
    )
    if manifest.get("active_cluster_count") != adapter.K:
        raise ValueError(
            "manifest active-cluster count differs from partition"
        )
    return {**summary, "graph_order_sha256": ordered_digest}


def parse_manuscript_model_annotations(
    manuscript_text: str,
) -> dict[str, tuple[int, int]]:
    """Read Cora Full partition settings from the current HPO longtable."""
    marker = r"\label{tab:optuna_hparams}"
    if marker not in manuscript_text:
        raise ValueError("manuscript lacks tab:optuna_hparams")
    after_marker = manuscript_text.split(marker, 1)[1]
    if r"\end{longtable}" not in after_marker:
        raise ValueError("manuscript HPO table has no longtable terminator")
    table = after_marker.split(r"\end{longtable}", 1)[0]
    heading = re.compile(r"\\multicolumn\{9\}\{c\}\{\\textbf\{([^}]+)\}\}")
    matches = list(heading.finditer(table))
    selected = {}
    for index, match in enumerate(matches):
        block = table[
            match.end() : matches[index + 1].start()
            if index + 1 < len(matches)
            else len(table)
        ]
        row = re.search(
            r"\\multirow\{2\}\{\*\}\{Cora Full\}[^\n]*\n\s*(& P &[^\n]+)",
            block,
        )
        if row is None:
            continue
        cells = row.group(1).split("&")
        if len(cells) < 8:
            raise ValueError(
                f"incomplete Cora Full P row for {match.group(1)}"
            )
        values = []
        for cell in cells[6:8]:
            value = re.fullmatch(r"\s*\$(\d+)\$\s*", cell)
            if value is None:
                raise ValueError(
                    f"unreadable Cora Full K,q for {match.group(1)}"
                )
            values.append(int(value.group(1)))
        selected[match.group(1)] = (values[0], values[1])
    return selected


def preflight_model_annotations(
    manuscript_text: str,
) -> dict[str, tuple[int, int]]:
    """Reject drift in the manuscript's selected Cora Full model settings."""
    selected = parse_manuscript_model_annotations(manuscript_text)
    expected = {
        "GCN": (32, 8),
        "EDHNN": (32, 4),
        "UniGNN": (32, 4),
        "CWN": (32, 4),
        "TopoTune": (64, 8),
        "SCN": (32, 16),
        "SCCNN": (32, 8),
    }
    for model, value in expected.items():
        if selected.get(model) != value:
            raise ValueError(
                f"{model} Cora Full K,q annotation differs from manuscript"
            )
    return selected


def preflight_pinned_training_split(
    adapter: RecoveryGraphAdapter,
    train_mask_perm: np.ndarray,
    config: dict,
) -> int:
    """Independently reproduce TopoBench's cached random fold zero.

    ``random_splitting`` seeds NumPy's legacy RNG with global_data_seed=42
    before generating fold 0 with its first ``np.random.permutation(n)``.
    This local RandomState produces the same permutation without changing
    process-global RNG state. Compare in original node coordinates.
    """
    validate_recovery_config(config)
    n = config["expected_num_nodes"]
    if adapter.num_nodes != n:
        raise ValueError("node count differs from pinned random split")
    observed = np.asarray(train_mask_perm)
    if observed.shape != (n,) or observed.dtype != np.bool_:
        raise ValueError("pinned training mask must be a Boolean node mask")
    expected = np.zeros(n, dtype=bool)
    permutation = np.random.RandomState(42).permutation(n)
    train_count = int(n * config["train_prop"])
    expected[permutation[:train_count]] = True
    if not np.array_equal(observed, expected[adapter.perm_to_global]):
        raise ValueError(
            "training mask differs from TopoBench random split "
            "fold 0 (global_data_seed=42)"
        )
    return train_count


def preflight_cora_full_run(
    adapter: RecoveryGraphAdapter,
    *,
    config: dict,
    schedule: str,
    edge_index: np.ndarray,
    partition_sidecar: dict,
    manifest: dict,
    train_mask_perm: np.ndarray,
    manuscript_text: str,
) -> dict[str, int | str]:
    """Bind generic graph checks to the pinned Cora Full experiment.

    The caller must supply the current processed edge-index order, a verified
    partition sidecar, and current manuscript text. Task 8 wires these inputs
    to its command-line runner.
    """
    validate_recovery_config(config)
    if schedule == "primary":
        requested = config["primary"]
    elif schedule == "additional[0]":
        requested = config["additional"][0]
    else:
        raise ValueError("unknown pinned recovery schedule")
    preflight_partition(
        adapter,
        expected_nodes=config["expected_num_nodes"],
        K=requested["K"],
        train_mask=train_mask_perm,
    )
    preflight_pinned_training_split(adapter, train_mask_perm, config)
    preflight_model_annotations(manuscript_text)
    return preflight_provenance(
        adapter,
        edge_index=edge_index,
        partition_metadata=partition_sidecar,
        manifest=manifest,
        train_mask_perm=train_mask_perm,
        expected_dataset=config["dataset"],
        expected_cycle_cap=config["cycle_max_length"],
    )


def reference_fingerprints(
    families: Mapping[str, Iterable],
) -> dict[str, dict[str, int | str]]:
    """Count and hash complete frozen identities and supports per family."""
    result = {}
    for family, references in families.items():
        records = sorted(
            (
                (reference.identity, sorted(reference.support))
                for reference in references
            ),
            key=repr,
        )
        payload = json.dumps(records, separators=(",", ":")).encode("utf-8")
        result[family] = {
            "count": len(records),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }
    return result


def verify_reference_snapshot(
    families: Mapping[str, Iterable],
    expected: Mapping[str, Mapping[str, int | str]],
) -> dict[str, dict[str, int | str]]:
    """Reject reference-count or identity drift from a frozen manifest."""
    observed = reference_fingerprints(families)
    if set(observed) != set(expected):
        raise ValueError("reference families differ from frozen snapshot")
    for family in observed:
        if observed[family] != expected[family]:
            raise ValueError(f"{family} reference count or hash drift")
    return observed


def audit_native_full_batch_order(
    adapter: RecoveryGraphAdapter,
    edge_index: np.ndarray,
    *,
    cluster_order: tuple[int, ...] | None = None,
    cycle_max_length: int = 9,
) -> dict[str, bool]:
    """Report, without repairing, q=K order and cycle-basis differences."""
    edges = np.asarray(edge_index, dtype=np.int64)
    if edges.ndim != 2 or edges.shape[0] != 2:
        raise ValueError("original edge_index must have shape (2, E)")
    if cluster_order is None:
        cluster_order = tuple(range(adapter.K))
    if (
        set(cluster_order) != set(range(adapter.K))
        or len(cluster_order) != adapter.K
    ):
        raise ValueError("q=K audit requires every cluster exactly once")
    batch = adapter.batch_for_clusters(cluster_order)
    original_pairs = tuple(
        zip(edges[0].tolist(), edges[1].tolist(), strict=True)
    )
    batch_pairs = tuple(
        (batch.global_nodes[int(source)], batch.global_nodes[int(target)])
        for source, target in batch.edge_index.T.tolist()
    )
    original_graph = nx.Graph()
    original_graph.add_nodes_from(range(adapter.num_nodes))
    original_graph.add_edges_from(original_pairs)
    return {
        "same_directed_edge_multiset": Counter(original_pairs)
        == Counter(batch_pairs),
        "same_insertion_order": (
            batch.global_nodes == tuple(range(adapter.num_nodes))
            and original_pairs == batch_pairs
        ),
        "same_cycle_basis": local_cycle_basis_ids(
            original_graph, max_length=cycle_max_length
        )
        == local_basis_ids_in_global_coordinates(
            batch, max_length=cycle_max_length
        ),
    }


@dataclass(frozen=True)
class RecoveryRunInputs:
    """Verified topology and protocol inputs for one structure-only run."""

    dataset: str
    edge_index: np.ndarray
    adapters: Mapping[int, RecoveryGraphAdapter]
    partition_sidecars: Mapping[int, dict]
    provenance_manifests: Mapping[int, dict]
    train_masks_perm: Mapping[int, np.ndarray]
    configuration_matrix: list[dict]
    seeds: list[int]
    epochs: int
    sample_every: int
    loader_config: str
    split_config: dict
    partition_source: str


def _ordered_full_graph(edge_index: np.ndarray, num_nodes: int) -> nx.Graph:
    """Build the reference graph from the original processed column order."""
    graph = nx.Graph()
    graph.add_nodes_from(range(num_nodes))
    graph.add_edges_from(
        (int(source), int(target))
        for source, target in np.asarray(edge_index).T
    )
    return graph


def _reference_families(
    graph: nx.Graph, requested: set[str]
) -> dict[str, list[ReferenceStructure]]:
    references = {}
    if "hypergraph" in requested:
        references["hypergraph"] = neighbourhood_references(graph)
    if "simplicial" in requested:
        references["simplicial"] = triangle_references(graph)
    if "cellular" in requested:
        references["cellular"] = cycle_basis_references(graph, max_length=9)
    return references


def _configuration_keys(inputs: RecoveryRunInputs) -> set[int]:
    if not inputs.configuration_matrix:
        raise ValueError("configuration matrix cannot be empty")
    keys = set()
    for row in inputs.configuration_matrix:
        if not isinstance(row, dict) or set(row) != {"K", "q", "families"}:
            raise ValueError("configuration matrix rows need K, q, families")
        K, q, families = row["K"], row["q"], row["families"]
        if type(K) is not int or type(q) is not int or K < 1 or q < 1:
            raise ValueError("K and q must be positive integers")
        if K % q or q > K:
            raise ValueError("q must divide K")
        if (
            not isinstance(families, list)
            or not families
            or set(families) - {"hypergraph", "simplicial", "cellular"}
            or len(set(families)) != len(families)
        ):
            raise ValueError("unknown or duplicate recovery family")
        keys.add(K)
    if type(inputs.epochs) is not int or inputs.epochs < 1:
        raise ValueError("epochs must be positive")
    if type(inputs.sample_every) is not int or inputs.sample_every < 1:
        raise ValueError("sample_every must be positive")
    if (
        not inputs.seeds
        or any(type(seed) is not int or seed < 0 for seed in inputs.seeds)
        or len(set(inputs.seeds)) != len(inputs.seeds)
    ):
        raise ValueError("seeds must be distinct nonnegative integers")
    if inputs.partition_source not in {"saved_experiment", "regenerated"}:
        raise ValueError("partition source must be explicit")
    for name, values in (
        ("adapters", inputs.adapters),
        ("partition sidecars", inputs.partition_sidecars),
        ("provenance manifests", inputs.provenance_manifests),
        ("training masks", inputs.train_masks_perm),
    ):
        if set(values) != keys:
            raise ValueError(f"{name} must cover exactly the configured K")
    return keys


def preflight_recovery_inputs(
    inputs: RecoveryRunInputs,
    *,
    pinned_config: dict | None = None,
    manuscript_text: str | None = None,
    reference_snapshot: Mapping[str, Mapping[str, int | str]] | None = None,
) -> dict:
    """Verify saved topology/order, partition, split, and fixed references."""
    keys = _configuration_keys(inputs)
    edge_index = np.asarray(inputs.edge_index)
    if (
        edge_index.ndim != 2
        or edge_index.shape[0] != 2
        or not np.issubdtype(edge_index.dtype, np.integer)
    ):
        raise ValueError("processed edge_index must be a (2,E) integer array")
    expected_nodes = {inputs.adapters[K].num_nodes for K in keys}
    if len(expected_nodes) != 1:
        raise ValueError("all partitions must refer to the same graph")
    num_nodes = expected_nodes.pop()
    if pinned_config is not None:
        validate_recovery_config(pinned_config)
        if manuscript_text is None:
            raise ValueError(
                "current manuscript text is required for Cora Full"
            )
        if inputs.dataset != pinned_config["dataset"]:
            raise ValueError("Cora Full run cannot substitute another dataset")
    summary = {}
    audits = {}
    for K in sorted(keys):
        sidecar = inputs.partition_sidecars[K]
        provenance = inputs.provenance_manifests[K]
        if sidecar.get("edge_order_sha256") != provenance.get(
            "edge_order_sha256"
        ):
            raise ValueError(
                "verified sidecar edge order differs from manifest"
            )
        if sidecar.get("partition_assignment_sha256") != provenance.get(
            "partition_assignment_sha256"
        ):
            raise ValueError("verified sidecar partition assignment differs")
        if pinned_config is None:
            summary[K] = preflight_provenance(
                inputs.adapters[K],
                edge_index=edge_index,
                partition_metadata=sidecar,
                manifest=provenance,
                train_mask_perm=inputs.train_masks_perm[K],
                expected_dataset=inputs.dataset,
                expected_cycle_cap=9,
            )
        else:
            label = (
                "primary"
                if pinned_config["primary"]["K"] == K
                else "additional[0]"
            )
            summary[K] = preflight_cora_full_run(
                inputs.adapters[K],
                config=pinned_config,
                schedule=label,
                edge_index=edge_index,
                partition_sidecar=sidecar,
                manifest=provenance,
                train_mask_perm=inputs.train_masks_perm[K],
                manuscript_text=manuscript_text,
            )
        audits[K] = audit_native_full_batch_order(
            inputs.adapters[K], edge_index, cycle_max_length=9
        )
        if not audits[K]["same_directed_edge_multiset"]:
            raise ValueError(
                "native full batch changes the graph edge multiset"
            )
    full_graph = _ordered_full_graph(edge_index, num_nodes)
    requested = {
        family
        for schedule in inputs.configuration_matrix
        for family in schedule["families"]
    }
    references = _reference_families(full_graph, requested)
    same_order_verified = False
    if "cellular" in references:
        identity_batch = RecoveryBatch(
            global_nodes=tuple(range(num_nodes)),
            permuted_nodes=tuple(range(num_nodes)),
            edge_index=torch.as_tensor(
                np.array(edge_index, dtype=np.int64, copy=True),
                dtype=torch.long,
            ),
        )
        if {
            reference.identity for reference in references["cellular"]
        } != local_basis_ids_in_global_coordinates(
            identity_batch, max_length=9
        ):
            raise ValueError(
                "same-order full-graph cycle-basis identity failed "
                "through the identity-order batch path"
            )
        same_order_verified = True
    if reference_snapshot is None:
        fingerprints = reference_fingerprints(references)
    else:
        fingerprints = verify_reference_snapshot(
            references, reference_snapshot
        )
    return {
        "graph": full_graph,
        "references": references,
        "reference_fingerprints": fingerprints,
        "partitions": summary,
        "native_full_batch_order": audits,
        "same_order_full_graph_basis_identity_verified": same_order_verified,
    }


def _indexed_references(
    adapter: RecoveryGraphAdapter,
    references: Mapping[str, list[ReferenceStructure]],
) -> tuple[dict[str, Counter], dict[str, dict[int, list[tuple]]]]:
    """Assign each reference one anchor cluster, so it is checked once/batch."""
    labels = {}
    for cluster in range(adapter.K):
        for permuted in range(
            int(adapter.partptr[cluster]), int(adapter.partptr[cluster + 1])
        ):
            labels[int(adapter.perm_to_global[permuted])] = cluster
    histograms = {}
    indexed = {}
    for family, family_references in references.items():
        histogram = Counter()
        anchors: dict[int, list[tuple]] = {}
        for reference in family_references:
            required = frozenset(labels[node] for node in reference.support)
            if not required:
                raise ValueError(
                    "reference structure cannot have empty support"
                )
            histogram[cluster_span(reference.support, labels)] += 1
            anchors.setdefault(min(required), []).append(
                (reference.identity, required)
            )
        histograms[family] = histogram
        indexed[family] = anchors
    return histograms, indexed


def _supported_in_group(
    anchors: Mapping[int, list[tuple]], group: tuple[int, ...]
) -> set[tuple]:
    chosen = frozenset(group)
    return {
        identity
        for cluster in group
        for identity, required in anchors.get(cluster, ())
        if required <= chosen
    }


def _full_hyperedges_in_batch(
    batch: RecoveryBatch,
    local_graph: nx.Graph,
    reference_support: Mapping[tuple, frozenset[int]],
    supported: set[tuple],
) -> set[tuple]:
    present = set()
    for local_centre, global_centre in enumerate(batch.global_nodes):
        identity = ("hyperedge", global_centre)
        if identity not in supported:
            continue
        local_support = frozenset(
            {global_centre}
            | {
                batch.global_nodes[node]
                for node in local_graph.neighbors(local_centre)
            }
        )
        if local_support == reference_support[identity]:
            present.add(identity)
    if present != supported:
        raise ValueError("induced batch changed full hyperedge membership")
    return present


def _observation_row(
    dataset: str,
    K: int,
    q: int,
    seed: int,
    epoch: int,
    family: str,
    measurement: str,
    count: int,
    reference_count: int,
    observable_count: int,
) -> dict:
    return {
        "dataset": dataset,
        "K": K,
        "q": q,
        "seed": seed,
        "epoch": epoch,
        "family": family,
        "measurement": measurement,
        "recovered_count": count,
        "reference_count": reference_count,
        "observable_count": observable_count,
        "coverage": count / reference_count if reference_count else None,
    }


def _append_epoch_observations(
    rows: list[dict],
    *,
    dataset: str,
    K: int,
    q: int,
    seed: int,
    epoch: int,
    seen: Mapping[tuple[str, str], set[tuple]],
    reference_ids: Mapping[str, frozenset[tuple]],
    observable: Mapping[str, int],
) -> None:
    for (family, measurement), identities in seen.items():
        rows.append(
            _observation_row(
                dataset,
                K,
                q,
                seed,
                epoch,
                family,
                measurement,
                len(identities),
                len(reference_ids[family]),
                observable[family],
            )
        )


def _aggregate_observations(
    observations: Iterable[dict],
    *,
    expected_seeds: Iterable[int],
    reference_ids: Mapping[str, set | frozenset],
    histograms_by_K: Mapping[int, Mapping[str, Mapping[int, int]]],
    dataset: str,
) -> list[dict]:
    """Aggregate one cumulative coverage value per seed and configuration."""
    seeds = set(expected_seeds)
    grouped: dict[tuple, dict[int, float | None]] = {}
    for row in observations:
        key = (
            row["K"],
            row["q"],
            row["epoch"],
            row["family"],
            row["measurement"],
        )
        seed = row["seed"]
        if seed not in seeds:
            raise ValueError("unexpected aggregation seed")
        by_seed = grouped.setdefault(key, {})
        if seed in by_seed:
            raise ValueError("duplicate aggregation seed row")
        by_seed[seed] = row["coverage"]

    summary = []
    for (K, q, epoch, family, measurement), by_seed in sorted(grouped.items()):
        if set(by_seed) != seeds:
            raise ValueError("missing aggregation seed row")
        total = len(reference_ids[family])
        repetition = (
            summarize_repetitions(by_seed[seed] for seed in sorted(seeds))
            if total
            else None
        )
        summary.append(
            {
                "dataset": dataset,
                "K": K,
                "q": q,
                "epoch": epoch,
                "family": family,
                "measurement": measurement,
                "n_repetitions": len(seeds),
                "coverage_mean": repetition.mean if repetition else None,
                "coverage_sample_sd": (
                    repetition.sample_sd if repetition else None
                ),
                "reference_count": total,
                "observable_fraction": (
                    sum(
                        count
                        for span, count in histograms_by_K[K][family].items()
                        if span <= q
                    )
                    / total
                    if total
                    else None
                ),
            }
        )
    return summary


def _code_provenance(root: Path | None = None) -> tuple[str, str | None]:
    """Capture the base commit and a digest of uncommitted local code."""
    root = Path(__file__).resolve().parents[2] if root is None else Path(root)
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if not status:
        return commit, None
    digest = hashlib.sha256(status.encode("utf-8"))
    relevant_sources = (
        "scripts/structural_coverage/run_recovery_diagnostic.py",
        "scripts/structural_coverage/recovery_core.py",
        "scripts/structural_coverage/recovery_io.py",
        "scripts/structural_coverage/configs/corafull_recovery_v2.json",
        "topobench/dataloader/dataload_cluster.py",
        "topobench/data/preprocessor/preprocessor.py",
        "topobench/data/utils/split_utils.py",
        "topobench/transforms/liftings/liftings.py",
        "topobench/transforms/liftings/graph2cell/cycle_lifting.py",
        "topobench/transforms/liftings/graph2hypergraph/khop_lifting_large_scale.py",
        "topobench/transforms/liftings/graph2simplicial/clique_lifting_fast.py",
    )
    for relative in relevant_sources:
        path = root / relative
        if path.is_file():
            digest.update(relative.encode("utf-8"))
            digest.update(path.read_bytes())
    return commit, digest.hexdigest()


def run_recovery_experiment(
    inputs: RecoveryRunInputs,
    *,
    output_dir: str | Path,
    run_mode: str,
    implementation_commit: str | None = None,
    local_patch_hash_if_dirty: str | None = None,
    max_wall_seconds: float | None = None,
    pinned_config: dict | None = None,
    manuscript_text: str | None = None,
    reference_snapshot: Mapping[str, Mapping[str, int | str]] | None = None,
) -> dict:
    """Run shared seeded schedules serially and write one validated bundle."""
    output = Path(output_dir)
    if output.exists():
        raise FileExistsError(f"result directory already exists: {output}")
    if run_mode not in {"smoke", "publication"}:
        raise ValueError("run_mode must be smoke or publication")
    if run_mode == "publication" and pinned_config is None:
        raise ValueError(
            "publication mode requires the pinned Cora Full config"
        )
    if pinned_config is not None and reference_snapshot is None:
        raise ValueError(
            "a frozen reference snapshot is required for real runs"
        )
    if max_wall_seconds is not None and (
        run_mode != "smoke" or max_wall_seconds <= 0
    ):
        raise ValueError("positive wall-time limit is available in smoke mode")
    started = time.monotonic()

    snapshot_for_preflight = reference_snapshot
    if run_mode == "smoke" and reference_snapshot is not None:
        requested_families = {
            family
            for row in inputs.configuration_matrix
            for family in row["families"]
        }
        missing_families = requested_families - reference_snapshot.keys()
        if missing_families:
            raise ValueError(
                "requested reference families missing from frozen snapshot: "
                + ", ".join(sorted(missing_families))
            )
        snapshot_for_preflight = {
            family: reference_snapshot[family] for family in requested_families
        }

    prepared = preflight_recovery_inputs(
        inputs,
        pinned_config=pinned_config,
        manuscript_text=manuscript_text,
        reference_snapshot=snapshot_for_preflight,
    )
    references = prepared["references"]
    keys = sorted({row["K"] for row in inputs.configuration_matrix})
    histograms_by_K = {}
    indexed_by_K = {}
    for K in keys:
        histograms_by_K[K], indexed_by_K[K] = _indexed_references(
            inputs.adapters[K], references
        )
    reference_support = {
        reference.identity: reference.support
        for reference in references.get("hypergraph", ())
    }
    reference_ids = {
        family: frozenset(reference.identity for reference in members)
        for family, members in references.items()
    }
    theory = []
    observations = []
    span_histogram = []
    for K in keys:
        for family in sorted(
            {
                family
                for schedule in inputs.configuration_matrix
                if schedule["K"] == K
                for family in schedule["families"]
            }
        ):
            span_histogram.extend(
                {
                    "dataset": inputs.dataset,
                    "K": K,
                    "family": family,
                    "span": span,
                    "count": count,
                    "reference_hash": prepared["reference_fingerprints"][
                        family
                    ]["sha256"],
                }
                for span, count in sorted(histograms_by_K[K][family].items())
            )
    checks = {
        "passed": True,
        "graph_partition_consistency_verified": True,
        "cellular_set_inclusion_verified": True,
        "native_full_batch_order": prepared["native_full_batch_order"],
        "same_order_full_graph_basis_identity_verified": prepared[
            "same_order_full_graph_basis_identity_verified"
        ],
    }
    for schedule in inputs.configuration_matrix:
        K, q = schedule["K"], schedule["q"]
        families = schedule["families"]
        adapter = inputs.adapters[K]
        observable = {
            family: sum(
                count
                for span, count in histograms_by_K[K][family].items()
                if span <= q
            )
            for family in families
        }
        for family in families:
            total = len(reference_ids[family])
            histogram = histograms_by_K[K][family]
            theory.extend(
                {
                    "dataset": inputs.dataset,
                    "K": K,
                    "q": q,
                    "epoch": epoch,
                    "family": family,
                    "reference_count": total,
                    "observable_count": observable[family],
                    "expected_coverage": expected_coverage(
                        histogram, K, q, epoch
                    ),
                    "entropy_nats_per_reference": recovery_entropy(
                        histogram, K, q, epoch
                    ),
                }
                for epoch in range(inputs.epochs + 1)
            )
        for seed in inputs.seeds:
            schedules = generate_epoch_schedules(K, q, inputs.epochs, seed)
            seen = {
                (family, measurement): set()
                for family in families
                for measurement in (
                    ("support_available", "actual_basis_recovery")
                    if family == "cellular"
                    else ("support_available",)
                )
            }

            _append_epoch_observations(
                observations,
                dataset=inputs.dataset,
                K=K,
                q=q,
                seed=seed,
                epoch=0,
                seen=seen,
                reference_ids=reference_ids,
                observable=observable,
            )
            for epoch, groups in enumerate(schedules, start=1):
                for group in groups:
                    if (
                        max_wall_seconds is not None
                        and time.monotonic() - started > max_wall_seconds
                    ):
                        raise TimeoutError(
                            "smoke wall-time limit reached (checked between "
                            "batches; preflight and a single slow lifting "
                            "cannot be interrupted)"
                        )
                    batch = adapter.batch_for_clusters(group)
                    local_graph = (
                        batch.local_graph()
                        if "hypergraph" in families or "cellular" in families
                        else None
                    )
                    for family in families:
                        supported = _supported_in_group(
                            indexed_by_K[K][family], group
                        )
                        if family == "hypergraph":
                            supported = _full_hyperedges_in_batch(
                                batch,
                                local_graph,
                                reference_support,
                                supported,
                            )
                        seen[family, "support_available"].update(supported)
                        if family == "cellular":
                            local = local_basis_ids_in_global_coordinates(
                                batch, max_length=9, local_graph=local_graph
                            )
                            actual = local & supported & reference_ids[family]
                            seen[family, "actual_basis_recovery"].update(
                                actual
                            )
                if (
                    "cellular" in families
                    and not seen["cellular", "actual_basis_recovery"]
                    <= seen["cellular", "support_available"]
                ):
                    raise ValueError("actual cell recovery exceeds support")
                _append_epoch_observations(
                    observations,
                    dataset=inputs.dataset,
                    K=K,
                    q=q,
                    seed=seed,
                    epoch=epoch,
                    seen=seen,
                    reference_ids=reference_ids,
                    observable=observable,
                )
    summary = _aggregate_observations(
        observations,
        expected_seeds=inputs.seeds,
        reference_ids=reference_ids,
        histograms_by_K=histograms_by_K,
        dataset=inputs.dataset,
    )
    if implementation_commit is None:
        implementation_commit, local_patch_hash_if_dirty = _code_provenance()
    first_sidecar = inputs.partition_sidecars[keys[0]]
    matrix = [dict(schedule) for schedule in inputs.configuration_matrix]
    manifest = {
        "schema_version": 2,
        "dataset": inputs.dataset,
        "loader_config": inputs.loader_config,
        "num_nodes": len(prepared["graph"]),
        "canonical_edge_count": prepared["graph"].number_of_edges(),
        "graph_hash": first_sidecar["topology_sha256"],
        "graph_order_hash": first_sidecar["edge_order_sha256"],
        "partition_hashes": {
            str(K): inputs.partition_sidecars[K]["partition_assignment_sha256"]
            for K in keys
        },
        "split_config": inputs.split_config,
        "active_cluster_counts": {str(K): K for K in keys},
        "family_definitions": {
            family: CANONICAL_FAMILY_DEFINITIONS[family]
            for family in sorted(references)
        },
        "reference_hashes": {
            family: data["sha256"]
            for family, data in prepared["reference_fingerprints"].items()
        },
        "cycle_max_length": 9,
        "hypergraph_identity": "center_indexed",
        "basis_order_policy": first_sidecar["basis_order_policy"],
        "configuration_matrix": matrix,
        "seeds": list(inputs.seeds),
        "epochs": inputs.epochs,
        "sample_every": inputs.sample_every,
        "entropy_normalization": "all_reference",
        "implementation_commit": implementation_commit,
        "local_patch_hash_if_dirty": local_patch_hash_if_dirty,
        "package_versions": {
            name: importlib.metadata.version(name)
            for name in ("numpy", "networkx", "torch")
        }
        | {"torch_geometric": torch_geometric.__version__},
        "partition_source": inputs.partition_source,
        "run_mode": run_mode,
        "completed": False,
    }
    write_recovery_bundle(
        output,
        manifest=manifest,
        partition_metadata={
            str(K): inputs.partition_sidecars[K] for K in keys
        },
        reference_summary=[
            {
                "dataset": inputs.dataset,
                "family": family,
                "reference_count": data["count"],
                "reference_hash": data["sha256"],
            }
            for family, data in sorted(
                prepared["reference_fingerprints"].items()
            )
        ],
        span_histogram=span_histogram,
        theory=theory,
        observations=observations,
        summary=summary,
        checks=checks,
    )
    return load_recovery_bundle(output, publication_only=False)


def _read_json(path: str | Path) -> dict:
    with Path(path).open(encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return value


def _require_cli_paths(parser: argparse.ArgumentParser, args, Ks: set[int]):
    required = ["processed_edge_index", "manuscript", "partition_source"]
    for K in sorted(Ks):
        required.extend(
            f"partition_k{K}_{suffix}"
            for suffix in ("dir", "sidecar", "provenance", "train_mask")
        )
    absent = [name for name in required if getattr(args, name) is None]
    if absent:
        parser.error(
            "missing verified Cora Full inputs: "
            + ", ".join("--" + name.replace("_", "-") for name in absent)
        )


def _select_smoke_schedule(
    config: dict, *, K: int, q: int, epochs: int
) -> tuple[dict, int, int]:
    """Select one pinned diagnostic row without changing the sweep config."""
    if not 1 <= epochs <= config["epochs"]:
        raise ValueError(
            f"smoke epochs must be between 1 and {config['epochs']}"
        )
    for schedule in [config["primary"], *config["additional"]]:
        if schedule["K"] == K:
            if q not in schedule["q_values"]:
                raise ValueError(f"K={K}, q={q} is not a pinned configuration")
            return schedule, q, epochs
    raise ValueError(f"K={K} is not a pinned configuration")


def main(argv: list[str] | None = None) -> int:
    """Run strict Cora Full preflight, a bounded smoke, or the pinned sweep."""
    parser = argparse.ArgumentParser(
        description=(
            "CPU-only recovery of frozen Cora Full structures. Supply the exact "
            "processed edge_index in original column order and independently "
            "saved partition provenance. The runner never downloads data."
        )
    )
    parser.add_argument(
        "--config",
        default=str(
            Path(__file__).parent / "configs" / "corafull_recovery_v2.json"
        ),
    )
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--preflight-only", action="store_true")
    modes.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--smoke-k",
        type=int,
        help="Diagnostic-only K from the locked configuration (default: 32)",
    )
    parser.add_argument(
        "--smoke-q",
        type=int,
        help="Diagnostic-only q from the selected K row (default: 4)",
    )
    parser.add_argument(
        "--smoke-epochs",
        type=int,
        help="Diagnostic-only horizon, at most the locked horizon (default: 2)",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--processed-edge-index",
        help="Exact processed (2,E) .npy tensor, preserving column order",
    )
    parser.add_argument(
        "--manuscript",
        help="Current manuscript .tex file for Cora Full K,q cross-check",
    )
    parser.add_argument(
        "--partition-source", choices=("saved_experiment", "regenerated")
    )
    parser.add_argument(
        "--reference-snapshot",
        help="Reference fingerprints emitted by --preflight-only",
    )
    parser.add_argument(
        "--max-wall-seconds",
        type=float,
        help=(
            "Positive cooperative smoke-run wall-time limit, checked between "
            "batches; it cannot interrupt preflight or one slow lifting"
        ),
    )
    for K in (32, 64):
        parser.add_argument(
            f"--partition-k{K}-dir",
            help=f"Native k{K} CSR .npy array directory",
        )
        parser.add_argument(
            f"--partition-k{K}-sidecar",
            help=f"Previously saved, verified k{K} sidecar JSON",
        )
        parser.add_argument(
            f"--partition-k{K}-provenance",
            help=f"Saved k{K} graph/partition provenance JSON",
        )
        parser.add_argument(
            f"--partition-k{K}-train-mask",
            help=f"Saved k{K} training mask in permuted coordinates",
        )
    args = parser.parse_args(argv)
    config = load_recovery_config(args.config)
    smoke_overrides = (
        args.smoke_k is not None
        or args.smoke_q is not None
        or args.smoke_epochs is not None
    )
    if smoke_overrides and not args.smoke:
        parser.error("a diagnostic override requires --smoke")
    if args.smoke:
        try:
            schedule, smoke_q, smoke_epochs = _select_smoke_schedule(
                config,
                K=32 if args.smoke_k is None else args.smoke_k,
                q=4 if args.smoke_q is None else args.smoke_q,
                epochs=2 if args.smoke_epochs is None else args.smoke_epochs,
            )
        except ValueError as exc:
            parser.error(str(exc))
        schedules = [schedule]
    else:
        schedules = [config["primary"], *config["additional"]]
    Ks = {schedule["K"] for schedule in schedules}
    _require_cli_paths(parser, args, Ks)
    if args.max_wall_seconds is not None and not args.smoke:
        parser.error("--max-wall-seconds applies only to --smoke")
    if args.max_wall_seconds is not None and args.max_wall_seconds <= 0:
        parser.error("--max-wall-seconds must be positive")
    if not args.preflight_only and not args.reference_snapshot:
        parser.error(
            "run --preflight-only first, then pass its reference_snapshot.json "
            "with --reference-snapshot"
        )
    output = Path(args.output_dir).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"result directory already exists: {output}")
    edge_index = np.load(
        args.processed_edge_index, allow_pickle=False, mmap_mode="r"
    )
    adapters = {}
    sidecars = {}
    provenance = {}
    masks = {}
    for K in sorted(Ks):
        adapters[K] = RecoveryGraphAdapter.from_memmap_dir(
            getattr(args, f"partition_k{K}_dir")
        )
        sidecars[K] = _read_json(getattr(args, f"partition_k{K}_sidecar"))
        provenance[K] = _read_json(getattr(args, f"partition_k{K}_provenance"))
        masks[K] = np.load(
            getattr(args, f"partition_k{K}_train_mask"),
            allow_pickle=False,
            mmap_mode="r",
        )
    matrix = [
        {"K": schedule["K"], "q": q, "families": schedule["families"]}
        for schedule in schedules
        for q in ([smoke_q] if args.smoke else schedule["q_values"])
    ]
    inputs = RecoveryRunInputs(
        dataset=config["dataset"],
        edge_index=edge_index,
        adapters=adapters,
        partition_sidecars=sidecars,
        provenance_manifests=provenance,
        train_masks_perm=masks,
        configuration_matrix=matrix,
        seeds=[config["seeds"][0]] if args.smoke else config["seeds"],
        epochs=smoke_epochs if args.smoke else config["epochs"],
        sample_every=config["sample_every"],
        loader_config="configs/dataset/graph/cocitation_cora_full_for_partitioning.yaml",
        split_config={
            "seed": config["split_seed"],
            "type": config["split_type"],
            "learning_setting": config["learning_setting"],
            "train_prop": config["train_prop"],
        },
        partition_source=args.partition_source,
    )
    manuscript_text = Path(args.manuscript).read_text(encoding="utf-8")
    snapshot = (
        _read_json(args.reference_snapshot)
        if args.reference_snapshot
        else None
    )
    if args.preflight_only:
        prepared = preflight_recovery_inputs(
            inputs,
            pinned_config=config,
            manuscript_text=manuscript_text,
            reference_snapshot=snapshot,
        )
        report = {
            "dataset": config["dataset"],
            "num_nodes": inputs.adapters[min(Ks)].num_nodes,
            "partition_source": args.partition_source,
            "partition_checks": prepared["partitions"],
            "native_full_batch_order": prepared["native_full_batch_order"],
            "reference_fingerprints": prepared["reference_fingerprints"],
            "configuration_matrix": matrix,
        }
        output.mkdir(parents=True, exist_ok=False)
        (output / "preflight.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (output / "reference_snapshot.json").write_text(
            json.dumps(
                prepared["reference_fingerprints"], indent=2, sort_keys=True
            )
            + "\n",
            encoding="utf-8",
        )
        return 0
    run_recovery_experiment(
        inputs,
        output_dir=output,
        run_mode="smoke" if args.smoke else "publication",
        max_wall_seconds=(args.max_wall_seconds or 120)
        if args.smoke
        else None,
        pinned_config=config,
        manuscript_text=manuscript_text,
        reference_snapshot=snapshot,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
