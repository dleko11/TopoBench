"""Support-only recovery over frozen full-graph structure references.

This counter measures cluster co-occurrence, not local cycle-basis selection.
It therefore does not construct or lift a graph for each shuffled mini-batch.
"""

import hashlib
import statistics
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence

import numpy as np

from scripts.structural_coverage.recovery_core import (
    ReferenceStructure,
    expected_coverage,
    generate_epoch_schedules,
)


def build_signature_groups(
    references: Iterable[ReferenceStructure], labels: Sequence[int], *, q: int
) -> tuple[dict[int, tuple[np.ndarray, np.ndarray]], dict[int, int]]:
    """Count distinct cluster signatures and retain each multiplicity."""
    if type(q) is not int or q < 1:
        raise ValueError("q must be a positive integer")
    signatures: dict[int, Counter[tuple[int, ...]]] = defaultdict(Counter)
    histogram: Counter[int] = Counter()
    for reference in references:
        if not reference.support:
            raise ValueError("reference support cannot be empty")
        if any(node < 0 or node >= len(labels) for node in reference.support):
            raise ValueError("node outside partition")
        signature = tuple(sorted({int(labels[node]) for node in reference.support}))
        span = len(signature)
        histogram[span] += 1
        if span <= q:
            signatures[span][signature] += 1
    groups = {
        span: (
            np.asarray(list(counts), dtype=np.int32).reshape(-1, span),
            np.asarray(list(counts.values()), dtype=np.int64),
        )
        for span, counts in signatures.items()
    }
    return groups, dict(histogram)


def count_support_by_epoch(
    groups: dict[int, tuple[np.ndarray, np.ndarray]],
    *,
    K: int,
    q: int,
    schedules: Iterable[Iterable[Iterable[int]]],
) -> list[int]:
    """Count distinct supported references after each complete epoch."""
    if type(K) is not int or type(q) is not int or K < 1 or q < 1 or K % q:
        raise ValueError("K must be positive and divisible by q")
    seen = {
        span: np.zeros(len(weights), dtype=bool)
        for span, (_, weights) in groups.items()
    }
    cumulative = 0
    counts = [0]
    for epoch_groups in schedules:
        batch_of_cluster = np.full(K, -1, dtype=np.int32)
        for batch_index, group in enumerate(epoch_groups):
            cluster_ids = tuple(group)
            if len(cluster_ids) != q or any(
                cluster < 0 or cluster >= K for cluster in cluster_ids
            ):
                raise ValueError("invalid cluster group")
            if np.any(batch_of_cluster[list(cluster_ids)] != -1):
                raise ValueError("cluster appears in multiple groups")
            batch_of_cluster[list(cluster_ids)] = batch_index
        if np.any(batch_of_cluster == -1):
            raise ValueError("epoch omits a cluster")
        for span, (signatures, weights) in groups.items():
            pending = np.flatnonzero(~seen[span])
            if not len(pending):
                continue
            assignments = batch_of_cluster[signatures[pending]]
            newly_seen = pending[
                np.all(assignments == assignments[:, :1], axis=1)
            ]
            seen[span][newly_seen] = True
            cumulative += int(weights[newly_seen].sum())
        counts.append(cumulative)
    return counts


def labels_from_partition(*, partptr, perm_to_global, train_mask_perm):
    """Map each original node to its partition, requiring all parts active."""
    ptr = np.asarray(partptr, dtype=np.int64)
    perm = np.asarray(perm_to_global, dtype=np.int64)
    train = np.asarray(train_mask_perm, dtype=bool)
    nodes = len(perm)
    if (
        ptr.ndim != 1
        or len(ptr) < 2
        or ptr[0] != 0
        or ptr[-1] != nodes
        or np.any(np.diff(ptr) <= 0)
    ):
        raise ValueError("partition offsets must cover nonempty clusters")
    if (
        np.any(perm < 0)
        or np.any(perm >= nodes)
        or len(np.unique(perm)) != nodes
    ):
        raise ValueError("partition permutation must cover every node once")
    if train.shape != (nodes,):
        raise ValueError("training mask shape disagrees with partition")
    labels = np.empty(nodes, dtype=np.int32)
    for cluster in range(len(ptr) - 1):
        begin, end = int(ptr[cluster]), int(ptr[cluster + 1])
        if not np.any(train[begin:end]):
            raise ValueError("inactive cluster violates fixed-K schedule")
        labels[perm[begin:end]] = cluster
    return labels


def validate_ordered_graph(edge_index, *, num_nodes, manifest):
    """Reject a processed graph that differs from its saved partition source."""
    edges = np.asarray(edge_index, dtype=np.int64)
    if edges.ndim != 2 or edges.shape[0] != 2:
        raise ValueError("edge_index must have shape (2,E)")
    if (
        num_nodes != manifest["num_nodes"]
        or edges.shape[1] != manifest["num_edges_directed"]
        or np.any(edges < 0)
        or np.any(edges >= num_nodes)
    ):
        raise ValueError("graph node or edge counts disagree with partition")
    actual = hashlib.sha256(edges.tobytes()).hexdigest()
    if actual != manifest["edge_order_sha256"]:
        raise ValueError("ordered edge hash differs from partition source")


def analyze_recovery(references, *, labels, K, q, seeds, epochs):
    """Return theory and ten-seed support counts for one fixed partition."""
    if (
        type(K) is not int
        or type(q) is not int
        or K < 1
        or q < 1
        or K % q
        or type(epochs) is not int
        or epochs < 1
    ):
        raise ValueError("invalid K, q, or epoch horizon")
    if (
        not seeds
        or any(type(seed) is not int or seed < 0 for seed in seeds)
        or len(set(seeds)) != len(seeds)
    ):
        raise ValueError("seeds must be distinct nonnegative integers")
    if not references:
        raise ValueError("at least one reference family is required")
    labels = np.asarray(labels, dtype=np.int32)
    if labels.ndim != 1 or np.any(labels < 0) or np.any(labels >= K):
        raise ValueError("partition labels must lie in [0,K)")
    prepared = {}
    for family, items in references.items():
        if not items:
            raise ValueError(f"{family} has no reference structures")
        groups, histogram = build_signature_groups(items, labels, q=q)
        prepared[family] = {
            "groups": groups,
            "span_histogram": histogram,
            "reference_count": len(items),
            "observable_count": sum(
                count for span, count in histogram.items() if span <= q
            ),
            "expected_coverage": [
                expected_coverage(histogram, K, q, epoch)
                for epoch in range(epochs + 1)
            ],
            "counts_by_seed": {},
        }
    for seed in seeds:
        schedules = generate_epoch_schedules(K, q, epochs, seed)
        for family, data in prepared.items():
            data["counts_by_seed"][seed] = count_support_by_epoch(
                data["groups"], K=K, q=q, schedules=schedules
            )
    families = {}
    for family, data in prepared.items():
        per_epoch = [
            [data["counts_by_seed"][seed][epoch] for seed in seeds]
            for epoch in range(epochs + 1)
        ]
        total = data["reference_count"]
        families[family] = {
            "reference_count": total,
            "observable_count": data["observable_count"],
            "span_histogram": data["span_histogram"],
            "expected_coverage": data["expected_coverage"],
            "counts_by_seed": data["counts_by_seed"],
            "coverage_mean": [
                statistics.mean(counts) / total for counts in per_epoch
            ],
            "coverage_sample_sd": [
                statistics.stdev(counts) / total if len(seeds) > 1 else None
                for counts in per_epoch
            ],
        }
    return {"K": K, "q": q, "seeds": list(seeds), "epochs": epochs,
            "families": families}


def analyze_q_sweep(references, *, labels, K, q_values, seeds, epochs):
    """Evaluate paired 200-epoch support recovery over a valid q grid.

    The same seed generates the same cluster permutation in every epoch for
    each q. Only the grouping of that permutation changes with q.
    """
    if (
        type(K) is not int or K < 1
        or type(epochs) is not int or epochs < 1
        or not q_values
        or any(type(q) is not int or q < 1 or K % q for q in q_values)
        or list(q_values) != sorted(set(q_values))
    ):
        raise ValueError("q grid must be strictly increasing divisors of K")
    if (
        not seeds
        or any(type(seed) is not int or seed < 0 for seed in seeds)
        or len(set(seeds)) != len(seeds)
    ):
        raise ValueError("seeds must be distinct nonnegative integers")
    if not references:
        raise ValueError("at least one reference family is required")
    labels = np.asarray(labels, dtype=np.int32)
    if labels.ndim != 1 or np.any(labels < 0) or np.any(labels >= K):
        raise ValueError("partition labels must lie in [0,K)")

    prepared = {}
    for family, items in references.items():
        if not items:
            raise ValueError(f"{family} has no reference structures")
        groups, histogram = build_signature_groups(items, labels, q=q_values[-1])
        prepared[family] = (groups, histogram, len(items))

    sweep = {}
    for q in q_values:
        schedules_by_seed = (
            {}
            if q in (1, K)
            else {
                seed: generate_epoch_schedules(K, q, epochs, seed)
                for seed in seeds
            }
        )
        family_data = {}
        for family, (groups_all, histogram, total) in prepared.items():
            groups = {span: group for span, group in groups_all.items()
                      if span <= q}
            observable = sum(count for span, count in histogram.items()
                             if span <= q)
            counts_by_seed = {}
            for seed in seeds:
                if q == 1:
                    counts = [0] + [observable] * epochs
                elif q == K:
                    counts = [0] + [total] * epochs
                else:
                    counts = count_support_by_epoch(
                        groups, K=K, q=q, schedules=schedules_by_seed[seed]
                    )
                counts_by_seed[seed] = counts
            per_epoch = [
                [counts_by_seed[seed][epoch] for seed in seeds]
                for epoch in range(epochs + 1)
            ]
            family_data[family] = {
                "reference_count": total,
                "observable_count": observable,
                "span_histogram": histogram,
                "expected_coverage": [
                    expected_coverage(histogram, K, q, epoch)
                    for epoch in range(epochs + 1)
                ],
                "counts_by_seed": counts_by_seed,
                "coverage_mean": [statistics.mean(row) / total
                                  for row in per_epoch],
                "coverage_sample_sd": [
                    statistics.stdev(row) / total if len(seeds) > 1 else None
                    for row in per_epoch
                ],
            }
        sweep[q] = {"K": K, "q": q, "seeds": list(seeds),
                    "epochs": epochs, "families": family_data}
    return sweep
