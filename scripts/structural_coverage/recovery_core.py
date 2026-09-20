"""Frozen full-graph reference structures for the Cora Full recovery study.

Identities refer to global node IDs. The reference universe is built once on the
full graph; later batch-level recovery checks compare against these identities.
"""

import math
import random
import statistics
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from heapq import heappop, heappush
from numbers import Integral

import networkx as nx


@dataclass(frozen=True)
class ReferenceStructure:
    """A structural identity and the complete node support it requires."""

    identity: tuple
    support: frozenset[int]


@dataclass(frozen=True)
class RepetitionSummary:
    """Mean and sample SD across independent run seeds, not batch rows."""

    n: int
    mean: float
    sample_sd: float | None


@dataclass(frozen=True)
class EntropyMilestone:
    """Integer analytical peak and final relative-decay crossing, if known."""

    status: str
    peak_epoch: int | None = None
    peak_value: float | None = None
    final_decay_epoch: int | None = None
    reason: str | None = None


def summarize_repetitions(values: Iterable[float]) -> RepetitionSummary:
    """Summarize one per-seed value at a fixed configuration and epoch."""
    per_seed = tuple(values)
    if not per_seed:
        raise ValueError("at least one repetition is required")
    return RepetitionSummary(
        n=len(per_seed),
        mean=statistics.mean(per_seed),
        sample_sd=statistics.stdev(per_seed) if len(per_seed) > 1 else None,
    )


def cycle_identity(nodes: Iterable[int]) -> tuple:
    """Identify an undirected cycle by its boundary, not merely its nodes."""
    cycle = tuple(int(node) for node in nodes)
    boundary = {
        tuple(sorted((cycle[index], cycle[(index + 1) % len(cycle)])))
        for index in range(len(cycle))
    }
    return ("cycle", tuple(sorted(boundary)))


def cluster_span(support: Iterable[int], labels: Mapping[int, int]) -> int:
    """Count distinct partition labels across the supporting global nodes."""
    return len({int(labels[int(node)]) for node in support})


def neighbourhood_references(graph: nx.Graph) -> list[ReferenceStructure]:
    """Keep one indexed, closed one-hop hyperedge per global centre node."""
    return [
        ReferenceStructure(
            ("hyperedge", int(centre)),
            frozenset(
                {int(centre), *(int(node) for node in graph.neighbors(centre))}
            ),
        )
        for centre in sorted(graph.nodes)
    ]


def triangle_references(graph: nx.Graph) -> list[ReferenceStructure]:
    """Return the graph's filled triangles, each identified by three nodes."""
    neighbours = {
        int(node): {
            int(other) for other in graph.neighbors(node) if other != node
        }
        for node in graph.nodes
    }
    references = []
    for u in sorted(neighbours):
        for v in sorted(node for node in neighbours[u] if node > u):
            references.extend(
                ReferenceStructure(("triangle", u, v, w), frozenset({u, v, w}))
                for w in sorted(
                    node for node in neighbours[u] & neighbours[v] if node > v
                )
            )
    return references


def cycle_basis_references(
    graph: nx.Graph, *, max_length: int = 9
) -> list[ReferenceStructure]:
    """Select one full-graph cycle basis and retain cycles within the length cap."""
    references = [
        ReferenceStructure(cycle_identity(cycle), frozenset(map(int, cycle)))
        for cycle in nx.cycle_basis(graph)
        if len(cycle) != 1 and len(cycle) <= max_length
    ]
    return sorted(references, key=lambda reference: reference.identity)


def local_cycle_basis_ids(
    graph: nx.Graph, *, max_length: int = 9
) -> set[tuple]:
    """Extract identities from one batch-local basis with the reference cap."""
    return {
        reference.identity
        for reference in cycle_basis_references(graph, max_length=max_length)
    }


def support_available_ids(
    references: Iterable[ReferenceStructure], batch_nodes: Iterable[int]
) -> set[tuple]:
    """Full-graph cells whose complete node support lies in one batch."""
    nodes = frozenset(batch_nodes)
    return {
        reference.identity
        for reference in references
        if reference.support <= nodes
    }


def matched_reference_ids(
    references: Iterable[ReferenceStructure], local_identities: Iterable[tuple]
) -> set[tuple]:
    """Locally selected basis cells that retain a frozen reference identity."""
    local = frozenset(local_identities)
    return {
        reference.identity
        for reference in references
        if reference.identity in local
    }


class CellRecoveryTracker:
    """Accumulate support availability and actual basis recovery separately."""

    def __init__(self, references: Iterable[ReferenceStructure]) -> None:
        self.references = tuple(references)
        self.reference_ids = frozenset(
            reference.identity for reference in self.references
        )
        if len(self.reference_ids) != len(self.references):
            raise ValueError("cell references must have distinct identities")
        self.support_seen: set[tuple] = set()
        self.actual_seen: set[tuple] = set()

    @property
    def reference_count(self) -> int:
        """Size of the frozen full-graph cycle-basis universe."""
        return len(self.references)

    @property
    def support_fraction(self) -> float | None:
        """Cumulative support coverage over the frozen reference universe."""
        return coverage_fraction(self.support_seen, self.reference_ids)

    @property
    def actual_fraction(self) -> float | None:
        """Cumulative local-basis recovery over the same universe."""
        return coverage_fraction(self.actual_seen, self.reference_ids)

    def record_batch(
        self, batch_nodes: Iterable[int], local_identities: Iterable[tuple]
    ) -> tuple[set[tuple], set[tuple]]:
        """Record one extracted local basis, without redefining the reference."""
        supported = support_available_ids(self.references, batch_nodes)
        actual = (
            matched_reference_ids(self.references, local_identities)
            & supported
        )
        self.support_seen.update(supported)
        self.actual_seen.update(actual)
        return supported, actual


def _is_integer_count(value: object) -> bool:
    """Accept Python and NumPy integers, but never booleans or floats."""
    return isinstance(value, Integral) and not isinstance(value, bool)


def per_epoch_probability(c: int, K: int, q: int) -> float:
    """Probability that one structure's supporting clusters share a batch."""
    if not all(_is_integer_count(value) for value in (c, K, q)):
        raise ValueError(
            "span, cluster count, and batch size must be integers"
        )
    if not (1 <= c <= K and 1 <= q <= K and K % q == 0):
        raise ValueError("invalid span or non-equal-batch configuration")
    if c > q:
        return 0.0
    return math.comb(q - 1, c - 1) / math.comb(K - 1, c - 1)


def generate_epoch_schedules(
    K: int, q: int, epochs: int, seed: int
) -> tuple[tuple[tuple[int, ...], ...], ...]:
    """Freeze seeded equal-size cluster groupings before lifting any domain.

    Generate once for each (K, q, seed), then reuse the returned schedules
    across lifting families so family iteration order cannot affect sampling.
    """
    per_epoch_probability(1, K, q)
    if not _is_integer_count(epochs) or epochs < 0:
        raise ValueError("epoch count must be a nonnegative integer")
    if not _is_integer_count(seed):
        raise ValueError("seed must be an integer")

    rng = random.Random(int(seed))
    schedules = []
    for _ in range(epochs):
        clusters = list(range(K))
        rng.shuffle(clusters)
        schedules.append(
            tuple(
                tuple(clusters[start : start + q]) for start in range(0, K, q)
            )
        )
    return tuple(schedules)


def cumulative_probability(p: float, T: int) -> float:
    """Probability of at least one recovery in T independent epochs."""
    if not _is_integer_count(T) or T < 0 or not 0 <= p <= 1:
        raise ValueError("invalid probability or horizon")
    if T == 0 or p == 0:
        return 0.0
    if p == 1:
        return 1.0
    return -math.expm1(T * math.log1p(-p))


def bernoulli_entropy(rho: float) -> float:
    """Binary entropy in nats, with exact zero at deterministic endpoints."""
    if not 0 <= rho <= 1:
        raise ValueError("invalid recovery probability")
    if rho == 0 or rho == 1:
        return 0.0
    return -rho * math.log(rho) - (1 - rho) * math.log1p(-rho)


def _reference_total(
    histogram: Mapping[int, int], K: int, q: int, T: int
) -> int:
    """Validate a span histogram and return the full reference-family size."""
    per_epoch_probability(1, K, q)
    cumulative_probability(0.0, T)
    for span, count in histogram.items():
        per_epoch_probability(span, K, q)
        if not _is_integer_count(count) or count < 0:
            raise ValueError("span counts must be nonnegative integers")
    return int(sum(histogram.values()))


def expected_coverage(
    histogram: Mapping[int, int], K: int, q: int, T: int
) -> float | None:
    """Expected recovered fraction among all full-graph reference structures."""
    total = _reference_total(histogram, K, q, T)
    if total == 0:
        return None
    return (
        sum(
            count
            * cumulative_probability(per_epoch_probability(span, K, q), T)
            for span, count in histogram.items()
        )
        / total
    )


def recovery_entropy(
    histogram: Mapping[int, int], K: int, q: int, T: int
) -> float | None:
    """Mean binary recovery entropy in nats per full-graph reference structure."""
    total = _reference_total(histogram, K, q, T)
    if total == 0:
        return None
    return _entropy_at_epoch(
        tuple(
            (per_epoch_probability(span, K, q), count)
            for span, count in histogram.items()
        ),
        total,
        T,
    )


def _entropy_at_epoch(
    terms: tuple[tuple[float, int], ...], reference_count: float, epoch: float
) -> float:
    """Evaluate the same all-reference entropy at integer or real epochs."""
    if reference_count == 0:
        return 0.0
    total = 0.0
    for probability, count in terms:
        if count == 0 or probability in (0.0, 1.0) or epoch == 0:
            continue
        log_survival = epoch * math.log1p(-probability)
        survival = math.exp(log_survival)
        recovered = -math.expm1(log_survival)
        if recovered > 0:
            total -= count * recovered * math.log(recovered)
        if survival > 0:
            total -= count * survival * math.log(survival)
    return total / reference_count


def _entropy_interval_upper(
    active: tuple[tuple[float, int], ...],
    component_peaks: tuple[float, ...],
    reference_count: float,
    lower: int,
    upper: int,
) -> float:
    """Bound a mixture by summing each component's maximum on an interval."""
    return sum(
        _entropy_at_epoch(
            ((probability, count),),
            reference_count,
            min(upper, max(lower, component_peak)),
        )
        for (probability, count), component_peak in zip(
            active, component_peaks, strict=True
        )
    )


def _entropy_precision_margin(value: float, components: int) -> float:
    """Budget roundoff across the terms in an entropy-mixture sum."""
    return 64 * components * math.ulp(max(abs(value), 1e-300))


def _split_integer_interval(lower: int, upper: int) -> int:
    """Divide enormous ranges approximately in log time, small ranges in half."""
    midpoint = math.isqrt(lower * upper)
    return midpoint if lower <= midpoint < upper else (lower + upper) // 2


def entropy_peak_decay_milestone(
    terms: Iterable[tuple[float, int]],
    *,
    reference_count: float,
    decay_fraction: float = 0.01,
    search_limit: int = 10**18,
    certificate_budget: int = 100_000,
) -> EntropyMilestone:
    """Find the global integer peak and final decay below a relative threshold.

    Terms are (one-epoch probability, count). The denominator is the complete
    frozen reference family. Each component is unimodal, so summing its
    interval maximum gives an upper bound on every unsampled integer epoch.
    After the latest component peak, every summand decreases. Inconclusive
    numerical comparisons or a finite search budget return ``unresolved``.
    """
    weighted = tuple(terms)
    if not math.isfinite(reference_count) or reference_count < 0:
        raise ValueError("reference count must be finite and nonnegative")
    if not 0 < decay_fraction < 1 or not math.isfinite(decay_fraction):
        raise ValueError(
            "decay fraction must lie strictly between zero and one"
        )
    if not _is_integer_count(search_limit) or search_limit < 2:
        raise ValueError("search limit must be an integer of at least two")
    if not _is_integer_count(certificate_budget) or certificate_budget < 1:
        raise ValueError("certificate budget must be a positive integer")
    for probability, count in weighted:
        if not math.isfinite(probability) or not 0 <= probability <= 1:
            raise ValueError("probabilities must lie in [0, 1]")
        if not _is_integer_count(count) or count < 0:
            raise ValueError("weights must be nonnegative integers")
    if reference_count == 0:
        if any(count for _, count in weighted):
            raise ValueError(
                "nonzero weights require a positive reference count"
            )
        return EntropyMilestone(
            status="not_applicable", reason="empty_reference"
        )
    active = tuple(
        (probability, count)
        for probability, count in weighted
        if 0 < probability < 1 and count > 0
    )
    if not active:
        return EntropyMilestone(status="not_applicable", reason="zero_entropy")

    component_peaks = tuple(
        math.log(0.5) / math.log1p(-probability) for probability, _ in active
    )
    latest_peak = max(component_peaks)
    if not math.isfinite(latest_peak) or latest_peak >= search_limit:
        return EntropyMilestone(
            status="unresolved", reason="component_peak_beyond_search_limit"
        )
    if latest_peak >= 2**53:
        return EntropyMilestone(
            status="unresolved", reason="integer_peak_precision_limit"
        )

    def entropy_at(epoch: int) -> float:
        return _entropy_at_epoch(active, reference_count, epoch)

    peak_upper = max(1, math.ceil(latest_peak))
    candidates = {1, peak_upper}
    for component_peak in component_peaks:
        candidates.update(
            (max(1, math.floor(component_peak)), math.ceil(component_peak))
        )
    peak_epoch = min(candidates, key=lambda epoch: (-entropy_at(epoch), epoch))
    peak_value = entropy_at(peak_epoch)
    if peak_value <= 0:
        return EntropyMilestone(status="not_applicable", reason="zero_entropy")

    pending: list[tuple[float, int, int]] = []

    def push_interval(lower: int, upper: int) -> None:
        if lower <= upper:
            bound = _entropy_interval_upper(
                active, component_peaks, reference_count, lower, upper
            )
            heappush(pending, (-bound, lower, upper))

    push_interval(1, peak_epoch - 1)
    push_interval(peak_epoch + 1, peak_upper)
    visited = 0
    while pending:
        visited += 1
        if visited > certificate_budget:
            return EntropyMilestone(
                status="unresolved", reason="peak_certificate_budget_exceeded"
            )
        negative_bound, lower, upper = heappop(pending)
        bound = -negative_bound
        margin = _entropy_precision_margin(max(bound, peak_value), len(active))
        if bound < peak_value - margin:
            continue
        if lower == upper:
            value = entropy_at(lower)
            if abs(value - peak_value) <= margin:
                return EntropyMilestone(
                    status="unresolved", reason="integer_peak_precision_limit"
                )
            if value > peak_value:
                peak_epoch, peak_value = lower, value
            continue
        midpoint = _split_integer_interval(lower, upper)
        push_interval(lower, midpoint)
        push_interval(midpoint + 1, upper)

    threshold = decay_fraction * peak_value
    margin = _entropy_precision_margin(peak_value, len(active))

    def threshold_side(epoch: int) -> int:
        value = entropy_at(epoch)
        if value > threshold + margin:
            return 1
        if value < threshold - margin:
            return -1
        return 0

    tail_start = max(peak_epoch, peak_upper)
    tail_side = threshold_side(tail_start)
    if tail_side == 0:
        return EntropyMilestone(
            status="unresolved", reason="integer_decay_precision_limit"
        )
    if tail_side > 0:
        lower = tail_start
        upper = min(search_limit, lower * 2)
        while threshold_side(upper) > 0:
            if upper == search_limit:
                return EntropyMilestone(
                    status="unresolved", reason="decay_beyond_search_limit"
                )
            upper = min(search_limit, upper * 2)
        if threshold_side(upper) == 0:
            return EntropyMilestone(
                status="unresolved", reason="integer_decay_precision_limit"
            )
        while lower + 1 < upper:
            midpoint = (lower + upper) // 2
            side = threshold_side(midpoint)
            if side == 0:
                return EntropyMilestone(
                    status="unresolved", reason="integer_decay_precision_limit"
                )
            if side > 0:
                lower = midpoint
            else:
                upper = midpoint
        final_decay = upper
    else:
        # Search from the right. Intervals whose component-wise upper bound is
        # below the threshold contain no later above-threshold excursion.
        stack = [(peak_epoch, tail_start - 1)]
        last_above = None
        while stack:
            visited += 1
            if visited > certificate_budget:
                return EntropyMilestone(
                    status="unresolved",
                    reason="decay_certificate_budget_exceeded",
                )
            lower, upper = stack.pop()
            if lower > upper:
                continue
            bound = _entropy_interval_upper(
                active, component_peaks, reference_count, lower, upper
            )
            if bound < threshold - margin:
                continue
            if lower == upper:
                side = threshold_side(lower)
                if side == 0:
                    return EntropyMilestone(
                        status="unresolved",
                        reason="integer_decay_precision_limit",
                    )
                if side > 0:
                    last_above = lower
                    break
                continue
            midpoint = _split_integer_interval(lower, upper)
            stack.append((lower, midpoint))
            stack.append((midpoint + 1, upper))
        if last_above is None:
            return EntropyMilestone(
                status="unresolved", reason="unbracketed_decay"
            )
        final_decay = last_above + 1
    if (
        threshold_side(final_decay - 1) != 1
        or threshold_side(final_decay) != -1
    ):
        return EntropyMilestone(
            status="unresolved", reason="unbracketed_decay"
        )
    return EntropyMilestone(
        status="resolved",
        peak_epoch=peak_epoch,
        peak_value=peak_value,
        final_decay_epoch=final_decay,
    )


def span_entropy_milestone(
    histogram: Mapping[int, int],
    K: int,
    q: int,
    *,
    decay_fraction: float = 0.01,
    search_limit: int = 10**18,
) -> EntropyMilestone:
    """Search milestones using exactly the theory/figure span histogram."""
    total = _reference_total(histogram, K, q, 0)
    return entropy_peak_decay_milestone(
        (
            (per_epoch_probability(span, K, q), count)
            for span, count in histogram.items()
        ),
        reference_count=total,
        decay_fraction=decay_fraction,
        search_limit=search_limit,
    )


def update_seen(
    seen: set, observed: Iterable, reference: set | frozenset
) -> None:
    """Record only distinct identities in the frozen reference family."""
    seen.update(identity for identity in observed if identity in reference)


def coverage_fraction(seen: set, reference: set | frozenset) -> float | None:
    """Observed-reference fraction, or missing for an empty reference family."""
    if not reference:
        return None
    return len(seen.intersection(reference)) / len(reference)
