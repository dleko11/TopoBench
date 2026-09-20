"""Reference identities and partition spans for structural recovery."""

import math
from dataclasses import FrozenInstanceError
from itertools import product

import networkx as nx
import numpy as np
import pytest

from scripts.structural_coverage import recovery_core
from scripts.structural_coverage.recovery_core import (
    ReferenceStructure,
    cluster_span,
    cycle_basis_references,
    cycle_identity,
    neighbourhood_references,
    summarize_repetitions,
    triangle_references,
)


def test_aggregation_uses_sample_sd_and_counts_repetitions():
    summary = summarize_repetitions([0.0, 1.0, 1.0])

    assert summary.n == 3
    assert summary.mean == pytest.approx(2 / 3)
    assert summary.sample_sd == pytest.approx(math.sqrt(1 / 3))


def test_single_repetition_has_undefined_sample_sd():
    summary = summarize_repetitions([0.75])

    assert summary.n == 1
    assert summary.mean == 0.75
    assert summary.sample_sd is None


def test_identical_memberships_keep_distinct_centres():
    graph = nx.Graph([(0, 1)])
    refs = neighbourhood_references(graph)

    assert len(refs) == 2
    assert refs[0].support == refs[1].support == frozenset({0, 1})
    assert {ref.identity for ref in refs} == {
        ("hyperedge", 0),
        ("hyperedge", 1),
    }


def test_isolated_centre_has_its_own_singleton_hyperedge():
    graph = nx.Graph([(11, 42)])
    graph.add_node(900)

    refs = neighbourhood_references(graph)

    assert len(refs) == graph.number_of_nodes()
    assert ReferenceStructure(("hyperedge", 900), frozenset({900})) in refs
    assert ReferenceStructure(("hyperedge", 11), frozenset({11, 42})) in refs


def test_hyperedge_identity_alone_does_not_imply_full_support():
    complete = ReferenceStructure(("hyperedge", 11), frozenset({11, 42}))
    truncated = ReferenceStructure(("hyperedge", 11), frozenset({11}))

    assert complete.identity == truncated.identity
    assert complete != truncated


def test_references_are_immutable():
    ref = ReferenceStructure(("hyperedge", 7), frozenset({7}))

    with pytest.raises(FrozenInstanceError):
        ref.support = frozenset()


def test_cycle_identity_uses_boundary_not_only_nodes():
    a = cycle_identity([0, 1, 2, 3])
    b = cycle_identity([0, 2, 1, 3])

    assert a != b
    assert a == cycle_identity([2, 3, 0, 1])
    assert a == cycle_identity([3, 2, 1, 0])


def test_cycle_identity_preserves_noncontiguous_global_node_ids():
    identity = cycle_identity([11, 42, 900])

    assert identity == (
        "cycle",
        ((11, 42), (11, 900), (42, 900)),
    )


def test_span_counts_distinct_clusters_not_nodes():
    labels = {0: 4, 1: 4, 2: 9}

    assert cluster_span({0, 1, 2}, labels) == 2


def test_span_uses_global_ids_and_arbitrary_cluster_labels():
    labels = {11: 100, 42: 100, 900: 250}

    assert cluster_span(frozenset({11, 42, 900}), labels) == 2


def test_triangle_references_keep_global_node_identity():
    graph = nx.Graph([(11, 42), (42, 900), (900, 11), (900, 901)])

    assert triangle_references(graph) == [
        ReferenceStructure(("triangle", 11, 42, 900), frozenset({11, 42, 900}))
    ]


def test_cycle_basis_includes_nine_cycle_but_excludes_ten_cycle():
    graph = nx.Graph()
    graph.add_edges_from((i, (i + 1) % 9) for i in range(9))
    graph.add_edges_from((i, 9 + ((i - 9 + 1) % 10)) for i in range(9, 19))

    refs = cycle_basis_references(graph, max_length=9)

    assert len(refs) == 1
    assert refs[0].support == frozenset(range(9))
    assert refs[0].identity == cycle_identity(list(range(9)))


def test_cycle_basis_excludes_self_loop_cycles_like_selective_lifting():
    graph = nx.Graph([(0, 0), (1, 2), (2, 3), (3, 1)])

    refs = cycle_basis_references(graph)

    assert refs == [
        ReferenceStructure(cycle_identity([1, 2, 3]), frozenset({1, 2, 3}))
    ]


def test_per_epoch_probability_golden_spans():
    assert recovery_core.per_epoch_probability(1, K=4, q=2) == 1.0
    assert recovery_core.per_epoch_probability(2, K=4, q=2) == pytest.approx(
        1 / 3
    )
    assert recovery_core.per_epoch_probability(3, K=4, q=2) == 0.0


GROUPINGS_K4_Q2 = (
    (frozenset({0, 1}), frozenset({2, 3})),
    (frozenset({0, 2}), frozenset({1, 3})),
    (frozenset({0, 3}), frozenset({1, 2})),
)


def fits_one_group(cluster_support, groups):
    """Independent finite-enumeration oracle, with no probability formula."""
    return any(set(cluster_support) <= set(group) for group in groups)


@pytest.mark.parametrize(
    ("support", "expected"),
    [({0}, 1.0), ({0, 1}, 1 / 3), ({0, 1, 2}, 0.0)],
)
def test_per_epoch_probability_matches_all_k4_q2_groupings(support, expected):
    measured = sum(
        fits_one_group(support, groups) for groups in GROUPINGS_K4_Q2
    ) / len(GROUPINGS_K4_Q2)

    assert measured == pytest.approx(expected)
    assert recovery_core.per_epoch_probability(
        len(support), K=4, q=2
    ) == pytest.approx(measured)


@pytest.mark.parametrize(
    ("support", "expected"),
    [({0}, 1.0), ({0, 1}, 5 / 9), ({0, 1, 2}, 0.0)],
)
def test_two_epoch_probability_matches_all_nine_ordered_pairs(
    support, expected
):
    grouping_pairs = list(product(GROUPINGS_K4_Q2, repeat=2))
    measured = sum(
        fits_one_group(support, first) or fits_one_group(support, second)
        for first, second in grouping_pairs
    ) / len(grouping_pairs)

    assert len(grouping_pairs) == 9
    assert measured == pytest.approx(expected)
    assert recovery_core.cumulative_probability(
        recovery_core.per_epoch_probability(len(support), K=4, q=2), T=2
    ) == pytest.approx(measured)


def test_shared_epoch_schedules_partition_clusters_into_equal_groups():
    schedules = recovery_core.generate_epoch_schedules(
        K=12, q=3, epochs=5, seed=17
    )

    assert len(schedules) == 5
    for epoch in schedules:
        assert len(epoch) == 4
        assert all(len(group) == 3 for group in epoch)
        assert sorted(cluster for group in epoch for cluster in group) == list(
            range(12)
        )


def test_shared_epoch_schedules_reproduce_from_seed():
    first = recovery_core.generate_epoch_schedules(
        K=32, q=8, epochs=10, seed=2026
    )
    second = recovery_core.generate_epoch_schedules(
        K=32, q=8, epochs=10, seed=2026
    )

    assert first == second


def test_shared_epoch_schedules_do_not_depend_on_lifting_family_order():
    def scheduled_families(family_order):
        schedules = recovery_core.generate_epoch_schedules(
            K=8, q=2, epochs=3, seed=123
        )
        return {family: schedules for family in family_order}

    forward = scheduled_families(("hypergraph", "cellular", "simplicial"))
    reverse = scheduled_families(("simplicial", "cellular", "hypergraph"))

    assert forward == reverse
    assert len({id(schedules) for schedules in forward.values()}) == 1
    assert isinstance(forward["hypergraph"], tuple)
    assert isinstance(forward["hypergraph"][0][0], tuple)


@pytest.mark.parametrize(
    ("K", "q", "epochs", "seed"),
    [
        (0, 1, 1, 1),
        (4, 0, 1, 1),
        (4, 3, 1, 1),
        (4, 2, -1, 1),
        (4.0, 2, 1, 1),
        (4, 2, 1, True),
    ],
)
def test_shared_epoch_schedules_reject_invalid_configuration(
    K, q, epochs, seed
):
    with pytest.raises(ValueError):
        recovery_core.generate_epoch_schedules(
            K=K, q=q, epochs=epochs, seed=seed
        )


@pytest.mark.parametrize(
    ("c", "K", "q"),
    [(0, 4, 2), (5, 4, 2), (1, 0, 1), (1, 4, 0), (1, 4, 5), (1, 5, 2)],
)
def test_per_epoch_probability_rejects_invalid_equal_batch_settings(c, K, q):
    with pytest.raises(ValueError):
        recovery_core.per_epoch_probability(c, K=K, q=q)


@pytest.mark.parametrize(
    ("c", "K", "q"),
    [
        (2.5, 4, 2),
        (float("nan"), 4, 2),
        (float("inf"), 4, 2),
        (True, 4, 2),
        (1, 4.0, 2),
        (1, float("nan"), 2),
        (1, True, 1),
        (1, 4, 2.0),
        (1, 4, float("inf")),
        (1, 4, True),
    ],
)
def test_per_epoch_probability_requires_integer_nonboolean_counts(c, K, q):
    with pytest.raises(ValueError):
        recovery_core.per_epoch_probability(c, K=K, q=q)


def test_per_epoch_probability_accepts_numpy_integer_counts():
    assert recovery_core.per_epoch_probability(
        np.int64(2), np.int64(4), np.int64(2)
    ) == pytest.approx(1 / 3)


def test_cumulative_probability_handles_exact_endpoints_and_large_horizon():
    assert recovery_core.cumulative_probability(0, T=0) == 0.0
    assert recovery_core.cumulative_probability(1, T=0) == 0.0
    assert recovery_core.cumulative_probability(0, T=10) == 0.0
    assert recovery_core.cumulative_probability(1, T=10) == 1.0
    assert recovery_core.cumulative_probability(0.5, T=100_000) == 1.0
    assert recovery_core.cumulative_probability(1 / 3, T=2) == pytest.approx(
        5 / 9
    )


@pytest.mark.parametrize(
    ("p", "T"), [(-0.1, 1), (1.1, 1), (float("nan"), 1), (0.5, -1)]
)
def test_cumulative_probability_rejects_invalid_inputs(p, T):
    with pytest.raises(ValueError):
        recovery_core.cumulative_probability(p, T=T)


@pytest.mark.parametrize("T", [2.5, float("nan"), float("inf"), True])
def test_cumulative_probability_requires_integer_nonboolean_epoch_horizon(T):
    with pytest.raises(ValueError):
        recovery_core.cumulative_probability(0.5, T=T)


def test_cumulative_probability_accepts_numpy_integer_epoch_horizon():
    assert recovery_core.cumulative_probability(
        0.5, T=np.int64(2)
    ) == pytest.approx(0.75)


def test_bernoulli_entropy_uses_natural_logarithms_and_exact_endpoints():
    expected = -(1 / 3) * math.log(1 / 3) - (2 / 3) * math.log(2 / 3)
    assert recovery_core.bernoulli_entropy(0.0) == 0.0
    assert recovery_core.bernoulli_entropy(1.0) == 0.0
    assert recovery_core.bernoulli_entropy(1 / 3) == pytest.approx(expected)


def _manual_weighted_entropy(terms, reference_count, epoch):
    total = 0.0
    for probability, count in terms:
        rho = 1 - (1 - probability) ** epoch
        if 0 < rho < 1:
            total += count * (
                -rho * math.log(rho) - (1 - rho) * math.log(1 - rho)
            )
    return total / reference_count


def test_entropy_milestone_single_probability_checks_neighbouring_epochs():
    probability = 0.2
    continuous_peak = math.log(0.5) / math.log1p(-probability)
    neighbours = (math.floor(continuous_peak), math.ceil(continuous_peak))

    milestone = recovery_core.entropy_peak_decay_milestone(
        [(probability, 1)], reference_count=1
    )

    assert milestone.status == "resolved"
    assert milestone.peak_epoch == max(
        neighbours,
        key=lambda epoch: _manual_weighted_entropy(
            [(probability, 1)], 1, epoch
        ),
    )
    assert milestone.peak_value == pytest.approx(
        _manual_weighted_entropy([(probability, 1)], 1, milestone.peak_epoch)
    )


def test_entropy_milestone_matches_independent_integer_enumeration():
    terms = [(0.5, 3), (0.08, 1), (0.0, 2)]
    values = [
        _manual_weighted_entropy(terms, 6, epoch) for epoch in range(201)
    ]
    peak = max(range(len(values)), key=lambda epoch: values[epoch])
    final_crossing = (
        max(
            epoch
            for epoch, value in enumerate(values)
            if value >= 0.01 * values[peak]
        )
        + 1
    )

    milestone = recovery_core.entropy_peak_decay_milestone(
        terms, reference_count=6
    )

    assert (milestone.peak_epoch, milestone.final_decay_epoch) == (
        peak,
        final_crossing,
    )
    assert values[final_crossing] < 0.01 * values[peak]


def test_entropy_milestone_uses_final_not_first_threshold_crossing():
    terms = [(0.5, 100), (0.0001, 2)]
    values = [
        _manual_weighted_entropy(terms, 102, epoch) for epoch in range(100_001)
    ]
    peak = max(range(len(values)), key=lambda epoch: values[epoch])
    threshold = 0.01 * values[peak]
    first_dip = next(
        epoch for epoch in range(peak + 1, 1000) if values[epoch] < threshold
    )
    final_crossing = (
        max(epoch for epoch, value in enumerate(values) if value >= threshold)
        + 1
    )

    milestone = recovery_core.entropy_peak_decay_milestone(
        terms, reference_count=102
    )

    assert first_dip < 1500 < final_crossing
    assert values[1500] > threshold
    assert (milestone.peak_epoch, milestone.final_decay_epoch) == (
        peak,
        final_crossing,
    )


def test_entropy_milestone_is_invariant_to_positive_curve_scaling():
    terms = [(0.5, 3), (0.08, 1)]
    first = recovery_core.entropy_peak_decay_milestone(
        terms, reference_count=4
    )
    doubled = recovery_core.entropy_peak_decay_milestone(
        terms, reference_count=2
    )

    assert (first.peak_epoch, first.final_decay_epoch) == (
        doubled.peak_epoch,
        doubled.final_decay_epoch,
    )
    assert doubled.peak_value == pytest.approx(2 * first.peak_value)


@pytest.mark.parametrize(
    ("terms", "reference_count"),
    [
        ([], 0),
        ([(0.0, 3)], 3),
        ([(1.0, 3)], 3),
        ([(0.0, 2), (1.0, 1)], 3),
    ],
)
def test_zero_entropy_milestone_is_not_applicable(terms, reference_count):
    milestone = recovery_core.entropy_peak_decay_milestone(
        terms, reference_count=reference_count
    )

    assert milestone.status == "not_applicable"
    assert milestone.peak_epoch is None
    assert milestone.peak_value is None
    assert milestone.final_decay_epoch is None


def test_entropy_milestone_returns_unresolved_when_search_cannot_bracket():
    milestone = recovery_core.entropy_peak_decay_milestone(
        [(1e-20, 1)], reference_count=1, search_limit=10**12
    )

    assert milestone.status == "unresolved"
    assert milestone.peak_epoch is None
    assert milestone.final_decay_epoch is None


def test_entropy_milestone_handles_very_small_resolvable_probability():
    probability = 5e-6
    milestone = recovery_core.entropy_peak_decay_milestone(
        [(probability, 1)], reference_count=1, search_limit=10**8
    )

    assert milestone.status == "resolved"
    assert 100_000 < milestone.peak_epoch < 1_000_000
    assert milestone.final_decay_epoch > 1_000_000


def test_entropy_milestone_does_not_claim_indistinguishable_integer_peak():
    milestone = recovery_core.entropy_peak_decay_milestone(
        [(1e-17, 1)], reference_count=1
    )

    assert milestone.status == "unresolved"
    assert milestone.peak_epoch is None
    assert milestone.final_decay_epoch is None
    assert milestone.reason in {
        "integer_peak_precision_limit",
        "peak_certificate_budget_exceeded",
    }


def test_many_components_use_conservative_roundoff_margin():
    milestone = recovery_core.entropy_peak_decay_milestone(
        [(5e-7, 1)] * 64, reference_count=64
    )

    assert milestone.status == "unresolved"
    assert milestone.reason == "integer_peak_precision_limit"


def test_entropy_milestone_does_not_fabricate_result_on_certificate_budget():
    milestone = recovery_core.entropy_peak_decay_milestone(
        [(0.5, 1), (0.1, 1)],
        reference_count=2,
        certificate_budget=1,
    )

    assert milestone.status == "unresolved"
    assert milestone.peak_epoch is None
    assert milestone.final_decay_epoch is None


def test_span_milestone_matches_same_integer_entropy_as_theory_csv():
    histogram = {1: 1, 2: 1, 3: 1}
    milestone = recovery_core.span_entropy_milestone(histogram, K=4, q=2)

    assert milestone.status == "resolved"
    assert milestone.peak_value == pytest.approx(
        recovery_core.recovery_entropy(
            histogram, K=4, q=2, T=milestone.peak_epoch
        )
    )
    assert (
        recovery_core.recovery_entropy(
            histogram, K=4, q=2, T=milestone.final_decay_epoch
        )
        < 0.01 * milestone.peak_value
    )
    assert (
        recovery_core.recovery_entropy(
            histogram, K=4, q=2, T=milestone.final_decay_epoch - 1
        )
        >= 0.01 * milestone.peak_value
    )


@pytest.mark.parametrize("rho", [-0.1, 1.1, float("nan")])
def test_bernoulli_entropy_rejects_invalid_probabilities(rho):
    with pytest.raises(ValueError):
        recovery_core.bernoulli_entropy(rho)


def test_expected_coverage_uses_every_reference_structure_in_denominator():
    histogram = {1: 1, 2: 1, 3: 1}
    assert recovery_core.expected_coverage(histogram, K=4, q=2, T=0) == 0.0
    assert recovery_core.expected_coverage(
        histogram, K=4, q=2, T=1
    ) == pytest.approx(4 / 9)
    assert recovery_core.expected_coverage(
        histogram, K=4, q=2, T=2
    ) == pytest.approx(14 / 27)
    assert recovery_core.expected_coverage(
        histogram, K=4, q=2, T=100_000
    ) == pytest.approx(2 / 3)


def test_expected_coverage_is_missing_for_empty_reference_family():
    assert recovery_core.expected_coverage({}, K=4, q=2, T=1) is None
    assert recovery_core.expected_coverage({1: 0}, K=4, q=2, T=1) is None


@pytest.mark.parametrize("histogram", [{1: -1}, {0: 1}, {5: 1}])
def test_expected_coverage_rejects_invalid_histograms(histogram):
    with pytest.raises(ValueError):
        recovery_core.expected_coverage(histogram, K=4, q=2, T=1)


def test_expected_coverage_rejects_invalid_batching_even_if_reference_is_empty():
    with pytest.raises(ValueError):
        recovery_core.expected_coverage({}, K=4, q=5, T=1)


def test_expected_coverage_accepts_numpy_integer_histogram_counts():
    assert (
        recovery_core.expected_coverage({1: np.int64(2)}, K=4, q=2, T=1) == 1.0
    )


@pytest.mark.parametrize(
    "histogram", [{True: 1}, {2.5: 1}, {1: True}, {1: 1.0}, {1: float("nan")}]
)
def test_expected_coverage_requires_integer_nonboolean_histogram_entries(
    histogram,
):
    with pytest.raises(ValueError):
        recovery_core.expected_coverage(histogram, K=4, q=2, T=1)


def test_recovery_entropy_uses_all_reference_structures_not_only_observable_ones():
    histogram = {1: 1, 2: 1, 3: 1}
    h = -(1 / 3) * math.log(1 / 3) - (2 / 3) * math.log(2 / 3)
    value = recovery_core.recovery_entropy(histogram, K=4, q=2, T=1)

    assert value == pytest.approx(h / 3)
    assert value != pytest.approx(h / 2)
    assert recovery_core.recovery_entropy(histogram, K=4, q=2, T=0) == 0.0
    assert (
        recovery_core.recovery_entropy(histogram, K=4, q=2, T=100_000) == 0.0
    )


def test_recovery_entropy_is_missing_for_empty_reference_family():
    assert recovery_core.recovery_entropy({}, K=4, q=2, T=1) is None


def test_recovery_entropy_rejects_invalid_histogram():
    with pytest.raises(ValueError):
        recovery_core.recovery_entropy({2: -1}, K=4, q=2, T=1)


def test_repeated_observation_does_not_inflate_recovered_numerator():
    reference = {"a", "b", "c"}
    seen = set()
    recovery_core.update_seen(seen, {"a", "b"}, reference)
    recovery_core.update_seen(seen, {"b", "new_local_cell"}, reference)

    assert seen == {"a", "b"}
    assert recovery_core.coverage_fraction(seen, reference) == pytest.approx(
        2 / 3
    )


def test_coverage_fraction_excludes_nonreference_ids_and_missing_reference():
    assert (
        recovery_core.coverage_fraction({"a", "unrelated"}, {"a", "b"}) == 0.5
    )
    assert recovery_core.coverage_fraction(set(), set()) is None


def test_cell_support_and_basis_recovery_use_one_frozen_reference_universe():
    graph = nx.Graph([(0, 1), (1, 2), (2, 0), (0, 3), (3, 2)])
    a = ReferenceStructure(cycle_identity([0, 1, 2]), frozenset({0, 1, 2}))
    b = ReferenceStructure(cycle_identity([0, 2, 3]), frozenset({0, 2, 3}))
    outer = cycle_identity([0, 1, 2, 3])
    refs = (a, b)

    assert set(graph.nodes) == {0, 1, 2, 3}
    supported = recovery_core.support_available_ids(refs, set(graph.nodes))
    actual = recovery_core.matched_reference_ids(refs, {a.identity, outer})

    assert supported == {a.identity, b.identity}
    assert actual == {a.identity}
    assert len(supported) / len(refs) == 1.0
    assert len(actual) / len(refs) == 0.5


def test_cell_tracker_keeps_support_and_actual_distinct_across_batches():
    a = ReferenceStructure(cycle_identity([0, 1, 2]), frozenset({0, 1, 2}))
    b = ReferenceStructure(cycle_identity([0, 2, 3]), frozenset({0, 2, 3}))
    refs = [a, b]
    tracker = recovery_core.CellRecoveryTracker(refs)
    refs.clear()

    first_support, first_actual = tracker.record_batch(
        {0, 1, 2, 3}, {a.identity, cycle_identity([0, 1, 2, 3])}
    )
    assert first_support == {a.identity, b.identity}
    assert first_actual == {a.identity}
    assert first_actual <= first_support
    assert tracker.support_fraction == 1.0
    assert tracker.actual_fraction == 0.5

    second_support, second_actual = tracker.record_batch(
        {0, 2, 3}, {a.identity, b.identity}
    )
    assert second_support == {b.identity}
    assert second_actual == {b.identity}
    assert second_actual <= second_support
    assert tracker.actual_seen <= tracker.support_seen
    assert tracker.actual_fraction == 1.0
    assert tracker.reference_count == 2


def test_cell_tracker_excludes_nonreference_cycles_at_every_batch():
    triangle = ReferenceStructure(
        cycle_identity([0, 1, 2]), frozenset({0, 1, 2})
    )
    tracker = recovery_core.CellRecoveryTracker([triangle])

    support, actual = tracker.record_batch(
        {0, 1}, {triangle.identity, cycle_identity([0, 1, 2, 3])}
    )

    assert support == actual == set()
    assert tracker.support_fraction == tracker.actual_fraction == 0.0


def test_cell_tracker_rejects_duplicate_reference_identity():
    triangle = ReferenceStructure(
        cycle_identity([0, 1, 2]), frozenset({0, 1, 2})
    )

    with pytest.raises(ValueError, match="distinct identities"):
        recovery_core.CellRecoveryTracker([triangle, triangle])


def test_cell_basis_extraction_uses_inclusive_nine_cycle_cap():
    graph = nx.Graph()
    graph.add_edges_from((i, (i + 1) % 9) for i in range(9))
    graph.add_edges_from((i, 9 + ((i - 9 + 1) % 10)) for i in range(9, 19))

    assert recovery_core.local_cycle_basis_ids(graph, max_length=9) == {
        cycle_identity(range(9))
    }


def test_cell_same_order_full_batch_matches_reference_but_singletons_do_not():
    graph = nx.Graph([(0, 1), (1, 2), (2, 0), (0, 3), (3, 2)])
    references = cycle_basis_references(graph, max_length=9)
    tracker = recovery_core.CellRecoveryTracker(references)

    for node in graph.nodes:
        singleton = graph.subgraph([node]).copy()
        support, actual = tracker.record_batch(
            set(singleton.nodes),
            recovery_core.local_cycle_basis_ids(singleton),
        )
        assert support == actual == set()
    assert tracker.support_fraction == tracker.actual_fraction == 0.0

    support, actual = tracker.record_batch(
        set(graph.nodes), recovery_core.local_cycle_basis_ids(graph)
    )
    assert support == actual == {ref.identity for ref in references}
    assert tracker.support_fraction == tracker.actual_fraction == 1.0
