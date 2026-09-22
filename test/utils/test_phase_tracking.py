"""Tests for phase-specific resource tracking."""

from __future__ import annotations

import os
from typing import Any

import pytest

from topobench.utils.phase_tracking import PhaseResourceTracker


class _Run:
    def __init__(self) -> None:
        self.summary: dict[str, Any] = {}
        self.history: list[dict[str, Any]] = []

    def log(self, payload: dict[str, Any]) -> None:
        self.history.append(payload)


class WandbLogger:
    def __init__(self, run: _Run) -> None:
        self.experiment = run


def _snapshot(rss_mb: float, tree_rss_mb: float) -> dict[str, float]:
    return {
        "tracking/resource/rss_mb": rss_mb,
        "tracking/resource/tree_rss_mb": tree_rss_mb,
    }


def test_rejects_nonpositive_cpu_sample_interval() -> None:
    with pytest.raises(ValueError, match="must be positive"):
        PhaseResourceTracker([], cpu_memory_sample_interval_sec=0)


def test_logs_sampled_cpu_peaks_for_nested_phases(monkeypatch) -> None:
    run = _Run()
    tracker = PhaseResourceTracker(
        WandbLogger(run), cpu_memory_sample_interval_sec=0.5
    )
    snapshots = iter(
        (
            _snapshot(100, 200),
            _snapshot(150, 250),
            _snapshot(130, 220),
            _snapshot(180, 300),
            _snapshot(160, 270),
            _snapshot(170, 290),
        )
    )
    monkeypatch.setattr(tracker, "_ensure_cpu_sampler", lambda: None)
    monkeypatch.setattr(
        tracker,
        "_cpu_memory_snapshot",
        lambda: next(snapshots),
    )
    monkeypatch.setattr(tracker, "_resource_snapshot", lambda: {})

    tracker.start_phase("fit")
    tracker._sample_cpu_peaks()
    tracker.start_phase("train_epoch")
    tracker._sample_cpu_peaks()
    tracker.end_phase("train_epoch")
    tracker.end_phase("fit")

    end_rows = {
        row["tracking/phase_id"]: row
        for row in run.history
        if row["tracking/is_end"] == 1
    }
    train_row = end_rows[110]
    fit_row = end_rows[100]
    assert train_row["tracking/resource/rss_peak_mb"] == 180
    assert train_row["tracking/resource/tree_rss_peak_mb"] == 300
    assert fit_row["tracking/resource/rss_peak_mb"] == 180
    assert fit_row["tracking/resource/tree_rss_peak_mb"] == 300


def test_initialization_records_cpu_sample_interval(monkeypatch) -> None:
    run = _Run()
    tracker = PhaseResourceTracker(
        WandbLogger(run), cpu_memory_sample_interval_sec=0.5
    )
    monkeypatch.setattr(tracker, "_ensure_cpu_sampler", lambda: None)

    tracker.initialize()

    assert run.summary["tracking/cpu_memory_sample_interval_sec"] == 0.5


def test_tracking_is_disabled_outside_owner_process(monkeypatch) -> None:
    run = _Run()
    owner_pid = os.getpid()
    tracker = PhaseResourceTracker(WandbLogger(run))
    monkeypatch.setattr(
        "topobench.utils.phase_tracking.os.getpid",
        lambda: owner_pid + 1,
    )

    tracker.initialize()
    with tracker.track("feature_lifting"):
        pass

    assert not tracker.enabled
    assert run.summary == {}
    assert run.history == []


def test_cuda_parent_peak_survives_nested_resets(monkeypatch) -> None:
    run = _Run()
    tracker = PhaseResourceTracker(WandbLogger(run))
    allocated = "tracking/resource/cuda_peak_allocated_mb"
    reserved = "tracking/resource/cuda_peak_reserved_mb"
    counters = {allocated: 900, reserved: 1000}
    monkeypatch.setattr(tracker, "_cuda_peak_snapshot", lambda: dict(counters))
    monkeypatch.setattr(
        tracker,
        "_reset_cuda_peak_stats",
        lambda: counters.update({allocated: 10, reserved: 20}),
    )
    monkeypatch.setattr(tracker, "_resource_snapshot", dict)
    monkeypatch.setattr(
        tracker, "_start_cpu_peak_tracking", lambda phase: None
    )
    monkeypatch.setattr(tracker, "_finish_cpu_peak_tracking", lambda phase: {})

    tracker.start_phase("fit")
    counters.update({allocated: 500, reserved: 600})
    tracker.start_phase("train_epoch")
    counters.update({allocated: 200, reserved: 300})
    tracker.start_phase("feature_lifting")
    counters.update({allocated: 50, reserved: 80})
    tracker.end_phase("feature_lifting")
    tracker.end_phase("train_epoch")
    tracker.start_phase("validation_epoch")
    counters.update({allocated: 30, reserved: 40})
    tracker.end_phase("validation_epoch")
    tracker.end_phase("fit")

    ends = {
        row["tracking/phase_id"]: row
        for row in run.history
        if row["tracking/is_end"]
    }
    assert ends[100][allocated] == 500
    assert ends[100][reserved] == 600
    assert ends[110][allocated] == 200
    assert ends[110][reserved] == 300
    assert ends[120][allocated] == 30
    assert tracker._active_cuda_peaks == {}
