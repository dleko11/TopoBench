"""W&B phase and resource tracking helpers."""

from __future__ import annotations

import contextlib
import json
import os
import time
from collections.abc import Iterator, Mapping
from typing import Any

import torch

try:  # pragma: no cover - optional runtime dependency
    import psutil
except Exception:  # pragma: no cover - psutil can be absent in minimal envs
    psutil = None

try:  # pragma: no cover - rank utility import differs across Lightning builds
    from lightning.pytorch.utilities.rank_zero import rank_zero_only
except Exception:  # pragma: no cover
    rank_zero_only = None

try:  # pragma: no cover - W&B may be disabled in some environments
    from lightning.pytorch.loggers.wandb import WandbLogger
except Exception:  # pragma: no cover
    WandbLogger = None


PHASE_IDS: dict[str, int] = {
    "dataset_load": 10,
    "full_graph_preprocessing": 20,
    "partition_build": 30,
    "datamodule_init": 40,
    "model_init": 50,
    "callbacks_init": 60,
    "trainer_init": 70,
    "fit": 100,
    "train_epoch": 110,
    "validation_epoch": 120,
    "test_epoch": 130,
    "checkpoint_load": 200,
    "val_best_rerun": 210,
    "test_best_rerun": 220,
    "test_inference_batched": 230,
    "test_inference_full_graph": 240,
    "test_inference_ensemble": 250,
    "val_cache_build": 300,
}

EVENT_IDS: dict[str, int] = {
    f"{phase}_start": phase_id * 10 + 1
    for phase, phase_id in PHASE_IDS.items()
}
EVENT_IDS.update(
    {
        f"{phase}_end": phase_id * 10 + 2
        for phase, phase_id in PHASE_IDS.items()
    }
)

_CURRENT_TRACKER: PhaseResourceTracker | None = None


def set_current_phase_tracker(tracker: PhaseResourceTracker | None) -> None:
    """Set the process-local phase tracker used by callbacks/datamodules.

    Parameters
    ----------
    tracker : PhaseResourceTracker or None
        Tracker to make available through module-level helpers.
    """
    global _CURRENT_TRACKER
    _CURRENT_TRACKER = tracker


def get_current_phase_tracker() -> PhaseResourceTracker | None:
    """Return the process-local phase tracker.

    Returns
    -------
    PhaseResourceTracker or None
        Configured tracker, or ``None`` if tracking has not been initialized.
    """
    return _CURRENT_TRACKER


@contextlib.contextmanager
def track_phase(
    phase: str,
    *,
    epoch: int | None = None,
    global_step: int | None = None,
    extra: Mapping[str, Any] | None = None,
) -> Iterator[None]:
    """Track a phase with the current process-local tracker if available.

    Parameters
    ----------
    phase : str
        Name of the phase being tracked.
    epoch : int or None, optional
        Current epoch to log with the phase marker.
    global_step : int or None, optional
        Current global step to log with the phase marker.
    extra : Mapping[str, Any] or None, optional
        Additional W&B metrics to include in the marker rows.

    Yields
    ------
    None
        Control to the wrapped block.
    """
    tracker = get_current_phase_tracker()
    if tracker is None:
        yield
        return

    with tracker.track(
        phase,
        epoch=epoch,
        global_step=global_step,
        extra=extra,
    ):
        yield


class PhaseResourceTracker:
    """Log phase boundary markers and resource snapshots to W&B.

    The tracker is deliberately best-effort: failures while collecting or
    logging diagnostics must not interrupt training.

    Parameters
    ----------
    loggers : Any
        Lightning logger or list of loggers from which W&B runs are collected.
    """

    def __init__(self, loggers: Any) -> None:
        self._wandb_runs = self._collect_wandb_runs(loggers)
        self._active_starts: dict[str, float] = {}
        self._summary_written = False

    @property
    def enabled(self) -> bool:
        """Return whether at least one W&B run is available for logging.

        Returns
        -------
        bool
            ``True`` when W&B marker logging can be attempted.
        """
        return bool(self._wandb_runs)

    def initialize(self) -> None:
        """Force W&B initialization and store stable ID maps in summary."""
        if not self.enabled or self._summary_written:
            return

        payload = {
            "tracking/phase_id_map": json.dumps(PHASE_IDS, sort_keys=True),
            "tracking/event_id_map": json.dumps(EVENT_IDS, sort_keys=True),
            "tracking/resource_tracking_enabled": True,
            "tracking/system_metrics_note": (
                "W&B system metrics are sampled separately; tracking/* rows "
                "mark phase boundaries and explicit resource snapshots."
            ),
        }
        for run in self._wandb_runs:
            with contextlib.suppress(Exception):
                for key, value in payload.items():
                    run.summary[key] = value
        self._summary_written = True

    def start_phase(
        self,
        phase: str,
        *,
        epoch: int | None = None,
        global_step: int | None = None,
        extra: Mapping[str, Any] | None = None,
    ) -> None:
        """Log a phase-start marker and reset CUDA peak counters.

        Parameters
        ----------
        phase : str
            Name of the phase being started.
        epoch : int or None, optional
            Current epoch to attach to the marker.
        global_step : int or None, optional
            Current global step to attach to the marker.
        extra : Mapping[str, Any] or None, optional
            Additional W&B metrics to include in the marker row.
        """
        if not self.enabled:
            return
        self._reset_cuda_peak_stats()
        self._active_starts[phase] = time.perf_counter()
        self._log_event(
            phase,
            event="start",
            is_start=True,
            is_end=False,
            duration_sec=0.0,
            epoch=epoch,
            global_step=global_step,
            extra=extra,
        )

    def end_phase(
        self,
        phase: str,
        *,
        epoch: int | None = None,
        global_step: int | None = None,
        extra: Mapping[str, Any] | None = None,
    ) -> None:
        """Log a phase-end marker with elapsed time and resource peaks.

        Parameters
        ----------
        phase : str
            Name of the phase being ended.
        epoch : int or None, optional
            Current epoch to attach to the marker.
        global_step : int or None, optional
            Current global step to attach to the marker.
        extra : Mapping[str, Any] or None, optional
            Additional W&B metrics to include in the marker row.
        """
        if not self.enabled:
            return
        start = self._active_starts.pop(phase, None)
        duration_sec = (
            time.perf_counter() - start if start is not None else 0.0
        )
        self._log_event(
            phase,
            event="end",
            is_start=False,
            is_end=True,
            duration_sec=duration_sec,
            epoch=epoch,
            global_step=global_step,
            extra=extra,
        )

    @contextlib.contextmanager
    def track(
        self,
        phase: str,
        *,
        epoch: int | None = None,
        global_step: int | None = None,
        extra: Mapping[str, Any] | None = None,
    ) -> Iterator[None]:
        """Context manager that logs start/end markers for a phase.

        Parameters
        ----------
        phase : str
            Name of the phase being tracked.
        epoch : int or None, optional
            Current epoch to log with the phase markers.
        global_step : int or None, optional
            Current global step to log with the phase markers.
        extra : Mapping[str, Any] or None, optional
            Additional W&B metrics to include in marker rows.

        Yields
        ------
        None
            Control to the wrapped block.
        """
        self.start_phase(
            phase,
            epoch=epoch,
            global_step=global_step,
            extra=extra,
        )
        try:
            yield
        except Exception:
            failed_extra = dict(extra or {})
            failed_extra["tracking/phase_failed"] = 1
            self.end_phase(
                phase,
                epoch=epoch,
                global_step=global_step,
                extra=failed_extra,
            )
            raise
        else:
            self.end_phase(
                phase,
                epoch=epoch,
                global_step=global_step,
                extra=extra,
            )

    def _collect_wandb_runs(self, loggers: Any) -> list[Any]:
        """Collect initialized W&B run objects from Lightning loggers.

        Parameters
        ----------
        loggers : Any
            Logger or list of loggers to inspect.

        Returns
        -------
        list[Any]
            W&B run-like objects available on rank zero.
        """
        if not loggers:
            return []

        logger_list = loggers if isinstance(loggers, list) else [loggers]
        runs = []
        for logger in logger_list:
            is_wandb = (
                WandbLogger is not None and isinstance(logger, WandbLogger)
            ) or logger.__class__.__name__ == "WandbLogger"
            if not is_wandb or not self._is_rank_zero():
                continue
            with contextlib.suppress(Exception):
                runs.append(logger.experiment)
        return runs

    def _log_event(
        self,
        phase: str,
        *,
        event: str,
        is_start: bool,
        is_end: bool,
        duration_sec: float,
        epoch: int | None,
        global_step: int | None,
        extra: Mapping[str, Any] | None,
    ) -> None:
        """Log a single phase marker row to all W&B runs.

        Parameters
        ----------
        phase : str
            Name of the phase being logged.
        event : str
            Event kind, either ``start`` or ``end``.
        is_start : bool
            Whether this row marks phase start.
        is_end : bool
            Whether this row marks phase end.
        duration_sec : float
            Elapsed phase duration in seconds.
        epoch : int or None
            Epoch value to store with the marker.
        global_step : int or None
            Global step value to store with the marker.
        extra : Mapping[str, Any] or None
            Extra metrics to merge into the payload.
        """
        phase_id = PHASE_IDS.get(phase, -1)
        event_id = EVENT_IDS.get(f"{phase}_{event}", -1)
        payload: dict[str, Any] = {
            "tracking/phase_id": phase_id,
            "tracking/event_id": event_id,
            "tracking/is_start": int(is_start),
            "tracking/is_end": int(is_end),
            "tracking/epoch": -1 if epoch is None else int(epoch),
            "tracking/global_step": (
                -1 if global_step is None else int(global_step)
            ),
            "tracking/duration_sec": float(duration_sec),
        }
        payload.update(self._resource_snapshot())
        if extra:
            payload.update(dict(extra))

        for run in self._wandb_runs:
            with contextlib.suppress(Exception):
                run.log(payload)

    def _resource_snapshot(self) -> dict[str, float]:
        """Collect current process, child-process, and CUDA memory values.

        Returns
        -------
        dict[str, float]
            Resource metric payload for a W&B marker row.
        """
        payload: dict[str, float] = {}
        process = self._process()
        if process is not None:
            rss_mb = self._rss_mb(process)
            if rss_mb is not None:
                payload["tracking/resource/rss_mb"] = rss_mb
                payload["tracking/resource/tree_rss_mb"] = (
                    rss_mb + self._children_rss_mb(process)
                )

        if torch.cuda.is_available():
            with contextlib.suppress(Exception):
                device = torch.cuda.current_device()
                scale = 1024**2
                payload["tracking/resource/cuda_allocated_mb"] = (
                    torch.cuda.memory_allocated(device) / scale
                )
                payload["tracking/resource/cuda_reserved_mb"] = (
                    torch.cuda.memory_reserved(device) / scale
                )
                payload["tracking/resource/cuda_peak_allocated_mb"] = (
                    torch.cuda.max_memory_allocated(device) / scale
                )
                payload["tracking/resource/cuda_peak_reserved_mb"] = (
                    torch.cuda.max_memory_reserved(device) / scale
                )
        return payload

    def _reset_cuda_peak_stats(self) -> None:
        """Reset CUDA peak memory counters when CUDA is available."""
        if not torch.cuda.is_available():
            return
        with contextlib.suppress(Exception):
            torch.cuda.reset_peak_memory_stats(torch.cuda.current_device())

    def _process(self) -> Any | None:
        """Return a psutil process handle for the current process.

        Returns
        -------
        Any or None
            Current process handle, or ``None`` when psutil is unavailable.
        """
        if psutil is None:
            return None
        with contextlib.suppress(Exception):
            return psutil.Process(os.getpid())
        return None

    def _rss_mb(self, process: Any) -> float | None:
        """Return resident memory for a process in megabytes.

        Parameters
        ----------
        process : Any
            Psutil process-like object.

        Returns
        -------
        float or None
            Resident set size in megabytes, if available.
        """
        with contextlib.suppress(Exception):
            return process.memory_info().rss / 1024**2
        return None

    def _children_rss_mb(self, process: Any) -> float:
        """Return cumulative child-process resident memory.

        Parameters
        ----------
        process : Any
            Psutil process-like object.

        Returns
        -------
        float
            Sum of child-process RSS values in megabytes.
        """
        total = 0.0
        with contextlib.suppress(Exception):
            for child in process.children(recursive=True):
                child_rss = self._rss_mb(child)
                if child_rss is not None:
                    total += child_rss
        return total

    def _is_rank_zero(self) -> bool:
        """Return whether the current process should log global markers.

        Returns
        -------
        bool
            ``True`` on rank zero or when rank detection is unavailable.
        """
        if rank_zero_only is None:
            return True
        return getattr(rank_zero_only, "rank", 0) == 0
