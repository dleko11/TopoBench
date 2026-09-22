"""GPU memory capacity measurements for short, training-only runs."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from lightning import Callback, LightningModule, Trainer
from lightning.pytorch.loggers import WandbLogger

from topobench.utils.phase_tracking import get_current_phase_tracker


class GPUMemoryBenchmarkCallback(Callback):
    """Save allocator peaks locally and to the existing W&B run.

    Parameters
    ----------
    result_path : str
        JSON destination for this configuration's measurements.
    """

    def __init__(self, result_path: str) -> None:
        self.result_path = Path(result_path)
        self.result: dict = {"status": "running", "completed_epochs": 0}
        self.phase = "gpu_memory_benchmark"
        self.tracker = None

    def setup(
        self, trainer: Trainer, pl_module: LightningModule, stage: str
    ) -> None:
        """Begin tracking before the model and optimizer move to CUDA.

        Parameters
        ----------
        trainer : Trainer
            Active Lightning trainer.
        pl_module : LightningModule
            Model being trained.
        stage : str
            Lightning stage, which must be ``fit``.
        """
        if stage != "fit":
            raise ValueError("GPU memory benchmark requires training.")
        device = trainer.strategy.root_device
        if device.type != "cuda":
            raise ValueError("GPU memory benchmark requires a CUDA device.")
        if trainer.world_size != 1:
            raise ValueError("Use one process and one GPU per configuration.")
        self.tracker = get_current_phase_tracker()
        if self.tracker is None or not self.tracker.enabled:
            raise ValueError("GPU memory benchmark requires W&B tracking.")
        props = torch.cuda.get_device_properties(device)
        free_bytes, total_bytes = torch.cuda.mem_get_info(device)
        self.result.update(
            gpu_name=props.name,
            gpu_total_gib=total_bytes / 1024**3,
            gpu_free_at_start_gib=free_bytes / 1024**3,
            torch_version=torch.__version__,
            cuda_version=torch.version.cuda,
            precision=str(trainer.precision),
            expected_epochs=trainer.max_epochs,
        )
        self.tracker.start_phase(self.phase)
        self._save(trainer)

    def _save(self, trainer: Trainer) -> None:
        """Persist progress and peaks, including epochs that later fail.

        Parameters
        ----------
        trainer : Trainer
            Trainer providing optimizer steps and loggers.
        """
        self.result["optimizer_steps"] = int(trainer.global_step)
        if self.tracker is not None:
            peaks = self.tracker.cuda_phase_peaks(self.phase)
            for name in ("allocated", "reserved"):
                key = f"tracking/resource/cuda_peak_{name}_mb"
                if key in peaks:
                    self.result[f"peak_{name}_gib"] = peaks[key] / 1024
        for logger in trainer.loggers:
            if isinstance(logger, WandbLogger):
                run = logger.experiment
                self.result["wandb_url"] = run.url
                self.result["wandb_run_id"] = run.id
        self.result_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.result_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(self.result, indent=2) + "\n")
        temporary.replace(self.result_path)
        payload = {
            f"gpu_memory/{key}": value for key, value in self.result.items()
        }
        for logger in trainer.loggers:
            if isinstance(logger, WandbLogger):
                logger.experiment.summary.update(payload)
                logger.experiment.log(payload)

    def on_train_epoch_end(
        self, trainer: Trainer, pl_module: LightningModule
    ) -> None:
        """Keep the maximum across every epoch, including the first.

        Parameters
        ----------
        trainer : Trainer
            Active Lightning trainer.
        pl_module : LightningModule
            Model being trained.
        """
        self.result["completed_epochs"] += 1
        self._save(trainer)

    def on_fit_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Mark success only after all requested epochs complete.

        Parameters
        ----------
        trainer : Trainer
            Active Lightning trainer.
        pl_module : LightningModule
            Model being trained.
        """
        torch.cuda.synchronize(trainer.strategy.root_device)
        self.result["status"] = (
            "success"
            if self.result["completed_epochs"] == trainer.max_epochs
            else "incomplete"
        )
        self._save(trainer)
        self.tracker.end_phase(self.phase)

    def on_exception(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        exception: BaseException,
    ) -> None:
        """Record CUDA OOM separately without swallowing the exception.

        Parameters
        ----------
        trainer : Trainer
            Active Lightning trainer.
        pl_module : LightningModule
            Model being trained.
        exception : BaseException
            Exception that interrupted the run.
        """
        self.result["status"] = (
            "cuda_oom"
            if isinstance(exception, torch.cuda.OutOfMemoryError)
            and "cuda out of memory" in str(exception).lower()
            else "error"
        )
        self.result["error_type"] = type(exception).__name__
        self.result["error"] = str(exception)
        self._save(trainer)
