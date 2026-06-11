"""Lightning callback for W&B phase/resource tracking."""

from __future__ import annotations

from lightning import Callback, LightningModule, Trainer

from topobench.utils.phase_tracking import get_current_phase_tracker


class PhaseResourceTrackingCallback(Callback):
    """Log Lightning fit/train/validation/test phase boundaries."""

    def _start(self, phase: str, trainer: Trainer) -> None:
        """Log a phase-start marker.

        Parameters
        ----------
        phase : str
            Name of the phase being started.
        trainer : Trainer
            Active Lightning trainer.
        """
        tracker = get_current_phase_tracker()
        if tracker is None:
            return
        tracker.start_phase(
            phase,
            epoch=getattr(trainer, "current_epoch", None),
            global_step=getattr(trainer, "global_step", None),
        )

    def _end(self, phase: str, trainer: Trainer) -> None:
        """Log a phase-end marker.

        Parameters
        ----------
        phase : str
            Name of the phase being ended.
        trainer : Trainer
            Active Lightning trainer.
        """
        tracker = get_current_phase_tracker()
        if tracker is None:
            return
        tracker.end_phase(
            phase,
            epoch=getattr(trainer, "current_epoch", None),
            global_step=getattr(trainer, "global_step", None),
        )

    def on_fit_start(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
    ) -> None:
        """Log fit start.

        Parameters
        ----------
        trainer : Trainer
            Active Lightning trainer.
        pl_module : LightningModule
            Lightning module being trained.
        """
        self._start("fit", trainer)

    def on_fit_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
    ) -> None:
        """Log fit end.

        Parameters
        ----------
        trainer : Trainer
            Active Lightning trainer.
        pl_module : LightningModule
            Lightning module being trained.
        """
        self._end("fit", trainer)

    def on_train_epoch_start(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
    ) -> None:
        """Log train epoch start.

        Parameters
        ----------
        trainer : Trainer
            Active Lightning trainer.
        pl_module : LightningModule
            Lightning module being trained.
        """
        self._start("train_epoch", trainer)

    def on_train_epoch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
    ) -> None:
        """Log train epoch end.

        Parameters
        ----------
        trainer : Trainer
            Active Lightning trainer.
        pl_module : LightningModule
            Lightning module being trained.
        """
        self._end("train_epoch", trainer)

    def on_validation_epoch_start(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
    ) -> None:
        """Log validation epoch start.

        Parameters
        ----------
        trainer : Trainer
            Active Lightning trainer.
        pl_module : LightningModule
            Lightning module being validated.
        """
        self._start("validation_epoch", trainer)

    def on_validation_epoch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
    ) -> None:
        """Log validation epoch end.

        Parameters
        ----------
        trainer : Trainer
            Active Lightning trainer.
        pl_module : LightningModule
            Lightning module being validated.
        """
        self._end("validation_epoch", trainer)

    def on_test_epoch_start(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
    ) -> None:
        """Log test epoch start.

        Parameters
        ----------
        trainer : Trainer
            Active Lightning trainer.
        pl_module : LightningModule
            Lightning module being tested.
        """
        self._start("test_epoch", trainer)

    def on_test_epoch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
    ) -> None:
        """Log test epoch end.

        Parameters
        ----------
        trainer : Trainer
            Active Lightning trainer.
        pl_module : LightningModule
            Lightning module being tested.
        """
        self._end("test_epoch", trainer)
