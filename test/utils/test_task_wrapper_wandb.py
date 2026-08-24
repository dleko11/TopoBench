"""Tests for W&B task failure finalization."""

from unittest.mock import MagicMock, patch

import pytest
from omegaconf import DictConfig

from topobench.utils.utils import task_wrapper


def test_task_wrapper_marks_wandb_run_successful() -> None:
    """A successful task closes the active W&B run with exit code zero."""
    cfg = DictConfig({"paths": {"output_dir": "logs/"}})

    with (
        patch("topobench.utils.utils.find_spec", return_value=True),
        patch("wandb.run", MagicMock()),
        patch("wandb.finish") as finish,
    ):
        task_wrapper(lambda cfg: ({}, {}))(cfg)

    finish.assert_called_once_with(exit_code=0)


def test_task_wrapper_marks_wandb_run_failed() -> None:
    """An exception closes the active W&B run with a failure exit code."""
    cfg = DictConfig({"paths": {"output_dir": "logs/"}})
    task = MagicMock(side_effect=RuntimeError("failed"))

    with (
        patch("topobench.utils.utils.find_spec", return_value=True),
        patch("wandb.run", MagicMock()),
        patch("wandb.finish") as finish,
        pytest.raises(RuntimeError, match="failed"),
    ):
        task_wrapper(task)(cfg)

    finish.assert_called_once_with(exit_code=1)
