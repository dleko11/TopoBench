"""Keep grid coordinates, retry identity and evaluation protocols distinct."""

import csv
from types import SimpleNamespace

import pytest

from scripts.partitioning.analyze_qk_performance import (
    ProjectSpec,
    aggregate,
    extract_run,
    main,
    plot_cells,
)


def make_run(run_id="first", state="finished", score=0.7, **stream):
    return SimpleNamespace(
        id=run_id,
        name=run_id,
        state=state,
        url="https://example.test/run",
        created_at="2026-11-10",
        heartbeat_at="2026-11-10",
        metadata={},
        config={
            "seed": 0,
            "model": {"model_name": "gcn"},
            "dataset": {
                "split_params": {
                    "data_seed": 0,
                    "split_type": "random",
                    "learning_setting": "transductive",
                },
                "loader": {
                    "parameters": {
                        "cluster": {"num_parts": 32},
                        "stream": {"q": 4, "q_val": 4, "q_test": 4, **stream},
                    }
                },
            },
            "test_inference": {"ensemble_runs": 10},
        },
        summary={
            "test_inference/batched/auroc": score,
            "test_inference/ensemble/auroc": 0.8,
            "_runtime": 100,
        },
    )


def rows(run):
    return extract_run(ProjectSpec("questions", "partitioning", "test"), run)


def test_grid_protocols_seeds_and_retries():
    first = make_run()
    retry = make_run("retry", score=0.75)
    retry.created_at = "2026-11-11"
    failed = make_run("failed", state="failed", score=0.99)
    failed.created_at = "2026-11-12"
    other_seed = make_run("seed1", score=0.85)
    other_seed.config["seed"] = 1
    other_q = make_run("q8", q=8, q_val=8, q_test=8)
    other_k = make_run("k64")
    other_k.config["dataset"]["loader"]["parameters"]["cluster"][
        "num_parts"
    ] = 64
    cells = aggregate(
        sum(
            (
                rows(run)
                for run in (first, retry, failed, other_seed, other_q, other_k)
            ),
            [],
        )
    )
    assert len(cells) == 6
    cell = next(
        cell
        for cell in cells
        if cell["num_parts"] == 32
        and cell["stream_q"] == 4
        and cell["protocol"] == "batched"
    )
    assert cell["n"] == 2
    assert cell["test_mean"] == pytest.approx(0.8)
    assert cell["test_std"] == pytest.approx(0.1 / 2**0.5)
    assert set(cell["run_ids"].split(",")) == {"retry", "seed1"}
    assert all(cell["test_std"] is None for cell in cells if cell["n"] == 1)


@pytest.mark.parametrize(
    "score,state",
    [(None, "finished"), (float("nan"), "finished"), (0.9, "running")],
)
def test_incomplete_metrics_are_not_zero(score, state):
    extracted = rows(make_run(score=score, state=state))
    assert not extracted[0]["usable"]
    cells = aggregate(extracted)
    assert not any(cell["protocol"] == "batched" for cell in cells)


def test_missing_identity_fails_visibly():
    run = make_run()
    del run.config["dataset"]["split_params"]["data_seed"]
    with pytest.raises(ValueError, match="identity"):
        aggregate(rows(run))


def test_heatmap_exports_missing_cells_and_keeps_anchors(tmp_path):
    anchor = make_run("anchor", q=20, q_val=20, q_test=20)
    anchor.config["dataset"]["loader"]["parameters"]["cluster"][
        "num_parts"
    ] = 500
    cells = aggregate(rows(make_run()) + rows(anchor))
    assert len(cells) == 4
    paths = plot_cells(cells, tmp_path, [32, 64, 128], [4, 8, 16])
    assert len(paths) == 6
    assert all(path.stat().st_size > 1000 for path in paths)
    assert (
        "missing"
        in next(path for path in paths if path.suffix == ".svg").read_text()
    )


def test_cli_exports_raw_and_cell_csv(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        [
            "analyze",
            "--project",
            "questions:test",
            "--output-dir",
            str(tmp_path),
        ],
    )
    monkeypatch.setattr(
        "wandb.Api", lambda: SimpleNamespace(runs=lambda project: [make_run()])
    )
    main()
    with (tmp_path / "cells.csv").open() as stream:
        cells = list(csv.DictReader(stream))
    assert len(cells) == 2
    assert {cell["metric"] for cell in cells} == {"auroc"}
    assert all(cell["test_std"] == "" for cell in cells)
    with (tmp_path / "runs.csv").open() as stream:
        assert len(list(csv.DictReader(stream))) == 2
