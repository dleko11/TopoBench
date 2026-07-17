# Structural Coverage Experiments

This folder contains the structural-recovery experiment, the restartable Cora
sweep, and the plotting code used for the appendix.

## Run experiments

Run an individual configuration through `run.sh` from the TopoBench
environment. The dataset, model, lifting, seeds, epochs, and result root can be
set through environment variables and Hydra overrides.

```bash
MODEL_CONFIG="simplicial/scn" \
Q_VALUES="8" \
DATA_SEEDS="0 1 2 3 4" \
MAX_EPOCHS="200" \
RESULTS_ROOT="scripts/structural_coverage/results/example" \
uv run scripts/structural_coverage/run.sh \
  dataset.loader.parameters.cluster.num_parts=64
```

The complete Cora sweep is restartable:

```bash
uv run python -m scripts.structural_coverage.sweep run
uv run python -m scripts.structural_coverage.sweep status
uv run python -m scripts.structural_coverage.sweep export
```

It contains four lifting profiles, `q = 1, 2, 4, 8, 16, 32`, and ten seeds,
for 240 runs in total. Full checkpoints and logs are written under `results/`
and remain ignored by Git.

## Plot the appendix sweep

Portable plotting data are stored under
`results_for_plotting/cora_np64_sweep/`. Each profile/`q`/seed directory
contains:

```text
empirical_coverage.csv
theory_curves.csv
span_histogram.csv
run_metadata.json
metrics.csv
status.json
```

Generate the current appendix figure set with:

```bash
uv run python -m scripts.structural_coverage.plot_appendix_sweep_results_classic
```

The main paper-facing outputs are written to
`results_for_plotting/cora_np64_sweep/appendix_figures_classic/`:

```text
classic_recovery_q2_q4_q8_q16_every5.pdf
classic_entropy_q2_q4_q8_q16_every5.pdf
classic_entropy_peak_to_1pct_analytic.pdf
```

The recovery and entropy grids use `q = 2, 4, 8, 16`, sample every five epochs
through `T = 200`, and aggregate ten seeds. Recovery uses empirical means and
sample standard deviations together with the theoretical expectation. The
entropy grid reads the normalized entropy in nats from `theory_curves.csv`.
The peak-to-1% figure is calculated analytically from `span_histogram.csv` and
`K_eff` in `run_metadata.json`, so its endpoints may extend beyond `T = 200`.

For a single legacy result root, use `plot_results.py` instead. The other
appendix plotting scripts are retained as design experiments; new figure work
should use `plot_appendix_sweep_results_classic.py`.
