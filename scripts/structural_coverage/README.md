# Structural Coverage Experiments

Training-integrated experiments for the appendix structural recovery theorem.

The main interface is:

```bash
scripts/structural_coverage/run.sh
```

By default this runs Cora with `simplicial/scn` and the matching
`graph2simplicial_default` lifting. For most runs, choose only the dataset,
model, seeds, epochs, and partition/batch sizes:

```bash
DATASET_CONFIG="graph/cocitation_cora_for_partitioning" \
MODEL_CONFIG="simplicial/scn" \
Q_VALUES="8" \
DATA_SEEDS="0 1 2 3 4" \
DATA_SPLIT_SEED="0" \
MAX_EPOCHS="50" \
RESULTS_ROOT="scripts/structural_coverage/results/cora_scn_np64_q8_5seeds" \
scripts/structural_coverage/run.sh \
  dataset.loader.parameters.cluster.num_parts=64 \
  callbacks.early_stopping=null \
  trainer.min_epochs=50
```

Run the script from an active TopoBench environment. If you use `uv`, call the
launcher through `uv run`:

```bash
uv run scripts/structural_coverage/run.sh \
  dataset.loader.parameters.cluster.num_parts=64
```

`run.sh` infers the default graph lifting from `MODEL_CONFIG`:

- `simplicial/*` -> `liftings/graph2simplicial_default`
- `cell/*` -> `liftings/graph2cell_default`
- `hypergraph/*` -> `liftings/graph2hypergraph_default`

Override `TRANSFORMS_CONFIG` or pass Hydra overrides when needed.

Plot a result root with:

```bash
uv run python -m scripts.structural_coverage.plot_results \
  --results-root scripts/structural_coverage/results/cora_scn_np64_q8_5seeds \
  --aggregate-seeds
```

Generated artifacts under `results/` are intentionally ignored by Git.
