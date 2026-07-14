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

For generated cell-space structures, opt into bounded simple cycles:

```bash
MODEL_CONFIG="cell/cwn" \
Q_VALUES="8" \
DATA_SEEDS="0 1 2" \
DATA_SPLIT_SEED="0" \
MAX_EPOCHS="200" \
RESULTS_ROOT="scripts/structural_coverage/results/cora_cell_simple_cycles_n8_np64_q8_3seeds" \
uv run scripts/structural_coverage/run.sh \
  dataset.loader.parameters.cluster.num_parts=64 \
  coverage.structure_family=cell_simple_cycles \
  +coverage.max_support_nodes=8 \
  callbacks.early_stopping=null \
  trainer.min_epochs=50
```

On Cora, `max_support_nodes=8` already generates roughly one million bounded
simple-cycle structures. Smoke-test larger values before launching multi-seed
runs.

Plot a result root with:

```bash
uv run python -m scripts.structural_coverage.plot_results \
  --results-root scripts/structural_coverage/results/cora_scn_np64_q8_5seeds \
  --aggregate-seeds
```

Generated artifacts under `results/` are intentionally ignored by Git.

## Restartable Cora cluster sweep

The cluster sweep covers four structural profiles, six train partition batch
sizes, and ten seeds (240 runs):

- `simplicial`: SCN with clique simplicial structures;
- `hypergraph`: UniGNN with 1-hop hyperedges;
- `cell_basis`: CWN with the standard cycle-basis cells;
- `cell_simple_coverage`: the same convenient CWN training run while coverage
  is measured against all simple cycles on at most eight support nodes.

Training is fixed to 200 epochs without early stopping. Cora uses 64 global
partitions, the data split is fixed at seed 0, train `q` takes values
`1,2,4,8,16,32`, and validation/test grouping is held fixed at 64 so that the
training curves are comparable across `q`. The configured random split uses a
50% training mask; local checks across split seeds 0--9 showed that this already
places training nodes in all 64 partitions.

Run the sweep from inside an allocated GPU job:

```bash
uv run python -m scripts.structural_coverage.sweep run
```

The default is ten concurrent processes on physical GPU 0. Override the
worker count or distribute workers round-robin across visible GPUs with:

```bash
uv run python -m scripts.structural_coverage.sweep run \
  --workers 20 \
  --gpus 0,1
```

The command is restartable: successful tasks with complete full and portable
artifacts are skipped. Use filters for pilots or targeted retries:

```bash
uv run python -m scripts.structural_coverage.sweep run \
  --profiles simplicial,cell_simple_coverage \
  --q-values 1,8,32 \
  --seeds 0 \
  --workers 2
```

Useful non-training commands are:

```bash
# List the deterministic task matrix.
uv run python -m scripts.structural_coverage.sweep plan

# Rebuild and display the completion manifest.
uv run python -m scripts.structural_coverage.sweep status

# Re-export portable artifacts from existing complete result directories.
uv run python -m scripts.structural_coverage.sweep export

# Print commands without running them.
uv run python -m scripts.structural_coverage.sweep run --dry-run
```

Complete artifacts, checkpoints, and logs are stored under
`results/cora_np64_sweep/`. The transferable tree under
`results_for_plotting/cora_np64_sweep/` contains only:

- `empirical_coverage.csv`;
- `theory_curves.csv`;
- `span_histogram.csv`;
- `metrics.csv`;
- `run_metadata.json`;
- per-run `status.json` and a sweep-level `manifest.csv`.

Both result roots are ignored by Git. Global structural universes are cached
under the full results root and protected by file locks, so concurrent workers
reuse the expensive bounded-simple-cycle enumeration safely.
