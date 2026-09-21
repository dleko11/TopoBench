# Reddit: structural support recovery versus q

This is the remaining **three-domain support-recovery plot**, not the older
epoch/entropy experiment. It measures the fraction of full-graph reference
structures whose complete supporting nodes occur in a common mini-batch at
least once during 200 epochs. Curves report means and sample standard deviations
over ten independent reshuffling seeds. No model is trained.

## Fixed settings and scope

- One fixed METIS partition with K=10,000 scheduling slots. METIS may leave
  some slots empty; all nonempty clusters participate without renumbering.
  Empty slots remain in the shuffle so the q grid and theoretical probabilities
  still use K=10,000. Actual nonempty and empty counts are logged and saved in
  `provenance.json`.
- q = 1, 2, 4, 8, 10, 20, 40, 100, 200, 500, 1,000, 2,000, 5,000, 10,000.
  Each q divides K. A seed uses the same epoch permutations across q values.
- One centre-indexed closed one-hop hyperedge per node, including identical
  memberships with different centres. Expected count: 232,965.
- All graph triangles as simplicial 2-cells. Expected count: 8,360,338,411.
- A full-graph NetworkX cycle basis, retaining cycles of length at most nine.
  Its supporting nodes remain fixed throughout the experiment. The plotted
  quantity is **not** identical batch-local cycle-basis recovery.
- Read only `reddit_graph.npz`, a SciPy sparse adjacency. Do not pass
  `reddit_data.npz` or a processed tensor file containing features.
- Edges are made unweighted, duplicate entries are combined, self-loops removed,
  and CSR indices sorted. The input must then have 232,965 nodes and 57,307,946
  undirected edges. Asymmetric input is rejected, not silently symmetrized.

No lifting implementation is edited. Reference supports are streamed into exact
cluster-signature counts, retaining multiplicity, instead of storing billions
of triangle objects or any feature matrices. Equal cluster signatures may be
aggregated because they have identical support-availability events. Cycle-basis
selection still uses `networkx.cycle_basis` and its full pre-filter basis.

**Important:** sorted CSR traversal can select a different valid cycle basis
from an older experiment with different insertion order. The actual cellular
reference count is reported, not forced to match the manuscript's old count.
Do not combine these curves with old curves from another partition or basis.
This runner regenerates all three families under one graph and partition.

## Server preparation

Use the server's existing TopoBench Python environment, including NumPy, SciPy,
NetworkX, PyTorch, PyG, its METIS-capable `pyg-lib` or `torch-sparse` backend,
Matplotlib, and pytest. No GPU is required. Do not blindly install a different
PyTorch/PyG build into an already working environment.

From the TopoBench clone, after the new branch is pushed:

```bash
git fetch origin
git switch --track origin/structural-recovery-server
```

If that branch already exists locally, use `git switch structural-recovery-server`
and `git pull --ff-only` instead. Do not discard existing server changes.

Activate the environment that runs the other TopoBench experiments. The commands
below use its `python`. If using the project's prepared uv environment, replace
`python` with `uv run --no-sync python`.

First run the small tests and synthetic smoke, which do not load Reddit:

```bash
MPLBACKEND=Agg python -m pytest test/scripts/test_reddit_support_recovery.py -q
MPLBACKEND=Agg python -u -m scripts.structural_coverage.run_reddit_support_recovery \
  --smoke --output-dir results/reddit_support_smoke
```

The smoke explicitly labels its figure as synthetic, not Reddit evidence.
Use a new output directory or `--resume` if repeating this command.

Find the existing raw graph file. It is normally under the Reddit dataset's
`raw` directory. If using ripgrep, include ignored dataset files:

```bash
rg --files --hidden --no-ignore datasets -g reddit_graph.npz
```

Set its actual server path, then check the input without partitioning or lifting:

```bash
REDDIT_GRAPH=/absolute/path/to/Reddit/raw/reddit_graph.npz
python -u -m scripts.structural_coverage.run_reddit_support_recovery \
  --adjacency "$REDDIT_GRAPH" --check-input \
  --output-dir results/reddit_support_200
```

## Full run

Reserve a **high-memory CPU server allocation** before launching. Input checking
does not estimate full enumeration RAM or guarantee feasibility. Streaming avoids
billions of stored Python references, but graph topology, NetworkX's complete
cycle basis, and distinct cluster-signature counts can still be very large.
Enumerating 8.36 billion triangles and running the exact sampling diagnostic may
take substantial time. No full-Reddit timing or peak-RAM claim has been verified.

```bash
mkdir -p logs
nohup env MPLBACKEND=Agg PYTHONUNBUFFERED=1 \
  python -u -m scripts.structural_coverage.run_reddit_support_recovery \
  --adjacency "$REDDIT_GRAPH" \
  --output-dir results/reddit_support_200 \
  > logs/reddit_support_200.log 2>&1 &
tail -f logs/reddit_support_200.log
```

On a managed cluster, use the same Python command inside the site's scheduler
allocation instead of running it on the login node. Choose memory and wall-time
according to the server allocation, not the synthetic test duration.

Optionally supply `--partition-labels /absolute/path/labels.npy` if an exact
existing partition is available. It must be a one-dimensional integer array in
original node-ID order, with every label in 0,...,9999; unused labels are allowed.
Otherwise a new featureless METIS partition is created, saved, and reused.
It is not claimed to be the partition of an earlier training run.

For an interrupted run, use the identical command plus `--resume`. Graph and
partition hashes, configuration, and numerical-library versions must match.
Completed family signature files and family/q results are reused. Interruption
during signature enumeration restarts that family. Interruption during one q
restarts that q. Checkpoints are atomic but are not per-triangle/per-seed snapshots.
Do not launch two processes into the same output directory.

## Outputs and interpretation

`results/reddit_support_200/` contains:

- `q_recovery.pdf`, `.png`, `.svg`: the three-domain q-axis plot.
- `q_recovery_source_data.csv`: full reference denominators, observable counts,
  theoretical expectations, empirical means and sample SDs, ten raw seed counts.
- `q_recovery_manifest.json`, `run_manifest.json`, `provenance.json`: configuration,
  graph and partition hashes, original source revision, traversal convention,
  and nonempty/empty cluster counts for new runs.
- `partition_labels.npy`: the common fixed partition, with no feature vectors.
- `*_signatures.npz`: restartable support-signature multiplicities.
- `*_q*.json`: per-epoch integer counts and summaries for each completed q.
- `completion.json`: written after the final exports complete.

Copy the figure, CSV and manifests back for review. Do not replace the manuscript
figure automatically. Compare the observed reference counts and clarify any
cycle-basis count difference before updating the paper's tables or captions.
