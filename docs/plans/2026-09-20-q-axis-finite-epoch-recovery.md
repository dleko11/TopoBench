# Q-axis finite-epoch recovery implementation plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Produce four validated plots of the fraction of fixed full-graph references recovered after 200 epochs as a function of the cluster-group size (q).

**Architecture:** Keep the existing frozen full-graph reference and partition loaders. Build cluster-signature groups once per dataset, then evaluate the same seeded epoch permutations for each valid (q). Export ten raw counts, mean, sample standard deviation, analytical expectation, and observability ceiling for every point before drawing the figures.

**Tech Stack:** Python 3.11, NumPy, NetworkX, PyTorch Geometric, Matplotlib, pytest. Use `/Users/leone/miniconda3/envs/tb_challenge/bin/python` and set `MPLCONFIGDIR=/private/tmp/cluster-tnn-recovery.7To8Le/mplconfig`.

---

### Task 1: Compute paired finite-epoch recovery for a q grid

**Files:**
- Modify: `scripts/structural_coverage/support_recovery_multidataset.py`
- Test: `test/scripts/test_support_recovery_multidataset.py`

**Step 1: Write failing tests.** Add a toy reference set with (K=4), `q_values=[1,2,4]`, and two seeds. For each q, compare the new q-sweep result with `analyze_recovery(..., q=q, epochs=3)` and an explicit batch-node oracle. Assert that the (q=1) endpoint equals the span-one fraction, that (q=K) reaches one at epoch 1, that all seed counts stay within the corresponding observable count, and that q order does not change any output.

**Step 2: Run the targeted tests to verify failure.**

Run: `MPLCONFIGDIR=/private/tmp/cluster-tnn-recovery.7To8Le/mplconfig /Users/leone/miniconda3/envs/tb_challenge/bin/python -m pytest test/scripts/test_support_recovery_multidataset.py -q`

Expected: the new tests fail because the q-sweep API does not yet exist.

**Step 3: Implement the minimal q-sweep API.** Add `analyze_q_sweep(references, labels, K, q_values, seeds, epochs)` returning one `analyze_recovery`-compatible result per q. Validate strictly increasing, unique, positive q values dividing K. Reuse the existing `build_signature_groups` at the largest q, filtering span groups and histograms for smaller q. Generate the same seeded shuffle sequence across q values. Keep full-graph reference counts and denominators invariant. Reuse the existing counter so the toy parity test constrains behavior.

**Step 4: Run the targeted tests to verify passing.**

**Step 5: Commit only the task's code and tests.**

### Task 2: Export and plot q-axis results

**Files:**
- Modify: `scripts/structural_coverage/run_support_recovery_multidataset.py`
- Test: `test/scripts/test_support_recovery_multidataset.py`

**Step 1: Write failing tests.** Use a toy q-sweep result. Assert that the exported CSV has one row per family and q, stores ten raw epoch-200 counts, fixed reference denominators, observable counts, mean, sample SD, and theoretical expectation. Assert each plotted line's x and y data match those CSV values and that its x-axis label is `Clusters per mini-batch, q`.

**Step 2: Run tests to verify failure.**

**Step 3: Implement.** Add the four approved q grids from the design document, a `run_q_sweep_one` entry point, a q-summary CSV/manifest exporter, and one figure per dataset using blue/orange/pink dashed lines, hollow markers, light one-sample-SD bands, a logarithmic q axis, and 0--100% y axis. Keep the existing epoch-axis output intact under a separate filename or directory. Record each input manifest and make no manuscript edits.

**Step 4: Run tests to verify passing.**

**Step 5: Commit only the task's code and tests.**

### Task 3: Execute, validate, and deliver the four figures

**Files:**
- Output: `/Users/leone/projects/6a01b3c0efa7a88126bf2a71/outputs/structural_recovery_q_sweep_2026-09-20/<dataset>/`

**Step 1: Run each approved grid.** Use the same processed graph and saved fixed partition as the completed epoch-axis runs. Use 200 epochs and reshuffling seeds 0--9. No OGBN Products or Reddit run is included.

**Step 2: Validate every output.** Check the graph fingerprints and paper reference counts, exact q grid, ten seed counts per point, monotone per-seed epoch counts, bounded counts, deterministic q=1 and q=K endpoints, theoretical expectation in [0,1], CSV means and SDs, and plotted data coordinates. Spot-check at the previously run selected q against the saved epoch-axis CSV's epoch-200 result for each family.

**Step 3: Review the PNGs visually.** Fix any legend overlap or unreadable q ticks, then regenerate only affected figures from stored summary data rather than rerunning the structural simulation.

**Step 4: Run all recovery tests.**

Run: `MPLCONFIGDIR=/private/tmp/cluster-tnn-recovery.7To8Le/mplconfig /Users/leone/miniconda3/envs/tb_challenge/bin/python -m pytest test/scripts/test_support_recovery_multidataset.py test/scripts/test_recovery_io.py test/scripts/test_recovery_runner.py test/scripts/test_recovery_core.py test/scripts/test_recovery_plots.py test/scripts/test_recovery_pipeline_parity.py -q`

Expected: all tests pass. Report the generated PNG/PDF/CSV links, descriptive numbers, and the support-versus-exact-cell-cycle distinction. Do not alter or push the paper.
