# Reddit structural support recovery server plan

**Goal:** Reproduce the three-domain fraction recovered by epoch 200 versus q on Reddit without loading or saving node features. No entropy run and no training.

**Architecture:** Add a standalone server CLI beside the existing four-dataset runner. Read a sparse topology-only adjacency, create or import a single fixed METIS assignment, and checkpoint full-graph cluster-support signatures with their multiplicities. Count empirical recovery using the existing tested reshuffling counter. Stream reference enumeration rather than retaining billions of Python triangle objects. Keep neighbourhood, triangle, and length-at-most-nine NetworkX cycle-basis definitions unchanged. Cycle basis construction still uses NetworkX's complete basis before filtering.

**Files:** `scripts/structural_coverage/run_reddit_support_recovery.py`, `test/scripts/test_reddit_support_recovery.py`, `scripts/structural_coverage/REDDIT_SUPPORT.md`.

## Task 1: Tests first

Test topology loading without feature files, indexed duplicate neighbourhoods, triangle and cycle support parity with existing reference extractors, empirical counts against the existing analyzer, q=1 and q=K, invalid partitions, and restart identity checks. Run targeted pytest and confirm failure before implementation.

## Task 2: Server entry point

Implement topology-only input validation, a fixed K=10000 partition (all clusters sampled), streamed reference signatures, checkpoint/restart, 200 epochs and seeds 0--9, endpoint exports and the established plot style. Provide an explicitly synthetic smoke test and an input-only check. Persist graph and partition hashes, library versions, reference counts, per-seed integer counts, and runtime by stage. Never reuse results from another partition. No features or model execution.

## Task 3: Verify and hand off

Run all recovery tests and the synthetic CLI end-to-end, inspect the exported plot, check help and restart behavior, and commit only relevant files on `structural-recovery-server`. Document server environment requirements, raw adjacency path, preflight, smoke, full run, outputs, and CPU/RAM caveats. Do not launch Reddit locally or push.
