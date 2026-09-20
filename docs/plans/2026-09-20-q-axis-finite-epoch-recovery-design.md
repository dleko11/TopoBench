# Finite-epoch recovery as a function of cluster-group size

## Purpose

The primary four-dataset figures should answer how the fraction of fixed full-graph references actually encountered during training changes with the number of clusters per mini-batch, (q). The existing epoch-axis figures answer a different question and remain separate diagnostics. No manuscript file is changed in this task.

## Experimental definition

For each dataset, keep the graph, METIS partition, (K), and reference structures fixed. For each (q), independently reshuffle all (K) clusters once per epoch, partition the permutation into equal groups of (q), and mark a reference seen when all clusters containing its supporting nodes occur in one group. Use exactly 200 complete epochs for each of 10 seeds (0--9). Divide the number of distinct references seen by epoch 200 by the number of all full-graph references in that family. Plot the mean of the ten fractions and a light band of plus/minus one sample standard deviation.

Cellular references are the globally selected cycle-basis cells of length at most nine. The plotted event is availability of their complete supporting nodes, not selection of an identical cycle in a locally recomputed basis. Hyperedges retain their centre identity even if two membership sets coincide. Simplicial 2-cells are graph triangles. The denominator and graph reference counts must stay fixed across all (q) values.

## Configurations

Use the saved partition assignments for Cora Full ((K=32)), Amazon Ratings ((K=32)), and Questions ((K=500)). Use the recorded local METIS assignment for Coauthor Physics ((K=2000)). Include only (q) values dividing (K), so every epoch covers all clusters in equal-sized batches. The grids are:

| Dataset | (q) values | Selected training (q) |
| --- | --- | ---: |
| Cora Full | 1, 2, 4, 8, 16, 32 | 4 |
| Amazon Ratings | 1, 2, 4, 8, 16, 32 | 8 |
| Questions | 1, 2, 5, 10, 20, 50, 100, 250, 500 | 50 |
| Coauthor Physics | 1, 2, 4, 10, 20, 40, 100, 200, 500, 1000, 2000 | 20 |

The (q=K) point is the full-graph structural endpoint, not a claim that the complete lifted domain was built. The x-axis is the actual (q) value, with logarithmic spacing to keep small and large values legible. Separate panels or files may use different tick subsets, but plotted points must include every configured value.

## Computation and provenance

Reuse the validated ordered processed graphs and partition manifests from the completed four-dataset experiment. Construct each family of full-graph references and its cluster signatures once per dataset. For each seed, generate one fresh 200-epoch permutation stream and reuse the same permutation at every (q), slicing it into the corresponding group sizes. This makes the comparison across (q) paired while preserving independent reshufflings across seeds. Reuse reference signatures, without local lifting at each mini-batch.

Write one CSV row per dataset, family, and (q), containing full-graph reference count, (q)-observable count, ten epoch-200 recovered counts, their mean fraction and sample standard deviation, and the analytical expected fraction. Retain graph/partition fingerprints, seed list, epoch horizon, and the exact (q) grid in a manifest. Produce PNG, PDF, and SVG figures in the approved blue/orange/pink, hollow-marker style.

## Checks

Before accepting a figure, assert that all graph-order and partition checks pass and each family count agrees with the paper's structural table. Check (q=1) against its deterministic span-one count, (q=K) against complete recovery after the first epoch, and each seed count against the fixed denominator and observable ceiling. Verify a toy oracle against explicit batch-node membership and verify that the plotted coordinates equal the CSV means. Report the numerical results without interpreting support availability as predictive performance or exact cycle-basis recovery.
