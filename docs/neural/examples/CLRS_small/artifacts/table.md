| algorithm | seeds | ID `n = 16` | OOD `n = 64` (32 traj.) | OOD `n = 64` (128 traj.) ± s.e.m. | ± 95% CI (traj.) | at trained depth | floor (MPNN) | ceiling (Triplet-GMPNN) |
|---|---|---|---|---|---|---|---|---|
| `minimum` | 3 | 1.0000 | 0.6667 | 0.7005 ± 0.0751 | ± 0.0772 | 0.9479 | 0.8534 ± 0.0088 | 0.9778 ± 0.0055 |
| `bfs` | 3 | 0.9928 | 0.8592 | 0.8556 ± 0.0201 | ± 0.0121 | 0.8501 | 0.9989 ± 0.0005 | 0.9973 ± 0.0004 |
| `bellman_ford` | 3 | 0.9681 | 0.5658 | 0.5737 ± 0.0213 | ± 0.0120 | 0.5703 | 0.9201 ± 0.0028 | 0.9739 ± 0.0019 |
| `dijkstra` | 3 | 0.9694 | 0.0498 | 0.0610 ± 0.0218 | ± 0.0085 | 0.6258 | 0.9150 ± 0.0050 | 0.9605 ± 0.0060 |
| `mst_prim` | 3 | 0.9531 | 0.0352 | 0.0320 ± 0.0141 | ± 0.0035 | 0.2150 | 0.6908 ± 0.0756 | 0.8639 ± 0.0133 |
| `dag_shortest_paths` | 3 | 0.9889 | 0.5721 | 0.5631 ± 0.0929 | ± 0.0311 | 0.5591 | 0.9624 ± 0.0056 | 0.9819 ± 0.0030 |
| `floyd_warshall` | 3 | 0.8888 | 0.0741 | 0.0728 ± 0.0141 | ± 0.0020 | 0.2379 | 0.2674 ± 0.0177 | 0.4852 ± 0.0104 |
| `matrix_chain_order` | 3 | 0.9887 | 0.3834 | 0.3459 ± 0.0898 | ± 0.0289 | 0.6607 | 0.7984 ± 0.0140 | 0.9168 ± 0.0059 |

Every column is at `n = 64` out of distribution; a parenthesis in a header counts **trajectories**, not nodes. `± s.e.m.` is the standard error over seeds, the anchors' own convention (theirs: 3 seeds for the floor, 10 for the ceiling). `± 95% CI` is 1.96 standard errors over the 128 trajectories within a run, averaged over the seeds. `at trained depth` is the same models on the same split run for the number of rounds they trained at rather than the number the sample's trajectory asks for.
  8 of 8 rows written
