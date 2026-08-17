### arm R
| algorithm | order-free ID | order-free OOD | order-free drop | order-dep. ID | order-dep. OOD | order-dep. drop | scalar (MSE) ID → OOD |
|---|---|---|---|---|---|---|---|
| `bellman_ford` | 0.860 | 0.913 | -0.053 | 0.893 | 0.574 | 0.319 | 0.067 → 0.025 |
| `dijkstra` | 0.923 | 0.452 | 0.471 | 0.909 | 0.137 | 0.772 | 0.032 → 0.049 |
| `mst_prim` | 0.893 | 0.448 | 0.445 | 0.875 | 0.087 | 0.788 | 0.017 → 0.023 |
| `floyd_warshall` | 0.994 | 0.965 | 0.029 | 0.950 | 0.079 | 0.870 | 0.012 → 0.133 |

Order-free is a `mask` or a `categorical`, order-dependent is a `pointer` or a `mask_one` — an `argmax` whose candidate set is the node set. Both are scored over the hints and the output together. The `scalar` column is a mean squared error, lower is better, and it is pooled with neither: it is printed as two numbers rather than a drop for that reason.
| algorithm | seeds | ID `n = 16` | OOD `n = 64` (32 traj.) | OOD `n = 64` (128 traj.) ± s.e.m. | ± 95% CI (traj.) | at trained depth | floor (MPNN) | ceiling (Triplet-GMPNN) |
|---|---|---|---|---|---|---|---|---|
| `bellman_ford` | 3 | 0.9688 | 0.6768 | 0.6783 ± 0.0024 | ± 0.0117 | — | 0.9201 ± 0.0028 | 0.9739 ± 0.0019 |
| `dijkstra` | 3 | 0.9857 | 0.0677 | 0.0684 ± 0.0232 | ± 0.0073 | — | 0.9150 ± 0.0050 | 0.9605 ± 0.0060 |
| `mst_prim` | 3 | 0.9616 | 0.0435 | 0.0417 ± 0.0083 | ± 0.0035 | — | 0.6908 ± 0.0756 | 0.8639 ± 0.0133 |
| `floyd_warshall` | 3 | 0.8888 | 0.0741 | 0.0728 ± 0.0141 | ± 0.0020 | — | 0.2674 ± 0.0177 | 0.4852 ± 0.0104 |

Every column is at `n = 64` out of distribution; a parenthesis in a header counts **trajectories**, not nodes. `± s.e.m.` is the standard error over seeds, the anchors' own convention (theirs: 3 seeds for the floor, 10 for the ceiling). `± 95% CI` is 1.96 standard errors over the 128 trajectories within a run, averaged over the seeds. `at trained depth` is the same models on the same split run for the number of rounds they trained at rather than the number the sample's trajectory asks for.
  4 of 4 rows written
### arm S
| algorithm | order-free ID | order-free OOD | order-free drop | order-dep. ID | order-dep. OOD | order-dep. drop | scalar (MSE) ID → OOD |
|---|---|---|---|---|---|---|---|
| `bellman_ford` | 0.859 | 0.900 | -0.041 | 0.899 | 0.605 | 0.293 | 0.065 → 0.023 |
| `dijkstra` | 0.920 | 0.444 | 0.476 | 0.901 | 0.158 | 0.743 | 0.034 → 0.033 |
| `mst_prim` | 0.895 | 0.455 | 0.440 | 0.877 | 0.097 | 0.780 | 0.017 → 0.017 |
| `floyd_warshall` | 0.994 | 0.965 | 0.028 | 0.942 | 0.074 | 0.868 | 0.012 → 0.148 |

Order-free is a `mask` or a `categorical`, order-dependent is a `pointer` or a `mask_one` — an `argmax` whose candidate set is the node set. Both are scored over the hints and the output together. The `scalar` column is a mean squared error, lower is better, and it is pooled with neither: it is printed as two numbers rather than a drop for that reason.
| algorithm | seeds | ID `n = 16` | OOD `n = 64` (32 traj.) | OOD `n = 64` (128 traj.) ± s.e.m. | ± 95% CI (traj.) | at trained depth | floor (MPNN) | ceiling (Triplet-GMPNN) |
|---|---|---|---|---|---|---|---|---|
| `bellman_ford` | 3 | 0.9720 | 0.7140 | 0.7302 ± 0.0181 | ± 0.0122 | — | 0.9201 ± 0.0028 | 0.9739 ± 0.0019 |
| `dijkstra` | 3 | 0.9824 | 0.0931 | 0.0948 ± 0.0157 | ± 0.0090 | — | 0.9150 ± 0.0050 | 0.9605 ± 0.0060 |
| `mst_prim` | 3 | 0.9635 | 0.0495 | 0.0459 ± 0.0105 | ± 0.0036 | — | 0.6908 ± 0.0756 | 0.8639 ± 0.0133 |
| `floyd_warshall` | 3 | 0.8662 | 0.0556 | 0.0571 ± 0.0036 | ± 0.0020 | — | 0.2674 ± 0.0177 | 0.4852 ± 0.0104 |

Every column is at `n = 64` out of distribution; a parenthesis in a header counts **trajectories**, not nodes. `± s.e.m.` is the standard error over seeds, the anchors' own convention (theirs: 3 seeds for the floor, 10 for the ceiling). `± 95% CI` is 1.96 standard errors over the 128 trajectories within a run, averaged over the seeds. `at trained depth` is the same models on the same split run for the number of rounds they trained at rather than the number the sample's trajectory asks for.
  4 of 4 rows written
### arm O
| algorithm | order-free ID | order-free OOD | order-free drop | order-dep. ID | order-dep. OOD | order-dep. drop | scalar (MSE) ID → OOD |
|---|---|---|---|---|---|---|---|
| `bellman_ford` | 0.862 | 0.882 | -0.021 | 0.882 | 0.599 | 0.283 | 0.089 → 0.036 |
| `dijkstra` | 0.704 | 0.363 | 0.341 | 0.689 | 0.340 | 0.349 | 0.110 → 0.147 |
| `mst_prim` | 0.649 | 0.353 | 0.296 | 0.657 | 0.279 | 0.378 | 0.031 → 0.030 |
| `floyd_warshall` | 0.872 | 0.968 | -0.096 | 0.386 | 0.070 | 0.316 | 0.212 → 0.126 |

Order-free is a `mask` or a `categorical`, order-dependent is a `pointer` or a `mask_one` — an `argmax` whose candidate set is the node set. Both are scored over the hints and the output together. The `scalar` column is a mean squared error, lower is better, and it is pooled with neither: it is printed as two numbers rather than a drop for that reason.
| algorithm | seeds | ID `n = 16` | OOD `n = 64` (32 traj.) | OOD `n = 64` (128 traj.) ± s.e.m. | ± 95% CI (traj.) | at trained depth | floor (MPNN) | ceiling (Triplet-GMPNN) |
|---|---|---|---|---|---|---|---|---|
| `bellman_ford` | 3 | 0.9798 | 0.7236 | 0.7323 ± 0.0144 | ± 0.0118 | — | 0.9201 ± 0.0028 | 0.9739 ± 0.0019 |
| `dijkstra` | 3 | 0.9909 | 0.5081 | 0.4773 ± 0.1762 | ± 0.0226 | — | 0.9150 ± 0.0050 | 0.9605 ± 0.0060 |
| `mst_prim` | 3 | 0.9661 | 0.4556 | 0.4633 ± 0.0496 | ± 0.0144 | — | 0.6908 ± 0.0756 | 0.8639 ± 0.0133 |
| `floyd_warshall` | 3 | 0.5671 | 0.0680 | 0.0705 ± 0.0290 | ± 0.0022 | — | 0.2674 ± 0.0177 | 0.4852 ± 0.0104 |

Every column is at `n = 64` out of distribution; a parenthesis in a header counts **trajectories**, not nodes. `± s.e.m.` is the standard error over seeds, the anchors' own convention (theirs: 3 seeds for the floor, 10 for the ceiling). `± 95% CI` is 1.96 standard errors over the 128 trajectories within a run, averaged over the seeds. `at trained depth` is the same models on the same split run for the number of rounds they trained at rather than the number the sample's trajectory asks for.
  4 of 4 rows written
### arm F
| algorithm | order-free ID | order-free OOD | order-free drop | order-dep. ID | order-dep. OOD | order-dep. drop | scalar (MSE) ID → OOD |
|---|---|---|---|---|---|---|---|
| `bellman_ford` | 0.856 | 0.873 | -0.017 | 0.412 | 0.208 | 0.204 | 0.129 → 0.103 |
| `dijkstra` | 0.464 | 0.356 | 0.107 | 0.297 | 0.107 | 0.190 | 0.166 → 0.102 |
| `mst_prim` | 0.453 | 0.354 | 0.099 | 0.262 | 0.053 | 0.209 | 0.041 → 0.024 |
| `floyd_warshall` | 0.831 | 0.960 | -0.129 | 0.221 | 0.060 | 0.161 | 0.247 → 0.202 |

Order-free is a `mask` or a `categorical`, order-dependent is a `pointer` or a `mask_one` — an `argmax` whose candidate set is the node set. Both are scored over the hints and the output together. The `scalar` column is a mean squared error, lower is better, and it is pooled with neither: it is printed as two numbers rather than a drop for that reason.
| algorithm | seeds | ID `n = 16` | OOD `n = 64` (32 traj.) | OOD `n = 64` (128 traj.) ± s.e.m. | ± 95% CI (traj.) | at trained depth | floor (MPNN) | ceiling (Triplet-GMPNN) |
|---|---|---|---|---|---|---|---|---|
| `bellman_ford` | 3 | 0.4635 | 0.2041 | 0.2140 ± 0.0062 | ± 0.0080 | — | 0.9201 ± 0.0028 | 0.9739 ± 0.0019 |
| `dijkstra` | 3 | 0.4199 | 0.1318 | 0.1418 ± 0.0000 | ± 0.0065 | — | 0.9150 ± 0.0050 | 0.9605 ± 0.0060 |
| `mst_prim` | 3 | 0.3477 | 0.0462 | 0.0446 ± 0.0002 | ± 0.0026 | — | 0.6908 ± 0.0756 | 0.8639 ± 0.0133 |
| `floyd_warshall` | 3 | 0.2205 | 0.0572 | 0.0582 ± 0.0399 | ± 0.0023 | — | 0.2674 ± 0.0177 | 0.4852 ± 0.0104 |

Every column is at `n = 64` out of distribution; a parenthesis in a header counts **trajectories**, not nodes. `± s.e.m.` is the standard error over seeds, the anchors' own convention (theirs: 3 seeds for the floor, 10 for the ceiling). `± 95% CI` is 1.96 standard errors over the 128 trajectories within a run, averaged over the seeds. `at trained depth` is the same models on the same split run for the number of rounds they trained at rather than the number the sample's trajectory asks for.
  4 of 4 rows written
