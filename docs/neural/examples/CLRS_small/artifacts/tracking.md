| algorithm | probe | kind | ID (mean over steps) | OOD first | OOD best | OOD last | reached |
|---|---|---|---|---|---|---|---|
| `minimum` | `pred_h` | node/pointer | 1.000 | 0.038 | 0.141 | 0.134 | 0.14 |
| `minimum` | `min_h` | node/mask_one | 0.963 | 0.375 | 0.875 | 0.812 | 0.91 |
| `minimum` | `i` | node/mask_one | 0.996 | 0.000 | 0.344 | 0.281 | 0.35 |
| `bfs` | `pi_h` | node/pointer | 0.845 | 0.679 | 0.845 | 0.845 | 1.00 |
| `bellman_ford` | `pi_h` | node/pointer | 0.825 | 0.418 | 0.510 | 0.344 | 0.62 |
| `dijkstra` | `pi_h` | node/pointer | 0.917 | 0.286 | 0.604 | 0.010 | 0.66 |
| `dijkstra` | `u` | node/mask_one | 0.733 | 1.000 | 1.000 | 0.031 | 1.36 |
| `mst_prim` | `pi_h` | node/pointer | 0.885 | 0.294 | 0.492 | 0.017 | 0.56 |
| `mst_prim` | `u` | node/mask_one | 0.802 | 1.000 | 1.000 | 0.000 | 1.25 |
| `dag_shortest_paths` | `pi_h` | node/pointer | 0.945 | 0.777 | 0.855 | 0.109 | 0.90 |
| `dag_shortest_paths` | `topo_h` | node/pointer | 0.970 | 0.770 | 0.866 | 0.125 | 0.89 |
| `dag_shortest_paths` | `topo_head_h` | node/mask_one | 0.685 | 0.938 | 0.938 | 0.000 | 1.37 |
| `dag_shortest_paths` | `s_prev` | node/pointer | 0.981 | 0.714 | 0.845 | 0.438 | 0.86 |
| `dag_shortest_paths` | `u` | node/mask_one | 0.919 | 1.000 | 1.000 | 0.500 | 1.09 |
| `dag_shortest_paths` | `v` | node/mask_one | 0.934 | 1.000 | 1.000 | 0.000 | 1.07 |
| `dag_shortest_paths` | `s_last` | node/mask_one | 0.887 | 1.000 | 1.000 | 0.000 | 1.13 |
| `floyd_warshall` | `Pi_h` | edge/pointer | 0.960 | 0.806 | 0.806 | 0.083 | 0.84 |
| `floyd_warshall` | `k` | node/mask_one | 1.000 | 0.000 | 0.094 | 0.031 | 0.09 |
| `matrix_chain_order` | `pred_h` | node/pointer | 1.000 | 0.032 | 0.099 | 0.078 | 0.10 |
| `matrix_chain_order` | `s_h` | edge/pointer | 0.985 | 0.858 | 0.858 | 0.605 | 0.87 |

`reached` is the best out-of-distribution step over the mean in-distribution one: near 1 the model tracks the algorithm somewhere and the failure is in the iteration, near 0 it never tracks it at any step and the rounds are not approximating the steps at all.
