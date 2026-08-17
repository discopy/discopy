| algorithm | steps: trained → out of distribution | at the trained depth | at half | at its own depth |
|---|---|---|---|---|
| `minimum` | 16 → 64 | 0.9479 | 0.8542 | 0.6667 |
| `bfs` | 7 → 4 | 0.8501 | 0.2607 | 0.8592 |
| `bellman_ford` | 7 → 8 | 0.5703 | 0.4619 | 0.5658 |
| `dijkstra` | 17 → 65 | 0.6258 | 0.2262 | 0.0498 |
| `mst_prim` | 17 → 65 | 0.2150 | 0.1538 | 0.0352 |
| `floyd_warshall` | 16 → 64 | 0.2379 | 0.1006 | 0.0741 |
| `matrix_chain_order` | 15 → 63 | 0.6607 | 0.5826 | 0.3834 |
