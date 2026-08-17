#!/bin/bash
# Scores each row of the primary campaign as soon as its three seeds
# exist, one row at a time so that a scoring pass never competes with
# itself, and draws the tables and figures when the last one lands.
cd /home/tommaso.salvatori/pc_new/discopy/docs/neural/examples/CLRS_small
export PYTHONPATH=/home/tommaso.salvatori/pc_new/discopy
PY=/home/tommaso.salvatori/miniconda3/envs/disc/bin/python
score () {  # <algorithm> <tag> <flags...>
  ALG=$1; TAG=$2; shift 2
  until [ -f artifacts/$TAG-$ALG-seed0.pt ] \
     && [ -f artifacts/$TAG-$ALG-seed1.pt ] \
     && [ -f artifacts/$TAG-$ALG-seed2.pt ]; do sleep 120; done
  sleep 20
  echo "[$(date +%H:%M:%S)] scoring $TAG-$ALG"
  $PY -u evaluate.py --algorithms $ALG --seeds 0 1 2 --pool max "$@" \
      --device cuda:1 > artifacts/log-eval-$TAG-$ALG.txt 2>&1
  echo "[$(date +%H:%M:%S)] scored $TAG-$ALG (exit $?)"
}
for a in bellman_ford minimum bfs dijkstra mst_prim \
         floyd_warshall matrix_chain_order dag_shortest_paths; do
  score $a full-max
done
for a in floyd_warshall matrix_chain_order; do
  score $a full-paired-max-nodeonly --node-only
done
$PY -u evaluate.py --table --pool max > artifacts/table.md 2>&1
for a in floyd_warshall matrix_chain_order; do
  $PY -u evaluate.py --h1 --algorithms $a --pool max >> artifacts/h1.md 2>&1
done
$PY -u figures.py --pool max > artifacts/log-figures.txt 2>&1
echo "[$(date +%H:%M:%S)] FINALISE DONE"
