#!/bin/bash
cd /home/tommaso.salvatori/pc_new/discopy/docs/neural/examples/CLRS_small
export PYTHONPATH=/home/tommaso.salvatori/pc_new/discopy
PY=/home/tommaso.salvatori/miniconda3/envs/disc/bin/python
for a in bfs bellman_ford; do
  $PY -u train.py --algorithms $a --seeds 0 --pool max --device cuda:0 \
      > artifacts/log-$a-max.txt 2>&1
done
$PY -u evaluate.py --algorithms bfs bellman_ford --seeds 0 --pool max \
    --device cuda:0 > artifacts/log-eval-max.txt 2>&1
echo "max campaign done"
