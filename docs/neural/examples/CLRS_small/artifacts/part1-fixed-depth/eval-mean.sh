#!/bin/bash
cd /home/tommaso.salvatori/pc_new/discopy/docs/neural/examples/CLRS_small
export PYTHONPATH=/home/tommaso.salvatori/pc_new/discopy
PY=/home/tommaso.salvatori/miniconda3/envs/disc/bin/python
while [ ! -f artifacts/full-bellman_ford-seed0.pt ]; do sleep 30; done
$PY -u evaluate.py --algorithms bfs bellman_ford --seeds 0 --device cuda:1 \
    > artifacts/log-eval-mean.txt 2>&1
echo "mean campaign evaluated"
