#!/bin/bash
cd /home/tommaso.salvatori/pc_new/discopy/docs/neural/examples/CLRS_small
export PYTHONPATH=/home/tommaso.salvatori/pc_new/discopy
PY=/home/tommaso.salvatori/miniconda3/envs/disc/bin/python
for a in bfs bellman_ford; do
  $PY -u train.py --algorithms $a --seeds 0 --device cuda:1 \
      > artifacts/log-$a-mean.txt 2>&1
done
echo "mean campaign done"
