#!/bin/bash
cd /home/tommaso.salvatori/pc_new/discopy/docs/neural/examples/CLRS_small
export PYTHONPATH=/home/tommaso.salvatori/pc_new/discopy
PY=/home/tommaso.salvatori/miniconda3/envs/disc/bin/python
$PY -u evaluate.py --algorithms minimum --seeds 0 --device cuda:0 \
    > artifacts/log-eval-minimum-mean.txt 2>&1
$PY -u evaluate.py --algorithms minimum --seeds 0 --pool max --device cuda:0 \
    > artifacts/log-eval-minimum-max.txt 2>&1
echo "minimum evaluated"
