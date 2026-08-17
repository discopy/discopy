#!/bin/bash
cd /home/tommaso.salvatori/pc_new/discopy/docs/neural/examples/CLRS_small
PY=/scratch/tommaso.salvatori/clrs/venv/bin/python
$PY -u dataset.py --generate 2>&1 | grep -v -E "absl|oneDNN|TF_ENABLE|external/local"
echo "GENERATION DONE"
