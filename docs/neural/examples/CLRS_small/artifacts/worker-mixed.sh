#!/bin/bash
# One slot of the T2+T3 campaign: the same eight algorithms retrained with
# mixed training sizes and termination supervision, one seed.  Pops
# "<algorithm>" off artifacts/queue-mixed.txt under a lock.
#   usage: worker-mixed.sh <device> <slot>
cd /home/tommaso.salvatori/pc_new/discopy/docs/neural/examples/CLRS_small
export PYTHONPATH=/home/tommaso.salvatori/pc_new/discopy
PY=/home/tommaso.salvatori/miniconda3/envs/disc/bin/python
DEVICE=$1
QUEUE=artifacts/queue-mixed.txt
LOCK=artifacts/queue-mixed.lock
touch $LOCK
while true; do
  JOB=$(flock $LOCK -c "head -n 1 $QUEUE && sed -i '1d' $QUEUE")
  [ -z "$JOB" ] && break
  set -- $JOB
  ALG=$1
  $PY train.py --algorithms $ALG --seeds 0 --pool max --mixed --settle \
      --device $DEVICE > artifacts/log-train-mixed-$ALG.txt 2>&1
  echo "$(date +%H:%M:%S) done $ALG on $DEVICE"
done
