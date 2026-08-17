#!/bin/bash
# One slot of a Part 3 campaign.  Pops a whole `train.py` argument line off
# a queue under a lock and runs it, so the queue is the campaign and the
# slot count is the schedule.
#   usage: worker.sh <queue name> <device>
# One H100 saturates at about 1.8x its single-job throughput on this
# workload -- measured, not assumed, and it is the GPU rather than the CPU
# that saturates: 8 cores and 48 cores give the same curve.  So four slots
# per device is the knee and more is waste.
cd /home/tommaso.salvatori/pc_new/discopy/docs/neural/examples/CLRS_small
export PYTHONPATH=/home/tommaso.salvatori/pc_new/discopy
PY=/home/tommaso.salvatori/miniconda3/envs/disc/bin/python
NAME=$1
DEVICE=$2
QUEUE=artifacts/queue-$NAME.txt
LOCK=artifacts/queue-$NAME.lock
touch $LOCK
while true; do
  JOB=$(flock $LOCK -c "head -n 1 $QUEUE && sed -i '1d' $QUEUE")
  [ -z "$JOB" ] && break
  SLUG=$(echo "$JOB" | tr -c 'a-zA-Z0-9' '-' | tr -s '-')
  echo "$(date +%H:%M:%S) start $JOB on $DEVICE"
  # -u so a running campaign's progress is visible in the log rather than
  # sitting in a stdio buffer until the process exits.
  $PY -u train.py $JOB --device $DEVICE \
      > artifacts/log-$NAME-$SLUG.txt 2>&1
  echo "$(date +%H:%M:%S) done  $JOB on $DEVICE (exit $?)"
done
