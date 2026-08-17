#!/bin/bash
# Waits for stage C of the campaign -- H2's four arms on the four rows
# that are not `dag_shortest_paths` -- and writes their reports, so that
# the four rows are readable rather than merely trained.  It runs while
# stage D is doing the dag probe, which occupies two slots of twelve.
cd /home/tommaso.salvatori/pc_new/discopy/docs/neural/examples/CLRS_small
export PYTHONPATH=/home/tommaso.salvatori/pc_new/discopy
PY=/home/tommaso.salvatori/miniconda3/envs/disc/bin/python
ROWS="bellman_ford dijkstra mst_prim floyd_warshall"

echo "$(date +%H:%M) waiting for stage C"
while [ ! -f artifacts/queue-h2.txt ] || [ -s artifacts/queue-h2.txt ] \
      || pgrep -f "train.py --algorithms .* --arm [RSOF]$" > /dev/null; do
  sleep 60
done
echo "$(date +%H:%M) stage C done, evaluating"

: > artifacts/queue-eval.txt
for arm in R S O F; do for row in $ROWS; do
  echo "$arm $row" >> artifacts/queue-eval.txt
done; done

worker () {
  # the device is captured *before* `set --` rewrites the positional
  # parameters with the job's fields; reading it off $3 afterwards is how
  # the first attempt passed an empty --device to all sixteen runs.
  local DEVICE=$1
  while true; do
    JOB=$(flock artifacts/queue-eval.lock -c \
      "head -n 1 artifacts/queue-eval.txt && sed -i '1d' artifacts/queue-eval.txt")
    [ -z "$JOB" ] && break
    set -- $JOB
    $PY -u evaluate.py --arm $1 --algorithms $2 --device $DEVICE \
        > artifacts/log-eval-$1-$2.txt 2>&1
    echo "$(date +%H:%M) evaluated $1/$2 on $DEVICE (exit $?)"
  done
}

touch artifacts/queue-eval.lock
for d in 0 1 2; do for s in 1 2; do worker cuda:$d & done; done
wait

echo "$(date +%H:%M) tables for the four rows"
for arm in R S O F; do
  echo "### arm $arm"
  $PY evaluate.py --arm $arm --algorithms $ROWS --heads
  $PY evaluate.py --arm $arm --algorithms $ROWS --table
done > artifacts/h2-four-rows.md 2>&1
echo "$(date +%H:%M) written to artifacts/h2-four-rows.md"
