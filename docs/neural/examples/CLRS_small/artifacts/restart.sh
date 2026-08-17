#!/bin/bash
# Restart of stages D and E on two devices, after the first attempt was
# killed mid-probe.  The four rows of stage C are complete on disk -- 52
# checkpoints, 16 reports, config.REGIME frozen for them -- so only
# `dag_shortest_paths` is rebuilt.
#
#   stage D  the dag regime probe, two jobs, one slot per device
#   stage D' the depth ladders the four finished rows still owe, on the
#            six slots the probe leaves idle: H2's depth-robustness
#            evidence, evaluation-only on checkpoints that already exist
#   stage E  freeze the dag regime, then its four arms x three seeds
#
# The probe is *not* speculated past: the arms wait for the frozen regime
# rather than assuming the "fixed" that the other five rows returned.
# Two shortcuts have already cost this session a wrong cost model and a
# silently failing evaluation driver, and the regime guard is the one
# piece of discipline the campaign rests on.
cd /home/tommaso.salvatori/pc_new/discopy/docs/neural/examples/CLRS_small
export PYTHONPATH=/home/tommaso.salvatori/pc_new/discopy
PY=/home/tommaso.salvatori/miniconda3/envs/disc/bin/python
ROWS="bellman_ford dijkstra mst_prim floyd_warshall"

launch () {  # launch <queue> <slots per device>
  for d in 0 1; do for s in $(seq 1 $2); do
    ./artifacts/worker.sh $1 cuda:$d >> artifacts/log-worker-$1-$d$s.txt 2>&1 &
  done; done
  wait
}

echo "$(date +%H:%M) stage D: the dag probe, one slot per device"
: > artifacts/queue-dagprobe.txt
for r in fixed mixed; do
  echo "--algorithms dag_shortest_paths --seeds 0 --arm R --regime $r" \
      >> artifacts/queue-dagprobe.txt
done
launch dagprobe 1 &
PROBE=$!

echo "$(date +%H:%M) stage D': the four rows' depth ladders, six slots"
: > artifacts/queue-ladder.txt
for arm in R S O F; do for row in $ROWS; do
  echo "$arm $row" >> artifacts/queue-ladder.txt
done; done
touch artifacts/queue-ladder.lock
ladder () {
  local DEVICE=$1
  while true; do
    JOB=$(flock artifacts/queue-ladder.lock -c \
      "head -n 1 artifacts/queue-ladder.txt && sed -i '1d' artifacts/queue-ladder.txt")
    [ -z "$JOB" ] && break
    set -- $JOB
    $PY -u evaluate.py --arm $1 --algorithms $2 --ladder --device $DEVICE \
        > artifacts/log-ladder-$1-$2.txt 2>&1
    echo "$(date +%H:%M) ladder $1/$2 on $DEVICE (exit $?)"
  done
}
for d in 0 1; do for s in 1 2 3; do ladder cuda:$d & done; done
wait $PROBE
echo "$(date +%H:%M) probe done"
wait

echo "$(date +%H:%M) stage E: freezing the dag regime"
$PY regime.py --algorithms dag_shortest_paths --write \
    >> artifacts/log-regime-decision.txt 2>&1
tail -3 artifacts/log-regime-decision.txt

: > artifacts/queue-dag.txt
for arm in R S O F; do for seed in 0 1 2; do
  echo "--algorithms dag_shortest_paths --seeds $seed --arm $arm" \
      >> artifacts/queue-dag.txt
done; done
launch dag 4

echo "$(date +%H:%M) campaign done"
