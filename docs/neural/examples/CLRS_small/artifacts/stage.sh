#!/bin/bash
# The Part 3 campaign, staged so that `dag_shortest_paths` is last and the
# other four rows are complete and evaluable on their own.
#
#   stage A  the regime probe for the four cheap rows   (running already)
#   stage B  freeze config.REGIME for those four rows
#   stage C  H2's four arms x four rows x three seeds
#   stage D  the regime probe for dag_shortest_paths
#   stage E  freeze it, then H2's four arms x dag x three seeds
#
# Four slots per device is the measured knee: one H100 saturates at about
# 1.8x its single-job throughput on this workload and it is the GPU that
# saturates, not the CPU.
cd /home/tommaso.salvatori/pc_new/discopy/docs/neural/examples/CLRS_small
export PYTHONPATH=/home/tommaso.salvatori/pc_new/discopy
PY=/home/tommaso.salvatori/miniconda3/envs/disc/bin/python
ROWS="bellman_ford dijkstra mst_prim floyd_warshall"
ARMS="R S O F"
SEEDS="0 1 2"

launch () {  # launch <queue name> ; blocks until the queue is drained
  for d in 0 1 2; do for s in 1 2 3 4; do
    ./artifacts/worker.sh $1 cuda:$d >> artifacts/log-worker-$1-$d$s.txt 2>&1 &
  done; done
  wait
}

echo "$(date +%H:%M) stage A: waiting for the four cheap probes"
while pgrep -f "train.py --algorithms .* --arm R --regime" > /dev/null; do
  sleep 60
done

echo "$(date +%H:%M) stage B: freezing config.REGIME for $ROWS"
$PY regime.py --algorithms $ROWS --write \
    > artifacts/log-regime-decision.txt 2>&1
cat artifacts/log-regime-decision.txt

echo "$(date +%H:%M) stage C: H2 on the four rows"
: > artifacts/queue-h2.txt
for arm in $ARMS; do for row in $ROWS; do for seed in $SEEDS; do
  echo "--algorithms $row --seeds $seed --arm $arm" >> artifacts/queue-h2.txt
done; done; done
launch h2

echo "$(date +%H:%M) stage D: the dag_shortest_paths probe"
: > artifacts/queue-dagprobe.txt
for r in mixed fixed; do
  echo "--algorithms dag_shortest_paths --seeds 0 --arm R --regime $r" \
      >> artifacts/queue-dagprobe.txt
done
launch dagprobe

echo "$(date +%H:%M) stage E: freezing dag and running its arms"
$PY regime.py --algorithms dag_shortest_paths --write \
    >> artifacts/log-regime-decision.txt 2>&1
: > artifacts/queue-dag.txt
for arm in $ARMS; do for seed in $SEEDS; do
  echo "--algorithms dag_shortest_paths --seeds $seed --arm $arm" \
      >> artifacts/queue-dag.txt
done; done
launch dag

echo "$(date +%H:%M) campaign done"
