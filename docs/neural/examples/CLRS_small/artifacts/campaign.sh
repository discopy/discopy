#!/bin/bash
# The primary campaign: eight tasks x three seeds, max-aggregated, plus
# H1's node-only arm on the two edge-state showcases.  Six slots over two
# H100s; the two that would land on top of the aggregator-ablation runs
# still in flight wait for them to exit first.
cd /home/tommaso.salvatori/pc_new/discopy/docs/neural/examples/CLRS_small
for slot in 1 2 3; do
  ./artifacts/worker.sh cuda:0 $slot >> artifacts/log-campaign.txt 2>&1 &
done
./artifacts/worker.sh cuda:1 4 >> artifacts/log-campaign.txt 2>&1 &
./artifacts/worker.sh cuda:1 5 >> artifacts/log-campaign.txt 2>&1 &
(
  while pgrep -f "train.py --algorithms bellman_ford --seeds 0 --device" \
        > /dev/null; do sleep 60; done
  ./artifacts/worker.sh cuda:1 6
) >> artifacts/log-campaign.txt 2>&1 &
wait
echo "[$(date +%H:%M:%S)] CAMPAIGN DRAINED" >> artifacts/log-campaign.txt
