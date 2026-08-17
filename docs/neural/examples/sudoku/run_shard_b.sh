#!/usr/bin/env bash
# Hand GPU 0 over from the search to the second evaluation shard, but only
# once trial 26 has finished: it is 23 checks of 30 into a run tracking
# second place, and worth more than the ninety minutes it costs to wait.
#
#     setsid nohup ./run_shard_b.sh > /dev/null 2>&1 &
set -u

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-$HOME/miniconda3/envs/disc/bin/python}"
ART="$HERE/../../artifacts"
LOG="$ART/testeval-B.log"
TRIAL="${TRIAL:-26}"

say () { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" >> "$LOG"; }

say "waiting for trial $TRIAL to leave RUNNING"
while true; do
    state=$(cd "$HERE" && env -u PYTHONPATH "$PYTHON" - <<PY 2>/dev/null
import sys; sys.argv=['x']
import optuna, optuna_act as o
optuna.logging.set_verbosity(optuna.logging.WARNING)
s = optuna.load_study(study_name='act-extreme',
                      storage=o.make_storage('$ART/optuna-act.journal'))
print(next((t.state.name for t in s.get_trials(deepcopy=False)
            if t.number == $TRIAL), 'GONE'))
PY
)
    [ "$state" != "RUNNING" ] && break
    sleep 120
done
say "trial $TRIAL is $state; stopping the search and taking GPU 0"

for p in $(pgrep -u "$USER" -f "optuna_act[.]py"); do kill -9 "$p" 2>/dev/null; done
pkill -9 -x -f "bash ./run_search.sh" 2>/dev/null
sleep 10

cd "$HERE" || exit 1
say "launching shard B on GPU 0"
CUDA_VISIBLE_DEVICES=0 env -u PYTHONPATH "$PYTHON" eval_best.py \
    ../../artifacts/optuna-act-extreme-trial27.pt \
    --computes 16 32 64 128 256 512 --skip-fixed --noiseless \
    --slice 25000:50000 --n-noise 25000 \
    --sigmas 1.0 --rollouts 32 --survivors 4 \
    --batch-size 2000 --stem testeval-trial27 >> "$LOG" 2>&1
say "shard B finished with exit code $?"
