#!/usr/bin/env bash
# The test-split report for the search winner, on two GPUs.
#
#     setsid nohup ./run_testeval.sh > /dev/null 2>&1 &
#
# Two stages that share nothing but the checkpoint, so they run at once:
#
#   A  the whole 422,786-puzzle test split at T=16, the anchor that says
#      how faithful the subsample is;
#   B  12,000 puzzles swept 16..512, noiseless and then under sigma=1.0
#      answer noise with the halt head selecting among 32 rollouts.
#
# B is the long one -- its beam carries four paths from 16 all the way to
# 512 -- so A's GPU comes free early and is handed to whatever runs next.
# Both write their numbers row by row, and the tables are merged at the
# end into one tidy CSV to plot from.
set -u

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-$HOME/miniconda3/envs/disc/bin/python}"
ART="$HERE/../../artifacts"
CKPT="${CKPT:-$ART/optuna-act-extreme-trial27.pt}"
LOG="${LOG:-$ART/testeval.log}"
BS="${BS:-4000}"
N_NOISE="${N_NOISE:-12000}"

say () { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" >> "$LOG"; }

cd "$HERE" || exit 1
say "checkpoint $(basename "$CKPT"), batch $BS, ${N_NOISE} puzzles for the grid"

say "A: full test split, noiseless, T=16 -> GPU 0"
CUDA_VISIBLE_DEVICES=0 env -u PYTHONPATH "$PYTHON" eval_best.py "$CKPT" \
    --computes 16 --n-fixed 0 --skip-noise \
    --batch-size "$BS" --stem trial27-fulltest \
    >> "$ART/testeval-fulltest.log" 2>&1 &
a_pid=$!

say "B: ${N_NOISE} puzzles, noiseless sweep + sigma=1.0 grid -> GPU 1"
CUDA_VISIBLE_DEVICES=1 env -u PYTHONPATH "$PYTHON" eval_best.py "$CKPT" \
    --computes 16 32 64 128 256 512 --skip-fixed --noiseless \
    --n-noise "$N_NOISE" --sigmas 1.0 --rollouts 32 --survivors 4 \
    --batch-size "$BS" --stem trial27-12k \
    >> "$ART/testeval-12k.log" 2>&1 &
b_pid=$!

wait "$a_pid"; say "A finished with exit code $?"
wait "$b_pid"; say "B finished with exit code $?"

env -u PYTHONPATH "$PYTHON" eval_best.py \
    --merge trial27-fulltest trial27-12k --stem trial27-eval >> "$LOG" 2>&1
say "merged into trial27-eval-records.csv"
