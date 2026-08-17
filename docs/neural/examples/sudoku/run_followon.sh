#!/usr/bin/env bash
# When the 10-epoch campaign finishes, start the 6-epoch one.
#
#     setsid nohup ./run_followon.sh > /dev/null 2>&1 &
#
# A budget is part of a recipe -- the learning-rate schedule decays over
# the run -- so the shorter trials go in their own study rather than
# beside the long ones, seeded from them so the pruner and the sampler
# keep everything the first campaign learned.  Cancel by killing this
# script before it fires:  pkill -f run_followon.sh
set -u

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG="${LOG:-$HERE/../../artifacts/optuna-act-extreme.log}"
PERIOD="${PERIOD:-300}"

say () { echo "[$(date '+%Y-%m-%d %H:%M:%S')] followon: $*" >> "$LOG"; }

say "waiting for the 10-epoch campaign to exit"
while pgrep -f "[o]ptuna_act.py" > /dev/null; do sleep "$PERIOD"; done
say "10-epoch campaign is done"

cd "$HERE" || exit 1
STUDY=act-extreme-6e \
SEED_FROM="$HERE/../../artifacts/optuna-act-extreme.db" \
SEED_STUDY=act-extreme \
DAYS_SECONDS="${DAYS_SECONDS:-172800}" \
    ./run_search.sh
