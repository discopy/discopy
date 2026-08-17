#!/usr/bin/env bash
# Three days of the ACT search, one worker per GPU, sharing one study.
#
#     setsid nohup ./run_search.sh > /dev/null 2>&1 &
#
# The GPUs of a SLURM allocation are not always idle -- another job of the
# same account can be holding one, and a framework that preallocates its
# VRAM leaves a torch worker to die of OOM -- so the run waits for each of
# them to be clear (FREE_MIB, STABLE checks PERIOD seconds apart) rather
# than starting into contention.  Everything it prints goes to LOG.
set -u

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-$HOME/miniconda3/envs/disc/bin/python}"
LEGACY="${LEGACY:-$HOME/pc_new/discopy_old/docs/optuna_trm_extreme_act.db}"
LOG="${LOG:-$HERE/../../artifacts/optuna-act-extreme.log}"

STUDY="${STUDY:-act-extreme}"
STORAGE="${STORAGE:-}"          # a .journal path pools workers across nodes
SEED_FROM="${SEED_FROM:-$LEGACY}"
SEED_STUDY="${SEED_STUDY:-trm-extreme-act-8k}"
GPUS="${GPUS:-2}"
WORKERS="${WORKERS:-1}"         # worker processes per GPU
FREE_MIB="${FREE_MIB:-70000}"   # free memory that counts as "clear"
STABLE="${STABLE:-2}"           # consecutive clear checks required
PERIOD="${PERIOD:-30}"          # seconds between checks
DAYS_SECONDS="${DAYS_SECONDS:-259200}"

say () { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" >> "$LOG"; }

say "waiting for ${GPUS} GPU(s) with >= ${FREE_MIB} MiB free, ${STABLE}x${PERIOD}s"
clear_count=0
while [ "$clear_count" -lt "$STABLE" ]; do
    busy="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits \
            | head -"$GPUS" | awk -v m="$FREE_MIB" '$1 < m' | wc -l)"
    if [ "$busy" -eq 0 ]; then
        clear_count=$((clear_count + 1))
        say "clear ${clear_count}/${STABLE}"
    else
        [ "$clear_count" -gt 0 ] && say "no longer clear (${busy} GPU busy)"
        clear_count=0
    fi
    [ "$clear_count" -lt "$STABLE" ] && sleep "$PERIOD"
done

say "launching ${STUDY}: ${GPUS} GPU(s) x ${WORKERS} worker(s), ${DAYS_SECONDS}s of new trials"
cd "$HERE" || exit 1
env -u PYTHONPATH "$PYTHON" optuna_act.py \
    --gpus "$GPUS" --workers-per-gpu "$WORKERS" \
    --trials 40 --timeout "$DAYS_SECONDS" --study-name "$STUDY" \
    --epochs "${EPOCHS:-6}" --schedule-epochs "${SCHEDULE_EPOCHS:-10}" \
    ${STORAGE:+--storage "$STORAGE"} \
    ${SEED_FROM:+--seed-from "$SEED_FROM" --seed-study "$SEED_STUDY"} \
    --pruner-startup 2 --pruner-warmup 8 >> "$LOG" 2>&1
say "finished with exit code $?"
