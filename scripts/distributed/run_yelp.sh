#!/usr/bin/env bash
# Exact SCN on Yelp (inductive, fixed split), encoder-first.
#
#   KIND=dist   : 6 processes, one per GPU, model seeds 42..46, with the
#                 full-batch protocol selected by run_yelp_grid.sh (constant
#                 lr 1e-2, dropout 0.5); resumable, logged to wandb when
#                 WANDB_API_KEY is set
#   KIND=probe  : whole graph on one GPU (records OOM if it does not fit)
#
#   PY=... TORCHRUN=... DATA_DIR=... KIND=dist bash scripts/distributed/run_yelp.sh
set -u
cd "$(dirname "$0")/../.."
export PYTHONPATH="$PWD" OMP_NUM_THREADS=4
PY="${PY:-python}"
TORCHRUN="${TORCHRUN:-torchrun}"
DATA_DIR="${DATA_DIR:-datasets}"
KIND="${KIND:-dist}"
NPROC="${NPROC:-6}"
OUT="${OUT:-logs/distributed/yelp}"
LR="${LR:-0.01}"
LR_STEP="${LR_STEP:-0}"
PD="${PD:-0.5}"
PROJECT="${WANDB_PROJECT:-partition-before-lift}"
GROUP="${WANDB_GROUP:-yelp-seeds}"
RUN_TIMEOUT="${RUN_TIMEOUT:-3600}"
mkdir -p "$OUT/logs"
export WANDB_CONFIG_DIR="${WANDB_CONFIG_DIR:-$OUT/.wandb_config}"
export WANDB_CACHE_DIR="${WANDB_CACHE_DIR:-$OUT/.wandb_cache}"

if [[ "$KIND" == "dist" ]]; then
    for i in 0 1 2 3 4; do
        name="lr${LR}_step${LR_STEP}_pd${PD}_seed$i"
        [[ -f "$OUT/$name.done" ]] && continue
        wandb=()
        if [[ -n "${WANDB_API_KEY:-}" ]]; then
            wandb=(--wandb-project "$PROJECT" --wandb-group "$GROUP"
                   --wandb-name "yelp_$name")
        fi
        timeout "$RUN_TIMEOUT" "$TORCHRUN" --nproc_per_node "$NPROC" \
            --master_port "2980$i" \
            scripts/distributed/train_scn.py --mode dist --encoder-first \
            --dataset yelp --data-dir "$DATA_DIR" --data-seed "$i" \
            --seed "$((42 + i))" --lr "$LR" --lr-step "$LR_STEP" \
            --proj-dropout "$PD" --weight-decay 0 --max-epochs 1000 \
            --val-every 20 --patience 6 ${wandb[@]+"${wandb[@]}"} \
            --out "$OUT" \
            > "$OUT/logs/dist_P${NPROC}_$name.log" 2>&1
        status=$?
        echo "$(date +%T) dist P$NPROC $name exit $status" >> "$OUT/progress.out"
        [[ $status -eq 0 ]] && touch "$OUT/$name.done"
    done
else
    "$PY" scripts/distributed/train_scn.py --mode first --parts 1 \
        --dataset yelp --data-dir "$DATA_DIR" --data-seed 0 --seed 42 \
        --out "$OUT" > "$OUT/logs/probe_first_P1_seed0.log" 2>&1
    echo "$(date +%T) probe exit $?" >> "$OUT/progress.out"
fi
touch "$OUT/DONE_$KIND"
