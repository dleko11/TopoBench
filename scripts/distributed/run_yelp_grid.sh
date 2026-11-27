#!/usr/bin/env bash
# Grid search of the full-batch training protocol for exact SCN on Yelp
# (inductive, fixed split, encoder-first, one process per GPU), logged to
# wandb. The model is the Cluster-TNN-tuned SCN (1 layer, hidden 128); only
# the training protocol is searched, because Cluster-TNN's lr and StepLR(50)
# schedule were tuned for ~350 optimizer steps per epoch, not one.
#
# Resumable: finished configurations are skipped on a rerun.
#
#   PY=... TORCHRUN=... DATA_DIR=... bash scripts/distributed/run_yelp_grid.sh
set -u
cd "$(dirname "$0")/../.."
export PYTHONPATH="$PWD" OMP_NUM_THREADS=4
PY="${PY:-python}"
TORCHRUN="${TORCHRUN:-torchrun}"
DATA_DIR="${DATA_DIR:-datasets}"
NPROC="${NPROC:-6}"
OUT="${OUT:-logs/distributed/yelp_grid}"
PROJECT="${WANDB_PROJECT:-partition-before-lift}"
GROUP="${WANDB_GROUP:-yelp-grid}"
RUN_TIMEOUT="${RUN_TIMEOUT:-3600}"
mkdir -p "$OUT/logs"
export WANDB_CONFIG_DIR="${WANDB_CONFIG_DIR:-$OUT/.wandb_config}"
export WANDB_CACHE_DIR="${WANDB_CACHE_DIR:-$OUT/.wandb_cache}"

if [[ -z "${WANDB_API_KEY:-}" && "${WANDB_MODE:-}" != "offline" ]]; then
    echo "WANDB_API_KEY is not set" | tee -a "$OUT/progress.out" >&2
    exit 1
fi
"$PY" -c "import wandb" || exit 1

# Most promising first, so a partial grid is still informative.
port=29700
for step in 0 200; do                  # 0 = constant lr; 200 = halve every 200
    for lr in 0.01 0.003 0.03; do
        for pd in 0.3 0.1 0.5; do
            name="lr${lr}_step${step}_pd${pd}"
            port=$((port + 1))
            [[ -f "$OUT/$name/DONE" ]] && continue
            timeout "$RUN_TIMEOUT" "$TORCHRUN" --nproc_per_node "$NPROC" \
                --master_port "$port" \
                scripts/distributed/train_scn.py --mode dist --encoder-first \
                --dataset yelp --data-dir "$DATA_DIR" --seed 42 \
                --lr "$lr" --lr-step "$step" --proj-dropout "$pd" \
                --weight-decay 0 --max-epochs 1000 --val-every 20 \
                --patience 6 --wandb-project "$PROJECT" \
                --wandb-group "$GROUP" --wandb-name "yelp_$name" \
                --out "$OUT/$name" > "$OUT/logs/$name.log" 2>&1
            status=$?
            echo "$(date +%T) $name exit $status" >> "$OUT/progress.out"
            [[ $status -eq 0 ]] && touch "$OUT/$name/DONE"
        done
    done
done
touch "$OUT/DONE"
