#!/usr/bin/env bash
# Exact SCN on Yelp (inductive, fixed split), encoder-first.
#
#   KIND=dist   : 6 processes, one per GPU, model seeds 42..46
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
mkdir -p "$OUT/logs"

if [[ "$KIND" == "dist" ]]; then
    for i in 0 1 2 3 4; do
        "$TORCHRUN" --nproc_per_node "$NPROC" --master_port "2980$i" \
            scripts/distributed/train_scn.py --mode dist --encoder-first \
            --dataset yelp --data-dir "$DATA_DIR" --data-seed "$i" \
            --seed "$((42 + i))" --out "$OUT" \
            > "$OUT/logs/dist_P${NPROC}_seed$i.log" 2>&1
        echo "$(date +%T) dist P$NPROC seed $i exit $?" >> "$OUT/progress.out"
    done
else
    "$PY" scripts/distributed/train_scn.py --mode first --parts 1 \
        --dataset yelp --data-dir "$DATA_DIR" --data-seed 0 --seed 42 \
        --out "$OUT" > "$OUT/logs/probe_first_P1_seed0.log" 2>&1
    echo "$(date +%T) probe exit $?" >> "$OUT/progress.out"
fi
touch "$OUT/DONE_$KIND"
