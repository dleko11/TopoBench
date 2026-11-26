#!/usr/bin/env bash
# Full-graph vs halo-distributed SCN on one 4-GPU node, paper protocol.
#
# Seed 0 runs alone (clean memory and timing); seeds 1-4 only feed the
# metric comparison and run concurrently.
#
#   PY=/path/to/python TORCHRUN=/path/to/torchrun \
#       bash scripts/distributed/run_pod_batch.sh
set -uo pipefail
cd "$(dirname "$0")/../.."
export PYTHONPATH="$PWD" OMP_NUM_THREADS=4
PY="${PY:-python}"
TORCHRUN="${TORCHRUN:-torchrun}"
OUT="${OUT:-logs/distributed/train}"
LOGS="$OUT/logs"
mkdir -p "$LOGS"
read -ra DATASETS <<< "${DATASETS:-cora_full amazon_ratings questions}"

full() {  # gpu dataset seed
    CUDA_VISIBLE_DEVICES="$1" "$PY" scripts/distributed/train_scn.py \
        --mode full --dataset "$2" --data-seed "$3" --out "$OUT" \
        > "$LOGS/$2_full_P1_seed$3.log" 2>&1
}

distributed() {  # gpus nproc port dataset seed
    CUDA_VISIBLE_DEVICES="$1" "$TORCHRUN" --nproc_per_node "$2" \
        --master_port "$3" scripts/distributed/train_scn.py \
        --mode dist --dataset "$4" --data-seed "$5" --out "$OUT" \
        > "$LOGS/$4_dist_P$2_seed$5.log" 2>&1
}

for dataset in "${DATASETS[@]}"; do
    echo "$(date +%T) seed 0 alone: $dataset"
    full 0 "$dataset" 0
    # Same distributed code on one GPU: the like-for-like memory baseline.
    distributed 0 1 29499 "$dataset" 0
    distributed 0,1 2 29500 "$dataset" 0
    distributed 0,1,2,3 4 29501 "$dataset" 0
done

for dataset in "${DATASETS[@]}"; do
    echo "$(date +%T) seeds 1-4: $dataset"
    for seed in 1 2 3 4; do full "$((seed - 1))" "$dataset" "$seed" & done
    wait
    distributed 0,1 2 29510 "$dataset" 1 &
    distributed 2,3 2 29511 "$dataset" 2 &
    wait
    distributed 0,1 2 29512 "$dataset" 3 &
    distributed 2,3 2 29513 "$dataset" 4 &
    wait
    for seed in 1 2 3 4; do
        distributed 0,1,2,3 4 29520 "$dataset" "$seed"
    done
done
echo "$(date +%T) all done"
touch "$OUT/DONE"
