#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

REPETITIONS="${PARTITION_BUILD_REPETITIONS:-0,1,2,3,4}"
MAX_CONCURRENT_BUILDS="${MAX_CONCURRENT_BUILDS:-1}"
WANDB_PROJECT_PREFIX="${WANDB_PROJECT_PREFIX:-partition_build_benchmark}"
BENCHMARK_NAMESPACE="${BENCHMARK_NAMESPACE:-partition_build_$(date -u +%Y%m%dT%H%M%SZ)}"
RESUME="${RESUME:-true}"
DRY_RUN="${DRY_RUN:-false}"

SPECS=(
    "cora_full:32"
    "cora_full:64"
    "amazon_ratings:32"
    "amazon_ratings:64"
    "amazon_ratings:128"
    "questions:500"
    "questions:1000"
)

if ! [[ "$MAX_CONCURRENT_BUILDS" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: MAX_CONCURRENT_BUILDS must be a positive integer." >&2
    exit 1
fi
if [[ "$RESUME" != "true" && "$RESUME" != "false" ]]; then
    echo "ERROR: RESUME must be true or false." >&2
    exit 1
fi
if [[ "$DRY_RUN" != "true" && "$DRY_RUN" != "false" ]]; then
    echo "ERROR: DRY_RUN must be true or false." >&2
    exit 1
fi
if [[ ! "$BENCHMARK_NAMESPACE" =~ ^[A-Za-z0-9_.-]+$ ]]; then
    echo "ERROR: BENCHMARK_NAMESPACE contains unsupported characters." >&2
    exit 1
fi

IFS=',' read -ra repetition_values <<< "$REPETITIONS"
if (( ${#repetition_values[@]} == 0 )); then
    echo "ERROR: PARTITION_BUILD_REPETITIONS must not be empty." >&2
    exit 1
fi
for repetition in "${repetition_values[@]}"; do
    if ! [[ "$repetition" =~ ^[0-9]+$ ]]; then
        echo "ERROR: invalid repetition '$repetition'." >&2
        exit 1
    fi
done

run_build() {
    local dataset="$1"
    local num_parts="$2"
    local repetition="$3"
    local run_namespace="${BENCHMARK_NAMESPACE}_${dataset}_k${num_parts}_rep${repetition}"
    local log_group="${run_namespace}_sweep"

    env \
        TRAINER=cpu \
        LOGGER=wandb \
        SELECTED_GPUS=0 \
        JOBS_PER_GPU_OVERRIDE=1 \
        MAX_CONCURRENT_RUNS=1 \
        DATASET_FILTER="$dataset" \
        MODEL_FILTER=gcn \
        DATA_SEEDS_OVERRIDE="$repetition" \
        PARTITION_GRID_OVERRIDE="${num_parts}:1" \
        PARTITION_CACHE_NAMESPACE="$run_namespace" \
        STREAM_NUM_WORKERS=0 \
        CACHE_NUM_WORKERS=0 \
        CACHE_VAL=false \
        VAL_SHUFFLE=true \
        MAX_EPOCHS=1 \
        MIN_EPOCHS=1 \
        CHECK_VAL_EVERY_N_EPOCH=1 \
        EARLY_STOPPING_PATIENCE=1 \
        TEST_INFERENCE_PROTOCOLS='[batched]' \
        ENSEMBLE_RUNS=1 \
        TRAIN=false \
        TEST=false \
        RESUME="$RESUME" \
        DRY_RUN="$DRY_RUN" \
        MAX_ATTEMPTS=1 \
        KEEP_SUCCESS_LOGS=true \
        WANDB_PROJECT_PREFIX="$WANDB_PROJECT_PREFIX" \
        RUN_NAME_PREFIX="$BENCHMARK_NAMESPACE" \
        LOG_GROUP="$log_group" \
        bash "$SCRIPT_DIR/final_partitioning.sh"
}

total_runs=$(( ${#SPECS[@]} * ${#repetition_values[@]} ))
echo "Fresh partition-build benchmark"
echo "Dataset-K combinations: ${#SPECS[@]}"
echo "Repetitions: ${repetition_values[*]}"
echo "Total builds: $total_runs"
echo "Maximum concurrent builds: $MAX_CONCURRENT_BUILDS"
echo "W&B project prefix: $WANDB_PROJECT_PREFIX"
echo "Benchmark namespace: $BENCHMARK_NAMESPACE"
echo "Training: false"
echo "Testing: false"

running=0
failures=0
for spec in "${SPECS[@]}"; do
    IFS=':' read -r dataset num_parts <<< "$spec"
    for repetition in "${repetition_values[@]}"; do
        echo "[BUILD] dataset=$dataset K=$num_parts repetition=$repetition"
        if [[ "$DRY_RUN" == "true" ]]; then
            run_build "$dataset" "$num_parts" "$repetition"
            continue
        fi

        run_build "$dataset" "$num_parts" "$repetition" &
        running=$(( running + 1 ))
        if (( running >= MAX_CONCURRENT_BUILDS )); then
            if ! wait -n; then
                failures=$(( failures + 1 ))
            fi
            running=$(( running - 1 ))
        fi
    done
done

while (( running > 0 )); do
    if ! wait -n; then
        failures=$(( failures + 1 ))
    fi
    running=$(( running - 1 ))
done

if (( failures > 0 )); then
    echo "ERROR: $failures partition-build job(s) failed." >&2
    exit 1
fi

echo "All $total_runs fresh partition builds completed."
