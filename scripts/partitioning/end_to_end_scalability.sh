#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

SELECTED_GPUS="${SELECTED_GPUS:-0}"
E2E_DATASET="${E2E_DATASET:-cora_full}"
E2E_MODEL="${E2E_MODEL:-cwn}"
E2E_PARTITION_GRID="${E2E_PARTITION_GRID:-32:4}"
E2E_SEEDS="${E2E_SEEDS:-0,1,2,3,4}"
E2E_EPOCHS="${E2E_EPOCHS:-10}"
ENSEMBLE_RUNS="${ENSEMBLE_RUNS:-10}"
STREAM_NUM_WORKERS="${STREAM_NUM_WORKERS:-1}"
WANDB_PROJECT_PREFIX="${WANDB_PROJECT_PREFIX:-manuscript_e2e}"
BENCHMARK_NAMESPACE="${BENCHMARK_NAMESPACE:-manuscript_e2e_$(date -u +%Y%m%dT%H%M%SZ)}"
DRY_RUN="${DRY_RUN:-false}"
TIME_BIN="${TIME_BIN:-/usr/bin/time}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/logs/manuscript_e2e/${E2E_DATASET}_${E2E_MODEL}_${BENCHMARK_NAMESPACE}}"

if [[ ! "$E2E_EPOCHS" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: E2E_EPOCHS must be a positive integer." >&2
    exit 1
fi
if [[ ! "$ENSEMBLE_RUNS" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: ENSEMBLE_RUNS must be a positive integer." >&2
    exit 1
fi
if [[ "$E2E_DATASET" == *","* || "$E2E_MODEL" == *","* ]]; then
    echo "ERROR: select exactly one dataset and one model per benchmark." >&2
    exit 1
fi
if [[ ! "$E2E_PARTITION_GRID" =~ ^[1-9][0-9]*:[1-9][0-9]*$ ]]; then
    echo "ERROR: E2E_PARTITION_GRID must have the form NUM_PARTS:Q." >&2
    exit 1
fi
IFS=':' read -r E2E_NUM_PARTS E2E_Q <<< "$E2E_PARTITION_GRID"
if (( E2E_Q > E2E_NUM_PARTS )); then
    echo "ERROR: q must not exceed the number of partitions." >&2
    exit 1
fi
if [[ ! "$BENCHMARK_NAMESPACE" =~ ^[A-Za-z0-9_.-]+$ ]]; then
    echo "ERROR: BENCHMARK_NAMESPACE contains unsupported characters." >&2
    exit 1
fi
if [[ "$DRY_RUN" != "true" && ! -x "$TIME_BIN" ]]; then
    echo "ERROR: GNU time was not found at $TIME_BIN." >&2
    exit 1
fi

mkdir -p "$OUTPUT_DIR/timing"

{
    echo "git_commit=$(git -C "$REPO_ROOT" rev-parse HEAD)"
    echo "selected_gpus=$SELECTED_GPUS"
    echo "epochs=$E2E_EPOCHS"
    echo "seeds=$E2E_SEEDS"
    echo "model=$E2E_MODEL"
    echo "dataset=$E2E_DATASET"
    echo "num_parts=$E2E_NUM_PARTS"
    echo "q=$E2E_Q"
    echo "validation_cache=false"
    echo "validation_shuffle=true"
    echo "partition_test_protocol=ensemble"
    echo "ensemble_runs=$ENSEMBLE_RUNS"
    echo "full_graph_test_protocol=batched"
    echo "partition_cache_namespace=$BENCHMARK_NAMESPACE"
    uname -a
    if command -v nvidia-smi >/dev/null 2>&1; then
        nvidia-smi --query-gpu=index,name,memory.total,driver_version \
            --format=csv,noheader
    fi
    if command -v lscpu >/dev/null 2>&1; then
        lscpu
    fi
    if command -v free >/dev/null 2>&1; then
        free -b
    fi
} > "$OUTPUT_DIR/hardware.txt"

run_pipeline() {
    local mode="$1"
    local seed="$2"
    local log_group="manuscript_e2e_${E2E_DATASET}_${E2E_MODEL}_${BENCHMARK_NAMESPACE}_${mode}_seed${seed}"
    local timing_file="$OUTPUT_DIR/timing/${mode}_seed${seed}.txt"
    local -a command=(
        env
        "SELECTED_GPUS=$SELECTED_GPUS"
        "JOBS_PER_GPU_OVERRIDE=1"
        "MAX_CONCURRENT_RUNS=1"
        "DATASET_FILTER=$E2E_DATASET"
        "MODEL_FILTER=$E2E_MODEL"
        "DATA_SEEDS_OVERRIDE=$seed"
        "STREAM_NUM_WORKERS=$STREAM_NUM_WORKERS"
        "CACHE_NUM_WORKERS=$STREAM_NUM_WORKERS"
        "CACHE_VAL=false"
        "VAL_SHUFFLE=true"
        "MAX_EPOCHS=$E2E_EPOCHS"
        "MIN_EPOCHS=$E2E_EPOCHS"
        "CHECK_VAL_EVERY_N_EPOCH=1"
        "EARLY_STOPPING_PATIENCE=$E2E_EPOCHS"
        "TRAIN=true"
        "TEST=true"
        "ENSEMBLE_RUNS=$ENSEMBLE_RUNS"
        "RESUME=false"
        "MAX_ATTEMPTS=1"
        "KEEP_SUCCESS_LOGS=true"
        "WANDB_PROJECT_PREFIX=$WANDB_PROJECT_PREFIX"
        "LOG_GROUP=$log_group"
        "DRY_RUN=$DRY_RUN"
    )

    if [[ "$mode" == "full" ]]; then
        command+=(
            "FULL_GRAPH_BASELINE=true"
            "FORCE_RELOAD_PREPROCESSING=true"
            "TEST_INFERENCE_PROTOCOLS=[batched]"
            "RUN_NAME_PREFIX=manuscript_e2e_${BENCHMARK_NAMESPACE}_full"
        )
    elif [[ "$mode" == "partitioning" ]]; then
        command+=(
            "FULL_GRAPH_BASELINE=false"
            "PARTITION_GRID_OVERRIDE=$E2E_PARTITION_GRID"
            "PARTITION_CACHE_NAMESPACE=${BENCHMARK_NAMESPACE}_seed${seed}"
            "TEST_INFERENCE_PROTOCOLS=[ensemble]"
            "RUN_NAME_PREFIX=manuscript_e2e_${BENCHMARK_NAMESPACE}_partitioning"
        )
    else
        echo "ERROR: unsupported pipeline mode $mode." >&2
        exit 1
    fi
    command+=(bash "$SCRIPT_DIR/final_partitioning.sh")

    echo "Running $mode pipeline for seed $seed"
    if [[ "$DRY_RUN" == "true" ]]; then
        "${command[@]}"
    else
        "$TIME_BIN" -v -o "$timing_file" "${command[@]}"
    fi

    local failed_log="$REPO_ROOT/logs/$log_group/$log_group/FAILED_RUNS.log"
    if [[ -s "$failed_log" ]]; then
        echo "ERROR: $mode pipeline failed; see $failed_log." >&2
        exit 1
    fi
}

IFS=',' read -ra seeds <<< "$E2E_SEEDS"
for seed in "${seeds[@]}"; do
    if [[ ! "$seed" =~ ^[0-9]+$ ]]; then
        echo "ERROR: invalid seed '$seed'." >&2
        exit 1
    fi
    run_pipeline full "$seed"
    run_pipeline partitioning "$seed"
done

echo "Matched end-to-end benchmark complete."
echo "Hardware: $OUTPUT_DIR/hardware.txt"
echo "External timings: $OUTPUT_DIR/timing"
