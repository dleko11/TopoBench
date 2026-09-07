#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

SELECTED_GPUS="${SELECTED_GPUS:-0}"
JOBS_PER_GPU_OVERRIDE="${JOBS_PER_GPU_OVERRIDE:-1}"
MAX_CONCURRENT_RUNS="${MAX_CONCURRENT_RUNS:-1}"
DATA_SEEDS_OVERRIDE="${DATA_SEEDS_OVERRIDE:-0}"
STREAM_NUM_WORKERS="${STREAM_NUM_WORKERS:-0}"
RESUME="${RESUME:-false}"
DRY_RUN="${DRY_RUN:-false}"
MAX_ATTEMPTS="${MAX_ATTEMPTS:-1}"
WANDB_PROJECT_PREFIX="${WANDB_PROJECT_PREFIX:-preprocessing_memory_fixed}"
RUN_NAME_PREFIX="${RUN_NAME_PREFIX:-full_graph_preprocessing_fixed}"
LOG_GROUP_PREFIX="${LOG_GROUP_PREFIX:-full_graph_preprocessing_fixed}"

ALL_MODELS="gcn,edgnn,unignn,cwn,cell_topotune,scn,sccnn"
REDDIT_MODELS="${REDDIT_MODEL_FILTER:-gcn,edgnn,unignn,cwn,cell_topotune}"

run_dataset() {
    local dataset="$1"
    local models="$2"

    echo
    echo "Running full-graph preprocessing: ${dataset}"
    echo "Models: ${models}"

    env \
        SELECTED_GPUS="$SELECTED_GPUS" \
        JOBS_PER_GPU_OVERRIDE="$JOBS_PER_GPU_OVERRIDE" \
        MAX_CONCURRENT_RUNS="$MAX_CONCURRENT_RUNS" \
        DATA_SEEDS_OVERRIDE="$DATA_SEEDS_OVERRIDE" \
        STREAM_NUM_WORKERS="$STREAM_NUM_WORKERS" \
        RESUME="$RESUME" \
        DRY_RUN="$DRY_RUN" \
        MAX_ATTEMPTS="$MAX_ATTEMPTS" \
        DATASET_FILTER="$dataset" \
        MODEL_FILTER="$models" \
        WANDB_PROJECT_PREFIX="$WANDB_PROJECT_PREFIX" \
        RUN_NAME_PREFIX="$RUN_NAME_PREFIX" \
        LOG_GROUP="${LOG_GROUP_PREFIX}_${dataset}_sweep" \
        bash "$SCRIPT_DIR/full_graph_preprocessing.sh"
}

echo "Full-graph preprocessing memory sweep"
echo "Datasets: amazon_ratings questions cora_full coauthor_physics reddit"
echo "W&B project prefix: ${WANDB_PROJECT_PREFIX}"
echo "Maximum concurrent runs: ${MAX_CONCURRENT_RUNS}"
echo "Force fresh preprocessing: true"

run_dataset "amazon_ratings" "$ALL_MODELS"
run_dataset "questions" "$ALL_MODELS"
run_dataset "cora_full" "$ALL_MODELS"
run_dataset "coauthor_physics" "$ALL_MODELS"
run_dataset "reddit" "$REDDIT_MODELS"

echo
echo "All full-graph preprocessing memory runs complete."
