#!/bin/bash

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

exec env \
    FULL_GRAPH_BASELINE=true \
    FORCE_RELOAD_PREPROCESSING=true \
    TRAIN=false \
    TEST=false \
    SELECTED_GPUS="${SELECTED_GPUS:-0}" \
    JOBS_PER_GPU_OVERRIDE="${JOBS_PER_GPU_OVERRIDE:-1}" \
    MAX_CONCURRENT_RUNS="${MAX_CONCURRENT_RUNS:-1}" \
    DATASET_FILTER="${DATASET_FILTER:-questions,cora_full,amazon_ratings,coauthor_physics}" \
    MODEL_FILTER="${MODEL_FILTER:-gcn,edgnn,unignn,cwn,cell_topotune,scn,sccnn}" \
    DATA_SEEDS_OVERRIDE="${DATA_SEEDS_OVERRIDE:-0}" \
    STREAM_NUM_WORKERS="${STREAM_NUM_WORKERS:-0}" \
    MAX_EPOCHS=1 \
    MIN_EPOCHS=1 \
    CHECK_VAL_EVERY_N_EPOCH=1 \
    EARLY_STOPPING_PATIENCE=1 \
    RESUME="${RESUME:-true}" \
    DRY_RUN="${DRY_RUN:-false}" \
    RUN_NAME_PREFIX="${RUN_NAME_PREFIX:-full_graph_preprocessing}" \
    LOG_GROUP="${LOG_GROUP:-full_graph_preprocessing_sweep}" \
    WANDB_PROJECT_PREFIX="${WANDB_PROJECT_PREFIX:-preprocessing_scaling}" \
    bash "$SCRIPT_DIR/final_partitioning.sh"
