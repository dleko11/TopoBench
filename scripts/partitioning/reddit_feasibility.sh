#!/bin/bash

SELECTED_GPUS="${SELECTED_GPUS:-0}"
RESUME="${RESUME:-true}"
DRY_RUN="${DRY_RUN:-false}"
wandb_entity="${wandb_entity:-topobench-scalability}"

LR_VALUES=(0.01)
WEIGHT_DECAY_VALUES=(0)
OUT_CHANNELS_VALUES=(64)
PROJ_DROPOUT_VALUES=(0.3)

DATA_SEEDS=(0)
Q_VALUES=(20)
NUM_CLUSTERS=(10000 7000 5000 2000 1000)
VAL_BATCHES_VALUES=("")
TEST_BATCHES_VALUES=("")
STREAM_NUM_WORKERS="${STREAM_NUM_WORKERS:-1}"

MAX_EPOCHS=1
MIN_EPOCHS=1
CHECK_VAL_EVERY_N_EPOCH=1
EARLY_STOPPING_PATIENCE=1

WANDB_PROJECT_PREFIX="${WANDB_PROJECT_PREFIX:-reddit_feasibility}"
WANDB_PROJECT_SUFFIX="${WANDB_PROJECT_SUFFIX:-}"

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
script_name="$(basename "${BASH_SOURCE[0]}" .sh)"
log_group="${script_name}_sweep"

DATASET_MODES=(
    "partitioning::graph/reddit_for_partitioning"
)

MODEL_SPECS=(
    "cwn::cell/cwn::cell"
)

append_run_args() {
    local mode="$1"
    local q_value="$2"
    local _num_clusters="$3"
    local _model_alias="$4"
    local data_seed="$5"

    if ! is_partitioning_mode "$mode"; then
        return
    fi

    cmd+=(
        "seed=${data_seed}"
        "++dataset.loader.parameters.stream.q_val=${q_value}"
        "++dataset.loader.parameters.stream.q_test=${q_value}"
        "++dataset.loader.parameters.stream.cache_num_workers=1"
        "++dataset.loader.parameters.stream.cache_val=false"
        "++dataset.loader.parameters.stream.val_shuffle=true"
        "+trainer.num_sanity_val_steps=0"
        "test=false"
        "extras.print_config=false"
        "extras.enforce_tags=false"
    )
}

source "$SCRIPT_DIR/common.sh"
run_dataset_suite
