#!/bin/bash

SELECTED_GPUS="${SELECTED_GPUS:-0}"
RESUME="${RESUME:-true}"
DRY_RUN="${DRY_RUN:-false}"
wandb_entity="${wandb_entity:-topobench-scalability}"

LR_VALUES=(0.001)
WEIGHT_DECAY_VALUES=(0)
OUT_CHANNELS_VALUES=(128)
PROJ_DROPOUT_VALUES=(0)

DATA_SEEDS=(0)
Q_VALUES=(64)
NUM_CLUSTERS=(2048)
VAL_BATCHES_VALUES=("")
TEST_BATCHES_VALUES=("")
STREAM_NUM_WORKERS="${STREAM_NUM_WORKERS:-8}"

MAX_EPOCHS="${MAX_EPOCHS:-1500}"
MIN_EPOCHS="${MIN_EPOCHS:-50}"
CHECK_VAL_EVERY_N_EPOCH="${CHECK_VAL_EVERY_N_EPOCH:-1}"
EARLY_STOPPING_PATIENCE="${EARLY_STOPPING_PATIENCE:-130}"

WANDB_PROJECT_PREFIX="${WANDB_PROJECT_PREFIX:-questions}"
WANDB_PROJECT_SUFFIX="${WANDB_PROJECT_SUFFIX:-_test}"

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
script_name="$(basename "${BASH_SOURCE[0]}" .sh)"
log_group="${script_name}_sweep"

DATASET_MODES=(
    "partitioning::graph/questions_for_partitioning"
)

MODEL_SPECS=(
    "scn::simplicial/scn::simplicial"
)

source "$SCRIPT_DIR/common.sh"
run_dataset_suite
