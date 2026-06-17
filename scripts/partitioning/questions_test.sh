#!/bin/bash

SELECTED_GPUS="${SELECTED_GPUS:-0}"
RESUME="${RESUME:-true}"
DRY_RUN="${DRY_RUN:-false}"
wandb_entity="${wandb_entity:-topobench-scalability}"

LR_VALUES=(0.1 0.001 0.001)
WEIGHT_DECAY_VALUES=(0 1e-3 1e-5)
OUT_CHANNELS_VALUES=(128)
PROJ_DROPOUT_VALUES=(0.0 0.1 0.2)

DATA_SEEDS=(0)
Q_VALUES=(500 1000 2000 3000 4000)
NUM_CLUSTERS=(10 20 30 40 50)
VAL_BATCHES_VALUES=("")
TEST_BATCHES_VALUES=("")
STREAM_NUM_WORKERS="${STREAM_NUM_WORKERS:-8}"

MAX_EPOCHS="${MAX_EPOCHS:-1500}"
MIN_EPOCHS="${MIN_EPOCHS:-50}"
CHECK_VAL_EVERY_N_EPOCH="${CHECK_VAL_EVERY_N_EPOCH:-5}"
EARLY_STOPPING_PATIENCE="${EARLY_STOPPING_PATIENCE:-15}"

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
script_name="$(basename "${BASH_SOURCE[0]}" .sh)"
log_group="${script_name}_sweep"

DATASET_MODES=(
    "partitioning::graph/questions_for_partitioning"
)

MODEL_SPECS=(
    "gcn::graph/gcn::graph"
    "edgnn::hypergraph/edgnn::hypergraph"
    "cwn::cell/cwn::cell"
    "scn::simplicial/scn::simplicial"
)

source "$SCRIPT_DIR/common.sh"
run_dataset_suite
