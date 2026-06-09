#!/bin/bash

SELECTED_GPUS="${SELECTED_GPUS:-0,1,2,3,4,5,6,7}"
RESUME="${RESUME:-true}"
DRY_RUN="${DRY_RUN:-false}"
wandb_entity="${wandb_entity:-topobench-scalability}"

LR_VALUES=(0.001)
WEIGHT_DECAY_VALUES=(0)
OUT_CHANNELS_VALUES=(128)
PROJ_DROPOUT_VALUES=(0)

DATA_SEEDS=(0 100 200 300 400)
Q_VALUES=(64)
NUM_CLUSTERS=(2048)
VAL_BATCHES_VALUES=("")
TEST_BATCHES_VALUES=("")
STREAM_NUM_WORKERS="${STREAM_NUM_WORKERS:-8}"

MAX_EPOCHS="${MAX_EPOCHS:-1500}"
MIN_EPOCHS="${MIN_EPOCHS:-50}"
CHECK_VAL_EVERY_N_EPOCH="${CHECK_VAL_EVERY_N_EPOCH:-1}"
EARLY_STOPPING_PATIENCE="${EARLY_STOPPING_PATIENCE:-130}"

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
script_name="$(basename "${BASH_SOURCE[0]}" .sh)"
log_group="${script_name}_sweep"

DATASET_MODES=(
    "full::graph/tolokers"
    "partitioning::graph/tolokers_for_partitioning"
)

MODEL_SPECS=(
    "gcn::graph/gcn::graph"
    "edgnn::hypergraph/edgnn::hypergraph"
    "unignn2::hypergraph/unignn2::hypergraph"
    "cwn::cell/cwn::cell"
    "cell_topotune::cell/topotune::cell"
    "scn::simplicial/scn::simplicial"
    "sccnn::simplicial/sccnn::simplicial"
)

source "$SCRIPT_DIR/common.sh"
run_dataset_suite
