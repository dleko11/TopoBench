#!/bin/bash

ABLATION_NAME="cluster_shuffling"
ABLATION_VALUES=(true false)

ablation_label() {
    if [[ "$1" == "true" ]]; then
        echo "reshuffle"
    else
        echo "fixed"
    fi
}

append_ablation_args() {
    cmd+=("++dataset.loader.parameters.stream.train_shuffle=$1")
}

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"
