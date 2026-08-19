#!/bin/bash

ABLATION_NAME="cross_cluster_edges"
ABLATION_VALUES=(true false)

ablation_label() {
    if [[ "$1" == "true" ]]; then
        echo "preserve"
    else
        echo "remove"
    fi
}

append_ablation_args() {
    cmd+=(
        "++dataset.loader.parameters.stream.reconstruct_cross_cluster_edges=$1"
    )
}

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"
