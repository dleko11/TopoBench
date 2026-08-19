#!/bin/bash

ABLATION_NAME="partition_method"
ABLATION_VALUES=(metis random)

ablation_label() {
    echo "$1"
}

append_ablation_args() {
    local method="$1"
    local seed="$2"
    cmd+=("++dataset.loader.parameters.cluster.partition_method=${method}")
    if [[ "$method" == "random" ]]; then
        cmd+=("++dataset.loader.parameters.cluster.partition_seed=${seed}")
    fi
}

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"
