#!/bin/bash

set -euo pipefail

OUTPUT_DIR="${OUTPUT_DIR:-outputs/structural_observability}"
DATASET_FILTER="${DATASET_FILTER:-amazon_ratings,questions,cora_full,coauthor_physics,reddit}"
FAMILIES="${FAMILIES:-hypergraph,simplicial,cell}"
MAX_CELL_LENGTH="${MAX_CELL_LENGTH:-9}"
RESUME="${RESUME:-true}"
WANDB_PROJECT="${WANDB_PROJECT:-structural_observability_scaling}"
WANDB_ENTITY="${WANDB_ENTITY:-topobench-scalability}"

AMAZON_RATINGS_NUM_PARTS="${AMAZON_RATINGS_NUM_PARTS:-32}"
QUESTIONS_NUM_PARTS="${QUESTIONS_NUM_PARTS:-500}"
CORA_FULL_NUM_PARTS="${CORA_FULL_NUM_PARTS:-32}"
COAUTHOR_PHYSICS_NUM_PARTS="${COAUTHOR_PHYSICS_NUM_PARTS:-2000}"
REDDIT_NUM_PARTS="${REDDIT_NUM_PARTS:-10000}"

matches_filter() {
    local candidate="$1"
    local entry
    IFS=',' read -ra entries <<< "$DATASET_FILTER"
    for entry in "${entries[@]}"; do
        if [[ "$candidate" == "$entry" ]]; then
            return 0
        fi
    done
    return 1
}

run_dataset() {
    local dataset="$1"
    local num_parts="$2"
    if ! matches_filter "$dataset"; then
        return
    fi

    local -a command=(
        uv run python -m scripts.partitioning.structural_observability
        --dataset "$dataset"
        --num-parts "$num_parts"
        --families "$FAMILIES"
        --max-cell-length "$MAX_CELL_LENGTH"
        --output-dir "$OUTPUT_DIR"
        --wandb-entity "$WANDB_ENTITY"
    )
    if [[ "$RESUME" == "true" ]]; then
        command+=(--resume)
    elif [[ "$RESUME" == "false" ]]; then
        command+=(--no-resume)
    else
        echo "ERROR: RESUME must be true or false." >&2
        exit 1
    fi
    if [[ -n "$WANDB_PROJECT" ]]; then
        command+=(--wandb-project "$WANDB_PROJECT")
    fi

    echo "Running ${dataset}: K=${num_parts}, families=${FAMILIES}"
    PYTHONUNBUFFERED=1 "${command[@]}"
}

echo "Exact structural observability sweep"
echo "Datasets: $DATASET_FILTER"
echo "Output: $OUTPUT_DIR"

run_dataset amazon_ratings "$AMAZON_RATINGS_NUM_PARTS"
run_dataset questions "$QUESTIONS_NUM_PARTS"
run_dataset cora_full "$CORA_FULL_NUM_PARTS"
run_dataset coauthor_physics "$COAUTHOR_PHYSICS_NUM_PARTS"
run_dataset reddit "$REDDIT_NUM_PARTS"

echo "Structural observability counting complete."
