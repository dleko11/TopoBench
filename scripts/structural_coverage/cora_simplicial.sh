#!/bin/bash

set -euo pipefail

Q_VALUES="${Q_VALUES:-1 2 4}"
DATA_SEEDS="${DATA_SEEDS:-0}"
DATA_SPLIT_SEED="${DATA_SPLIT_SEED:-}"
MAX_EPOCHS="${MAX_EPOCHS:-50}"
TRAINER="${TRAINER:-cpu}"
SAVE_BATCH_EVENTS="${SAVE_BATCH_EVENTS:-false}"
PYTHON_CMD="${PYTHON_CMD:-python}"
MODEL_CONFIG="${MODEL_CONFIG:-simplicial/scn}"
TRANSFORMS_CONFIG="${TRANSFORMS_CONFIG:-liftings/graph2simplicial_default}"
STRUCTURE_FAMILY="${STRUCTURE_FAMILY:-auto}"
LIFTING_OVERRIDE="${LIFTING_OVERRIDE:-}"
read -r -a PYTHON_BIN <<< "$PYTHON_CMD"

for q in ${Q_VALUES}; do
  for seed in ${DATA_SEEDS}; do
    split_seed="${DATA_SPLIT_SEED:-${seed}}"
    cmd=(
      "${PYTHON_BIN[@]}" -m scripts.structural_coverage.run_cora_simplicial
      "model=${MODEL_CONFIG}"
      "transforms=${TRANSFORMS_CONFIG}"
      "trainer=${TRAINER}"
      "seed=${seed}"
      "dataset.split_params.data_seed=${split_seed}"
      "dataset.loader.parameters.stream.q=${q}"
      "++dataset.loader.parameters.stream.q_val=${q}"
      "++dataset.loader.parameters.stream.q_test=${q}"
      "trainer.max_epochs=${MAX_EPOCHS}"
      "coverage.structure_family=${STRUCTURE_FAMILY}"
      "coverage.save_batch_events=${SAVE_BATCH_EVENTS}"
    )
    if [[ -n "${LIFTING_OVERRIDE}" ]]; then
      cmd+=("${LIFTING_OVERRIDE}")
    fi
    cmd+=("$@")
    "${cmd[@]}"
  done
done
