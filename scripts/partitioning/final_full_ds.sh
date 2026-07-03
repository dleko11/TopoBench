#!/bin/bash

SELECTED_GPUS="${SELECTED_GPUS:-0,1,2,3,4,5,6,7}"
RESUME="${RESUME:-true}"
DRY_RUN="${DRY_RUN:-false}"
MAX_ATTEMPTS="${MAX_ATTEMPTS:-1}"
KEEP_SUCCESS_LOGS="${KEEP_SUCCESS_LOGS:-true}"
wandb_entity="${wandb_entity:-topobench-scalability}"

TRAINER="${TRAINER:-gpu}"
LOGGER="${LOGGER:-wandb}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-}"

MAX_EPOCHS="${MAX_EPOCHS:-300}"
MIN_EPOCHS="${MIN_EPOCHS:-1}"
CHECK_VAL_EVERY_N_EPOCH="${CHECK_VAL_EVERY_N_EPOCH:-5}"
EARLY_STOPPING_PATIENCE="${EARLY_STOPPING_PATIENCE:-5}"
TEST_INFERENCE_PROTOCOLS="${TEST_INFERENCE_PROTOCOLS:-[batched]}"

DATASET_FILTER="${DATASET_FILTER:-}"
MODEL_FILTER="${MODEL_FILTER:-}"
WANDB_PROJECT_PREFIX="${WANDB_PROJECT_PREFIX:-final}"
WANDB_PROJECT_SUFFIX="${WANDB_PROJECT_SUFFIX:-}"

if [[ -n "${DATA_SEEDS_OVERRIDE:-}" ]]; then
    data_seeds_string="${DATA_SEEDS_OVERRIDE//,/ }"
    read -ra DATA_SEEDS <<< "$data_seeds_string"
else
    DATA_SEEDS=(0 1 2 3 4)
fi

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
script_name="$(basename "${BASH_SOURCE[0]}" .sh)"
log_group="${script_name}_sweep"

source "$SCRIPT_DIR/common.sh"

FINAL_SPECS=(
    "questions|graph/questions|gcn|graph/gcn|graph|0.01|0.0001|128|0.3|0.5"
    "questions|graph/questions|edgnn|hypergraph/edgnn|hypergraph|0.001|0.001|128|0.0|0.5"
    "questions|graph/questions|unignn|hypergraph/unignn|hypergraph|0.01|0|32|0.0|"
    "questions|graph/questions|cwn|cell/cwn|cell|0.01|0.001|128|0.3|"
    "questions|graph/questions|cell_topotune|cell/topotune|cell|0.01|1e-05|128|0.1|0.0"
    "questions|graph/questions|scn|simplicial/scn|simplicial|0.01|0.0001|128|0.3|"
    "questions|graph/questions|sccnn|simplicial/sccnn_custom|simplicial|0.001|1e-05|128|0.1|"
    "cora_full|graph/cocitation_cora_full|gcn|graph/gcn|graph|0.001|1e-05|128|0.3|0.0"
    "cora_full|graph/cocitation_cora_full|edgnn|hypergraph/edgnn|hypergraph|0.001|0.001|128|0.3|0.0"
    "cora_full|graph/cocitation_cora_full|unignn|hypergraph/unignn|hypergraph|0.001|0.0001|128|0.3|"
    "cora_full|graph/cocitation_cora_full|cwn|cell/cwn|cell|0.01|0.001|64|0.3|"
    "cora_full|graph/cocitation_cora_full|cell_topotune|cell/topotune|cell|0.01|0.001|128|0.2|0.2"
    "cora_full|graph/cocitation_cora_full|scn|simplicial/scn|simplicial|0.001|0.001|128|0.3|"
    "cora_full|graph/cocitation_cora_full|sccnn|simplicial/sccnn_custom|simplicial|0.001|0.001|128|0.3|"
    "amazon_ratings|graph/amazon_ratings|gcn|graph/gcn|graph|0.01|0.001|128|0.0|0.0"
    "amazon_ratings|graph/amazon_ratings|edgnn|hypergraph/edgnn|hypergraph|0.001|0.001|128|0.1|0.0"
    "amazon_ratings|graph/amazon_ratings|unignn|hypergraph/unignn|hypergraph|0.01|0.001|128|0.0|"
    "amazon_ratings|graph/amazon_ratings|cwn|cell/cwn|cell|0.01|0.001|128|0.3|"
    "amazon_ratings|graph/amazon_ratings|cell_topotune|cell/topotune|cell|0.01|1e-05|64|0.1|0.2"
    "amazon_ratings|graph/amazon_ratings|scn|simplicial/scn|simplicial|0.01|0|64|0.3|"
    "amazon_ratings|graph/amazon_ratings|sccnn|simplicial/sccnn_custom|simplicial|0.001|0|128|0.3|"
)

matches_filter() {
    local alias="$1"
    local filter="$2"

    if [[ -z "$filter" ]]; then
        return 0
    fi

    local item
    IFS=',' read -ra filter_items <<< "$filter"
    for item in "${filter_items[@]}"; do
        if [[ "$alias" == "$item" ]]; then
            return 0
        fi
    done

    return 1
}

quote_cmd() {
    printf '%q ' "$@"
}

append_dropout_arg() {
    local model_alias="$1"
    local dropout="$2"

    if [[ -z "$dropout" ]]; then
        return
    fi

    case "$model_alias" in
        gcn|edgnn)
            cmd+=("model.backbone.dropout=${dropout}")
            ;;
        cell_topotune)
            cmd+=("model.backbone.GNN.dropout=${dropout}")
            ;;
    esac
}

count_selected_runs() {
    local total=0
    local spec dataset_alias dataset_config model_alias model_config
    local transform_kind lr weight_decay out_channels proj_dropout dropout

    for spec in "${FINAL_SPECS[@]}"; do
        IFS='|' read -r dataset_alias dataset_config model_alias model_config \
            transform_kind lr weight_decay out_channels proj_dropout \
            dropout <<< "$spec"
        if ! matches_filter "$dataset_alias" "$DATASET_FILTER"; then
            continue
        fi
        if ! matches_filter "$model_alias" "$MODEL_FILTER"; then
            continue
        fi
        total=$(( total + ${#DATA_SEEDS[@]} ))
    done

    echo "$total"
}

run_final_full_ds_suite() {
    init_experiment_environment
    detect_gpu_slots

    local success_log="$LOG_DIR/$log_group/SUCCESSFUL_RUNS.log"
    local total_runs
    total_runs=$(count_selected_runs)
    local launched=0
    local skipped=0
    local run_counter=0

    if [[ "$total_runs" -eq 0 ]]; then
        echo "ERROR: no runs selected. Check DATASET_FILTER and MODEL_FILTER." >&2
        exit 1
    fi

    echo "Total final full-dataset runs planned: $total_runs"
    echo "Dry run: $DRY_RUN"
    echo "Datasets filter: ${DATASET_FILTER:-all}"
    echo "Models filter: ${MODEL_FILTER:-all}"
    echo "Data seeds: ${DATA_SEEDS[*]}"
    echo "Test inference protocols: $TEST_INFERENCE_PROTOCOLS"
    echo "Dataloader num_workers override: ${DATALOADER_NUM_WORKERS:-config default}"

    local spec dataset_alias dataset_config model_alias model_config
    local transform_kind lr weight_decay out_channels proj_dropout dropout
    local data_seed current_gpu project_name run_name assigned_slot cmd_string

    for spec in "${FINAL_SPECS[@]}"; do
        IFS='|' read -r dataset_alias dataset_config model_alias model_config \
            transform_kind lr weight_decay out_channels proj_dropout \
            dropout <<< "$spec"
        if ! matches_filter "$dataset_alias" "$DATASET_FILTER"; then
            continue
        fi
        if ! matches_filter "$model_alias" "$MODEL_FILTER"; then
            continue
        fi

        for data_seed in "${DATA_SEEDS[@]}"; do
            run_name="final_full_${dataset_alias}_${model_alias}_seed${data_seed}"

            if [[ "$RESUME" == "true" && -f "$success_log" ]] && grep -Fq "[SUCCESS] ${run_name}" "$success_log"; then
                skipped=$(( skipped + 1 ))
                continue
            fi

            run_counter=$(( run_counter + 1 ))
            echo "Run $run_counter / $total_runs: $run_name"

            assigned_slot=-1
            while [[ "$assigned_slot" -eq -1 ]]; do
                for i in "${!gpus[@]}"; do
                    local pid="${slot_pids[$i]}"
                    if [[ "$pid" -eq 0 ]] || ! kill -0 "$pid" 2>/dev/null; then
                        assigned_slot=$i
                        break
                    fi
                done
                if [[ "$assigned_slot" -eq -1 ]]; then
                    wait -n
                fi
            done

            current_gpu="${gpus[$assigned_slot]}"
            project_name="${WANDB_PROJECT_PREFIX}_${dataset_alias}_full${WANDB_PROJECT_SUFFIX}"

            cmd=(
                "python" "-m" "topobench"
                "dataset=${dataset_config}"
                "model=${model_config}"
                "trainer=${TRAINER}"
                "logger=${LOGGER}"
                "optimizer.parameters.lr=${lr}"
                "optimizer.parameters.weight_decay=${weight_decay}"
                "model.feature_encoder.out_channels=${out_channels}"
                "model.feature_encoder.proj_dropout=${proj_dropout}"
                "dataset.dataloader_params.batch_size=1"
                "dataset.split_params.data_seed=${data_seed}"
                "seed=${data_seed}"
                "trainer.max_epochs=${MAX_EPOCHS}"
                "trainer.min_epochs=${MIN_EPOCHS}"
                "trainer.check_val_every_n_epoch=${CHECK_VAL_EVERY_N_EPOCH}"
                "callbacks.early_stopping.patience=${EARLY_STOPPING_PATIENCE}"
                "test_inference.protocols=${TEST_INFERENCE_PROTOCOLS}"
                "+trainer.enable_progress_bar=false"
                "extras.print_config=false"
                "extras.enforce_tags=false"
            )

            append_dropout_arg "$model_alias" "$dropout"

            if [[ -n "$DATALOADER_NUM_WORKERS" ]]; then
                cmd+=("dataset.dataloader_params.num_workers=${DATALOADER_NUM_WORKERS}")
            fi

            if [[ "$TRAINER" == "gpu" ]]; then
                cmd+=("trainer.devices=[${current_gpu}]")
            fi

            if [[ "$LOGGER" == "wandb" ]]; then
                cmd+=(
                    "logger.wandb.project=${project_name}"
                    "+logger.wandb.name=${run_name}"
                )
                if [[ -n "${wandb_entity:-}" ]]; then
                    cmd+=("+logger.wandb.entity=${wandb_entity}")
                fi
            fi

            append_transform_args "$transform_kind"
            if [[ "$transform_kind" == "simplicial" ]]; then
                cmd+=("transforms.graph2simplicial_lifting.complex_dim=2")
            fi

            cmd_string=$(quote_cmd "${cmd[@]}")

            if [[ "$DRY_RUN" == "true" ]]; then
                printf '[DRY_RUN] %s\n' "$cmd_string"
                slot_pids[$assigned_slot]=0
            else
                run_and_log "$cmd_string" "$log_group" "$run_name" "$LOG_DIR" &
                slot_pids[$assigned_slot]=$!
                launched=$(( launched + 1 ))
            fi
        done
    done

    echo "All final full-dataset runs launched ($launched launched, $skipped skipped)."
    if [[ "$DRY_RUN" != "true" ]]; then
        echo "Waiting for remaining final full-dataset runs..."
        wait
        echo "All final full-dataset runs complete."
    fi
}

run_final_full_ds_suite
