#!/bin/bash

SELECTED_GPUS="${SELECTED_GPUS:-0,1,2,3,4,5,6,7}"
RESUME="${RESUME:-true}"
DRY_RUN="${DRY_RUN:-false}"
MAX_ATTEMPTS="${MAX_ATTEMPTS:-1}"
KEEP_SUCCESS_LOGS="${KEEP_SUCCESS_LOGS:-true}"
wandb_entity="${wandb_entity:-topobench-scalability}"

TRAINER="${TRAINER:-gpu}"
LOGGER="${LOGGER:-wandb}"
STREAM_NUM_WORKERS="${STREAM_NUM_WORKERS:-8}"

MAX_EPOCHS="${MAX_EPOCHS:-300}"
MIN_EPOCHS="${MIN_EPOCHS:-1}"
CHECK_VAL_EVERY_N_EPOCH="${CHECK_VAL_EVERY_N_EPOCH:-5}"
EARLY_STOPPING_PATIENCE="${EARLY_STOPPING_PATIENCE:-5}"
ENSEMBLE_RUNS="${ENSEMBLE_RUNS:-10}"
TEST_INFERENCE_PROTOCOLS="${TEST_INFERENCE_PROTOCOLS:-[batched,ensemble,full_graph]}"

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
    "questions|graph/questions_for_partitioning|gcn|graph/gcn|graph|0.0001|0|128|0.2|500|20|0.2"
    "questions|graph/questions_for_partitioning|edgnn|hypergraph/edgnn|hypergraph|0.01|0.001|32|0.2|500|50|0.0"
    "questions|graph/questions_for_partitioning|unignn|hypergraph/unignn|hypergraph|0.001|0.001|128|0.0|1000|40|"
    "questions|graph/questions_for_partitioning|cwn|cell/cwn|cell|0.01|0.001|32|0.3|1000|40|"
    "questions|graph/questions_for_partitioning|cell_topotune|cell/topotune|cell|0.01|0.001|128|0.3|500|50|0.2"
    "questions|graph/questions_for_partitioning|scn|simplicial/scn|simplicial|0.001|0.001|64|0.1|500|50|"
    "questions|graph/questions_for_partitioning|sccnn|simplicial/sccnn_custom|simplicial|0.001|1e-05|128|0.3|500|30|"
    "cora_full|graph/cocitation_cora_full_for_partitioning|gcn|graph/gcn|graph|0.001|1e-05|128|0.3|32|8|0.0"
    "cora_full|graph/cocitation_cora_full_for_partitioning|edgnn|hypergraph/edgnn|hypergraph|0.01|0.0001|64|0.2|32|4|0.2"
    "cora_full|graph/cocitation_cora_full_for_partitioning|unignn|hypergraph/unignn|hypergraph|0.0001|0.0001|128|0.1|32|4|"
    "cora_full|graph/cocitation_cora_full_for_partitioning|cwn|cell/cwn|cell|0.01|0.001|32|0.3|32|4|"
    "cora_full|graph/cocitation_cora_full_for_partitioning|cell_topotune|cell/topotune|cell|0.01|0.001|128|0.2|64|8|0.5"
    "cora_full|graph/cocitation_cora_full_for_partitioning|scn|simplicial/scn|simplicial|0.001|0.0001|128|0.3|32|16|"
    "cora_full|graph/cocitation_cora_full_for_partitioning|sccnn|simplicial/sccnn_custom|simplicial|0.001|0.001|128|0.1|32|8|"
    "amazon_ratings|graph/amazon_ratings_for_partitioning|gcn|graph/gcn|graph|0.01|0.0001|128|0.3|64|4|0.0"
    "amazon_ratings|graph/amazon_ratings_for_partitioning|edgnn|hypergraph/edgnn|hypergraph|0.001|0.0001|128|0.3|128|4|0.0"
    "amazon_ratings|graph/amazon_ratings_for_partitioning|unignn|hypergraph/unignn|hypergraph|0.01|0.0001|64|0.0|32|8|"
    "amazon_ratings|graph/amazon_ratings_for_partitioning|cwn|cell/cwn|cell|0.01|0.001|128|0.2|64|16|"
    "amazon_ratings|graph/amazon_ratings_for_partitioning|cell_topotune|cell/topotune|cell|0.01|1e-05|128|0.0|32|8|0.5"
    "amazon_ratings|graph/amazon_ratings_for_partitioning|scn|simplicial/scn|simplicial|0.001|0|128|0.2|32|8|"
    "amazon_ratings|graph/amazon_ratings_for_partitioning|sccnn|simplicial/sccnn_custom|simplicial|0.001|0.001|128|0.2|32|4|"
    "coauthor_physics|graph/coauthor_physics_for_partitioning|gcn|graph/gcn|graph|0.01|0.0001|64|0.3|500|20|0.2"
    "coauthor_physics|graph/coauthor_physics_for_partitioning|edgnn|hypergraph/edgnn|hypergraph|0.01|0|128|0.2|2000|30|0.5"
    "coauthor_physics|graph/coauthor_physics_for_partitioning|unignn|hypergraph/unignn|hypergraph|0.01|1e-05|32|0.2|2000|20|"
    "coauthor_physics|graph/coauthor_physics_for_partitioning|cwn|cell/cwn|cell|0.01|0|32|0.2|2000|20|"
    "coauthor_physics|graph/coauthor_physics_for_partitioning|cell_topotune|cell/topotune|cell|0.001|0.001|32|0.3|2000|20|0.5"
    "coauthor_physics|graph/coauthor_physics_for_partitioning|scn|simplicial/scn|simplicial|0.01|0|32|0.3|2000|20|"
    "coauthor_physics|graph/coauthor_physics_for_partitioning|sccnn|simplicial/sccnn_custom|simplicial|0.01|0|128|0.3|2000|20|"
    "reddit|graph/reddit_for_partitioning|gcn|graph/gcn|graph|0.01|0|64|0.3|500|20|0.5"
    "reddit|graph/reddit_for_partitioning|edgnn|hypergraph/edgnn|hypergraph|0.01|0|64|0.3|500|20|0.5"
    "reddit|graph/reddit_for_partitioning|unignn|hypergraph/unignn|hypergraph|0.01|0|64|0.3|500|20|"
    "reddit|graph/reddit_for_partitioning|cwn|cell/cwn|cell|0.01|0|64|0.3|500|20|"
    "reddit|graph/reddit_for_partitioning|cell_topotune|cell/topotune|cell|0.01|0|64|0.3|500|20|0.5"
    "reddit|graph/reddit_for_partitioning|scn|simplicial/scn|simplicial|0.01|0|64|0.3|500|20|"
    "reddit|graph/reddit_for_partitioning|sccnn|simplicial/sccnn_custom|simplicial|0.01|0|64|0.3|500|20|"
    "ogbn_products|graph/ogbn_products_for_partitioning|gcn|graph/gcn|graph|0.01|0|128|0.2|1000|20|0.0"
    "ogbn_products|graph/ogbn_products_for_partitioning|edgnn|hypergraph/edgnn|hypergraph|0.01|0|128|0.2|1000|20|0.0"
    "ogbn_products|graph/ogbn_products_for_partitioning|unignn|hypergraph/unignn|hypergraph|0.01|0|128|0.2|1000|20|"
    "ogbn_products|graph/ogbn_products_for_partitioning|cwn|cell/cwn|cell|0.01|0|128|0.2|1000|20|"
    "ogbn_products|graph/ogbn_products_for_partitioning|cell_topotune|cell/topotune|cell|0.01|0|128|0.2|1000|20|0.0"
    "ogbn_products|graph/ogbn_products_for_partitioning|scn|simplicial/scn|simplicial|0.01|0|128|0.2|1000|20|"
    "ogbn_products|graph/ogbn_products_for_partitioning|sccnn|simplicial/sccnn_custom|simplicial|0.01|0|128|0.2|1000|20|"
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
    local transform_kind lr weight_decay out_channels proj_dropout
    local num_parts q dropout

    for spec in "${FINAL_SPECS[@]}"; do
        IFS='|' read -r dataset_alias dataset_config model_alias model_config \
            transform_kind lr weight_decay out_channels proj_dropout \
            num_parts q dropout <<< "$spec"
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

run_final_partitioning_suite() {
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

    echo "Total final partitioning runs planned: $total_runs"
    echo "Dry run: $DRY_RUN"
    echo "Datasets filter: ${DATASET_FILTER:-all}"
    echo "Models filter: ${MODEL_FILTER:-all}"
    echo "Data seeds: ${DATA_SEEDS[*]}"
    echo "Test inference protocols: $TEST_INFERENCE_PROTOCOLS"
    echo "Ensemble runs: $ENSEMBLE_RUNS"

    local spec dataset_alias dataset_config model_alias model_config
    local transform_kind lr weight_decay out_channels proj_dropout
    local num_parts q dropout data_seed current_gpu project_name run_name
    local assigned_slot cmd_string

    for spec in "${FINAL_SPECS[@]}"; do
        IFS='|' read -r dataset_alias dataset_config model_alias model_config \
            transform_kind lr weight_decay out_channels proj_dropout \
            num_parts q dropout <<< "$spec"
        if ! matches_filter "$dataset_alias" "$DATASET_FILTER"; then
            continue
        fi
        if ! matches_filter "$model_alias" "$MODEL_FILTER"; then
            continue
        fi

        for data_seed in "${DATA_SEEDS[@]}"; do
            run_name="final_partitioning_${dataset_alias}_${model_alias}_seed${data_seed}_q${q}_clusters${num_parts}"

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
            project_name="${WANDB_PROJECT_PREFIX}_${dataset_alias}_partitioning${WANDB_PROJECT_SUFFIX}"

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
                "dataset.loader.parameters.stream.q=${q}"
                "++dataset.loader.parameters.stream.q_val=${q}"
                "++dataset.loader.parameters.stream.q_test=${q}"
                "dataset.loader.parameters.cluster.num_parts=${num_parts}"
                "dataset.loader.parameters.stream.num_workers=${STREAM_NUM_WORKERS}"
                "dataset.dataloader_params.num_workers=${STREAM_NUM_WORKERS}"
                "trainer.max_epochs=${MAX_EPOCHS}"
                "trainer.min_epochs=${MIN_EPOCHS}"
                "trainer.check_val_every_n_epoch=${CHECK_VAL_EVERY_N_EPOCH}"
                "callbacks.early_stopping.patience=${EARLY_STOPPING_PATIENCE}"
                "test_inference.protocols=${TEST_INFERENCE_PROTOCOLS}"
                "test_inference.ensemble_runs=${ENSEMBLE_RUNS}"
                "test_inference.ensemble_seed=${data_seed}"
                "+trainer.enable_progress_bar=false"
                "extras.print_config=false"
                "extras.enforce_tags=false"
            )

            append_dropout_arg "$model_alias" "$dropout"

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

    echo "All final partitioning runs launched ($launched launched, $skipped skipped)."
    if [[ "$DRY_RUN" != "true" ]]; then
        echo "Waiting for remaining final partitioning runs..."
        wait
        echo "All final partitioning runs complete."
    fi
}

run_final_partitioning_suite
