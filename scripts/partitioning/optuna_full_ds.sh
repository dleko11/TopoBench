#!/bin/bash

SELECTED_GPUS="${SELECTED_GPUS:-0,1,2,3,4,5,6,7}"
RESUME="${RESUME:-true}"
DRY_RUN="${DRY_RUN:-false}"
MAX_ATTEMPTS="${MAX_ATTEMPTS:-1}"
KEEP_SUCCESS_LOGS="${KEEP_SUCCESS_LOGS:-true}"
wandb_entity="${wandb_entity:-topobench-scalability}"

N_TRIALS="${N_TRIALS:-50}"
TRAINER="${TRAINER:-gpu}"
LOGGER="${LOGGER:-wandb}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-}"

MAX_EPOCHS="${MAX_EPOCHS:-300}"
MIN_EPOCHS="${MIN_EPOCHS:-1}"
CHECK_VAL_EVERY_N_EPOCH="${CHECK_VAL_EVERY_N_EPOCH:-5}"
EARLY_STOPPING_PATIENCE="${EARLY_STOPPING_PATIENCE:-5}"

LR_SPACE="${LR_SPACE:-choice(1e-5,1e-4,1e-3,1e-2)}"
WEIGHT_DECAY_SPACE="${WEIGHT_DECAY_SPACE:-choice(0,1e-5,1e-4,1e-3)}"
OUT_CHANNELS_SPACE="${OUT_CHANNELS_SPACE:-choice(32,64,128)}"
PROJ_DROPOUT_SPACE="${PROJ_DROPOUT_SPACE:-choice(0.0,0.1,0.2,0.3)}"

# Optional comma-separated aliases, e.g.
# DATASET_FILTER=questions,cora_full MODEL_FILTER=gcn,scn
DATASET_FILTER="${DATASET_FILTER:-}"
MODEL_FILTER="${MODEL_FILTER:-}"
STUDY_PREFIX="${STUDY_PREFIX:-full_ds_optuna}"
WANDB_PROJECT_PREFIX="${WANDB_PROJECT_PREFIX:-full_ds_optuna}"
WANDB_PROJECT_SUFFIX="${WANDB_PROJECT_SUFFIX:-}"
OPTUNA_STORAGE="${OPTUNA_STORAGE:-null}"

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
script_name="$(basename "${BASH_SOURCE[0]}" .sh)"
log_group="${script_name}_sweep"

DATASET_SPECS=(
    "questions::graph/questions"
    "coauthor_physics::graph/coauthor_physics"
    "cora_full::graph/cocitation_cora_full"
)

MODEL_SPECS=(
    "gcn::graph/gcn::graph"
    "edgnn::hypergraph/edgnn::hypergraph"
    "unignn::hypergraph/unignn::hypergraph"
    "cwn::cell/cwn::cell"
    "cell_topotune::cell/topotune::cell"
    "scn::simplicial/scn::simplicial"
    "sccnn::simplicial/sccnn_custom::simplicial"
)

source "$SCRIPT_DIR/common.sh"

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

append_model_search_args() {
    local model_alias="$1"

    case "$model_alias" in
        gcn|edgnn)
            add_sweeper_param "model.backbone.dropout" "choice(0.0,0.2,0.5)"
            ;;
        cell_topotune)
            add_sweeper_param "model.backbone.GNN.dropout" "choice(0.0,0.2,0.5)"
            ;;
        unignn|cwn|scn|sccnn)
            ;;
        *)
            echo "ERROR: unknown model alias: $model_alias" >&2
            exit 1
            ;;
    esac
}

add_sweeper_param() {
    local key="$1"
    local value="$2"

    sweeper_params+=("${key}:\"${value}\"")
}

join_sweeper_params() {
    local IFS=,
    printf '{%s}' "${sweeper_params[*]}"
}

count_selected_studies() {
    local total=0
    local dataset_spec model_spec
    local dataset_alias dataset_config
    local model_alias model_config transform_kind

    for dataset_spec in "${DATASET_SPECS[@]}"; do
        IFS="::" read -r dataset_alias _ dataset_config <<< "$dataset_spec"
        if ! matches_filter "$dataset_alias" "$DATASET_FILTER"; then
            continue
        fi

        for model_spec in "${MODEL_SPECS[@]}"; do
            IFS="::" read -r model_alias _ model_config _ transform_kind <<< "$model_spec"
            if ! matches_filter "$model_alias" "$MODEL_FILTER"; then
                continue
            fi
            ((total++))
        done
    done

    echo "$total"
}

run_optuna_suite() {
    init_experiment_environment
    detect_gpu_slots
    mkdir -p logs/optuna

    local success_log="$LOG_DIR/$log_group/SUCCESSFUL_RUNS.log"
    local total_studies
    total_studies=$(count_selected_studies)
    local launched=0
    local skipped=0
    local study_counter=0

    if [[ "$total_studies" -eq 0 ]]; then
        echo "ERROR: no studies selected. Check DATASET_FILTER and MODEL_FILTER." >&2
        exit 1
    fi

    echo "Total Optuna studies planned: $total_studies"
    echo "Dry run: $DRY_RUN"
    echo "Datasets filter: ${DATASET_FILTER:-all}"
    echo "Models filter: ${MODEL_FILTER:-all}"
    echo "Trials per study: $N_TRIALS"
    echo "Common LR space: $LR_SPACE"
    echo "Common weight decay space: $WEIGHT_DECAY_SPACE"
    echo "Common out_channels space: $OUT_CHANNELS_SPACE"
    echo "Common proj_dropout space: $PROJ_DROPOUT_SPACE"
    echo "Dataloader num_workers override: ${DATALOADER_NUM_WORKERS:-config default}"
    echo "Optuna storage: $OPTUNA_STORAGE"

    local dataset_spec model_spec
    local dataset_alias dataset_config
    local model_alias model_config transform_kind transform_name

    for dataset_spec in "${DATASET_SPECS[@]}"; do
        IFS="::" read -r dataset_alias _ dataset_config <<< "$dataset_spec"
        if ! matches_filter "$dataset_alias" "$DATASET_FILTER"; then
            continue
        fi

        for model_spec in "${MODEL_SPECS[@]}"; do
            IFS="::" read -r model_alias _ model_config _ transform_kind <<< "$model_spec"
            if ! matches_filter "$model_alias" "$MODEL_FILTER"; then
                continue
            fi

            transform_name=$(transform_alias "$transform_kind")
            local study_name="${STUDY_PREFIX}_${dataset_alias}_${model_alias}"
            local run_name="$study_name"

            if [[ "$RESUME" == "true" && -f "$success_log" ]] && grep -Fq "[SUCCESS] ${run_name}" "$success_log"; then
                ((skipped++))
                continue
            fi

            ((study_counter++))
            echo "Study $study_counter / $total_studies: $study_name"

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
            local project_name="${WANDB_PROJECT_PREFIX}_${dataset_alias}${WANDB_PROJECT_SUFFIX}"
            local optuna_storage="$OPTUNA_STORAGE"
            optuna_storage="${optuna_storage//\{study_name\}/$study_name}"
            sweeper_params=()
            add_sweeper_param "optimizer.parameters.lr" "$LR_SPACE"
            add_sweeper_param "optimizer.parameters.weight_decay" "$WEIGHT_DECAY_SPACE"
            add_sweeper_param "model.feature_encoder.out_channels" "$OUT_CHANNELS_SPACE"
            add_sweeper_param "model.feature_encoder.proj_dropout" "$PROJ_DROPOUT_SPACE"
            append_model_search_args "$model_alias"

            local sweeper_params_override
            sweeper_params_override=$(join_sweeper_params)

            cmd=(
                "python" "-m" "topobench" "--multirun"
                "hparams_search=partitioning_optuna"
                "dataset=${dataset_config}"
                "model=${model_config}"
                "trainer=${TRAINER}"
                "logger=${LOGGER}"
                "hydra.sweeper.storage=${optuna_storage}"
                "hydra.sweeper.study_name=${study_name}"
                "hydra.sweeper.n_trials=${N_TRIALS}"
                "hydra.sweeper.n_jobs=1"
                "+hydra.sweeper.params=${sweeper_params_override}"
                "dataset.dataloader_params.batch_size=1"
                "trainer.max_epochs=${MAX_EPOCHS}"
                "trainer.min_epochs=${MIN_EPOCHS}"
                "trainer.check_val_every_n_epoch=${CHECK_VAL_EVERY_N_EPOCH}"
                "callbacks.early_stopping.patience=${EARLY_STOPPING_PATIENCE}"
                "+trainer.enable_progress_bar=false"
                "test=false"
                "extras.print_config=false"
                "extras.enforce_tags=false"
            )

            if [[ -n "$DATALOADER_NUM_WORKERS" ]]; then
                cmd+=("dataset.dataloader_params.num_workers=${DATALOADER_NUM_WORKERS}")
            fi

            if [[ "$TRAINER" == "gpu" ]]; then
                cmd+=("trainer.devices=[${current_gpu}]")
            fi

            if [[ "$LOGGER" == "wandb" ]]; then
                cmd+=(
                    "logger.wandb.project=${project_name}"
                    "+logger.wandb.name=${study_name}"
                )
                if [[ -n "${wandb_entity:-}" ]]; then
                    cmd+=("+logger.wandb.entity=${wandb_entity}")
                fi
            fi

            append_transform_args "$transform_kind"
            if [[ "$transform_kind" == "simplicial" ]]; then
                cmd+=("transforms.graph2simplicial_lifting.complex_dim=2")
            fi

            local cmd_string
            cmd_string=$(quote_cmd "${cmd[@]}")

            if [[ "$DRY_RUN" == "true" ]]; then
                printf '[DRY_RUN] %s\n' "$cmd_string"
                slot_pids[$assigned_slot]=0
            else
                run_and_log "$cmd_string" "$log_group" "$run_name" "$LOG_DIR" &
                slot_pids[$assigned_slot]=$!
                ((launched++))
            fi
        done
    done

    echo "All Optuna studies launched ($launched launched, $skipped skipped)."
    if [[ "$DRY_RUN" != "true" ]]; then
        echo "Waiting for remaining studies..."
        wait
        echo "All Optuna studies complete."
    fi
}

run_optuna_suite
