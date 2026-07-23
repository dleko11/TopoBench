#!/bin/bash

# METIS-versus-random partitioning ablation for Questions and Amazon Ratings.
# Default: 2 partition methods × 2 datasets × 7 models × 5 seeds = 140 runs.

SELECTED_GPUS="${SELECTED_GPUS:-0,1,2,3,4,5,6,7}"
MAX_PARALLEL="${MAX_PARALLEL:-0}"
PROCESSES_PER_GPU="${JOBS_PER_GPU:-1}"
RESUME="${RESUME:-true}"
DRY_RUN="${DRY_RUN:-false}"
STREAM_NUM_WORKERS="${STREAM_NUM_WORKERS:-8}"
MAX_EPOCHS="${MAX_EPOCHS:-300}"
MIN_EPOCHS="${MIN_EPOCHS:-1}"
CHECK_VAL_EVERY_N_EPOCH="${CHECK_VAL_EVERY_N_EPOCH:-5}"
EARLY_STOPPING_PATIENCE="${EARLY_STOPPING_PATIENCE:-5}"
wandb_entity="${wandb_entity:-topobench-scalability}"
WANDB_PROJECT_PREFIX="${WANDB_PROJECT_PREFIX:-ablation_random_partitioning}"
DATASET_FILTER="${DATASET_FILTER:-}"
MODEL_FILTER="${MODEL_FILTER:-}"

if [[ -n "${DATA_SEEDS_OVERRIDE:-}" ]]; then
    read -ra DATA_SEEDS <<< "${DATA_SEEDS_OVERRIDE//,/ }"
else
    DATA_SEEDS=(0 1 2 3 4)
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
script_name="$(basename "${BASH_SOURCE[0]}" .sh)"
log_group="${script_name}_sweep"
source "$SCRIPT_DIR/common.sh"

# dataset|config|model|model_config|transform|lr|weight_decay|channels|proj_dropout|parts|q|backbone_dropout
SPECS=(
    "questions|graph/questions_for_partitioning|gcn|graph/gcn|graph|0.0001|0|128|0.2|500|20|0.2"
    "questions|graph/questions_for_partitioning|edgnn|hypergraph/edgnn|hypergraph|0.01|0.001|32|0.2|500|50|0.0"
    "questions|graph/questions_for_partitioning|unignn|hypergraph/unignn2|hypergraph|0.001|0.001|128|0.0|1000|40|"
    "questions|graph/questions_for_partitioning|cwn|cell/cwn|cell|0.01|0.001|32|0.3|1000|40|"
    "questions|graph/questions_for_partitioning|cell_topotune|cell/topotune|cell|0.01|0.001|128|0.3|500|50|0.2"
    "questions|graph/questions_for_partitioning|scn|simplicial/scn|simplicial|0.001|0.001|64|0.1|500|50|"
    "questions|graph/questions_for_partitioning|sccnn|simplicial/sccnn_custom|simplicial|0.001|1e-05|128|0.3|500|30|"
    "amazon_ratings|graph/amazon_ratings_for_partitioning|gcn|graph/gcn|graph|0.01|0.0001|128|0.3|64|4|0.0"
    "amazon_ratings|graph/amazon_ratings_for_partitioning|edgnn|hypergraph/edgnn|hypergraph|0.001|0.0001|128|0.3|128|4|0.0"
    "amazon_ratings|graph/amazon_ratings_for_partitioning|unignn|hypergraph/unignn2|hypergraph|0.01|0.0001|64|0.0|32|8|"
    "amazon_ratings|graph/amazon_ratings_for_partitioning|cwn|cell/cwn|cell|0.01|0.001|128|0.2|64|16|"
    "amazon_ratings|graph/amazon_ratings_for_partitioning|cell_topotune|cell/topotune|cell|0.01|1e-05|128|0.0|32|8|0.5"
    "amazon_ratings|graph/amazon_ratings_for_partitioning|scn|simplicial/scn|simplicial|0.001|0|128|0.2|32|8|"
    "amazon_ratings|graph/amazon_ratings_for_partitioning|sccnn|simplicial/sccnn_custom|simplicial|0.001|0.001|128|0.2|32|4|"
)

matches_filter() {
    local value="$1" filter="$2" item
    [[ -z "$filter" ]] && return 0
    IFS=',' read -ra items <<< "$filter"
    for item in "${items[@]}"; do [[ "$value" == "$item" ]] && return 0; done
    return 1
}

append_dropout_arg() {
    local model="$1" dropout="$2"
    [[ -z "$dropout" ]] && return
    case "$model" in
        gcn|edgnn) cmd+=("model.backbone.dropout=${dropout}") ;;
        cell_topotune) cmd+=("model.backbone.GNN.dropout=${dropout}") ;;
    esac
}

append_transform_args() {
    case "$1" in
        graph) ;;
        hypergraph) cmd+=("transforms=liftings/graph2hypergraph_default" "transforms/liftings/graph2hypergraph@transforms.graph2hypergraph_lifting=khop_large_scale") ;;
        cell) cmd+=("transforms=liftings/graph2cell_default" "transforms/liftings/graph2cell@transforms.graph2cell_lifting=cycle_selective") ;;
        simplicial) cmd+=("transforms=liftings/graph2simplicial_default" "transforms/liftings/graph2simplicial@transforms.graph2simplicial_lifting=clique_selective") ;;
    esac
}

quote_cmd() { printf '%q ' "$@"; }

run_ablation() {
    init_experiment_environment
    detect_gpu_slots
    if ! [[ "$PROCESSES_PER_GPU" =~ ^[1-9][0-9]*$ ]]; then
        echo "ERROR: JOBS_PER_GPU must be a positive integer." >&2
        exit 1
    fi
    if [[ "$PROCESSES_PER_GPU" -gt 1 ]]; then
        local -a physical_gpus=("${gpus[@]}") expanded_gpus=()
        local gpu slot
        for gpu in "${physical_gpus[@]}"; do
            for ((slot = 1; slot <= PROCESSES_PER_GPU; slot++)); do
                expanded_gpus+=("$gpu")
            done
        done
        gpus=("${expanded_gpus[@]}")
        slot_pids=()
        for slot in "${!gpus[@]}"; do slot_pids[$slot]=0; done
    fi
    if [[ "$MAX_PARALLEL" != "0" ]]; then
        if ! [[ "$MAX_PARALLEL" =~ ^[1-9][0-9]*$ ]]; then
            echo "ERROR: MAX_PARALLEL must be 0 or a positive integer." >&2
            exit 1
        fi
        if [[ "$MAX_PARALLEL" -lt "${#gpus[@]}" ]]; then
            gpus=("${gpus[@]:0:MAX_PARALLEL}")
            slot_pids=("${slot_pids[@]:0:MAX_PARALLEL}")
        fi
    fi
    echo "Requested jobs per GPU: $PROCESSES_PER_GPU"
    echo "Maximum concurrent runs: ${#gpus[@]}"
    local success_log="$LOG_DIR/$log_group/SUCCESSFUL_RUNS.log"
    local total=0 launched=0 skipped=0 counter=0
    local spec dataset config model model_config transform lr wd channels proj_dropout parts q backbone_dropout
    local method seed run_name project_name assigned_slot current_gpu cmd_string
    for spec in "${SPECS[@]}"; do
        IFS='|' read -r dataset config model model_config transform lr wd channels proj_dropout parts q backbone_dropout <<< "$spec"
        matches_filter "$dataset" "$DATASET_FILTER" && matches_filter "$model" "$MODEL_FILTER" && total=$((total + 2 * ${#DATA_SEEDS[@]}))
    done
    [[ "$total" -gt 0 ]] || { echo "ERROR: no runs selected." >&2; exit 1; }
    echo "Planned random-partitioning runs: $total"
    for spec in "${SPECS[@]}"; do
        IFS='|' read -r dataset config model model_config transform lr wd channels proj_dropout parts q backbone_dropout <<< "$spec"
        matches_filter "$dataset" "$DATASET_FILTER" || continue
        matches_filter "$model" "$MODEL_FILTER" || continue
        for method in metis random; do
            for seed in "${DATA_SEEDS[@]}"; do
                run_name="random_partitioning_${method}_${dataset}_${model}_seed${seed}_q${q}_clusters${parts}"
                if [[ "$RESUME" == "true" && -f "$success_log" ]] && grep -Fq "[SUCCESS] ${run_name}" "$success_log"; then skipped=$((skipped + 1)); continue; fi
                counter=$((counter + 1)); echo "Run $counter / $total: $run_name"
                assigned_slot=-1
                while [[ "$assigned_slot" -eq -1 ]]; do
                    for i in "${!gpus[@]}"; do
                        if [[ "${slot_pids[$i]}" -eq 0 ]] || ! kill -0 "${slot_pids[$i]}" 2>/dev/null; then assigned_slot=$i; break; fi
                    done
                    [[ "$assigned_slot" -eq -1 ]] && wait -n
                done
                current_gpu="${gpus[$assigned_slot]}"
                project_name="${WANDB_PROJECT_PREFIX}_${dataset}"
                cmd=("python" "-m" "topobench" "dataset=${config}" "model=${model_config}" "trainer=gpu" "logger=wandb"
                    "optimizer.parameters.lr=${lr}" "optimizer.parameters.weight_decay=${wd}" "model.feature_encoder.out_channels=${channels}" "model.feature_encoder.proj_dropout=${proj_dropout}"
                    "dataset.dataloader_params.batch_size=1" "dataset.split_params.data_seed=${seed}" "seed=${seed}"
                    "dataset.loader.parameters.cluster.num_parts=${parts}" "++dataset.loader.parameters.cluster.partition_method=${method}"
                    "dataset.loader.parameters.stream.q=${q}" "++dataset.loader.parameters.stream.q_val=${q}" "++dataset.loader.parameters.stream.q_test=${q}"
                    "dataset.loader.parameters.stream.num_workers=${STREAM_NUM_WORKERS}" "dataset.dataloader_params.num_workers=${STREAM_NUM_WORKERS}"
                    "trainer.max_epochs=${MAX_EPOCHS}" "trainer.min_epochs=${MIN_EPOCHS}" "trainer.check_val_every_n_epoch=${CHECK_VAL_EVERY_N_EPOCH}" "callbacks.early_stopping.patience=${EARLY_STOPPING_PATIENCE}"
                    "trainer.devices=[${current_gpu}]" "logger.wandb.project=${project_name}" "+logger.wandb.name=${run_name}" "+trainer.enable_progress_bar=false")
                [[ "$method" == "random" ]] && cmd+=("++dataset.loader.parameters.cluster.partition_seed=${seed}")
                [[ -n "$wandb_entity" ]] && cmd+=("+logger.wandb.entity=${wandb_entity}")
                append_dropout_arg "$model" "$backbone_dropout"
                append_transform_args "$transform"
                cmd_string=$(quote_cmd "${cmd[@]}")
                if [[ "$DRY_RUN" == "true" ]]; then printf '[DRY_RUN] %s\n' "$cmd_string"; slot_pids[$assigned_slot]=0; else run_and_log "$cmd_string" "$log_group" "$run_name" "$LOG_DIR" & slot_pids[$assigned_slot]=$!; launched=$((launched + 1)); fi
            done
        done
    done
    echo "All runs launched ($launched launched, $skipped skipped)."
    [[ "$DRY_RUN" == "true" ]] || wait
}

run_ablation
