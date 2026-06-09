#!/bin/bash

find_logging_script() {
    local dir="$1"
    while [[ "$dir" != "/" ]]; do
        if [[ -f "$dir/base/logging.sh" ]]; then
            echo "$dir/base/logging.sh"
            return 0
        fi
        if [[ -f "$dir/scripts/base/logging.sh" ]]; then
            echo "$dir/scripts/base/logging.sh"
            return 0
        fi
        dir="$(dirname "$dir")"
    done
    return 1
}

init_experiment_environment() {
    export HYDRA_FULL_ERROR=1
    export OMP_NUM_THREADS=1
    export MKL_NUM_THREADS=1
    export OPENBLAS_NUM_THREADS=1
    export VECLIB_MAXIMUM_THREADS=1
    export NUMEXPR_NUM_THREADS=1

    LOGGING_PATH=$(find_logging_script "$SCRIPT_DIR")
    if [[ -z "$LOGGING_PATH" ]]; then
        echo "ERROR: could not locate scripts/base/logging.sh" >&2
        exit 1
    fi
    source "$LOGGING_PATH"

    LOG_DIR="./logs/${log_group}"
    echo "Preparing log directory: $LOG_DIR"
    if [[ "$RESUME" == "true" ]]; then
        mkdir -p "$LOG_DIR"
    else
        rm -rf "$LOG_DIR"
        mkdir -p "$LOG_DIR"
    fi
}

mode_grid_size() {
    local mode="$1"
    if is_partitioning_mode "$mode"; then
        echo $(( ${#Q_VALUES[@]} * ${#NUM_CLUSTERS[@]} ))
    else
        echo 1
    fi
}

count_total_runs() {
    local total=0
    local dataset_mode mode dataset_config mode_multiplier

    for dataset_mode in "${DATASET_MODES[@]}"; do
        IFS="::" read -r mode _ dataset_config <<< "$dataset_mode"
        mode_multiplier=$(mode_grid_size "$mode")
        total=$(( total + mode_multiplier * ${#MODEL_SPECS[@]} * ${#DATA_SEEDS[@]} ))
    done

    echo "$total"
}

detect_gpu_slots() {
    export SELECTED_GPUS
    local gpu_info
    gpu_info=$(python3 -c "
import os
import subprocess

selected_env = os.environ.get('SELECTED_GPUS', '').strip()
allowed_gpus = [x.strip() for x in selected_env.split(',') if x.strip()] if selected_env else None

try:
    out = subprocess.check_output(
        ['nvidia-smi', '--query-gpu=index,memory.total', '--format=csv,noheader,nounits'],
        text=True,
    )
    indices = []
    mem_mb = []
    for line in out.strip().splitlines():
        idx, mem = line.split(',')
        idx = idx.strip()
        if allowed_gpus and idx not in allowed_gpus:
            continue
        indices.append(idx)
        mem_mb.append(int(mem.strip()))
    if not indices:
        print('0')
        raise SystemExit
    min_mem_gb = min(mem_mb) / 1024
    if min_mem_gb >= 80:
        jobs = 5
    elif min_mem_gb <= 10:
        jobs = 1
    elif min_mem_gb <= 30:
        jobs = 2
    else:
        jobs = 3
    print(jobs, ' '.join(indices))
except Exception:
    fallback = allowed_gpus[0] if allowed_gpus else '0'
    print(1, fallback)
")

    read -r JOBS_PER_GPU gpu_ids <<< "$gpu_info"
    read -ra physical_gpus <<< "$gpu_ids"

    if [[ "${#physical_gpus[@]}" -eq 0 || "$JOBS_PER_GPU" -eq 0 ]]; then
        echo "ERROR: no GPUs available from SELECTED_GPUS=$SELECTED_GPUS" >&2
        exit 1
    fi

    gpus=()
    for gpu in "${physical_gpus[@]}"; do
        for ((i = 1; i <= JOBS_PER_GPU; i++)); do
            gpus+=("$gpu")
        done
    done

    echo "Detected GPU(s): ${physical_gpus[*]}"
    echo "Jobs per GPU: $JOBS_PER_GPU"
    echo "Total virtual slots: ${#gpus[@]}"

    slot_pids=()
    for i in "${!gpus[@]}"; do
        slot_pids[$i]=0
    done
}

transform_alias() {
    case "$1" in
        graph) echo "graph" ;;
        hypergraph) echo "khop_large_scale" ;;
        cell) echo "cycle_selective" ;;
        simplicial) echo "clique_selective" ;;
        *)
            echo "ERROR: unknown transform kind: $1" >&2
            exit 1
            ;;
    esac
}

append_transform_args() {
    local transform_kind="$1"
    case "$transform_kind" in
        graph)
            ;;
        hypergraph)
            cmd+=(
                "transforms=liftings/graph2hypergraph_default"
                "transforms/liftings/graph2hypergraph@transforms.graph2hypergraph_lifting=khop_large_scale"
            )
            ;;
        cell)
            cmd+=(
                "transforms=liftings/graph2cell_default"
                "transforms/liftings/graph2cell@transforms.graph2cell_lifting=cycle_selective"
            )
            ;;
        simplicial)
            cmd+=(
                "transforms=liftings/graph2simplicial_default"
                "transforms/liftings/graph2simplicial@transforms.graph2simplicial_lifting=clique_selective"
            )
            ;;
        *)
            echo "ERROR: unknown transform kind: $transform_kind" >&2
            exit 1
            ;;
    esac
}

is_partitioning_mode() {
    [[ "$1" == "partitioning" ]]
}

run_dataset_suite() {
    init_experiment_environment
    detect_gpu_slots

    local success_log="$LOG_DIR/$log_group/SUCCESSFUL_RUNS.log"
    local total_runs
    total_runs=$(count_total_runs)
    local launched=0
    local skipped=0
    local run_counter=0
    local one_percent_step=$(( total_runs / 100 ))
    if [[ "$one_percent_step" -eq 0 ]]; then
        one_percent_step=1
    fi

    echo "Total runs planned: $total_runs"
    echo "Dry run: $DRY_RUN"
    echo "Partition q values: ${Q_VALUES[*]}"
    echo "Partition num_clusters values: ${NUM_CLUSTERS[*]}"

    for dataset_mode in "${DATASET_MODES[@]}"; do
        IFS="::" read -r mode _ dataset_config <<< "$dataset_mode"

        q_grid=("")
        cluster_grid=("")
        if is_partitioning_mode "$mode"; then
            q_grid=("${Q_VALUES[@]}")
            cluster_grid=("${NUM_CLUSTERS[@]}")
        fi

        for q_value in "${q_grid[@]}"; do
            for num_clusters in "${cluster_grid[@]}"; do
                local partition_suffix=""
                if is_partitioning_mode "$mode"; then
                    partition_suffix="_q${q_value}_clusters${num_clusters}"
                fi

                for model_spec in "${MODEL_SPECS[@]}"; do
                    IFS="::" read -r model_alias _ model_config _ transform_kind <<< "$model_spec"
                    local transform_name
                    transform_name=$(transform_alias "$transform_kind")

                    for data_seed in "${DATA_SEEDS[@]}"; do
                        local run_name="${mode}_${model_alias}_${transform_name}${partition_suffix}_seed${data_seed}_lr${LR}_wd${WEIGHT_DECAY}_h${OUT_CHANNELS}_pd${PROJ_DROPOUT}"

                        if [[ "$RESUME" == "true" && -f "$success_log" ]] && grep -Fq "[SUCCESS] ${run_name}" "$success_log"; then
                            ((skipped++))
                            continue
                        fi

                        ((run_counter++))
                        if (( run_counter % one_percent_step == 0 )); then
                            local percent=$(( (run_counter * 100) / total_runs ))
                            echo "Progress: ${percent}% ($run_counter / $total_runs considered)"
                        fi

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
                        local project_name="${script_name}_${mode}"

                        cmd=(
                            "python" "-m" "topobench"
                            "model=${model_config}"
                            "dataset=${dataset_config}"
                            "optimizer.parameters.lr=${LR}"
                            "optimizer.parameters.weight_decay=${WEIGHT_DECAY}"
                            "model.feature_encoder.out_channels=${OUT_CHANNELS}"
                            "model.feature_encoder.proj_dropout=${PROJ_DROPOUT}"
                            "dataset.dataloader_params.batch_size=1"
                            "dataset.split_params.data_seed=${data_seed}"
                            "trainer.max_epochs=${MAX_EPOCHS}"
                            "trainer.min_epochs=${MIN_EPOCHS}"
                            "trainer.check_val_every_n_epoch=${CHECK_VAL_EVERY_N_EPOCH}"
                            "callbacks.early_stopping.patience=${EARLY_STOPPING_PATIENCE}"
                            "logger=wandb"
                            "logger.wandb.project=${project_name}"
                            "+logger.wandb.name=${run_name}"
                            "trainer=gpu"
                            "trainer.devices=[${current_gpu}]"
                            "+trainer.enable_progress_bar=false"
                        )

                        if [[ -n "${wandb_entity:-}" ]]; then
                            cmd+=("+logger.wandb.entity=${wandb_entity}")
                        fi

                        append_transform_args "$transform_kind"

                        if is_partitioning_mode "$mode"; then
                            cmd+=(
                                "dataset.loader.parameters.stream.q=${q_value}"
                                "dataset.loader.parameters.cluster.num_parts=${num_clusters}"
                                "dataset.loader.parameters.stream.num_workers=${STREAM_NUM_WORKERS}"
                                "dataset.dataloader_params.num_workers=${STREAM_NUM_WORKERS}"
                            )
                        fi

                        if [[ "$DRY_RUN" == "true" ]]; then
                            printf '[DRY_RUN] %s\n' "${cmd[*]}"
                            slot_pids[$assigned_slot]=0
                        else
                            run_and_log "${cmd[*]}" "$log_group" "$run_name" "$LOG_DIR" &
                            slot_pids[$assigned_slot]=$!
                            ((launched++))
                        fi
                    done
                done
            done
        done
    done

    echo "All jobs launched ($launched launched, $skipped skipped)."
    if [[ "$DRY_RUN" != "true" ]]; then
        echo "Waiting for remaining jobs..."
        wait
        echo "All runs complete."
    fi
}
