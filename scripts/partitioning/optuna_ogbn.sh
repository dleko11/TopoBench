#!/bin/bash
set -euo pipefail

# MODEL_FILTER=edgnn DRY_RUN=true uv run --no-sync bash scripts/partitioning/optuna_ogbn.sh
# One model per invocation. N_TRIALS adds trials to an existing study.
MODEL_FILTER="${MODEL_FILTER:-gcn}"
GPU="${GPU:-0}"
SEED="${SEED:-0}"
OPTUNA_N_JOBS="${OPTUNA_N_JOBS:-1}"
MAX_EPOCHS="${MAX_EPOCHS:-300}"
EARLY_STOPPING_PATIENCE="${EARLY_STOPPING_PATIENCE:-10}"
DRY_RUN="${DRY_RUN:-false}"
search_config=(hparams_search=partitioning_optuna)

case "$MODEL_FILTER" in
    gcn)
        model_config="graph/gcn"
        transform_kind="graph"
        lr_space="choice(1e-3,3e-3,1e-2)"
        model_params=('model.backbone.num_layers=choice(2,3,4)')
        ;;
    edgnn)
        model_config="hypergraph/edgnn"
        transform_kind="hypergraph"
        lr_space="choice(3e-4,1e-3,3e-3)"
        model_params=(
            'model.backbone.All_num_layers=choice(1,2,3)'
            'model.backbone.MLP_num_layers=choice(1,2)'
        )
        ;;
    sccnn)
        model_config="simplicial/sccnn_custom"
        transform_kind="simplicial"
        lr_space="choice(1e-4,3e-4,1e-3)"
        model_params=()
        search_config=(
            hydra/sweeper=optuna hydra/sweeper/sampler=grid
            hydra.sweeper.direction=maximize
            +optimized_metric=best_monitored_score
        )
        ;;
    *)
        echo "ERROR: MODEL_FILTER must be one of gcn, edgnn, sccnn." >&2
        exit 1
        ;;
esac

if [[ "$MODEL_FILTER" == "sccnn" ]]; then
    N_TRIALS="${N_TRIALS:-6}"
    NUM_PARTS="${NUM_PARTS:-15000}"
    Q="${Q:-20}"
    width_space="choice(32,64)"
    proj_dropout=0.0
else
    N_TRIALS="${N_TRIALS:-48}"
    NUM_PARTS="${NUM_PARTS:-5000}"
    Q="${Q:-25}"
    width_space="choice(128,256)"
    proj_dropout=0.1
    model_params+=(
        'model.backbone.dropout=choice(0.0,0.2,0.5)'
        'optimizer.parameters.weight_decay=choice(0,1e-4,1e-3)'
    )
fi

for value in "$N_TRIALS" "$OPTUNA_N_JOBS" "$NUM_PARTS" "$Q" "$MAX_EPOCHS" "$EARLY_STOPPING_PATIENCE"; do
    [[ "$value" =~ ^[1-9][0-9]*$ ]] || { echo "ERROR: counts must be positive integers." >&2; exit 1; }
done
for value in "$GPU" "$SEED"; do
    [[ "$value" =~ ^(0|[1-9][0-9]*)$ ]] || { echo "ERROR: GPU and SEED must be nonnegative integers." >&2; exit 1; }
done
(( Q <= NUM_PARTS )) || { echo "ERROR: Q must not exceed NUM_PARTS." >&2; exit 1; }
[[ "$DRY_RUN" == "true" || "$DRY_RUN" == "false" ]] || { echo "ERROR: DRY_RUN must be true or false." >&2; exit 1; }

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$SCRIPT_DIR/../.."
source "$SCRIPT_DIR/common.sh"
study_name="${STUDY_NAME:-ogbn_official_${MODEL_FILTER}_K${NUM_PARTS}_q${Q}_seed${SEED}_e${MAX_EPOCHS}_p${EARLY_STOPPING_PATIENCE}}"
log_group="$study_name"
storage="${OPTUNA_STORAGE:-sqlite:///$PWD/logs/optuna/${study_name}.db}"
cmd=(
    python -m topobench --multirun
    "${search_config[@]}" hydra/launcher=joblib
    dataset=graph/ogbn_products_for_partitioning "model=$model_config"
    trainer=gpu logger=wandb
    "hydra.sweeper.study_name=$study_name" "hydra.sweeper.storage=$storage"
    "hydra.sweeper.n_trials=$N_TRIALS" "hydra.sweeper.n_jobs=$OPTUNA_N_JOBS"
    "hydra.launcher.n_jobs=$OPTUNA_N_JOBS"
    "optimizer.parameters.lr=$lr_space" "model.feature_encoder.out_channels=$width_space"
    "${model_params[@]}"
    dataset.split_params.split_type=fixed "dataset.split_params.data_seed=$SEED" "seed=$SEED"
    "dataset.loader.parameters.cluster.num_parts=$NUM_PARTS"
    "dataset.loader.parameters.stream.q=$Q"
    "++dataset.loader.parameters.stream.q_val=$Q" "++dataset.loader.parameters.stream.q_test=$Q"
    dataset.dataloader_params.batch_size=1 dataset.dataloader_params.num_workers=1
    dataset.loader.parameters.stream.num_workers=1 ++dataset.loader.parameters.stream.cache_num_workers=1
    ++dataset.loader.parameters.stream.cache_val=false ++dataset.loader.parameters.stream.val_shuffle=true
    ++dataset.loader.parameters.stream.cleanup_val_cache=false
    "model.feature_encoder.proj_dropout=$proj_dropout"
    "trainer.max_epochs=$MAX_EPOCHS" trainer.min_epochs=1 trainer.check_val_every_n_epoch=5
    "callbacks.early_stopping.patience=$EARLY_STOPPING_PATIENCE" "trainer.devices=[$GPU]"
    train=true test=false +trainer.enable_progress_bar=false extras.print_config=false extras.enforce_tags=false
    "logger.wandb.project=${WANDB_PROJECT_NAME:-ogbn_products_official_splits}"
    "logger.wandb.group=$study_name" "+logger.wandb.name=${study_name}_trial_\${hydra:job.num}"
    "+logger.wandb.entity=${wandb_entity:-topobench-scalability}"
)
append_transform_args "$transform_kind"
if [[ "$MODEL_FILTER" == "sccnn" ]]; then
    cmd+=(
        transforms.graph2simplicial_lifting.complex_dim=2 model.backbone.n_layers=1
        optimizer.parameters.weight_decay=0.0001
    )
fi

printf -v cmd_string '%q ' "${cmd[@]}"
if [[ "$DRY_RUN" == "true" ]]; then
    printf '[DRY_RUN] %s\n' "$cmd_string"
    exit 0
fi

RESUME=true
export MAX_ATTEMPTS=1 KEEP_SUCCESS_LOGS=true MPLBACKEND=Agg PYTHONUNBUFFERED=1
init_experiment_environment
mkdir -p logs/optuna
echo "Study: $study_name; trials: $N_TRIALS; parallel trials: $OPTUNA_N_JOBS; GPU: $GPU"
run_and_log "$cmd_string" "$log_group" "$study_name" "$LOG_DIR" || exit "$?"
