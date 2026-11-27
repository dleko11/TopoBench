#!/usr/bin/env bash
# Cluster-TNN SCN on Coauthor Physics with the paper's final settings
# (scripts/partitioning/final_partitioning.sh), standard vs encoder-first.
#
# Seed 0 of each variant runs alone for timing; seeds 1-4 run concurrently
# for metrics only. Logs carry timestamps for the phase breakdown.
#
#   PY=/path/to/python bash scripts/distributed/run_cluster_tnn_physics.sh
set -u
cd "$(dirname "$0")/../.."
PY="${PY:-python}"
OUT="${OUT:-logs/distributed/cluster_tnn_physics}"
DATA_DIR="${DATA_DIR:-}"
mkdir -p "$OUT"
export PROJECT_ROOT="$PWD" PYTHONPATH="$PWD"

common=(
    dataset=graph/coauthor_physics_for_partitioning model=simplicial/scn
    trainer=gpu "trainer.devices=[0]" logger=csv
    optimizer.parameters.lr=0.01 optimizer.parameters.weight_decay=0
    model.feature_encoder.out_channels=32 model.feature_encoder.proj_dropout=0.3
    dataset.dataloader_params.batch_size=1 dataset.dataloader_params.num_workers=1
    dataset.split_params.split_type=random dataset.split_params.learning_setting=transductive
    dataset.loader.parameters.stream.q=20 ++dataset.loader.parameters.stream.q_val=20
    ++dataset.loader.parameters.stream.q_test=20 dataset.loader.parameters.cluster.num_parts=2000
    dataset.loader.parameters.stream.num_workers=1 ++dataset.loader.parameters.stream.cache_num_workers=1
    ++dataset.loader.parameters.stream.cache_val=true ++dataset.loader.parameters.stream.val_shuffle=false
    ++dataset.loader.parameters.stream.cleanup_val_cache=true
    trainer.max_epochs=300 trainer.min_epochs=1 trainer.check_val_every_n_epoch=5
    "test_inference.protocols=[batched,ensemble]" test_inference.ensemble_runs=10
    train=true test=true +trainer.enable_progress_bar=false
    extras.print_config=false extras.enforce_tags=false callbacks.early_stopping.patience=5
    transforms=liftings/graph2simplicial_default
    "transforms/liftings/graph2simplicial@transforms.graph2simplicial_lifting=clique_selective"
    transforms.graph2simplicial_lifting.complex_dim=2
)
if [[ -n "$DATA_DIR" ]]; then
    common+=("paths.data_dir=$DATA_DIR")
fi

run() {  # variant seed
    local variant="$1" seed="$2" extra=()
    if [[ "$variant" == "first" ]]; then
        extra=(+model.feature_encoder.encoder_first=true
               transforms.graph2simplicial_lifting.feature_lifting=null)
    fi
    local name="${variant}_seed${seed}"
    local start end
    start=$(date +%s.%N)
    "$PY" -m topobench "${common[@]}" "${extra[@]}" \
        dataset.split_params.data_seed="$seed" seed="$seed" \
        test_inference.ensemble_seed="$seed" \
        hydra.run.dir="$OUT/runs/$name" > "$OUT/$name.log" 2>&1
    local status=$?
    end=$(date +%s.%N)
    echo "$name status=$status start=$start end=$end" >> "$OUT/wall.txt"
}

run standard 0
run first 0
for variant in standard first; do
    for seed in 1 2 3 4; do run "$variant" "$seed" & done
    wait
done
touch "$OUT/DONE"
