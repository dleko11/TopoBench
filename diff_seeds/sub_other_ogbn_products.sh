!/bin/bash

if [ -z "${1}" ]; then
echo "Must set number of CUDA device. Use nvidia-smi to get numbers of available devices!"
exit -1
fi

export CUDA_VISIBLE_DEVICES="$1"
echo $CUDA_VISIBLE_DEVICES
num_threads=1
export OMP_NUM_THREADS=$num_threads
export MKL_NUM_THREADS=$num_threads

#PYPATH=/home/lukab/miniconda3/envs/mace_env3/bin/

python3 -m topobench \
    --multirun \
    model=hypergraph/edgnn,hypergraph/unignn2,cell/cwn,cell/cccn,simplicial/scn \
    dataset=graph/ogbn_products_for_partitioning \
    optimizer.parameters.lr=0.001 \
    optimizer.parameters.weight_decay=0 \
    model.feature_encoder.out_channels=128 \
    model.feature_encoder.proj_dropout=0 \
    dataset.dataloader_params.batch_size=1 \
    dataset.split_params.data_seed=100,200,300,400 \
    trainer.max_epochs=5000 \
    trainer.min_epochs=50 \
    trainer.check_val_every_n_epoch=1 \
    callbacks.early_stopping.patience=130 \
    logger=tensorboard \
    trainer=gpu \
