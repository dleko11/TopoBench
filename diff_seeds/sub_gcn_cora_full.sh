#!/bin/bash

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
    model=graph/gcn_cluster \
    dataset=graph/cocitation_cora_full_for_partitioning dataset.loader.parameters.memory_type=on_disk_cluster \
    dataset.split_params.data_seed=100,200,300,400 trainer.devices=1 \
    trainer.max_epochs=1000 trainer.min_epochs=50 trainer.check_val_every_n_epoch=2 \
    callbacks.early_stopping.patience=50 logger=tensorboard \
    trainer=gpu

