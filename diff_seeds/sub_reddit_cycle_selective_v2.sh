#!/bin/bash

if [ -z "${1}" ]; then
echo "Must set number of CUDA device. Use nvidia-smi to get numbers of available devices!"
exit -1
fi

export CUDA_VISIBLE_DEVICES="$1"
echo $CUDA_VISIBLE_DEVICES

# CRITICAL: Keep this at 1 to prevent thread contention when PyTorch workers spawn!
num_threads=1
export OMP_NUM_THREADS=$num_threads
export MKL_NUM_THREADS=$num_threads

# Adjust number of workers here!!!
NUM_WORKERS=8

python -m topobench \
    --multirun \
    model=cell/cwn,cell/cccn \
    dataset=graph/reddit_for_partitioning \
    optimizer.parameters.lr=0.001 \
    optimizer.parameters.weight_decay=0 \
    model.feature_encoder.out_channels=128 \
    model.feature_encoder.proj_dropout=0 \
    dataset.loader.parameters.stream.num_workers=$NUM_WORKERS \
    dataset.dataloader_params.num_workers=$NUM_WORKERS \
    dataset.split_params.data_seed=100,200,300,400 \
    trainer.max_epochs=300 \
    trainer.min_epochs=50 \
    trainer.check_val_every_n_epoch=5 \
    callbacks.early_stopping.patience=5 \
    logger=wandb \
    logger.wandb.project=reddit \
    trainer=gpu \
    +trainer.enable_progress_bar=false
