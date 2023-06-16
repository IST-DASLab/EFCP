#!/bin/bash

# width = {128, 512, 2048}
# batch_size = {32, 128, 512, 2048}
# lr = {3e-1, 1e-1, 3e-2, 1e-2, 3e-3, 1e-3}
# num_epochs = 20

export WANDB_ENTITY=eldarkurtic

export WANDB_PROJECT=mlp_mnist
export WANDB_NAME=mlp_sparse_mfac
#python examples/arxiv_results/train.py --config examples/arxiv_results/configs/mlp/sparse_mfac.json

python examples/arxiv_results/train.py --dataset MNIST --model mlp --width 2048 --depth 3 --epochs 20 --batch_size 512 --train_size 45056 --weight_decay 1e-5 --clipping_norm 10 --lr 0.01 --momentum 0.9 --ema_decay -1

