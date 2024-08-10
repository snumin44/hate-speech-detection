#!/bin/sh

GPU_ID=0
CUDA_VISIBLE_DEVICES=$GPU_ID python3 train.py \
    	--train_path '../data/train_dataset.csv' \
        --valid_path '../data/valid_dataset.csv' \
        --test_path '../data/test_dataset.csv' \
        --output_path '../pretrained_model'\
        --vocab_path '../pretrained_model' \
        --embed_dim 100 \
        --num_kernels 100 \
        --kernel_sizes 3 4 5 \
        --stride 1 \
        --gru_hidden_dim 100 \
    	--epochs 30 \
        --dropout 0.2 \
        --batch_size 128 \
        --learning_rate 1e-3 \