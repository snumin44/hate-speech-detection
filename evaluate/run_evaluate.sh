#!/bin/sh

GPU_ID=0
CUDA_VISIBLE_DEVICES=$GPU_ID python3 evaluate.py \
        --test_path '../data/custom_dataset.csv' \
        --model_path '../pretrained_model'\
        --vocab_path '../pretrained_model' \
        --embed_dim 100 \
        --num_kernels 100 \
        --kernel_sizes 3 4 5 \
        --stride 1 \
        --gru_hidden_dim 100 \