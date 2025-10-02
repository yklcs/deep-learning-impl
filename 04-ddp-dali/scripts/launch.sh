#!/bin/bash

export DATA_DIR
export CKPT_DIR
export NSIGHT_LOG_DIR
export NSIGHT_FILE_NAME
export MODE
export LOCAL_GPU_IDS
export NUM_GPUS

###########################################################################
# Problem 0: Generate Nsight log
# Find the correct way to generate Nsight log.
###########################################################################

# Scaffold
CUDA_VISIBLE_DEVICES=$LOCAL_GPU_IDS \
nsys profile -o "$NSIGHT_LOG_DIR/$NSIGHT_FILE_NAME" \
    --trace=cuda,nvtx,cublas,cudnn,osrt \
    --pytorch=autograd-nvtx \
    --gpu-metrics-devices=$LOCAL_GPU_IDS \
python train_cifar.py \
    --num_gpu=$NUM_GPUS \
    --data="$DATA_DIR" \
    --ckpt="$CKPT_DIR" \
    --mode="$MODE" \
    --save_ckpt
