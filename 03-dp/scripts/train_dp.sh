#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python train_cifar.py --data dataset/cifar10 --ckpt runs/default/${TIMESTAMP}
CUDA_VISIBLE_DEVICES=0,1 python train_cifar.py --dp --data dataset/cifar10 --ckpt runs/dp_2/${TIMESTAMP}