#!/usr/bin/env bash
python train.py --proj_dir proj_log \
                --model vae \
                --gpu_ids 0 \
                --class_name chair \
                --pretrained_vae proj_log/vae \
                --data_directory data/shapenet \
                --batch 50 \
                --lr 0.0005 \
                --epochs 1000