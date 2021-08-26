#!/usr/bin/env bash
python train.py --proj_dir proj_log \
                --model vae \
                --gpu_ids 0 \
                --class_name chair \
                --data_directory data/shapenet \
                --batch 200 \
                --lr 0.0005 \
                --epochs 2000
