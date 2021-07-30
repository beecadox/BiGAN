#!/usr/bin/env bash
python train.py --proj_dir proj_log \
                --exp_name shapenet-table \
                --module bigan \
                --dataset_name shapenet \
                --category Table \
                --data_root /data/shapenet/train \
                --batch_size 200 \
                --lr 5e-4 \
                --lr_decay 0.999 \
                --save_frequency 500 \
                --nr_epochs 1000 \
                --num_workers 16 \
                -g 0