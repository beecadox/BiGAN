#!/usr/bin/env bash
python test.py --proj_dir proj_log \
                --exp_name shapenet-car \
                --module gan \
                --dataset_name shapenet \
                --category Car \
                --data_root /data/shapenet/train \
                --pretrain_ae_path proj_log/shapenet-car/ae/model/ckpt_epoch1000.pth \
                --pretrain_vae_path proj_log/shapenet-car/vae/model/ckpt_epoch1000.pth \
                --num_sample -1 \
                --num_z 10 \
                --ckpt 500 \
                -g 0