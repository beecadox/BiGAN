#!/usr/bin/env bash
python test.py --proj_dir proj_log \
                --exp_name shapenet-chair \
                --module gan \
                --dataset_name shapenet \
                --category Chair \
                --data_root /data/shapenet/train \
                --pretrain_ae_path proj_log/shapenet-chair/ae/model/ckpt_epoch1000.pth \
                --pretrain_vae_path proj_log/shapenet-chair/vae/model/ckpt_epoch1000.pth \
                --num_sample -1 \
                --num_z 10 \
                --ckpt 500 \
                -g 0