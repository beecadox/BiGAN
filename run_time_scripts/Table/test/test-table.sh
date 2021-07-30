#!/usr/bin/env bash
python test.py --proj_dir proj_log \
                --exp_name shapenet-table \
                --module gan \
                --dataset_name shapenet \
                --category Table \
                --data_root /data/shapenet/train \
                --pretrain_ae_path proj_log/shapenet-table/ae/model/ckpt_epoch1000.pth \
                --pretrain_vae_path proj_log/shapenet-table/vae/model/ckpt_epoch1000.pth \
                --num_sample -1 \
                --num_z 10 \
                --ckpt 500 \
                -g 0