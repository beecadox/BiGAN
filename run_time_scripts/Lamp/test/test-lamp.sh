#!/usr/bin/env bash
python test.py --proj_dir proj_log \
                --exp_name shapenet-lamp \
                --module gan \
                --dataset_name shapenet \
                --category Lamp \
                --data_root /data/shapenet/train \
                --pretrain_ae_path proj_log/shapenet-lamp/ae/model/ckpt_epoch1000.pth \
                --pretrain_vae_path proj_log/shapenet-lamp/vae/model/ckpt_epoch1000.pth \
                --num_sample -1 \
                --num_z 10 \
                --ckpt 500 \
                -g 0