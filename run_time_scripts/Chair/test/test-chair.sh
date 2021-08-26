#!/usr/bin/env bash
python test.py --proj_dir proj_log \
                --model gan \
                --gpu_ids 0 \
                --class_name chair \
                --pretrained_vae proj_log/vae \
                --data_directory data/shapenet