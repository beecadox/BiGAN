import sys
import os
import os.path as path
import shutil
import json
import argparse


def configuration(args, phase):
    config = {
        "training": phase == "train",
        "proj_dir": args.proj_dir,
        "model": args.model,
        "gpus": args.gpu_ids,
        "category": args.class_name,
        "log_dir": os.path.join(args.proj_dir, args.model, args.class_name, "log"),
        "model_dir": os.path.join(args.proj_dir, args.model, args.class_name, "models"),
        "data_root_path": args.data_directory,
        "workers": 8,
        "points": 2048,
        "pretrain_vae_path": "proj_log/vae/" + args.class_name + "/models/checkpoint_epoch5300.pth",
        "lr_decay": 0.9995,
        "continue_training": False,
        "checkpoint": 'latest',
        "visualization": True,
        "save_frequency": 100,
        "validation_frequency": 1,
        "visualization_frequency": 1,
        "kl_vae_weights": 10.0,
        "z_L1_weights": 7.5,
        "partial_rec_weights": 6,
        "pc_augm_scale": 0,
        "pc_augm_rot": 1,
        "pc_augm_mirror_prob": 0.5,
        "pc_augm_jitter": 0
    }
    if phase == "test":
        config["num_samples"] = 10,
        config["outpus"] = 2
    if phase == "train":
        config["epochs"] = args.epochs
        config["lr"] = args.lr
        config["batch_size"] = args.batch

    if config["gpus"] is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = config['gpus']

    if config["training"]:
        with open(os.path.join(config["proj_dir"], config["model"], config["category"], 'config.txt'), 'w') as f:
            json.dump(config, f, indent=2)

    return config
