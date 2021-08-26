from config import configuration
import argparse
import random
from utils.utils import cycle, write_ply
import torch
import os
from nn_architecture.agents import GAN
from data_processing.shapenet import get_data
from data_processing.visualization import plot_pcds
random.seed(7256)


def main(args):
    config = configuration(args, 'test')
    gan_test_agent = GAN(config)
    gan_test_agent.load_checkpoints(config['checkpoint'])
    gan_test_agent.eval()
    config['batch_size'] = 1
    config['workers'] = 1
    test_dataset = get_data("test", config)
    num_test = len(test_dataset)
    print("total number of test samples: {}".format(num_test))
    samples_to_use = num_test if config["num_samples"] == -1 else config["num_samples"]
    print("used number of test samples: {}".format(samples_to_use ))
    test_dataset = cycle(test_dataset)

    for i in range(samples_to_use):
        data = next(test_dataset)
        for j in range(config["outputs"]):
            with torch.no_grad():
                gan_test_agent.forward(data)
            real_pts, fake_pts, raw_pts = gan_test_agent.get_point_cloud()

            raw_id = data['shape_id'][0].split('.')[0]

            plot_pcds(filename='', pcds=[real_pts[0].transpose(1, 0), fake_pts[0].transpose(1, 0), raw_pts[0].transpose(1, 0)],
                      titles=['gt', 'partial', 'completed'], suptitle=raw_id + "_" + str(j), use_color=[0, 0, 0], color=[None, None, None])

            save_sample_dir = os.path.join("results", config["category"])

            # save completed shape
            save_path = os.path.join(save_sample_dir, "fake-z{}.ply".format(j))
            write_ply(fake_pts[0], save_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Give model configuration arguments.')
    parser.add_argument('--proj_dir', type=str, default='proj_log', help='project directory')
    parser.add_argument('--model', type=str, choices=['vae', 'gan'], default='gan', help='gan or vae.')
    parser.add_argument('-g', '--gpu_ids', type=str, default=None, help='positive integer')
    parser.add_argument('--class_name', type=str, default='chair', help='airplane, car, chair, lamp or table.')
    parser.add_argument('--pretrained_vae', type=str, default='', help='vae checkpoint directory.')
    parser.add_argument('--data_directory', type=str, default='data/shapenet', help='shapenet dataset directory.')
    args = parser.parse_args()
    main(args)
