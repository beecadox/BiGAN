import argparse
from config import configuration
from nn_architecture.agents import GAN, VAE


def main(args):
    config = configuration(args, "train")
    gan_train_agent = GAN(config)

    if config['continue_training']:
        gan_train_agent.load_ckpt(config['checkpoint'])

    clock = gan_train_agent.training_clock
    for e in range(clock.epoch, config['epochs']):
        for b, data in enumerate(training_data):
            gan_train_agent.train_model(data)


    print("HEHE")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Give model configuration arguments.')
    parser.add_argument('--proj_dir', type=str, default='proj_log', help='project directory')
    parser.add_argument('--model', type=str, choices=['ae', 'vae', 'gan'], required=True, help='gan or vae.')
    parser.add_argument('-g', '--gpu_ids', type=str, default=None, help='positive integer')
    parser.add_argument('--class_name', type=str, default='airplane', help='airplane, car, chair, lamp or table.')
    parser.add_argument('--pretrained_vae', type=str, default='', help='vae checkpoint directory.')
    parser.add_argument('--data_directory', type=str, default='data/shapenet', help='shapenet dataset directory.')
    parser.add_argument('--batch', type=int, default=50, help='positive integer')
    parser.add_argument('--lr', type=float, default=0.0005, help='learning rate ex. 0.001, 0.0005')
    parser.add_argument('--epochs', type=int, default=1000, help='positive integer')
    args = parser.parse_args()
    main(args)
