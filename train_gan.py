import argparse
from config import configuration
from nn_architecture.agents import GAN, VAE
from data_processing.shapenet import ShapenetDataProcess


def get_point_clouds(config):
    from multiprocessing import Queue
    parser = argparse.ArgumentParser(description='')
    args = parser.parse_args()
    args.dataset = 'shapenet'
    args.nworkers = 1
    args.batch_size = 1
    args.pc_augm_scale = 0
    args.pc_augm_rot = 0
    args.pc_augm_mirror_prob = 0
    args.pc_augm_jitter = 0
    args.inpts = 2048
    data_processes = []
    data_queue = Queue(1)
    for i in range(args.nworkers):
        data_processes.append(ShapenetDataProcess(data_queue, args, split='test',
                                                  repeat=False))
        data_processes[-1].start()
    return "train", "val"


def main(args):
    config = configuration(args, "train")
    gan_train_agent = GAN(config)

    if config['continue_training']:
        gan_train_agent.load_checkpoints(config['checkpoint'])
    train_dataset, val_dataset = get_point_clouds(config)
    clock = gan_train_agent.training_clock
    for e in range(clock.epoch, config['epochs']):
        for b, data in enumerate(train_dataset):
            gan_train_agent.train(data)
            if config.vis and clock.step % config.vis_frequency == 0:
                gan_train_agent.visualize_batch(data, "train")
                losses = gan_train_agent.collect_loss()

                # validation step
                if clock.step % config['val_frequency'] == 0:
                    data = next(val_dataset)
                    gan_train_agent.validation(data)

                    if config.vis and clock.step % config.vis_frequency == 0:
                        gan_train_agent.visualize_batch(data, "validation")

                clock.tick()

            clock.tock()

            if clock.epoch % config.save_frequency == 0:
                gan_train_agent.save_checkpoints()
            gan_train_agent.save_checkpoints('latest')


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
