import argparse
from config import configuration
from nn_architecture.agents import GAN, VAE
from data_processing.shapenet import get_data
from utils.utils import cycle


def get_point_clouds(config):
    train_dataloader = get_data("train", config)
    val_dataloader = get_data("val", config)
    val_dataloader = cycle(val_dataloader)
    return train_dataloader, val_dataloader


def main(args):
    config = configuration(args, "train")
    if config['model'] == 'gan':
        print("Loading GAN...")
        training_agent = GAN(config)
    elif config['model'] == 'vae':
        print("Loading VAE...")
        training_agent = VAE(config)
    else:
        print("Wrong model parameter...Loading default model..")
        training_agent = GAN(config)
    if config['continue_training']:
        training_agent.load_checkpoints(config['checkpoint'])
    train_dataset, val_dataset = get_point_clouds(config)
    clock = training_agent.training_clock
    for e in range(clock.epoch, config['epochs']):
        for b, data in enumerate(train_dataset):
            training_agent.train(data)
            if config.vis and clock.step % config.vis_frequency == 0:
                training_agent.visualize_batch(data, "train")
                losses = training_agent.collect_loss()
                with open("results/train_loses.txt", 'a+', newline='') as write_obj:
                    write_obj.write(losses['D_GAN'] + '\t' + losses['G_GAN'] + '\t' + losses['z_L1'] + '\t' +
                                    losses['partial_rec'] + '\t' + losses['emd_loss'])
                # validation step
                if clock.step % config['val_frequency'] == 0:
                    data = next(val_dataset)
                    training_agent.validation(data)

                    if config.vis and clock.step % config.vis_frequency == 0:
                        training_agent.visualize_batch(data, "validation")

                clock.tick()

            clock.tock()

            if clock.epoch % config.save_frequency == 0:
                training_agent.save_checkpoints()
            training_agent.save_checkpoints('latest')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Give model configuration arguments.')
    parser.add_argument('--proj_dir', type=str, default='proj_log', help='project directory')
    parser.add_argument('--model', type=str, choices=['vae', 'gan'],  default='gan', help='gan or vae.')
    parser.add_argument('-g', '--gpu_ids', type=str, default=None, help='positive integer')
    parser.add_argument('--class_name', type=str, default='chair', help='airplane, car, chair, lamp or table.')
    parser.add_argument('--pretrained_vae', type=str, default='', help='vae checkpoint directory.')
    parser.add_argument('--data_directory', type=str, default='data/shapenet', help='shapenet dataset directory.')
    parser.add_argument('--batch', type=int, default=50, help='positive integer')
    parser.add_argument('--lr', type=float, default=0.0005, help='learning rate ex. 0.001, 0.0005')
    parser.add_argument('--epochs', type=int, default=1000, help='positive integer')
    args = parser.parse_args()
    main(args)