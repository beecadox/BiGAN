import argparse


def main(args):
    print("HEHE")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Give model configuration arguments.')
    parser.add_argument('--proj_dir', type=str, default='proj_log', help='project directory')
    parser.add_argument('--model', type=str, default='vae', help='gan or vae.')
    parser.add_argument('--class', type=str, default='airplane', help='airplane, car, chair, lamp or table.')
    parser.add_argument('--pretrained_vae', type=str, default='', help='vae checkpoint directory.')
    parser.add_argument('--data_directory', type=str, default='data/shapenet', help='shapenet dataset directory.')
    parser.add_argument('--batch', type=int, default=200, help='positive integer')
    parser.add_argument('--lr', type=float, default=0.0005, help='learning rate ex. 0.001, 0.0005')
    parser.add_argument('--lr_decay', type=float, default=0.999, help='learning rate decay ex. 0.099, 0.0095')
    parser.add_argument('--epochs', type=int, default=2000, help='positive integer')
    parser.add_argument('--n_workers', type=int, default=10, help='positive integer')
    args = parser.parse_args()
    main(args)
