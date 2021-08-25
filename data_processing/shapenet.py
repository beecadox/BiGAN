import argparse
import os
import numpy as np

from data_processing.process import DataProcess, get_while_running, kill_data_processes
from data_processing.data_utils import load_h5, load_csv, augment_cloud, pad_cloudN
from data_processing.visualization import plot_pcds


class ShapenetDataProcess(DataProcess):

    def __init__(self, data_queue, args, split='test', repeat=True):
        """Shapenet dataloader.
        Args:
            data_queue: multiprocessing queue where data is stored at.
            split: str in ('train', 'val', 'test'). Loads corresponding dataset.
            repeat: repeats epoch if true. Terminates after one epoch otherwise.
        """
        self.args = args
        self.split = split
        args.DATA_PATH = '../data/%s' % (args.dataset)
        classmap = load_csv(args.DATA_PATH + '/synsetoffset2category.txt')
        args.classmap = {}
        for i in range(classmap.shape[0]):
            args.classmap[str(classmap[i][1]).zfill(8)] = classmap[i][0]

        self.data_paths = sorted([os.path.join(args.DATA_PATH, split, 'partial', k.rstrip() + '.h5') for k in
                                  open(args.DATA_PATH + '/%s.list' % (split)).readlines()])
        N = int(len(self.data_paths) / args.batch_size) * args.batch_size
        self.data_paths = self.data_paths[0:N]
        super().__init__(data_queue, self.data_paths, None, args.batch_size, repeat=repeat)

    def get_pair(self, args, fname, train):
        partial = load_h5(fname)
        gtpts = load_h5(fname.replace('partial', 'gt'))
        if train:
            gtpts, partial = augment_cloud([gtpts, partial], args)
        partial = pad_cloudN(partial, args.inpts)
        return {'partial': partial, 'gt': gtpts}

    def load_data(self, fname):
        pair = self.get_pair(self.args, fname, train=self.split == 'train')
        partial = pair['partial'].T
        target = pair['gt']
        cloud_meta = ['{}.{:d}'.format('/'.join(fname.split('/')[-2:]), 0), ]

        return target[np.newaxis, ...], cloud_meta, partial[np.newaxis, ...]

    def collate(self, batch):
        targets, clouds_meta, clouds = list(zip(*batch))
        targets = np.concatenate(targets, 0)
        if len(clouds_meta[0]) > 0:
            clouds = np.concatenate(clouds, 0)
            clouds_meta = [item for sublist in clouds_meta for item in sublist]
        return targets, (clouds_meta, clouds)


if __name__ == '__main__':
    pass
