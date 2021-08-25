from torch.utils.data import Dataset, DataLoader
import os
import trimesh
from data_processing.data_utils import augment_cloud, load_h5, pad_cloudN
import torch
import random

category_to_id = {
    'airplane': '02691156',
    'car': '02958343',
    'chair': '03001627',
    'table': '04379243',
    'lamp': '03636649'
}


def get_data(phase, config):
    augm_args = {x: config[x] for x in ['pc_augm_scale', 'pc_augm_rot', 'pc_augm_mirror_prob', 'pc_augm_jitter']}
    if config['model'] in ['gan', 'vae']:
        dataloader = Shapenet(config['model'], phase, config['data_complete_', phase], config['category'],
                              config['points'], augm_args)
    else:
        raise ValueError
    return dataloader


class Shapenet(Dataset):
    def __init__(self, model, phase, data_root, category, n_pts, augment_config):
        super(Shapenet, self).__init__()
        self.model = model
        self.phase = phase
        self.augment = phase == "train"
        self.augment_config = augment_config
        self.category_id = category_to_id[category]

        if self.model == "gan":
            self.gt_data_paths = sorted([os.path.join(data_root, self.phase, 'gt', path.rstrip() + '.h5') for path in
                                         open(data_root + '/%s.list' % self.phase).readlines() if
                                         path.split("/")[0] == self.category_id])
            self.partial_data_paths = list(map(lambda x: x.replace("gt", "partial"),
                                               self.gt_data_paths))  # replace all 'ground truth' paths with the equivalent 'partial' paths
        self.category_shapes = list(
            map(lambda x: x.split("/")[-1], self.gt_data_paths))  # get only the ids for each path

        if self.model == "vae":
            self.gt_data_paths = sorted([os.path.join(data_root, 'vae', path.rstrip() + '.ply') for path in
                                         open(data_root + '/%s.list' % self.phase).readlines() if
                                         path.split("/")[0] == self.category_id])
        self.n_pts = n_pts

    def __getitem__(self, index):
        pc_shape_name = self.category_shapes[index]

        if self.model == "gan":
            gtpts = load_h5(self.gt_data_paths[index])
            partial = load_h5(self.partial_data_paths[index])
            if self.augment:
                gtpts, partial = augment_cloud([gtpts, partial], self.augment_config)
            partial = pad_cloudN(partial, self.n_pts)

            return {"gt": torch.tensor(gtpts, dtype=torch.float32).transpose(1, 0), "partial": torch.tensor(partial, dtype=torch.float32).transpose(1, 0),  "shape_id": pc_shape_name}
        else:
            pc = trimesh.load(self.gt_data_paths[index])
            p_idx = list(range(pc.shape[0]))
            random.shuffle(p_idx)
            p_idx = p_idx[:self.n_pts]
            # p_idx = random.choices(list(range(pc.shape[0])), k=self.n_pts)

            pc = pc[p_idx]
            pc = torch.tensor(pc, dtype=torch.float32).transpose(1, 0)
            return {"points": pc, "shape_id": pc_shape_name}

    def __len__(self):
        return len(self.category_shapes)
