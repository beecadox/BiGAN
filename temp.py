from data_processing.visualization import plot_pcds
from data_processing.data_utils import load_h5, augment_cloud, pad_cloudN
import os
import torch
import trimesh
import random

category_to_id = {
    'airplane': '02691156',
    'car': '02958343',
    'chair': '03001627',
    'table': '04379243',
    'lamp': '03636649'
}

augment_config = {'pc_augm_scale': 0,
                  'pc_augm_rot': 1,
                  'pc_augm_mirror_prob': 0.5,
                  'pc_augm_jitter': 0}

# gt_data_paths = sorted([os.path.join("data/shapenet", "test", 'gt', path.rstrip() + '.h5') for path in
#                         open("data/shapenet" + '/test.list').readlines() if
#                         path.split("/")[0] == '02691156'])
#
# partial_data_paths = list(map(lambda x: x.replace("gt", "partial"), gt_data_paths))



# index = 0
# pc_shape_name = category_shapes[index]
#
# gtpts = load_h5(gt_data_paths[index])
# partial = load_h5(partial_data_paths[index])
#
# gtpts, partial = augment_cloud([gtpts, partial], augment_config)
# partial = pad_cloudN(partial, 6000)
# gtpts = pad_cloudN(gtpts, 6000)
# data = {"gt": torch.tensor(gtpts, dtype=torch.float32).transpose(1, 0),
#         "partial": torch.tensor(partial, dtype=torch.float32).transpose(1, 0), "shape_id": pc_shape_name}

# plot_pcds(filename='', pcds=[gtpts, partial, gtpts],
#           titles=['gt', 'partial', 'gtp'], suptitle='02691156/' + data['shape_id'], use_color=[0, 0, 0], color=[None, None, None])

gt_data_paths = sorted([os.path.join("data/shapenet", 'vae', path.rstrip() + '.ply') for path in
                                         open("data/shapenet" + '/test.list').readlines() if
                                         path.split("/")[0] == '03001627'])
category_shapes = list(map(lambda x: x.split("/")[-1], gt_data_paths))
index = 100
pc = trimesh.load(gt_data_paths[index])
p_idx = list(range(pc.shape[0]))
random.shuffle(p_idx)
pc_shape_name = category_shapes[index]

p_idx1 = p_idx[:1024]
pc1 = pc[p_idx1]
pc1 = torch.tensor(pc1, dtype=torch.float32)

p_idx2 = p_idx[:2048]
pc2 = pc[p_idx2]
pc2 = torch.tensor(pc2, dtype=torch.float32)

p_idx3 = p_idx[:4096]
pc3 = pc[p_idx3]
pc3 = torch.tensor(pc3, dtype=torch.float32)

p_idx4 = p_idx[:512]
pc4 = pc[p_idx4]
pc4 = torch.tensor(pc4, dtype=torch.float32)

p_idx5 = p_idx[:10000]
pc5 = pc[p_idx5]
pc5 = torch.tensor(pc5, dtype=torch.float32)
# data_vae = {"points": pc, "shape_id": pc_shape_name}

plot_pcds(filename='', pcds=[pc4, pc1, pc2, pc3, pc5],
          titles=['512', '1024', '2048', '4096', '10000'], suptitle='03001627/' + pc_shape_name, use_color=[0, 0, 0, 0, 0], color=[None, None, None, None, None])