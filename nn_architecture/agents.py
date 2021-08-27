import os
import numpy as np

import torch

torch.cuda.empty_cache()
import torch.optim
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.optim import Adam, lr_scheduler

from utils.hausdorff import hausdorff
from utils.emd import earth_mover_distance
from nn_architecture.vae_architecture import VariationalAutoencoder
from nn_architecture.gan_architecture import Generator, Discriminator

torch.autograd.set_detect_anomaly(True)


def set_requires_grad(nets, requires_grad=False):
    """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
    Parameters:
        nets (network list)   -- a list of networks
        requires_grad (bool)  -- whether the networks require gradients or not
    """
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad


class TrainingClock(object):
    def __init__(self):
        self.epoch = 1
        self.minibatch = 0
        self.step = 0

    def tick(self):
        self.minibatch += 1
        self.step += 1

    def tock(self):
        self.epoch += 1
        self.minibatch = 0

    def make_checkpoint(self):
        return {
            'epoch': self.epoch,
            'minibatch': self.minibatch,
            'step': self.step
        }

    def restore_checkpoint(self, clock_dict):
        self.epoch = clock_dict['epoch']
        self.minibatch = clock_dict['minibatch']
        self.step = clock_dict['step']


# GAN Training Agent
class GAN(object):
    def __init__(self, config):
        super(GAN, self).__init__()
        self.log_dir = config['log_dir']
        self.model_dir = config['model_dir']
        self.training_clock = TrainingClock()
        self.batch_size = config['batch_size']
        self.z_dim = 64
        # load pretrained pointVAE
        vae = VariationalAutoencoder()
        try:
            vae_weights = torch.load(config["pretrain_vae_path"])['model_state_dict']
        except Exception as e:
            raise ValueError("Check the path for pretrained model of point VAE. \n{}".format(e))
        vae.load_state_dict(vae_weights)
        self.vaeE = vae.encoder.eval().cuda()
        self.vaeD = vae.decoder.eval().cuda()
        set_requires_grad(self.vaeE, False)
        # build G, D
        self.Generator = Generator().cuda()
        self.Discriminator = Discriminator().cuda()

        # set loss function
        self.criterionGAN = nn.MSELoss().cuda()
        self.criterionL1 = nn.L1Loss().cuda()

        # set optimizer
        self.Generator_optimizer = Adam(self.Generator.parameters(), lr=config["lr"])
        self.Discriminator_optimizer = Adam(self.Discriminator.parameters(), lr=config["lr"])
        self.Encoder_optimizer = Adam(self.vaeE.parameters(), lr=config["lr"])

        self.z_L1_weights = config["z_L1_weights"]
        self.partial_rec_weights = config["partial_rec_weights"]

        # set tensorboard writer
        self.train_writer = SummaryWriter(os.path.join(self.log_dir, 'train.events'))
        self.val_writer = SummaryWriter(os.path.join(self.log_dir, 'val.events'))

    def collect_loss(self):
        loss_dict = {"D_GAN": self.loss_D,
                     "G_GAN": self.loss_G_GAN,
                     "z_L1": self.loss_z_L1,
                     "partial_rec": self.loss_partial_rec,
                     "emd_loss": self.emd_loss}
        return loss_dict

    def forward(self, data):
        self.partial_pc = data['partial'].cuda()
        self.gt_pc = data['gt'].cuda()

        with torch.no_grad():
            self.raw_latent, = self.vaeE.encode(self.partial_pc)
            self.real_latent, = self.vaeE.encode(self.gt_pc)

        self.z_random = torch.randn((self.raw_latent.size(0), self.z_dim)).cuda()

        self.fake_latent = self.Generator(self.raw_latent, self.z_random)
        self.predicted_pc = self.vaeD.decode(self.fake_latent)
        self.z_rec, z_mean, z_logvar = self.vaeE(self.predicted_pc)

    def update_D(self):
        set_requires_grad(self.Discriminator, True)

        self.Discriminator_optimizer.zero_grad()
        # fake
        pred_fake = self.Discriminator(self.fake_latent.detach())
        fake = torch.zeros_like(pred_fake).fill_(0.0).cuda()
        self.loss_D_fake = self.criterionGAN(pred_fake, fake)

        # real
        pred_real = self.Discriminator(self.real_latent.detach())
        real = torch.ones_like(pred_real).fill_(1.0).cuda()
        self.loss_D_real = self.criterionGAN(pred_real, real)

        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        self.loss_D.backward()
        self.Discriminator_optimizer.step()

    def update_G_and_E(self):
        set_requires_grad(self.Discriminator, False)
        self.Generator_optimizer.zero_grad()
        # 1. G(A) fool D
        pred_fake = self.Discriminator(self.fake_latent)
        real = torch.ones_like(pred_fake).fill_(1.0).cuda()
        self.loss_G_GAN = self.criterionGAN(pred_fake, real)

        # 2. noise reconstruction |E(G(A, z)) - z_random|
        self.loss_z_L1 = self.criterionL1(self.z_rec, self.z_random) * self.z_L1_weights

        # 3. partial scan reconstruction

        self.loss_partial_rec = hausdorff(self.partial_pc, self.predicted_pc) * self.partial_rec_weights

        self.emd_loss = torch.mean(earth_mover_distance(self.predicted_pc, self.gt_pc))

        self.loss_EG = self.loss_G_GAN + self.loss_z_L1 + self.loss_partial_rec
        self.loss_EG.backward()
        self.Generator_optimizer.step()

    def get_point_cloud(self):
        """get real/fake/raw point cloud of current batch"""
        gt_pts = self.gt_pc.transpose(1, 2).detach().cpu().numpy()
        predicted_pts = self.predicted_pc.transpose(1, 2).detach().cpu().numpy()
        partial_pts = self.partial_pc.transpose(1, 2).detach().cpu().numpy()
        return gt_pts, predicted_pts, partial_pts

    def training(self, data, mode='train'):
        """one step of training"""
        self.forward(data)
        self.update_G_and_E()
        self.update_D()

        loss_dict = self.collect_loss()
        losses_values = {k: v.item() for k, v in loss_dict.items()}

        tb = self.train_writer if mode == 'train' else self.val_writer
        for k, v in losses_values.items():
            tb.add_scalar(k, v, self.training_clock.step)

    def validation(self, data):
        """one step of validation"""
        with torch.no_grad():
            self.forward(data)

    def eval(self):
        """set G, D, E to eval mode"""
        self.Generator.eval()
        self.Discriminator.eval()
        self.vaeE.eval()

    def save_checkpoints(self, name=None):
        """save checkpoint during training for future restore"""
        if name is None:
            save_path = os.path.join(self.model_dir, "checkpoint_epoch{}.pth".format(self.training_clock.epoch))
            print("Saving checkpoint epoch {}...".format(self.training_clock.epoch))
        else:
            save_path = os.path.join(self.model_dir, "{}.pth".format(name))

        torch.save({
            'clock': self.training_clock.make_checkpoint(),
            'Generator_state_dict': self.Generator.cpu().state_dict(),
            'Discriminator_state_dict': self.Discriminator.cpu().state_dict(),
            'Encoder_state_dict': self.vaeE.cpu().state_dict(),
            'optimizer_Generator_state_dict': self.Generator_optimizer.state_dict(),
            'optimizer_Discriminator_state_dict': self.Discriminator_optimizer.state_dict(),
            'optimizer_Encoder_state_dict': self.Encoder_optimizer.state_dict(),
        }, save_path)

        self.Generator.cuda()
        self.Discriminator.cuda()
        self.vaeE.cuda()

    def load_checkpoints(self, name=None):

        name = name if name == 'latest' else "checkpoint_epoch{}".format(name)
        load_path = os.path.join(self.model_dir, "{}.pth".format(name))
        if not os.path.exists(load_path):
            raise ValueError("Checkpoint {} not exists.".format(load_path))

        checkpoint = torch.load(load_path)
        print("Loading checkpoint from {} ...".format(load_path))
        self.Generator.load_state_dict(checkpoint['Generator_state_dict'])
        self.Discriminator.load_state_dict(checkpoint['Discriminator_state_dict'])
        self.vaeE.load_state_dict(checkpoint['Encoder_state_dict'])
        self.Generator_optimizer.load_state_dict(checkpoint['optimizer_Generator_state_dict'])
        self.Discriminator_optimizer.load_state_dict(checkpoint['optimizer_Discriminator_state_dict'])
        self.Encoder_optimizer.load_state_dict(checkpoint['optimizer_Encoder_state_dict'])
        self.training_clock.restore_checkpoint(checkpoint['clock'])

    def visualize_batch(self, data, mode, **kwargs):
        tb = self.train_writer if mode == 'train' else self.val_writer

        num = 2

        gt_pts = data['gt'][:num].transpose(1, 2).detach().cpu().numpy()
        predicted_pts = self.predicted_pc[:num].transpose(1, 2).detach().cpu().numpy()
        partial_pts = self.partial_pc[:num].transpose(1, 2).detach().cpu().numpy()

        predicted_pts = torch.Tensor(np.clip(predicted_pts, -0.999, 0.999))

        tb.add_mesh("ground_truth", vertices=gt_pts, global_step=self.training_clock.step)
        tb.add_mesh("completion", vertices=predicted_pts, global_step=self.training_clock.step)
        tb.add_mesh("input", vertices=partial_pts, global_step=self.training_clock.step)


# VAE Training Agent
class VAE(object):
    def __init__(self, config):
        super(VAE, self).__init__()
        self.log_dir = config['log_dir']
        self.model_dir = config['model_dir']
        self.training_clock = TrainingClock()
        self.batch_size = config['batch_size']
        self.weight_kl_vae = config["kl_vae_weights"]
        self.z_dim = 64

        # build network
        self.vae = VariationalAutoencoder().cuda()
        # print('-----pointVAE architecture-----')
        # print(self.vae)

        # set loss function
        self.criterion = nn.MSELoss().cuda()

        # set optimizer
        self.base_lr = config["lr"]
        self.optimizer = Adam(self.vae.parameters(), config["lr"])

        self.scheduler = lr_scheduler.ExponentialLR(self.optimizer, config['lr_decay'])

        # set tensorboard writer
        self.train_writer = SummaryWriter(os.path.join(self.log_dir, 'train.events'))
        self.val_writer = SummaryWriter(os.path.join(self.log_dir, 'val.events'))

    def forward(self, data):
        input_pts = data['points'].cuda()

        target_pts = input_pts.clone()

        self.output_pts, mean, log_variance = self.vae(input_pts)

        self.emd_loss = torch.mean(earth_mover_distance(self.output_pts, target_pts))

        self.kl_loss = -0.5 * torch.mean(1 + log_variance - mean ** 2 - torch.exp(log_variance)) * self.weight_kl_vae

    def save_checkpoints(self, name=None):
        """save checkpoint during training for future restore"""
        if name is None:
            save_path = os.path.join(self.model_dir, "checkpoint_epoch{}.pth".format(self.training_clock.epoch))
            print("Saving checkpoint epoch {}...".format(self.training_clock.epoch))
            print(save_path)
        else:
            save_path = os.path.join(self.model_dir, "{}.pth".format(name))

        if isinstance(self.vae, nn.DataParallel):
            model_state_dict = self.vae.module.cpu().state_dict()
        else:
            model_state_dict = self.vae.cpu().state_dict()

        torch.save({
            'clock': self.training_clock.make_checkpoint(),
            'model_state_dict': model_state_dict,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
        }, save_path)

        self.vae.cuda()

    def load_checkpoints(self, name=None):
        """load checkpoint from saved checkpoint"""
        name = name if name == 'latest' else "checkpoint_epoch{}".format(name)
        load_path = os.path.join(self.model_dir, "{}.pth".format(name))
        if not os.path.exists(load_path):
            raise ValueError("Checkpoint {} not exists.".format(load_path))

        checkpoint = torch.load(load_path)
        print("Loading checkpoint from {} ...".format(load_path))
        if isinstance(self.vae, nn.DataParallel):
            self.vae.module.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.vae.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.training_clock.restore_checkpoint(checkpoint['clock'])

    def collect_loss(self):
        loss_dict = {"emd": self.emd_loss, "kl": self.kl_loss}
        return loss_dict

    def random_sample(self, num):
        z = torch.normal(torch.zeros((num, self.z_dim)), torch.ones((num, self.z_dim))).cuda()
        gen_pts = self.vae.decoder(z)
        return gen_pts

    def update_learning_rate(self):
        """record and update learning rate"""
        self.train_writer.add_scalar('learning_rate', self.optimizer.param_groups[-1]['lr'], self.training_clock.epoch)
        if not self.optimizer.param_groups[-1]['lr'] < self.base_lr / 5.0:
            self.scheduler.step(self.training_clock.epoch)

    def record_losses(self, loss_dict, mode='train'):
        """record loss to tensorboard"""
        losses_values = {k: v.item() for k, v in loss_dict.items()}

        tb = self.train_writer if mode == 'train' else self.val_writer
        for k, v in losses_values.items():
            tb.add_scalar(k, v, self.training_clock.step)

    def training(self, data):
        """one step of training"""
        self.vae.train()

        self.forward(data)

        losses = self.collect_loss()
        self.optimizer.step()
        self.optimizer.zero_grad()
        loss = sum(losses.values())

        loss.backward()

        self.record_losses(losses, 'train')

    def validation(self, data):
        """one step of validation"""
        self.vae.eval()

        with torch.no_grad():
            self.forward(data)

        losses = self.collect_loss()
        self.record_losses(losses, 'validation')

    def visualize_batch(self, data, mode, **kwargs):
        tb = self.train_writer if mode == 'train' else self.val_writer

        num = 2

        target_pts = data['points'][:num].transpose(1, 2).detach().cpu().numpy()
        outputs_pts = self.output_pts[:num].transpose(1, 2).detach().cpu().numpy()

        tb.add_mesh("gt", vertices=target_pts, global_step=self.training_clock.step)
        tb.add_mesh("output", vertices=outputs_pts, global_step=self.training_clock.step)

        self.vae.eval()
        with torch.no_grad():
            gen_pts = self.random_sample(num)
        gen_pts = gen_pts.transpose(1, 2).detach().cpu().numpy()
        tb.add_mesh("generated", vertices=gen_pts, global_step=self.training_clock.step)
