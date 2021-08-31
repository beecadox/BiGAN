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
from data_processing.visualization import plot_pcds
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


class GAN(object):
    def __init__(self, config):
        self.log_dir = config["log_dir"]
        self.model_dir = config["model_dir"]
        self.training_clock = TrainingClock()
        self.batch_size = config["batch_size"]

        self.z_dim = 64

        # build network
        self.build_net(config)

        # set loss function
        self.set_loss_function()

        # set optimizer
        self.set_optimizer(config)

        # set tensorboard writer
        self.train_tb = SummaryWriter(os.path.join(self.log_dir, 'train.events'))
        self.val_tb = SummaryWriter(os.path.join(self.log_dir, 'val.events'))

        self.weight_z_L1 = config["z_L1_weights"]
        self.weight_partial_rec = config["partial_rec_weights"]

    def build_net(self, config):

        # load pretrained pointVAE
        pointVAE = VariationalAutoencoder()
        try:
            vae_weights = torch.load(config["pretrain_vae_path"])['model_state_dict']
        except Exception as e:
            raise ValueError("Check the path for pretrained model of point VAE. \n{}".format(e))
        pointVAE.load_state_dict(vae_weights, strict=False)
        self.netE = pointVAE.encoder.eval().cuda()
        self.vaeD = pointVAE.decoder.eval().cuda()
        set_requires_grad(self.netE, False)  # netE remains fixed
        # print(self.netE)
        # print("---------")
        # print(self.vaeD)
        # print("---------")
        # build G, D
        self.netG = Generator(64, 64, (256, 512)).cuda()
        # print(self.netG)
        # print("---------")
        self.netD = Discriminator(64).cuda()
        # print(self.netD)

    def set_loss_function(self):
        """set loss function used in training"""
        self.criterionGAN = nn.MSELoss()  # LSGAN
        self.criterionL1 = nn.L1Loss()

    def collect_loss(self):
        loss_dict = {"D_GAN": self.loss_D,
                     "G_GAN": self.loss_G_GAN,
                     "z_L1": self.loss_z_L1,
                     "partial_rec": self.loss_partial_rec,
                     "emd_loss": self.emd_loss}
        return loss_dict

    def set_optimizer(self, config):
        """set optimizer and lr scheduler used in training"""
        self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=config["lr"], betas=(0.5, 0.999))
        self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=config["lr"], betas=(0.5, 0.999))
        self.optimizer_E = torch.optim.Adam(self.netE.parameters(), lr=config["lr"], betas=(0.5, 0.999))

    def forward(self, data):
        self.raw_pc = data['partial'].cuda()
        self.real_pc = data['gt'].cuda()

        with torch.no_grad():
            self.raw_latent,_,_ = self.netE(self.raw_pc).cuda()
            self.real_latent,_,_ = self.netE(self.real_pc).cuda()

        self.forward_GE()

    def forward_GE(self):
        self.z_random = self.get_random_noise(self.raw_latent.size(0))
        self.fake_latent = self.netG(self.raw_latent, self.z_random)
        self.fake_pc = self.vaeD(self.fake_latent)
        self.z_rec, z_mu, z_logvar = self.netE(self.fake_pc)

    def backward_D(self):
        # fake
        pred_fake = self.netD(self.fake_latent.detach())
        fake = torch.zeros_like(pred_fake).fill_(0.0).cuda()
        self.loss_D_fake = self.criterionGAN(pred_fake, fake)

        # real
        pred_real = self.netD(self.real_latent.detach())
        real = torch.ones_like(pred_real).fill_(1.0).cuda()
        self.loss_D_real = self.criterionGAN(pred_real, real)

        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        self.loss_D.backward()
    def update_G_and_E(self):
        set_requires_grad(self.netD, False)
        self.optimizer_G.zero_grad()
        self.backward_EG()
        self.optimizer_G.step()

    def backward_EG(self):
        # 1. G(A) fool D
        pred_fake = self.netD(self.fake_latent)
        real = torch.ones_like(pred_fake).fill_(1.0).cuda()
        self.loss_G_GAN = self.criterionGAN(pred_fake, real)

        # 2. noise reconstruction |E(G(A, z)) - z_random|
        self.loss_z_L1 = self.criterionL1(self.z_rec, self.z_random) * self.weight_z_L1

        # 3. partial scan reconstruction
        self.loss_partial_rec = hausdorff(self.raw_pc, self.fake_pc) * self.weight_partial_rec
        self.emd_loss = torch.mean(earth_mover_distance(self.real_pc, self.fake_pc))
        self.loss_EG = self.loss_G_GAN + self.loss_z_L1 + self.loss_partial_rec
        self.loss_EG.backward()

    def update_D(self):
        set_requires_grad(self.netD, True)

        self.optimizer_D.zero_grad()
        self.backward_D()
        self.optimizer_D.step()

    def get_point_cloud(self):
        """get real/fake/raw point cloud of current batch"""
        real_pts = self.real_pc.transpose(1, 2).detach().cpu().numpy()
        fake_pts = self.fake_pc.transpose(1, 2).detach().cpu().numpy()
        raw_pts = self.raw_pc.transpose(1, 2).detach().cpu().numpy()
        return real_pts, fake_pts, raw_pts

    def optimize_network(self):
        self.update_G_and_E()
        self.update_D()

    def training(self, data):
        """one step of training"""
        self.forward(data)
        self.optimize_network()

        loss_dict = self.collect_loss()
        self.record_losses(loss_dict, "train")

    def validation(self, data):
        """one step of validation"""
        with torch.no_grad():
            self.forward(data)

    def update_learning_rate(self):
        """record and update learning rate"""
        pass

    def eval(self):
        """set G, D, E to eval mode"""
        self.netG.eval()
        self.netD.eval()
        self.netE.eval()

    def get_random_noise(self, batch_size):
        """sample random z from gaussian"""
        z = torch.randn((batch_size, self.z_dim))
        return z

    def save_checkpoints(self, name=None):
        """save checkpoint during training for future restore"""
        if name is None:
            save_path = os.path.join(self.model_dir, "ckpt_epoch{}.pth".format(self.training_clock.epoch))
            print("Saving checkpoint epoch {}...".format(self.training_clock.epoch))
        else:
            save_path = os.path.join(self.model_dir, "{}.pth".format(name))

        torch.save({
            'clock': self.training_clock.make_checkpoint(),
            'netG_state_dict': self.netG.cpu().state_dict(),
            'netD_state_dict': self.netD.cpu().state_dict(),
            'netE_state_dict': self.netE.cpu().state_dict(),
            'optimizerG_state_dict': self.optimizer_G.state_dict(),
            'optimizerD_state_dict': self.optimizer_D.state_dict(),
            'optimizerE_state_dict': self.optimizer_E.state_dict(),
        }, save_path)

        self.netG.cuda()
        self.netD.cuda()
        self.netE.cuda()

    def load_checkpoints(self, name=None):
        """load checkpoint from saved checkpoint"""
        name = name if name == 'latest' else "ckpt_epoch{}".format(name)
        load_path = os.path.join(self.model_dir, "{}.pth".format(name))
        if not os.path.exists(load_path):
            raise ValueError("Checkpoint {} not exists.".format(load_path))

        checkpoint = torch.load(load_path)
        print("Loading checkpoint from {} ...".format(load_path))
        self.netG.load_state_dict(checkpoint['netG_state_dict'])
        self.netD.load_state_dict(checkpoint['netD_state_dict'])
        self.netE.load_state_dict(checkpoint['netE_state_dict'])
        self.optimizer_G.load_state_dict(checkpoint['optimizerG_state_dict'])
        self.optimizer_D.load_state_dict(checkpoint['optimizerD_state_dict'])
        self.optimizer_E.load_state_dict(checkpoint['optimizerE_state_dict'])
        self.training_clock.restore_checkpoint(checkpoint['clock'])

    def record_losses(self, loss_dict, mode='train'):
        """record loss to tensorboard"""
        losses_values = {k: v.item() for k, v in loss_dict.items()}

        tb = self.train_tb if mode == 'train' else self.val_tb
        for k, v in losses_values.items():
            tb.add_scalar(k, v, self.training_clock.step)

    def visualize_batch(self, data, mode, **kwargs):
        tb = self.train_tb if mode == 'train' else self.val_tb

        num = 3

        real_pts = data['gt'][:num].transpose(1, 2).detach().cpu().numpy()
        fake_pts = self.fake_pc[:num].transpose(1, 2).detach().cpu().numpy()
        raw_pts = self.raw_pc[:num].transpose(1, 2).detach().cpu().numpy()

        fake_pts = torch.from_numpy(np.clip(fake_pts, -0.999, 0.999))

        tb.add_mesh("real", vertices=real_pts, global_step=self.training_clock.step)
        tb.add_mesh("fake", vertices=fake_pts, global_step=self.training_clock.step)
        tb.add_mesh("input", vertices=raw_pts, global_step=self.training_clock.step)


        plot_pcds(filename='gan', pcds=[real_pts[0], raw_pts[0], fake_pts[0]], titles=["gt", "partial", "completion"], use_color=[0, 0, 0],
                  color=[None, None, None], suptitle=str(str(self.training_clock.epoch) + "_0"))
        plot_pcds(filename='gan', pcds=[real_pts[1], raw_pts[1], fake_pts[1]], titles=["gt", "partial", "completion"], use_color=[0, 0],
                  color=[None, None, None], suptitle=str(self.training_clock.step) + "_1")
        plot_pcds(filename='gan', pcds=[real_pts[2], raw_pts[2], fake_pts[2]], titles=["gt", "partial", "completion"], use_color=[0, 0, 0],
                  color=[None, None, None], suptitle=str(self.training_clock.step) + "_2")


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
        # set_requires_grad(self.vae, False)
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

        target_pts = input_pts.detach().clone()
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
        print(checkpoint['clock'])
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

    def update_network(self, loss_dict):
        """update network by back propagation"""
        loss = sum(loss_dict.values())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def training(self, data):
        """one step of training"""
        self.vae.train()

        self.forward(data)

        losses = self.collect_loss()
        self.update_network(losses)
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

        num = 3

        target_pts = data['points'][:num].transpose(1, 2).detach().cpu().numpy()
        outputs_pts = self.output_pts[:num].transpose(1, 2).detach().cpu().numpy()

        tb.add_mesh("gt", vertices=target_pts, global_step=self.training_clock.step)
        tb.add_mesh("output", vertices=outputs_pts, global_step=self.training_clock.step)
        plot_pcds(filename='vae', pcds=[target_pts[0], outputs_pts[0]], titles=["gt", "output"], use_color=[0, 0], color=[None, None], suptitle=str(str(self.training_clock.epoch) + "_0"))
        plot_pcds(filename='vae', pcds=[target_pts[1], outputs_pts[1]], titles=["gt", "output"], use_color=[0, 0], color=[None, None], suptitle=str(self.training_clock.step) + "_1")
        plot_pcds(filename='vae', pcds=[target_pts[2], outputs_pts[2]], titles=["gt", "output"], use_color=[0, 0], color=[None, None], suptitle=str(self.training_clock.step) + "_2")
        self.vae.eval()
        with torch.no_grad():
            gen_pts = self.random_sample(num)
        gen_pts = gen_pts.transpose(1, 2).detach().cpu().numpy()
        tb.add_mesh("generated", vertices=gen_pts, global_step=self.training_clock.step)
