import os
import numpy as np
import torch
import torch.optim
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.optim import Adamax, lr_scheduler
from utils.hausdorff import hausdorff
from utils.emd import earth_mover_distance
from nn_architecture.vae_architecture import VariationalAutoencoder
from nn_architecture.gan_architecture import Generator, Discriminator


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
        super(GAN, self).__init__(config)
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
        self.optimizer_G = Adamax(self.Generator.parameters(), lr=config["lr"])
        self.optimizer_D = Adamax(self.Discriminator.parameters(), lr=config["lr"])
        self.optimizer_E = Adamax(self.vaeE.parameters(), lr=config["lr"])

        self.z_L1_weights = config["z_L1_weights"]
        self.partial_rec_weights = config["partial_rec_weights"]

        # set tensorboard writer
        self.train_writer = SummaryWriter(os.path.join(self.log_dir, 'train.events'))
        self.val_writer = SummaryWriter(os.path.join(self.log_dir, 'val.events'))

    def collect_loss(self):
        loss_dict = {"D_GAN": self.loss_D,
                     "G_GAN": self.loss_G_GAN,
                     "z_L1": self.loss_z_L1,
                     "partial_rec": self.loss_partial_rec}
        return loss_dict

    def forward(self, data):
        self.raw_pc = data['raw'].cuda()
        self.real_pc = data['real'].cuda()

        with torch.no_grad():
            self.raw_latent, = self.vaeE.encode(self.raw_pc)
            self.real_latent, = self.vaeE.encode(self.real_pc)

        self.z_random = torch.randn((self.raw_latent.size(0), self.z_dim)).cuda()

        self.fake_latent = self.Generator(self.raw_latent, self.z_random)
        self.fake_pc = self.vaeD.decode(self.fake_latent)
        self.z_rec, z_mean, z_logvar = self.vaeE(self.fake_pc)

    def update_D(self):
        set_requires_grad(self.Discriminator, True)

        self.optimizer_D.zero_grad()
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
        self.optimizer_D.step()

    def update_G_and_E(self):
        set_requires_grad(self.Discriminator, False)
        self.optimizer_G.zero_grad()
        # 1. G(A) fool D
        pred_fake = self.Discriminator(self.fake_latent)
        real = torch.ones_like(pred_fake).fill_(1.0).cuda()
        self.loss_G_GAN = self.criterionGAN(pred_fake, real)

        # 2. noise reconstruction |E(G(A, z)) - z_random|
        self.loss_z_L1 = self.criterionL1(self.z_rec, self.z_random) * self.z_L1_weights

        # 3. partial scan reconstruction
        self.loss_partial_rec = hausdorff(self.raw_pc, self.fake_pc) * self.partial_rec_weights

        self.loss_EG = self.loss_G_GAN + self.loss_z_L1 + self.loss_partial_rec
        self.loss_EG.backward()
        self.optimizer_G.step()

    def get_point_cloud(self):
        """get real/fake/raw point cloud of current batch"""
        real_pts = self.real_pc.transpose(1, 2).detach().cpu().numpy()
        fake_pts = self.fake_pc.transpose(1, 2).detach().cpu().numpy()
        raw_pts = self.raw_pc.transpose(1, 2).detach().cpu().numpy()
        return real_pts, fake_pts, raw_pts

    def train_model(self, data, mode='train'):
        """one step of training"""
        self.forward(data)
        self.update_G_and_E()
        self.update_D()

        loss_dict = self.collect_loss()
        losses_values = {k: v.item() for k, v in loss_dict.items()}

        tb = self.train_writer if mode == 'train' else self.val_writer
        for k, v in losses_values.items():
            tb.add_scalar(k, v, self.training_clock.step)

    def val_func(self, data):
        """one step of validation"""
        with torch.no_grad():
            self.forward(data)

    def update_learning_rate(self):
        """record and update learning rate"""
        pass

    def eval(self):
        """set G, D, E to eval mode"""
        self.Generator.eval()
        self.Discriminator.eval()
        self.vaeE.eval()

    def save_ckpt(self, name=None):
        """save checkpoint during training for future restore"""
        if name is None:
            save_path = os.path.join(self.model_dir, "ckpt_epoch{}.pth".format(self.training_clock.epoch))
            print("Saving checkpoint epoch {}...".format(self.training_clock.epoch))
        else:
            save_path = os.path.join(self.model_dir, "{}.pth".format(name))

        torch.save({
            'clock': self.training_clock.make_checkpoint(),
            'netG_state_dict': self.Generator.cpu().state_dict(),
            'netD_state_dict': self.Discriminator.cpu().state_dict(),
            'netE_state_dict': self.vaeE.cpu().state_dict(),
            'optimizerG_state_dict': self.optimizer_G.state_dict(),
            'optimizerD_state_dict': self.optimizer_D.state_dict(),
            'optimizerE_state_dict': self.optimizer_E.state_dict(),
        }, save_path)

        self.Generator.cuda()
        self.Discriminator.cuda()
        self.vaeE.cuda()

    def load_ckpt(self, name=None):
        """load checkpoint from saved checkpoint"""
        name = name if name == 'latest' else "ckpt_epoch{}".format(name)
        load_path = os.path.join(self.model_dir, "{}.pth".format(name))
        if not os.path.exists(load_path):
            raise ValueError("Checkpoint {} not exists.".format(load_path))

        checkpoint = torch.load(load_path)
        print("Loading checkpoint from {} ...".format(load_path))
        self.Generator.load_state_dict(checkpoint['netG_state_dict'])
        self.Discriminator.load_state_dict(checkpoint['netD_state_dict'])
        self.vaeE.load_state_dict(checkpoint['netE_state_dict'])
        self.optimizer_G.load_state_dict(checkpoint['optimizerG_state_dict'])
        self.optimizer_D.load_state_dict(checkpoint['optimizerD_state_dict'])
        self.optimizer_E.load_state_dict(checkpoint['optimizerE_state_dict'])
        self.training_clock.restore_checkpoint(checkpoint['clock'])

    def visualize_batch(self, data, mode, **kwargs):
        tb = self.train_tb if mode == 'train' else self.val_tb

        num = 2

        real_pts = data['real'][:num].transpose(1, 2).detach().cpu().numpy()
        fake_pts = self.fake_pc[:num].transpose(1, 2).detach().cpu().numpy()
        raw_pts = self.raw_pc[:num].transpose(1, 2).detach().cpu().numpy()

        fake_pts = np.clip(fake_pts, -0.999, 0.999)

        tb.add_mesh("real", vertices=real_pts, global_step=self.training_clock.step)
        tb.add_mesh("fake", vertices=fake_pts, global_step=self.training_clock.step)
        tb.add_mesh("input", vertices=raw_pts, global_step=self.training_clock.step)

# VAE Training Agent
class VAE(object):
    def __init__(self, config):
        super(VAE, self).__init__(config)
        self.z_dim = config.z_dim
        self.weight_kl_vae = config.weight_kl_vae
        self.log_dir = "proj_log"
        self.model_dir = config.model_dir
        self.training_clock = TrainingClock()
        self.batch_size = config.batch_size

        # build network
        self.net = self.build_net(config)

        # set loss function
        self.criterion = nn.MSELoss().cuda()

        # set optimizer
        self.base_lr = config.lr
        self.optimizer = Adamax(self.net.parameters(), config.lr)

        # set lr scheduler
        self.scheduler = lr_scheduler.ExponentialLR(self.optimizer, config.lr_decay)

        # set tensorboard writer
        self.train_tb = SummaryWriter(os.path.join(self.log_dir, 'train.events'))
        self.val_tb = SummaryWriter(os.path.join(self.log_dir, 'val.events'))

    def build_net(self, config):
        # customize your build_net function
        net = VariationalAutoencoder()
        print('-----pointVAE architecture-----')
        print(net)
        if config.parallel:
            net = nn.DataParallel(net)
        net = net.cuda()
        return net

    def forward(self, data):
        input_pts = data['points'].cuda()
        target_pts = input_pts.clone().detach()

        self.output_pts, mu, logvar = self.net(input_pts)

        self.emd_loss = earth_mover_distance(self.output_pts, target_pts)
        self.emd_loss = torch.mean(self.emd_loss)

        self.kl_loss = -0.5 * torch.mean(1 + logvar - mu ** 2 - torch.exp(logvar)) * self.weight_kl_vae

    def save_ckpt(self, name=None):
        """save checkpoint during training for future restore"""
        if name is None:
            save_path = os.path.join(self.model_dir, "ckpt_epoch{}.pth".format(self.training_clock.epoch))
            print("Saving checkpoint epoch {}...".format(self.training_clock.epoch))
        else:
            save_path = os.path.join(self.model_dir, "{}.pth".format(name))

        if isinstance(self.net, nn.DataParallel):
            model_state_dict = self.net.module.cpu().state_dict()
        else:
            model_state_dict = self.net.cpu().state_dict()

        torch.save({
            'clock': self.training_clock.make_checkpoint(),
            'model_state_dict': model_state_dict,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
        }, save_path)

        self.net.cuda()

    def load_ckpt(self, name=None):
        """load checkpoint from saved checkpoint"""
        name = name if name == 'latest' else "ckpt_epoch{}".format(name)
        load_path = os.path.join(self.model_dir, "{}.pth".format(name))
        if not os.path.exists(load_path):
            raise ValueError("Checkpoint {} not exists.".format(load_path))

        checkpoint = torch.load(load_path)
        print("Loading checkpoint from {} ...".format(load_path))
        if isinstance(self.net, nn.DataParallel):
            self.net.module.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.net.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.training_clock.restore_checkpoint(checkpoint['clock'])

    def collect_loss(self):
        loss_dict = {"emd": self.emd_loss, "kl": self.kl_loss}
        return loss_dict

    def random_sample(self, num):
        z = torch.normal(torch.zeros((num, self.z_dim)), torch.ones((num, self.z_dim))).cuda()
        gen_pts = self.net.decoder(z)
        return gen_pts

    def update_network(self, loss_dict):
        """update network by back propagation"""
        loss = sum(loss_dict.values())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_learning_rate(self):
        """record and update learning rate"""
        self.train_tb.add_scalar('learning_rate', self.optimizer.param_groups[-1]['lr'], self.training_clock.epoch)
        if not self.optimizer.param_groups[-1]['lr'] < self.base_lr / 5.0:
            self.scheduler.step(self.training_clock.epoch)

    def record_losses(self, loss_dict, mode='train'):
        """record loss to tensorboard"""
        losses_values = {k: v.item() for k, v in loss_dict.items()}

        tb = self.train_tb if mode == 'train' else self.val_tb
        for k, v in losses_values.items():
            tb.add_scalar(k, v, self.training_clock.step)

    def train_func(self, data):
        """one step of training"""
        self.net.train()

        self.forward(data)

        losses = self.collect_loss()
        self.update_network(losses)
        self.record_losses(losses, 'train')

    def val_func(self, data):
        """one step of validation"""
        self.net.eval()

        with torch.no_grad():
            self.forward(data)

        losses = self.collect_loss()
        self.record_losses(losses, 'validation')

    def visualize_batch(self, data, mode, **kwargs):
        tb = self.train_tb if mode == 'train' else self.val_tb

        num = 2

        target_pts = data['points'][:num].transpose(1, 2).detach().cpu().numpy()
        outputs_pts = self.output_pts[:num].transpose(1, 2).detach().cpu().numpy()

        tb.add_mesh("gt", vertices=target_pts, global_step=self.training_clock.step)
        tb.add_mesh("output", vertices=outputs_pts, global_step=self.training_clock.step)

        self.net.eval()
        with torch.no_grad():
            gen_pts = self.random_sample(num)
        gen_pts = gen_pts.transpose(1, 2).detach().cpu().numpy()
        tb.add_mesh("generated", vertices=gen_pts, global_step=self.training_clock.step)
