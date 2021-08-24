import torch
import torch.nn as nn
import os

MODEL_PATH = "models"
CHECKPOINT_PATH = os.path.join(MODEL_PATH, 'checkpoints')


class Lambda(nn.Module):
    def __init__(self, function):
        super(Lambda, self).__init__()
        self.function = function

    def forward(self, x):
        return self.function(x)


def initialize_weights(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


class SavableModule(nn.Module):
    def __init__(self, filename):
        super(SavableModule, self).__init__()
        self.filename = filename

    def get_filename(self, epoch=None, filename=None):
        if filename is None:
            filename = self.filename
        if epoch is None:
            return os.path.join(MODEL_PATH, filename)
        else:
            filename = filename.split('.')
            filename[-2] += '-epoch-{:05d}'.format(epoch)
            filename = '.'.join(filename)
            return os.path.join(CHECKPOINT_PATH, filename)

    def load(self, epoch=None):
        self.load_state_dict(torch.load(self.get_filename(epoch=epoch)), strict=False)

    def save(self, epoch=None):
        if epoch is not None and not os.path.exists(CHECKPOINT_PATH):
            os.mkdir(CHECKPOINT_PATH)
        torch.save(self.state_dict(), self.get_filename(epoch=epoch))

    @property
    def device(self):
        return next(self.parameters()).device


class BaseAgent(object):
    """Base trainer that provides common training behavior.
        All customized trainer should be subclass of this class.
    """

    def __init__(self, config):
        self.log_dir = config.log_dir
        self.model_dir = config.model_dir
        self.clock = TrainClock()
        self.batch_size = config.batch_size

        # build network
        self.net = self.build_net(config)

        # set loss function
        self.set_loss_function()

        # set optimizer
        self.set_optimizer(config)

        # set lr scheduler
        self.set_scheduler(config)

        # set tensorboard writer
        self.train_tb = SummaryWriter(os.path.join(self.log_dir, 'train.events'))
        self.val_tb = SummaryWriter(os.path.join(self.log_dir, 'val.events'))

    @abstractmethod
    def build_net(self, config):
        raise NotImplementedError

    def set_loss_function(self):
        """set loss function used in training"""
        self.criterion = nn.MSELoss().cuda()

    @abstractmethod
    def collect_loss(self):
        """collect all losses into a dict"""
        raise NotImplementedError

    def set_optimizer(self, config):
        """set optimizer used in training"""
        self.base_lr = config.lr
        self.optimizer = optim.Adam(self.net.parameters(), config.lr)

    def set_scheduler(self, config):
        """set lr scheduler used in training"""
        # self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, config.lr_step_size)
        self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, config.lr_decay)

    def save_ckpt(self, name=None):
        """save checkpoint during training for future restore"""
        if name is None:
            save_path = os.path.join(self.model_dir, "ckpt_epoch{}.pth".format(self.clock.epoch))
            print("Saving checkpoint epoch {}...".format(self.clock.epoch))
        else:
            save_path = os.path.join(self.model_dir, "{}.pth".format(name))

        if isinstance(self.net, nn.DataParallel):
            model_state_dict = self.net.module.cpu().state_dict()
        else:
            model_state_dict = self.net.cpu().state_dict()

        torch.save({
            'clock': self.clock.make_checkpoint(),
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
        self.clock.restore_checkpoint(checkpoint['clock'])

    def forward(self, data):
        """forward logic for your network"""
        raise NotImplementedError

    def update_network(self, loss_dict):
        """update network by back propagation"""
        loss = sum(loss_dict.values())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_learning_rate(self):
        """record and update learning rate"""
        self.train_tb.add_scalar('learning_rate', self.optimizer.param_groups[-1]['lr'], self.clock.epoch)
        if not self.optimizer.param_groups[-1]['lr'] < self.base_lr / 5.0:
            self.scheduler.step(self.clock.epoch)

    def record_losses(self, loss_dict, mode='train'):
        """record loss to tensorboard"""
        losses_values = {k: v.item() for k, v in loss_dict.items()}

        tb = self.train_tb if mode == 'train' else self.val_tb
        for k, v in losses_values.items():
            tb.add_scalar(k, v, self.clock.step)

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

    def visualize_batch(self, data, tb, **kwargs):
        """write visualization results to tensorboard writer"""
        raise NotImplementedError
