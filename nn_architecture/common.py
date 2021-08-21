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


def _initialize_weights(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
