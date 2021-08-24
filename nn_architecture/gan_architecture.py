import torch
import torch.nn as nn
from nn_architecture.common import initialize_weights


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(136, 256),
            nn.LeakyReLU(True),
            nn.Linear(256, 512),
            nn.LeakyReLU(True),
            nn.Linear(512, 128)
        )
        self.apply(initialize_weights)

    def forward(self, x, noise):
        x = torch.cat([x, noise], 1)
        x = self.model(x)
        return x


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(128, 256),
            nn.LeakyReLU(True),
            nn.Linear(256, 512),
            nn.LeakyReLU(True),
            nn.Linear(512, 1)
        )
        self.apply(initialize_weights)

    def forward(self, x):
        x = self.model(x).view(-1)
        return x


class LatentEncoder(nn.Module):
    def __init__(self):
        super(LatentEncoder, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(128, 256),
            nn.LeakyReLU(True),
            nn.Linear(256, 512),
            nn.LeakyReLU(True)
        )

        self.fc_mean = nn.Linear(512, 8)
        self.fc_logvar = nn.Linear(512, 8)


    def forward(self, x):
        x = self.model(x)
        mean = self.fc_mean(x)
        log_variance = self.fc_logvar(x)
        var = torch.exp(log_variance / 2.)
        eps = torch.distributions.normal.Normal(0, 1).sample(mean.shape).to(x.device)
        z = mean + var * eps
        return z, mean, log_variance


class LatentEncoderPointNet(nn.Module):
    def __init__(self):
        super(LatentEncoderPointNet, self).__init__()

        self.model = nn.Sequential(
            nn.Conv1d(3, 64, (1, 1), (1, 1)),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2, True),

            nn.Conv1d(64, 128, (1, 1), (1, 1)),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2, True),

            nn.Conv1d(128, 128, (1, 1), (1, 1)),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2, True),

            nn.Conv1d(128, 256, (1, 1), (1, 1)),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2, True),
            nn.Conv1d(256, 128, (1, 1), (1, 1))
        )

        self.fc_mean = nn.Linear(128, 64)
        self.fc_logvar = nn.Linear(128, 64)

    def forward(self, x):
        x = self.model(x)
        x = torch.max(x, 2)[0]

        mean = self.fc_mean(x)
        log_variance = self.fc_logvar(x)
        var = torch.exp(log_variance / 2.)
        eps = torch.distributions.normal.Normal(0, 1).sample(mean.shape).to(x.device)
        z = mean + var * eps
        return z, mean, log_variance



