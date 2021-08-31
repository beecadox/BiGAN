import torch
import torch.nn as nn
from nn_architecture.common import initialize_weights


class Generator(nn.Module):
    def __init__(self, latent_dim=64, noise_dim=64, n_features=(256, 512)):
        super(Generator, self).__init__()
        self.n_features = list(n_features)

        model = []
        prev_nf = latent_dim + noise_dim
        for idx, nf in enumerate(self.n_features):
            model.append(nn.Linear(prev_nf, nf))
            model.append(nn.LeakyReLU(inplace=True))
            prev_nf = nf
        model.append(nn.Linear(self.n_features[-1], latent_dim))
        self.model = nn.Sequential(*model)
        self.apply(initialize_weights)

    def forward(self, x, noise):
        x = torch.cat([x, noise], dim=1)
        x = self.model(x)
        return x


class Discriminator(nn.Module):
    def __init__(self, latent_dim=128, n_features=(256, 512)):
        super(Discriminator, self).__init__()
        self.n_features = list(n_features)

        model = []
        prev_nf = latent_dim
        for idx, nf in enumerate(self.n_features):
            model.append(nn.Linear(prev_nf, nf))
            model.append(nn.LeakyReLU(inplace=True))
            prev_nf = nf

        model.append(nn.Linear(self.n_features[-1], 1))

        self.model = nn.Sequential(*model)
        self.apply(initialize_weights)

    def forward(self, x):
        x = self.model(x)
        x = x.view(-1)
        return x


class LatentEncoder(nn.Module):
    def __init__(self, latent_dim=128, z_dim=8, n_features=(256, 512)):
        super(LatentEncoder, self).__init__()
        self.n_features = list(n_features)

        model = []
        prev_nf = latent_dim
        for idx, nf in enumerate(self.n_features):
            model.append(nn.Linear(prev_nf, nf))
            model.append(nn.LeakyReLU(inplace=True))
            prev_nf = nf

        self.model = nn.Sequential(*model)

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
    def __init__(self, n_filters=(64, 128, 128, 256), latent_dim=128, z_dim=64):
        super(LatentEncoderPointNet, self).__init__()

        self.n_filters = list(n_filters) + [latent_dim]
        self.latent_dim = latent_dim
        self.z_dim = z_dim

        model = []
        prev_nf = 3
        for idx, nf in enumerate(self.n_filters):
            conv_layer = nn.Conv1d(prev_nf, nf, kernel_size=1, stride=1)
            model.append(conv_layer)

            bn_layer = nn.BatchNorm1d(nf)
            model.append(bn_layer)

            act_layer = nn.LeakyReLU(0.1, True)
            model.append(act_layer)
            prev_nf = nf

        self.model = nn.Sequential(*model)

        self.fc_mean = nn.Linear(prev_nf // 2, z_dim)
        self.fc_logvar = nn.Linear(prev_nf // 2, z_dim)

    def forward(self, x):
        x = self.model(x)
        x = torch.max(x, 2)[0]

        mean = self.fc_mean(x)
        log_variance = self.fc_logvar(x)
        var = torch.exp(log_variance / 2.)
        eps = torch.distributions.normal.Normal(0, 1).sample(mean.shape)
        z = mean + var * eps
        return z, mean, log_variance



