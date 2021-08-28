import torch
import torch.nn as nn
from nn_architecture.common import SavableModule, Lambda


class EncoderPointNet(nn.Module):
    def __init__(self, n_filters=(64, 128, 128, 256), latent_dim=128, z_dim=64, bn=True):
        super(EncoderPointNet, self).__init__()
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

            act_layer = nn.LeakyReLU(inplace=True)
            model.append(act_layer)
            prev_nf = nf

        self.model = nn.Sequential(*model)

        self.fc_mu = nn.Linear(latent_dim, z_dim)
        self.fc_logvar = nn.Linear(latent_dim, z_dim)

    def forward(self, x):
        x = self.model(x)
        x = torch.max(x, dim=2)[0]

        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        var = torch.exp(logvar / 2.)
        # N ~ N(0,1)
        z_size = mu.size()
        N = torch.normal(torch.zeros(z_size), torch.ones(z_size)).cuda()
        z = mu + var * N
        return z, mu, logvar


class DecoderFC(nn.Module):
    def __init__(self, n_features=(256, 256), latent_dim=128, z_dim=64, output_pts=2048, bn=False):
        super(DecoderFC, self).__init__()
        self.n_features = list(n_features)
        self.output_pts = output_pts
        self.latent_dim = latent_dim
        self.z_dim = z_dim

        model = []
        prev_nf = self.latent_dim
        for idx, nf in enumerate(self.n_features):
            fc_layer = nn.Linear(prev_nf, nf)
            model.append(fc_layer)

            bn_layer = nn.BatchNorm1d(nf)
            model.append(bn_layer)
            act_layer = nn.LeakyReLU(inplace=True)
            model.append(act_layer)
            prev_nf = nf

        fc_layer = nn.Linear(self.n_features[-1], output_pts*3)
        model.append(fc_layer)

        self.model = nn.Sequential(*model)

        self.expand = nn.Linear(z_dim, latent_dim)

    def forward(self, x):
        x = self.expand(x)
        x = self.model(x)
        x = x.view((-1, 3, self.output_pts))
        return x


class VariationalAutoencoder(nn.Module):
    def __init__(self):
        super(VariationalAutoencoder, self).__init__()
        self.encoder = EncoderPointNet()
        self.decoder = DecoderFC()

    def encode(self, x):
        return self.encoder(x)

    def decode(self, x):
        return self.decoder(x)

    def forward(self, x):
        z, mu, logvar = self.encoder(x)
        x = self.decoder(z)
        return x, mu, logvar


class VariationalAutoencoder3D(SavableModule):
    def __init__(self):
        super(VariationalAutoencoder3D, self).__init__(filename="vae-{:d}.to".format(128))
        self.encoder = EncoderVAE3D()
        self.decoder = DecoderVAE3D()

    def encode(self, x):
        return self.encoder(x)

    def decode(self, x):
        return self.decoder(x)

    def forward(self, x):
        z, mean, log_variance = self.encoder(x)
        x = self.decoder(z)
        return x, mean, log_variance


class EncoderVAE3D(nn.Module):
    def __init__(self):
        super(EncoderVAE3D, self).__init__()
        self.latent_dim = 128
        self.z_dim = 64

        self.model = nn.Sequential(
            nn.Conv3d(1, 24, 4, 2, 1),
            nn.BatchNorm3d(24),
            nn.LeakyReLU(0.2, True),

            nn.Conv3d(24, 48, 4, 2, 1),
            nn.BatchNorm3d(48),
            nn.LeakyReLU(0.2, True),

            nn.Conv3d(48, 96, 4, 2, 1),
            nn.BatchNorm3d(96),
            nn.LeakyReLU(0.2, True),

            nn.Conv3d(96, 256, 4, 1),
            nn.BatchNorm3d(256),
            nn.LeakyReLU(0.2, True),

            Lambda(lambda x: x.reshape(x.shape[0], -1)),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2, True)
        )

        self.encode_mean = nn.Linear(128, 128)
        self.encode_log_variance = nn.Linear(128, 128)

    def forward(self, x):
        x = self.model(x)
        x = torch.max(x, 2)[0]

        mean = self.encode_mean(x)
        log_variance = self.encode_log_varianc(x)
        var = torch.exp(log_variance / 2.)
        eps = torch.distributions.normal.Normal(0, 1).sample(mean.shape).to(x.device)
        z = mean + var * eps
        return z, mean, log_variance


class DecoderVAE3D(nn.Module):
    def __init__(self):
        super(DecoderVAE3D, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2, True),

            Lambda(lambda x: x.reshape(-1, 256, 1, 1, 1)),

            nn.ConvTranspose3d(256, 96, 4, 1),
            nn.BatchNorm3d(96),
            nn.LeakyReLU(0.2, True),

            nn.ConvTranspose3d(96, 48, 4, 2, 1),
            nn.BatchNorm3d(48),
            nn.LeakyReLU(0.2, True),

            nn.ConvTranspose3d(48, 24, 4, 2, 1),
            nn.BatchNorm3d(24),
            nn.LeakyReLU(0.2, True),

            nn.ConvTranspose3d(24, 1, 4, 2, 1)
        )

    def forward(self, x):
        # if len(x.shape) == 1:
        #     x = x.unsqueeze(dim=0)  # add dimension for channels
        x = self.model(x)
        x = x.view((-1, 3, 2048))
        return x

