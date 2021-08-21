import torch
import torch.nn as nn
from common import Lambda, SavableModule


VAE_MULTIPLIER = 24
standard_normal_distribution = torch.distributions.normal.Normal(0, 1)
LATENT_CODE_SIZE = 128


class VariationalAutoencoder(SavableModule):
    def __init__(self):
        super(VariationalAutoencoder, self).__init__(filename="vae-{:d}.to".format(LATENT_CODE_SIZE))

        self.encoder = nn.Sequential(
            nn.Conv3d(in_channels=1, out_channels=1 * VAE_MULTIPLIER, kernel_size=(4, 4), stride=(2, 2), padding=1),
            nn.BatchNorm3d(1 * VAE_MULTIPLIER),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.Conv3d(in_channels=1 * VAE_MULTIPLIER, out_channels=2 * VAE_MULTIPLIER, kernel_size=(4, 4), stride=(2, 2), padding=1),
            nn.BatchNorm3d(2 * VAE_MULTIPLIER),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.Conv3d(in_channels=2 * VAE_MULTIPLIER, out_channels=4 * VAE_MULTIPLIER, kernel_size=(4, 4), stride=(2, 2), padding=1),
            nn.BatchNorm3d(4 * VAE_MULTIPLIER),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.Conv3d(in_channels=4 * VAE_MULTIPLIER, out_channels=2 * LATENT_CODE_SIZE, kernel_size=(4, 4), stride=1),
            nn.BatchNorm3d(2* LATENT_CODE_SIZE),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            Lambda(lambda x: x.reshape(x.shape[0], -1)),

            nn.Linear(in_features=LATENT_CODE_SIZE * 2, out_features=LATENT_CODE_SIZE)
        )

        self.encoder.add_module('vae-bn', nn.BatchNorm1d(LATENT_CODE_SIZE))
        self.encoder.add_module('vae-lr', nn.LeakyReLU(negative_slope=0.2, inplace=True))

        self.encode_mean = nn.Linear(in_features=LATENT_CODE_SIZE, out_features=LATENT_CODE_SIZE)
        self.encode_log_variance = nn.Linear(in_features=LATENT_CODE_SIZE, out_features=LATENT_CODE_SIZE)

        self.decoder = nn.Sequential(
            nn.Linear(in_features=LATENT_CODE_SIZE, out_features=LATENT_CODE_SIZE * 2),
            nn.BatchNorm1d(LATENT_CODE_SIZE * 2),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            Lambda(lambda x: x.reshape(-1, LATENT_CODE_SIZE * 2, 1, 1, 1)),

            nn.ConvTranspose3d(in_channels=LATENT_CODE_SIZE * 2, out_channels=4 * VAE_MULTIPLIER, kernel_size=(4, 4), stride=(1, 1)),
            nn.BatchNorm3d(4 * VAE_MULTIPLIER),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.ConvTranspose3d(in_channels=4 * VAE_MULTIPLIER, out_channels=2 * VAE_MULTIPLIER, kernel_size=(4, 4), stride=(2, 2), padding=1),
            nn.BatchNorm3d(2 * VAE_MULTIPLIER),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.ConvTranspose3d(in_channels=2 * VAE_MULTIPLIER, out_channels=1 * VAE_MULTIPLIER, kernel_size=(4, 4), stride=(2, 2), padding=1),
            nn.BatchNorm3d(1 * VAE_MULTIPLIER),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.ConvTranspose3d(in_channels=1 * VAE_MULTIPLIER, out_channels=1, kernel_size=(4, 4), stride=(2, 2), padding=1)
        )
        self.cuda()

    def encode(self, x, return_mean_and_log_variance=False):
        x = x.reshape((-1, 1, 32, 32, 32))
        x = self.encoder(x)

        if not self.is_variational:
            return x

        mean = self.encode_mean(x).squeeze()

        if self.training or return_mean_and_log_variance:
            log_variance = self.encode_log_variance(x).squeeze()
            standard_deviation = torch.exp(log_variance * 0.5)
            eps = standard_normal_distribution.sample(mean.shape).to(x.device)

        if self.training:
            x = mean + standard_deviation * eps
        else:
            x = mean

        if return_mean_and_log_variance:
            return x, mean, log_variance
        else:
            return x

    def decode(self, x):
        if len(x.shape) == 1:
            x = x.unsqueeze(dim=0)  # add dimension for channels
        x = self.decoder(x)
        return x.squeeze()

    def forward(self, x):
        if not self.is_variational:
            z = self.encode(x)
            x = self.decode(z)
            return x

        z, mean, log_variance = self.encode(x, return_mean_and_log_variance=True)
        x = self.decode(z)
        return x, mean, log_variance
