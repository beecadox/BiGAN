import torch
import torch.nn as nn
from nn_architecture.common import SavableModule


class VariationalAutoencoder(SavableModule):
    def __init__(self):
        super(VariationalAutoencoder, self).__init__(filename="vae-{:d}.to".format(128))
        self.encoder = EncoderVAE()
        self.decoder = DecoderVAE()

    def encode(self, x):
        return self.encoder(x)

    def decode(self, x):
        return self.decoder(x)

    def forward(self, x):
        z, mean, log_variance = self.encoder(x)
        x = self.decoder(z)
        return x, mean, log_variance


class EncoderVAE(nn.Module):
    def __init__(self):
        super(EncoderVAE, self).__init__()
        self.latent_dim = 128
        self.z_dim = 64

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

            nn.Conv1d(256, 128, (1, 1), (1, 1)),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2, True)
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


class DecoderVAE(nn.Module):
    def __init__(self):
        super(DecoderVAE, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2, True),

            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2, True),

            nn.Linear(256, 6144)
        )
        self.expand = nn.Linear(64, 128)

    def forward(self, x):
        x = self.expand(x)
        x = self.model(x)
        x = x.view((-1, 3, 2048))
        return x
