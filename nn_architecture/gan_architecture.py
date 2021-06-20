import torch
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, latent_dim=128, noise_dim=8, n_features=(256, 512)):
        super(Generator, self).__init__()

        # def block(, ):

        self.n_features = list(n_features)

        model = []
        prev_nf = latent_dim + noise_dim
        for idx, nf in enumerate(self.n_features):
            model.append(nn.Linear(prev_nf, nf))
            model.append(nn.LeakyReLU(inplace=True))
            prev_nf = nf

        model.append(nn.Linear(self.n_features[-1], latent_dim))

        self.model = nn.Sequential(*model)

        self._initialize_weights()

    def forward(self, x, noise, y=None):
        y = torch.cat([x, noise], dim=1)
        y = self.model(y)
        return y

    def _initialize_weights(self) -> None:
        for m in self.modules():
            classname = m.__class__.__name__
            if classname.find('Linear') != -1:
                m.weight.data.normal_(0.0, 0.02)
                m.bias.data.fill_(0)
            elif classname.find('BatchNorm') != -1:
                m.weight.data.normal_(1.0, 0.02)
                m.bias.data.fill_(0)


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

        self._initialize_weights()

    def forward(self, x):
        x = self.model(x).view(-1)
        return x

    def _initialize_weights(self) -> None:
        for m in self.modules():
            classname = m.__class__.__name__
            if classname.find('Linear') != -1:
                m.weight.data.normal_(0.0, 0.02)
                m.bias.data.fill_(0)
            elif classname.find('BatchNorm') != -1:
                m.weight.data.normal_(1.0, 0.02)
                m.bias.data.fill_(0)
