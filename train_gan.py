import torch
import torch.nn as nn
from torch.optim import Adamax as amax
import numpy as np

from nn_architecture.BicGAN import Generator, Discriminator

generator = Generator()
discriminator = Discriminator()

generator.load()
discriminator.load()


generator_optimizer = amax(generator.parameters(), lr=0.001)

discriminator_criterion = torch.nn.functional.binary_cross_entropy
discriminator_optimizer = amax(discriminator.parameters(), lr=0.00005)

valid_target_default = torch.ones(BATCH_SIZE, requires_grad=False).to(device)
fake_target_default = torch.zeros(BATCH_SIZE, requires_grad=False).to(device)