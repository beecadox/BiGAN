import os
import sys
from torch.optim import Adamax as amax
from nn_architecture.autoencoder import *
import time
import numpy as np
sys.path.append(os.getcwd())

import random

random.seed(0)
torch.manual_seed(0)

BATCH_SIZE = 32
def save_model(args, epoch):
    torch.save({'epoch': epoch + 1, 'args': args, 'state_dict': args.model.state_dict(),
                'optimizer': args.optimizer.state_dict()},
               os.path.join(args.odir, 'models/model_%d.pth.tar' % (epoch + 1)))


vae = VariationalAutoencoder()
vae.load()
optimizer = amax(vae.parameters(), lr=0.0001)  # https://pytorch.org/docs/stable/generated/torch.optim.Adamax.html

criterion = nn.CrossEntropyLoss

def kld_loss(mean, log_variance):
    return -0.5 * torch.sum(1 + log_variance - mean.pow(2) - log_variance.exp()) / mean.nelement()

def get_reconstruction_loss(input, target):
    difference = input - target
    wrong_signs = target < 0
    difference[wrong_signs] *= 32

    return torch.mean(torch.abs(difference))

def train():
    for epoch in count():
        batch_index = 0
        epoch_start_time = time.time()
        for batch in data_loader, desc='Epoch {:d}'.format(epoch):
            try:
                batch = batch.to(device)

                vae.zero_grad()
                vae.train()

                output, mean, log_variance = vae(batch)
                kld = kld_loss(mean, log_variance)

                reconstruction_loss = get_reconstruction_loss(output, batch)
                loss = reconstruction_loss + kld

                reconstruction_error_history.append(reconstruction_loss.item())
                kld_error_history.append(kld.item() if IS_VARIATIONAL else 0)

                loss.backward()
                optimizer.step()


                print("epoch " + str(epoch) + ", batch " + str(batch_index) \
                      + ', reconstruction loss: {0:.4f}'.format(reconstruction_loss.item()) \
                      + ' (average: {0:.4f}), '.format(np.mean(reconstruction_error_history)) \
                      + 'KLD loss: {0:.4f}'.format(np.mean(kld_error_history)))

                batch_index += 1

        vae.save()
        if epoch % 20 == 0:
            vae.save(epoch=epoch)

