import argparse

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision.utils import make_grid, save_image

from datasets.bmnist import bmnist

from torch.nn import functional as F
import math
from scipy.stats import norm


class Encoder(nn.Module):

    def __init__(self, hidden_dim=500, z_dim=20):
        super().__init__()

        self.encoder = nn.Sequential(nn.Linear(784, hidden_dim), nn.ReLU(),
                                    nn.Linear(hidden_dim, hidden_dim), nn.ReLU())

        self.mean_layer = nn.Linear(hidden_dim, z_dim)
        self.std_layer = nn.Linear(hidden_dim, z_dim)


    def forward(self, input):
        """
        Perform forward pass of encoder.

        Returns mean and std with shape [batch_size, z_dim]. Make sure
        that any constraints are enforced.
        """
        hidden_out = self.encoder(input)
        mean = self.mean_layer(hidden_out)
        std = self.std_layer(hidden_out)

        return mean, std


class Decoder(nn.Module):

    def __init__(self, hidden_dim=500, z_dim=20):
        super().__init__()
        self.decoder = nn.Sequential(nn.Linear(z_dim, hidden_dim), nn.ReLU(), 
                                        nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), 
                                        nn.Linear(hidden_dim, 784))

    def forward(self, input):
        """
        Perform forward pass of encoder.

        Returns mean with shape [batch_size, 784].
        """
        mean = self.decoder(input)

        return torch.sigmoid(mean)


class VAE(nn.Module):

    def __init__(self, hidden_dim=500, z_dim=20):
        super().__init__()

        self.z_dim = z_dim
        self.encoder = Encoder(hidden_dim, z_dim)
        self.decoder = Decoder(hidden_dim, z_dim)

    def forward(self, input, batch_size):
        """
        Given input, perform an encoding and decoding step and return the
        negative average elbo for the given batch.
        """

        mean, log_var = self.encoder(input)
        sampled_std = torch.randn_like(log_var)
        latent_vec = mean + (sampled_std * torch.exp(0.5 * log_var)) # multiplied with 0.5 while var is returned and std wanted

        out = self.decoder(latent_vec)
        recon_loss = F.binary_cross_entropy(out, input, reduction='sum')
        reg_loss = -0.5 * torch.sum(1 + log_var - torch.pow(mean, 2) - torch.exp(log_var))
        average_negative_elbo = recon_loss + reg_loss

        return average_negative_elbo / batch_size

    def sample(self, n_samples, random_latent_vec = None):
        """
        Sample n_samples from the model. Return both the sampled images
        (from bernoulli) and the means for these bernoullis (as these are
        used to plot the data manifold).
        """
        if random_latent_vec is None:
            random_latent_vec = torch.randn((n_samples, self.z_dim))

        im_means = self.decoder(random_latent_vec)
        sampled_ims = torch.bernoulli(im_means)

        return sampled_ims, im_means


def epoch_iter(model, data, optimizer):
    """
    Perform a single epoch for either the training or validation.
    use model.training to determine if in 'training mode' or not.

    Returns the average elbo for the complete epoch.
    """
    elbow = 0
    for batch in data:
        batch_size, _, _, _ = batch.shape
        batch = batch.view(batch_size, 784)

        out = model.forward(batch, batch_size)

        elbow += out/batch_size
        if model.training:
            optimizer.zero_grad()
            out.backward()
            optimizer.step()

    average_epoch_elbo = elbow/len(data)

    return average_epoch_elbo


def run_epoch(model, data, optimizer):
    """
    Run a train and validation epoch and return average elbo for each.
    """
    traindata, valdata = data

    model.train()
    train_elbo = epoch_iter(model, traindata, optimizer)

    model.eval()
    val_elbo = epoch_iter(model, valdata, optimizer)

    return train_elbo, val_elbo


def save_elbo_plot(train_curve, val_curve, filename):
    plt.figure(figsize=(12, 6))
    plt.plot(train_curve, label='train elbo')
    plt.plot(val_curve, label='validation elbo')
    plt.legend()
    plt.xlabel('epochs')
    plt.ylabel('ELBO')
    plt.tight_layout()
    plt.savefig(filename)


def main():
    data = bmnist()[:2]  # ignore test split
    model = VAE(z_dim=ARGS.zdim)
    optimizer = torch.optim.Adam(model.parameters())

    train_curve, val_curve = [], []
    for epoch in range(ARGS.epochs):
        elbos = run_epoch(model, data, optimizer)
        train_elbo, val_elbo = elbos
        train_curve.append(train_elbo)
        val_curve.append(val_elbo)
        print(f"[Epoch {epoch}] train elbo: {train_elbo} val_elbo: {val_elbo}")

        # --------------------------------------------------------------------
        #  Add functionality to plot samples from model during training.
        #  You can use the make_grid functioanlity that is already imported.
        # --------------------------------------------------------------------
        if epoch % 10 == 0:
            generated_samples = model.sample(n_samples=16)[0]
            generated_images = generated_samples.view(16, 1, 28, 28)
            save_image(generated_images, "vae_images/vae_epoch_{}.png".format(epoch))
    # --------------------------------------------------------------------
    #  Add functionality to plot plot the learned data manifold after
    #  if required (i.e., if zdim == 2). You can use the make_grid
    #  functionality that is already imported.
    # --------------------------------------------------------------------
    if ARGS.zdim == 2:
        dens = 15
        x_vals = torch.linspace(.01, .99, dens)
        y_vals = torch.linspace(.01, .99, dens)

        samples = torch.empty(dens*dens, 1, 28, 28)
        for i, y in enumerate(x_vals):
            for j, x in enumerate(y_vals):
                random_latent_vec = torch.tensor([[norm.ppf(x), norm.ppf(y)]])
                samples[i*dens + j,:,:,:] = model.sample(random_latent_vec=random_latent_vec)[1].view(1, 1, 28, 28)
        
        save_image(samples.view(dens*dens, 1, 28, 28), "manifold.png", nrow=dens)


    save_elbo_plot(train_curve, val_curve, 'elbo.pdf')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', default=80, type=int,
                        help='max number of epochs')
    parser.add_argument('--zdim', default=20, type=int,
                        help='dimensionality of latent space')

    ARGS = parser.parse_args()

    main()
