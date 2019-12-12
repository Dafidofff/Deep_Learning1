import argparse
import os

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torchvision import datasets


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        # Construct generator. You are free to experiment with your model,
        # but the following is a good start:
        #   Linear args.latent_dim -> 128
        #   LeakyReLU(0.2)
        #   Linear 128 -> 256
        #   Bnorm
        #   LeakyReLU(0.2)
        #   Linear 256 -> 512
        #   Bnorm
        #   LeakyReLU(0.2)
        #   Linear 512 -> 1024
        #   Bnorm
        #   LeakyReLU(0.2)
        #   Linear 1024 -> 768
        #   Output non-linearity
        self.model = nn.Sequential(nn.Linear(args.latent_dim, 128), nn.LeakyReLU(0.2),
                                    nn.Linear(128, 256), nn.BatchNorm1d(256), nn.LeakyReLU(0.2),
                                    nn.Linear(256, 512), nn.BatchNorm1d(512), nn.LeakyReLU(0.2),
                                    nn.Linear(512, 1024), nn.BatchNorm1d(1024), nn.LeakyReLU(0.2),
                                    nn.Linear(1024, 784), nn.Tanh()) 

    def forward(self, z):
        # Generate images from z
        out = self.model(z)
        batch_size = out.shape[0]
        return out.view(batch_size, 1, 28, 28)


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        # Construct distriminator. You are free to experiment with your model,
        # but the following is a good start:
        #   Linear 784 -> 512
        #   LeakyReLU(0.2)
        #   Linear 512 -> 256
        #   LeakyReLU(0.2)
        #   Linear 256 -> 1
        #   Output non-linearity
        self.model = nn.Sequential(nn.Linear(784, 512), nn.LeakyReLU(0.2),
                                    nn.Linear(512, 256), nn.LeakyReLU(0.2),
                                    nn.Linear(256, 1), nn.Sigmoid())

    def forward(self, img):
        # return discriminator score for img
        batch_size, _, width, height = img.shape
        input_ = img.view(batch_size, width*height)
        return self.model(input_)


def train(dataloader, discriminator, generator, optimizer_G, optimizer_D):
    loss_function = nn.BCELoss().to(device)
    discriminator = discriminator.to(device)
    generator = generator.to(device)

    for epoch in range(args.n_epochs):
        for i, (imgs, _) in enumerate(dataloader):
            # imgs.cuda()
            device = args.device
            imgs.to(device)

            batch_size = imgs.shape[0]

            latent_vectors = torch.randn(batch_size, args.latent_dim).to(device)
            fake_images = generator.forward(latent_vectors)

            disc_fake_out = discriminator.forward(fake_images).to(device)
            disc_real_out = discriminator.forward(imgs).to(device)

            # Train Generator
            # ---------------
            optimizer_G.zero_grad()

            gen_loss = loss_function(disc_fake_out, torch.ones(batch_size, 1))
            gen_loss.backward(retain_graph=True)
            optimizer_G.step()
            
            
            # Train Discriminator
            # -------------------
            optimizer_D.zero_grad()

            disc_loss_real = loss_function(disc_real_out, torch.ones(batch_size, 1))
            disc_loss_fake = loss_function(disc_fake_out, torch.zeros(batch_size, 1))
            disc_total_loss = disc_loss_real + disc_loss_fake
            disc_total_loss.backward()
            optimizer_D.step()
            
            
            batches_done = epoch * len(dataloader) + i
            if batches_done % 200 == 0:
                print(f"[Epoch {epoch}] batch: {i} gen_loss: {gen_loss.item()} disc_loss: {disc_total_loss.item()}")
            
            # Save Images
            # ----------- 
            if batches_done % args.save_interval == 0:
                # You can use the function save_image(Tensor (shape Bx1x28x28),
                # filename, number of rows, normalize) to save the generated
                # images, e.g.:
                # save_image(fake_images[:25],
                #            'images/{}.png'.format(batches_done),
                #            nrow=5, normalize=True)
                pass


def main():
    # Create output image directory
    os.makedirs('gan_images', exist_ok=True)

    # load data
    dataloader = torch.utils.data.DataLoader(
        datasets.MNIST('./data/mnist', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.5,),(0.5,))])),
        batch_size=args.batch_size, shuffle=True)

    # Initialize models and optimizers
    generator = Generator()
    discriminator = Discriminator()
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=args.lr)
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=args.lr)

    # Start training
    train(dataloader, discriminator, generator, optimizer_G, optimizer_D)

    # You can save your generator here to re-use it to generate images for your
    # report, e.g.:
    # torch.save(generator.state_dict(), "mnist_generator.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_epochs', type=int, default=200,
                        help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='batch size')
    parser.add_argument('--lr', type=float, default=0.0002,
                        help='learning rate')
    parser.add_argument('--latent_dim', type=int, default=100,
                        help='dimensionality of the latent space')
    parser.add_argument('--save_interval', type=int, default=500,
                        help='save every SAVE_INTERVAL iterations')
    parser.add_argument('--device', type=str, default="cpu",
                        help='do not train a model, load one instead.')
    args = parser.parse_args()

    main()
