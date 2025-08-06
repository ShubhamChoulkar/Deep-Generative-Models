
import os
import random
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt


# 1. CONFIGURATION

class Config:
    # Centralized configuration for all settings.
    dataroot = "./data"
    output_dir = "./dcgan_cifar10_output" 
    workers = 2
    batch_size = 128
    image_size = 32
    nc = 3      # Number of channels in the training images.
    nz = 100    # Size of the latent z vector.
    ngf = 64    # Size of feature maps in the generator.
    ndf = 64    # Size of feature maps in the discriminator.
    num_epochs = 50
    lr = 0.0002
    beta1 = 0.5
    ngpu = 1
    seed = 999 # just a random seed for reproducibility.
    print_freq = 100 # How often to print stats and save images (in iterations).

# 2. MODEL DEFINITIONS

# Custom weights initialization for the models.
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

# The Generator Network.
class Generator(nn.Module):
    def __init__(self, config):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(config.nz, config.ngf * 4, 4, 1, 0, bias=False),
            nn.BatchNorm2d(config.ngf * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(config.ngf * 4, config.ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(config.ngf * 2),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.ConvTranspose2d(config.ngf * 2, config.ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(config.ngf),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.ConvTranspose2d(config.ngf, config.nc, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, input):
        return self.main(input)

# The Discriminator Network.
class Discriminator(nn.Module):
    def __init__(self, config):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(config.nc, config.ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(config.ndf, config.ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(config.ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5),
            nn.Conv2d(config.ndf * 2, config.ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(config.ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5),
            nn.Conv2d(config.ndf * 4, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)


# 3. MAIN EXECUTION BLOCK

if __name__ == '__main__':
    # --- Setup ---
    config = Config()
    print("--- PyTorch DCGAN Project ---")
    print(f"Random Seed: {config.seed}")
    random.seed(config.seed)
    torch.manual_seed(config.seed)

    device = torch.device("cuda:0" if (torch.cuda.is_available() and config.ngpu > 0) else "cpu")
    print(f"Using device: {device}")

    # Create the single output directory.
    os.makedirs(config.output_dir, exist_ok=True)
    print(f"All results will be saved to: {config.output_dir}")

    # --- Data Loading ---
    dataset = dset.CIFAR10(root=config.dataroot, download=True,
                           transform=transforms.Compose([
                               transforms.Resize(config.image_size),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=config.batch_size,
                                             shuffle=True, num_workers=config.workers)

    # --- Model Initialization ---
    netG = Generator(config).to(device)
    netG.apply(weights_init)
    netD = Discriminator(config).to(device)
    netD.apply(weights_init)

    # --- Loss, Optimizers, and Fixed Noise ---
    criterion = nn.BCELoss()
    fixed_noise = torch.randn(64, config.nz, 1, 1, device=device)
    real_label, fake_label = 0.9, 0.1 # Using label smoothing for stability.

    optimizerD = optim.Adam(netD.parameters(), lr=config.lr, betas=(config.beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=config.lr, betas=(config.beta1, 0.999))

    # --- Training Loop ---
    print("\nStarting Training Loop...")
    start_time = time.time()
    G_losses, D_losses = [], []

    for epoch in range(config.num_epochs):
        for i, data in enumerate(dataloader, 0):

            # (1) Update Discriminator
            netD.zero_grad()
            real_images = data[0].to(device)
            b_size = real_images.size(0)
            
            # Train with real images.
            label = torch.full((b_size,), real_label, device=device)
            output = netD(real_images).view(-1)
            errD_real = criterion(output, label)
            errD_real.backward()
            D_x = output.mean().item()

            # Train with fake images.
            noise = torch.randn(b_size, config.nz, 1, 1, device=device)
            fake_images = netG(noise)
            label.fill_(fake_label)
            output = netD(fake_images.detach()).view(-1)
            errD_fake = criterion(output, label)
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            
            errD = errD_real + errD_fake
            optimizerD.step()

            # (2) Update Generator
            netG.zero_grad()
            label.fill_(1.0) # Generator wants D to think fakes are real.
            output = netD(fake_images).view(-1)
            errG = criterion(output, label)
            errG.backward()
            D_G_z2 = output.mean().item()
            optimizerG.step()

            # --- Logging and Visualization ---
            if i % config.print_freq == 0:
                print(f'[{epoch+1}/{config.num_epochs}][{i}/{len(dataloader)}] | '
                      f'Loss_D: {errD.item():.4f} | Loss_G: {errG.item():.4f} | '
                      f'D(x): {D_x:.4f} | D(G(z)): {D_G_z1:.4f} / {D_G_z2:.4f}')
                
                # Save a grid of generated images to the output folder.
                with torch.no_grad():
                    fake_grid = netG(fixed_noise).detach().cpu()
                vutils.save_image(fake_grid,
                                  f"{config.output_dir}/epoch_{epoch+1:03d}_iter_{i:04d}.png",
                                  normalize=True)

            G_losses.append(errG.item())
            D_losses.append(errD.item())

    # --- Finalization ---
    total_time = time.time() - start_time
    print(f"\n--- Training Finished ---")
    print(f"Total Training Time: {total_time/60:.2f} minutes")

    # Save the final generated image grid.
    with torch.no_grad():
        final_fake = netG(fixed_noise).detach().cpu()
    vutils.save_image(final_fake, f"{config.output_dir}/final_output.png", normalize=True)
    print(f"Final image grid saved to '{config.output_dir}/final_output.png'")

    # Plot final loss curve and save it.
    plt.figure(figsize=(10,5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(G_losses,label="G")
    plt.plot(D_losses,label="D")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(f"{config.output_dir}/loss_curve.png")
    print(f"Loss curve plot saved to '{config.output_dir}/loss_curve.png'")
    plt.show()