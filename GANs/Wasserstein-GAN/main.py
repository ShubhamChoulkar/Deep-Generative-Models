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
    dataroot = "./data"
    output_dir = "./wgan_cifar10_output"
    workers = 2
    batch_size = 128
    image_size = 32
    nc = 3
    nz = 100
    ngf = 64
    ndf = 64
    num_epochs = 50
    lr = 0.00005
    ngpu = 1
    seed = 999
    print_freq = 100
    clip_value = 0.01
    critic_iters = 5

# 2. WEIGHT INITIALIZATION
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

# 3. GENERATOR
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
            nn.ConvTranspose2d(config.ngf * 2, config.ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(config.ngf),
            nn.ReLU(True),
            nn.ConvTranspose2d(config.ngf, config.nc, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, input):
        return self.main(input)

# 4. CRITIC (DISCRIMINATOR)
class Discriminator(nn.Module):
    def __init__(self, config):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(config.nc, config.ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(config.ndf, config.ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(config.ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(config.ndf * 2, config.ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(config.ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(config.ndf * 4, 1, 4, 1, 0, bias=False)
        )

    def forward(self, input):
        return self.main(input).view(-1)

# 5. TRAINING LOOP
if __name__ == '__main__':
    config = Config()
    os.makedirs(config.output_dir, exist_ok=True)
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    device = torch.device("cuda:0" if (torch.cuda.is_available() and config.ngpu > 0) else "cpu")

    # Data loading
    dataset = dset.CIFAR10(root=config.dataroot, download=True,
        transform=transforms.Compose([
            transforms.Resize(config.image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=config.batch_size,
                                             shuffle=True, num_workers=config.workers)

    # Models
    netG = Generator(config).to(device)
    netG.apply(weights_init)
    netD = Discriminator(config).to(device)
    netD.apply(weights_init)

    fixed_noise = torch.randn(64, config.nz, 1, 1, device=device)
    optimizerD = optim.RMSprop(netD.parameters(), lr=config.lr)
    optimizerG = optim.RMSprop(netG.parameters(), lr=config.lr)

    print("\n[INFO] Starting WGAN Training...")
    start_time = time.time()
    G_losses, D_losses = [], []

    for epoch in range(config.num_epochs):
        for i, data in enumerate(dataloader, 0):
            ############################
            # (1) Update Critic (D)
            ############################
            for _ in range(config.critic_iters):
                netD.zero_grad()
                real_data = data[0].to(device)
                b_size = real_data.size(0)

                # Real loss
                D_real = netD(real_data)
                D_real_loss = -D_real.mean()

                # Fake loss
                noise = torch.randn(b_size, config.nz, 1, 1, device=device)
                fake_data = netG(noise).detach()
                D_fake = netD(fake_data)
                D_fake_loss = D_fake.mean()

                D_loss = D_real_loss + D_fake_loss
                D_loss.backward()
                optimizerD.step()

                # Weight clipping
                for p in netD.parameters():
                    p.data.clamp_(-config.clip_value, config.clip_value)

            ############################
            # (2) Update Generator (G)
            ############################
            netG.zero_grad()
            noise = torch.randn(config.batch_size, config.nz, 1, 1, device=device)
            fake = netG(noise)
            G_loss = -netD(fake).mean()
            G_loss.backward()
            optimizerG.step()

            ############################
            # (3) Logging
            ############################
            if i % config.print_freq == 0:
                print(f"[{epoch+1}/{config.num_epochs}][{i}/{len(dataloader)}] "
                      f"D_loss: {D_loss.item():.4f} | G_loss: {G_loss.item():.4f} | "
                      f"D(real): {-D_real_loss.item():.4f} | D(fake): {D_fake_loss.item():.4f}")

                with torch.no_grad():
                    fake_grid = netG(fixed_noise).detach().cpu()
                vutils.save_image(fake_grid,
                    f"{config.output_dir}/epoch_{epoch+1:03d}_iter_{i:04d}.png",
                    normalize=True)

                G_losses.append(G_loss.item())
                D_losses.append(D_loss.item())

    # Final results
    total_time = time.time() - start_time
    print(f"\n[INFO] Training completed in {total_time/60:.2f} minutes.")

    with torch.no_grad():
        final_fake = netG(fixed_noise).detach().cpu()
    vutils.save_image(final_fake, f"{config.output_dir}/final_output.png", normalize=True)

    # Plot loss
    plt.figure(figsize=(10,5))
    plt.title("WGAN Loss During Training")
    plt.plot(G_losses,label="G")
    plt.plot(D_losses,label="D")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(f"{config.output_dir}/loss_curve.png")
    plt.show()
