import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader

try:
    df = pd.read_csv("face_data.csv")
except FileNotFoundError:
    print("Oops! File not found. Please check the file path.")
    df = None

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        ninputs = 48 * 48
        self.model = nn.Sequential(
            nn.Conv2d(1, 32, (7,7), padding=(3,3)), # 32 X 48X48
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 32, (7, 7), padding=(3, 3)), # 32 X 48X48
            nn.LeakyReLU(0.2),
            nn.Dropout2d(0.3),
            nn.MaxPool2d((4,4)), # 32 X 12X12
            nn.Conv2d(32, 16, (3, 3), padding=(1, 1)), # 16 X 12X12
            nn.LeakyReLU(0.2),
            nn.Dropout2d(0.3),
            nn.Conv2d(16, 8, (3, 3), padding=(1, 1)),  # 8 X 12X12
            nn.LeakyReLU(0.2),
            nn.Dropout2d(0.3),
            nn.MaxPool2d((2, 2)),  # 8 X 6X6
            nn.Flatten(1),
            nn.Linear(8*6*6, 1),
        )

    def forward(self, x):
        return self.model(x)

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        ninputs = 100
        self.model = nn.Sequential(
            # Start with a 100-dim noise vector, project and reshape to (256, 6, 6)
            nn.Linear(100, 256 * 6 * 6),
            nn.ReLU(True),
            nn.BatchNorm1d(256 * 6 * 6),

            # Reshape to 256 x 6 x 6
            nn.Unflatten(1, (256, 6, 6)),

            # Upsample to 128 x 12 x 12
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(True),
            nn.BatchNorm2d(128),

            # Upsample to 64 x 24 x 24
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(True),
            nn.BatchNorm2d(64),

            # Upsample to 1 x 48 x 48
            nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()  # Output in range [0, 1]
        )

    def forward(self, x):
        return self.model(x)

def show(data):
    # Display image
    image = data.reshape((48, 48))
    plt.imshow(image, cmap='grey', vmin=0, vmax=255)
    plt.show()


# Save model
def save_checkpoint(gen, dis, gen_opt, dis_opt, epoch, path='checkpoint3.pth'):
    torch.save({
        'epoch': epoch,
        'gen_state_dict': gen.state_dict(),
        'dis_state_dict': dis.state_dict(),
        'gen_optimizer': gen_opt.state_dict(),
        'dis_optimizer': dis_opt.state_dict()
    }, path)

# Load model
def load_checkpoint(gen, dis, gen_opt, dis_opt, path='checkpoint3.pth'):
    if os.path.exists(path):
        if torch.cuda.is_available():
            checkpoint = torch.load(path)
        else:
            checkpoint = torch.load(path, map_location=torch.device('cpu'))
        gen.load_state_dict(checkpoint['gen_state_dict'])
        dis.load_state_dict(checkpoint['dis_state_dict'])
        gen_opt.load_state_dict(checkpoint['gen_optimizer'])
        dis_opt.load_state_dict(checkpoint['dis_optimizer'])
        print(f"Checkpoint loaded from epoch {checkpoint['epoch']}")
        return checkpoint['epoch']
    return 0

def trainNN(epochs=100, batch_size=16, lr=0.0002, save_time = 500, device='cuda' if torch.cuda.is_available() else 'cpu'):
    gen = Generator().to(device)
    dis = Discriminator().to(device)
    criterion = nn.BCEWithLogitsLoss()
    dis_opt = torch.optim.Adam(dis.parameters(), lr=lr, betas=(0.5, 0.999))
    gen_opt = torch.optim.Adam(gen.parameters(), lr=lr, betas=(0.5, 0.999))
    noise_dim = 100

    start_epoch = load_checkpoint(gen, dis, gen_opt, dis_opt)

    for epoch in range(start_epoch, start_epoch + epochs):
        real = df.sample(batch_size // 2).to_numpy() / 255
        real = torch.tensor(real, dtype=torch.float).view(-1, 1, 48, 48)
        real = real.to(device)
        noise = torch.randn(batch_size // 2, noise_dim, device=device)#.view(-1, 1, 10, 10)
        fake = gen(noise).view(-1, 1, 48, 48).detach()
        dis_opt.zero_grad()
        gen_opt.zero_grad()

        # Discriminator on real
        real_labels = torch.ones(batch_size // 2, 1).to(device)
        real_preds = dis(real)
        real_loss = criterion(real_preds, real_labels)

        # Discriminator on fake
        fake_labels = torch.zeros(batch_size // 2, 1).to(device)
        fake_preds = dis(fake)
        fake_loss = criterion(fake_preds, fake_labels)

        d_loss = real_loss + fake_loss
        d_loss.backward()
        dis_opt.step()

        # Train Generator
        noise = torch.randn(batch_size // 2, noise_dim, device=device)
        real_labels = torch.ones(batch_size // 2, 1).to(device)
        fake = gen(noise).view(-1, 1, 48, 48)

        # Generator wants discriminator to output 1
        fake_preds = dis(fake)
        g_loss = criterion(fake_preds, real_labels)
        g_loss.backward()
        gen_opt.step()

        if (epoch + 1) % save_time == 0:
            save_checkpoint(gen, dis, gen_opt, dis_opt, epoch)
            print(f"Epoch {epoch} - D Loss: {d_loss.item():.4f}, G Loss: {g_loss.item():.4f}")

    gen.eval()
    for i in range(100):
        r = torch.randn(2, 100).to(device)
        im = gen(r).detach().cpu().numpy()[0] * 255
        show(im)

print("CUDA Available:", torch.cuda.is_available())
trainNN(0, 16)
