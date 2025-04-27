import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import functional as TF
import cv2
import helper


img_dir = "./Flickr.npy"


class FaceDataset(Dataset):
    def __init__(self, npy_file):
        self.data = np.load(npy_file)  # , mmap_mode='r' stays on disk, only loads slices

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = self.data[idx].astype(np.float32) / 127.5 - 1.0
        return torch.from_numpy(img)


class D_Block(nn.Module):
    def __init__(self, in_channels, out_channels, end=False):
        super(D_Block, self).__init__()

        if end:
            self.model = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True),
                nn.LeakyReLU(0.2, inplace=False),
                nn.Dropout2d(0.3),
            )
        else:
            self.model = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=True),
                nn.LeakyReLU(0.2, inplace=False),
            )


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.noise_std = 0.1
        self.depth = 0
        self.alpha = 1
        self.grow_rate = 0
        self.downsample = nn.AvgPool2d(kernel_size=2, stride=2)

        self.layers = nn.ModuleList(
            [
                D_Block(512, 512),  # 16x16 -> 8x8
                D_Block(512, 512),  # 32x32 -> 16x16
                D_Block(256, 512),  # 64x64 -> 32x32
                D_Block(128, 256, True),  # 128x128 -> 64x64
            ]
        )

        self.from_rgbs = nn.ModuleList(
            [
                nn.Conv2d(3, 512, kernel_size=3, stride=1, padding=1, bias=True),
                nn.Conv2d(3, 512, kernel_size=3, stride=1, padding=1, bias=True),
                nn.Conv2d(3, 512, kernel_size=3, stride=1, padding=1, bias=True),
                nn.Conv2d(3, 256, kernel_size=3, stride=1, padding=1, bias=True),
            ]
        )

        self.map_output = nn.Sequential(nn.Conv2d(512, 1, kernel_size=8, stride=1, padding=0, bias=True))

    def forward(self, x_rgb):
        if self.training:
            x_rgb = x_rgb + torch.randn_like(x_rgb) * self.noise_std

        x = self.from_rgbs[self.depth](x_rgb)
        x = self.layers[self.depth](x)

        if self.alpha < 1:
            # Blend with previous layer
            x_rgb = self.downsample(x_rgb)
            x_old = self.from_rgbs[self.depth - 1](x_rgb)
            x = self.alpha * x + (1 - self.alpha) * x_old
            self.alpha += self.grow_rate

        for i in range(self.depth, 0, -1):
            if self.training:
                x = x + torch.randn_like(x) * self.noise_std
            x = self.layers[i - 1](x)

        x = self.map_output(x)
        x = x.view(-1)

        return x


class G_Block(nn.Module):
    def __init__(self, in_channels, out_channels, start=False):
        super(G_Block, self).__init__()

        if start:
            self.model = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(True),
            )
        else:
            self.model = nn.Sequential(
                nn.Upsample(scale_factor=2, mode="nearest"),
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(True),
            )

    def forward(self, x):
        return self.model(x)


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.depth = 0
        self.alpha = 1
        self.grow_rate = 0
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")

        self.map_noise = nn.Sequential(
            nn.Linear(100, 512 * 8 * 8, bias=False),
            nn.BatchNorm1d(512 * 8 * 8),
            nn.ReLU(True),
            nn.Unflatten(1, (512, 8, 8)),
        )

        self.layers = nn.ModuleList(
            [
                G_Block(512, 512, True),  # 8x8 -> 16x16
                G_Block(512, 512),  # 16x16 -> 32x32
                G_Block(512, 256),  # 32x32 -> 64x64
                G_Block(256, 128),  # 64x64 -> 128x128
            ]
        )

        self.to_rgbs = nn.ModuleList(
            [
                nn.Conv2d(512, 3, kernel_size=3, stride=1, padding=1, bias=True),
                nn.Conv2d(512, 3, kernel_size=3, stride=1, padding=1, bias=True),
                nn.Conv2d(256, 3, kernel_size=3, stride=1, padding=1, bias=True),
                nn.Conv2d(128, 3, kernel_size=3, stride=1, padding=1, bias=True),
            ]
        )

    def grow_network(self, num_iters):
        self.grow_rate = 1 / num_iters
        self.alpha = self.grow_rate
        self.depth += 1

    def forward(self, x):
        x = self.map_noise(x)

        for i in range(self.depth):
            x = self.layers[i](x)

        out = self.layers[self.depth](x)
        x_rgb = self.to_rgbs[self.depth](out)

        if self.alpha < 1:
            # Blend with previous layer
            x_old = self.upsample(out)
            old_rgb = self.to_rgbs[self.depth - 1](x_old)
            x_rgb = self.alpha * x_rgb + (1 - self.alpha) * old_rgb
            self.alpha += self.grow_rate

        return F.tanh(x_rgb)


def show(img):
    # Display image
    img = (img + 1) * 127.5
    img = img.clip(0, 255).astype(np.uint8)
    img = np.transpose(img, (1, 2, 0))
    plt.imshow(img, cmap="grey", vmin=0, vmax=255)
    plt.show()


def save_checkpoint(gen, dis, gen_opt, dis_opt, epoch, path):
    torch.save(
        {
            "epoch": epoch,
            "gen_state_dict": gen.state_dict(),
            "dis_state_dict": dis.state_dict(),
            "gen_optimizer": gen_opt.state_dict(),
            "dis_optimizer": dis_opt.state_dict(),
        },
        path,
    )


def load_checkpoint(gen, dis, gen_opt, dis_opt, path):
    if os.path.exists(path):
        if torch.cuda.is_available():
            checkpoint = torch.load(path, weights_only=True)
        else:
            checkpoint = torch.load(path, map_location=torch.device("cpu"), weights_only=True)
        gen.load_state_dict(checkpoint["gen_state_dict"])
        dis.load_state_dict(checkpoint["dis_state_dict"])
        gen_opt.load_state_dict(checkpoint["gen_optimizer"])
        dis_opt.load_state_dict(checkpoint["dis_optimizer"])
        print(f"Checkpoint loaded from epoch {checkpoint['epoch']}")
        return checkpoint["epoch"]
    print("Checkpoint not found")
    return 0


def init_weights(gen):
    def init_weights_helper(m):
        # If the submodule is nn.Conv2d or nn.ConvTranspose2d
        # set weights to normal distribution with mean=0 and std=0.02
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    # Apply the function to all submodules of the generator (i.e. all layers)
    gen.apply(init_weights_helper)


def slider_window(gen):
    def nothing(x):
        pass

    cv2.namedWindow("image")
    cv2.createTrackbar("im1", "image", 50, 100, nothing)
    cv2.createTrackbar("im2", "image", 50, 100, nothing)
    cv2.createTrackbar("im3", "image", 50, 100, nothing)
    cv2.createTrackbar("im4", "image", 50, 100, nothing)
    cv2.createTrackbar("im5", "image", 50, 100, nothing)

    r1 = ((torch.randn(2, 100))).to(device)
    r2 = ((torch.randn(2, 100))).to(device)
    r3 = ((torch.randn(2, 100))).to(device)
    r4 = ((torch.randn(2, 100))).to(device)
    r5 = ((torch.randn(2, 100))).to(device)
    r6 = ((torch.randn(2, 100))).to(device) * 0.2
    r7 = ((torch.randn(2, 100))).to(device) * 0.2
    r8 = ((torch.randn(2, 100))).to(device) * 0.2
    r9 = ((torch.randn(2, 100))).to(device) * 0.2

    img = np.zeros((128, 128, 3), np.uint8)

    while True:
        big_img = cv2.resize(img, (128 * 4, 128 * 4), interpolation=cv2.INTER_NEAREST)
        cv2.imshow("image", big_img)
        k = cv2.waitKey(1) & 0xFF
        if k == 27:  # Escape
            break
        if k == ord("s"):  # Save
            cv2.imwrite("output.png", img)

        v1 = cv2.getTrackbarPos("im1", "image") / 100 - 0.5
        v2 = cv2.getTrackbarPos("im2", "image") / 100 - 0.5
        v3 = cv2.getTrackbarPos("im3", "image") / 100 - 0.5
        v4 = cv2.getTrackbarPos("im4", "image") / 100 - 0.5
        v5 = cv2.getTrackbarPos("im5", "image") / 100 - 0.5

        bias = r6 + r7 + r8 + r9
        noise = v1 * r1 + v2 * r2 + v3 * r3 + v4 * r4 + v5 * r5 + bias
        img = gen(noise).detach().cpu().numpy()[1, :, :, :]
        img = ((img + 1) * 127.5).clip(0, 255).astype("uint8")
        img = np.transpose(img, (1, 2, 0))[:, :, ::-1]  # CWH -> WHC and RGB -> BGR

    cv2.destroyAllWindows()


def display_random_samples(gen, num_samples=10):
    noise_dim = 100
    r = torch.randn(max(num_samples, 2), noise_dim).to(device)
    im = gen(r).detach().cpu().numpy()
    for i in range(num_samples):
        show(im[i])


def trainNN(epochs=0, batch_size=16, lr=0.0002, save_time=1, save_dir=""):
    noise_dim = 100

    gen = Generator().to(device)
    dis = Discriminator().to(device)
    lossfn = nn.BCEWithLogitsLoss()

    init_weights(gen)

    dis_opt = torch.optim.Adam(dis.parameters(), lr=lr, betas=(0.5, 0.999), weight_decay=1e-4)
    gen_opt = torch.optim.Adam(gen.parameters(), lr=lr, betas=(0.5, 0.999))
    start_epoch = load_checkpoint(gen, dis, gen_opt, dis_opt, save_dir)

    if epochs > 0:
        dataset = FaceDataset(img_dir)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)

        for epoch in range(start_epoch, start_epoch + epochs):
            for real in loader:
                real = real.to(device, non_blocking=True)

                # === Discriminator ===
                noise = torch.randn(batch_size, noise_dim, device=device)
                fake = gen(noise).detach()

                dis_opt.zero_grad()
                gen_opt.zero_grad()

                real_preds = dis(real)
                fake_preds = dis(fake)

                real_labels = (torch.ones_like(real_preds) * 0.9).to(device)
                fake_labels = (torch.zeros_like(fake_preds) + 0.1).to(device)

                real_loss = lossfn(real_preds, real_labels)
                fake_loss = lossfn(fake_preds, fake_labels)
                d_loss = real_loss + fake_loss
                d_loss.backward()
                dis_opt.step()

                # === Generator ===
                noise = torch.randn(batch_size, noise_dim, device=device)
                fake = gen(noise)

                fake_preds = dis(fake)
                gen_labels = torch.ones_like(fake_preds)
                g_loss = lossfn(fake_preds, gen_labels)
                g_loss.backward()
                gen_opt.step()

            if (epoch + 1) % save_time == 0:
                save_checkpoint(gen, dis, gen_opt, dis_opt, epoch + 1, save_dir)
                folder_path = save_dir[:-4]
                if not os.path.exists(folder_path):
                    os.makedirs(folder_path)
                r = torch.randn(2, 100).to(device)
                im = gen(r * 0.5).detach().cpu().numpy()[0]
                im = np.transpose(im, (1, 2, 0))
                im = ((im + 1) * 127.5).clip(0, 255).astype(np.uint8)
                plt.imsave(f"{folder_path}/epoch{epoch+1}.png", im)
                print(f"Epoch {epoch+1} - D Loss: {d_loss.item():.4f}, G Loss: {g_loss.item():.4f}")

    return gen


if __name__ == "__main__":
    import multiprocessing

    multiprocessing.freeze_support()  # Optional but recommended on Windows
    print("CUDA Available:", torch.cuda.is_available())

    global device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    gen = trainNN(0, 128, save_time=1, save_dir="PGAN.pth")
    gen.eval()

    # slider_window(gen)
    display_random_samples(gen, 100)
