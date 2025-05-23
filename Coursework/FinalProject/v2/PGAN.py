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
    def __init__(self, in_channels, out_channels):
        super(D_Block, self).__init__()

        self.model = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True),
            nn.LeakyReLU(0.2, inplace=False),
            nn.AvgPool2d(kernel_size=2, stride=2),
        )

    def forward(self, x):
        return self.model(x)


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.noise_std = 0.1
        self.depth = 0
        self.alpha = 1
        self.grow_rate = 0
        self.downsample = nn.AvgPool2d(kernel_size=2, stride=2)

        self.from_rgbs = nn.ModuleList(
            [
                nn.Sequential( # 8x8 -> 4x4
                    nn.Conv2d(3, 1024, kernel_size=3, stride=2, padding=1),
                    nn.LeakyReLU(0.2, inplace=False)
                ),
                nn.Sequential( # 16x16 -> 8x8
                    nn.Conv2d(3, 512, kernel_size=3, stride=2, padding=1),
                    nn.LeakyReLU(0.2, inplace=False)
                ),
                nn.Sequential( # 32x32 -> 16x16
                    nn.Conv2d(3, 256, kernel_size=3, stride=2, padding=1),
                    nn.LeakyReLU(0.2, inplace=False)
                ),
                nn.Sequential( # 64x64 -> 32x32
                    nn.Conv2d(3, 128, kernel_size=3, stride=2, padding=1),
                    nn.LeakyReLU(0.2, inplace=False)
                ),
                nn.Sequential( # 128x128 -> 64x64
                    nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
                    nn.LeakyReLU(0.2, inplace=False)
                ),
            ]
        )

        self.layers = nn.ModuleList(
            [
                nn.Identity(), # 4x4 -> 4x4
                nn.Sequential( # 8x8 -> 4x4
                    nn.Conv2d(512, 1024, kernel_size=3, stride=2, padding=1),
                    nn.LeakyReLU(0.2, inplace=False),
                ),
                nn.Sequential( # 16x16 -> 8x8
                    nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
                    nn.LeakyReLU(0.2, inplace=False),
                ),
                nn.Sequential( # 32x32 -> 16x16
                    nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
                    nn.LeakyReLU(0.2, inplace=False),
                ),
                nn.Sequential( # 64x64 -> 32x32
                    nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
                    nn.LeakyReLU(0.2, inplace=False),
                ),
            ]
        )

        self.map_output = nn.Sequential(nn.Conv2d(1024, 1, kernel_size=4, padding=0))

    def grow_network(self, num_iters):
        self.grow_rate = 1 / num_iters
        self.alpha = self.grow_rate
        self.depth += 1

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
    def __init__(self, in_channels, out_channels):
        super(G_Block, self).__init__()

        self.model = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
            # nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True),
            # nn.ReLU(True),
        )

    def forward(self, x):
        # sz = int(x.shape[-1] * 2)
        # x = TF.resize(x, (sz, sz))
        return self.model(x)


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.depth = 0
        self.alpha = 1
        self.grow_rate = 0
        # self.upsample = nn.Upsample(scale_factor=2, mode="nearest")

        self.map_noise = nn.Sequential(
            nn.Linear(100, 1024 * 4 * 4, bias=False),
            nn.BatchNorm1d(1024 * 4 * 4),
            nn.ReLU(True),
            nn.Unflatten(1, (1024, 4, 4)),
        )

        self.layers = nn.ModuleList(
            [
                G_Block(1024, 512),
                G_Block(512, 256),
                G_Block(256, 128),
                G_Block(128, 64),
                G_Block(64, 32),
                # nn.Sequential( # 4x4 -> 8x8
                #     nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2, padding=1, bias=True),
                #     nn.BatchNorm2d(512),
                #     nn.ReLU(True),
                # ),
                # nn.Sequential( # 8x8 -> 16x16
                #     nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=True),
                #     nn.BatchNorm2d(256),
                #     nn.ReLU(True),
                # ),
                # nn.Sequential( # 16x16 -> 32x32
                #     nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=True),
                #     nn.BatchNorm2d(128),
                #     nn.ReLU(True),
                # ),
                # nn.Sequential( # 32x32 -> 64x64
                #     nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=True),
                #     nn.BatchNorm2d(64),
                #     nn.ReLU(True),
                # )
            ]
        )

        self.to_rgbs = nn.ModuleList(
            [
                nn.Conv2d(512, 3, kernel_size=3, stride=1, padding=1, bias=True),
                nn.Conv2d(256, 3, kernel_size=3, stride=1, padding=1, bias=True),
                nn.Conv2d(128, 3, kernel_size=3, stride=1, padding=1, bias=True),
                nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1, bias=True),
                nn.Conv2d(32, 3, kernel_size=3, stride=1, padding=1, bias=True),
            ]
        )

    def load_state(self, state):
        if state is None:
            return

        self.depth = state["depth"]
        self.alpha = state["alpha"]
        self.grow_rate = state["grow_rate"]

    def save_state(self):
        return {"depth": self.depth, "alpha": self.alpha, "grow_rate": self.grow_rate}

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)

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
        avg = 0

        if self.alpha < 1:
            # Blend with previous layer
            # x_old = self.upsample(x)
            sz = int(x.shape[-1] * 2)
            x_old = TF.resize(x, (sz, sz))
            old_rgb = self.to_rgbs[self.depth - 1](x_old)
            avg = self.alpha * F.tanh(x_rgb) + (1 - self.alpha) * F.tanh(old_rgb)
            self.alpha += self.grow_rate

        return avg


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
            "gen_state": gen.save_state(),
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
        gen.load_state(checkpoint["gen_state"])
        print(f"Checkpoint loaded from epoch {checkpoint['epoch']}")
        return checkpoint["epoch"]
    print("Checkpoint not found")
    return 0


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
    gen.init_weights()

    dis_opt = torch.optim.Adam(dis.parameters(), lr=lr, betas=(0.5, 0.999), weight_decay=1e-4)
    gen_opt = torch.optim.Adam(gen.parameters(), lr=lr, betas=(0.5, 0.999))
    start_epoch = load_checkpoint(gen, dis, gen_opt, dis_opt, save_dir)

    # Update image_resize
    image_resize = 8 * 2**gen.depth
    # epoch_growth_stops = [50, 100, 120, 140, 160]  # 8, 16, 32, 64, 128
    # epoch_growth_stops = [50, 80, 95, 105, 110]  # 8, 16, 32, 64, 128
    # epoch_growth_stops = [20, 30, 40, 45, 50]  # 8, 16, 32, 64, 128
    epoch_growth_stops = [5, 10, 15, 20, 25]  # 8, 16, 32, 64, 128
    final_size = 128

    print(f"Image size: {image_resize}x{image_resize}")

    if epochs > 0:
        dataset = FaceDataset(img_dir)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)

        for epoch in range(start_epoch, start_epoch + epochs):
            if gen.depth < len(epoch_growth_stops) and epoch_growth_stops[gen.depth] and epoch == epoch_growth_stops[gen.depth]:
                # Grow network
                num_iters = 1.5 * (len(dataset) // batch_size) * epoch_growth_stops[gen.depth]
                gen.grow_network(num_iters)
                dis.grow_network(num_iters)
                image_resize *= 2
                print(f"Growing network to {image_resize}x{image_resize}")

            i = 0
            for real in loader:
                real = real.to(device, non_blocking=True)
                real = F.interpolate(real, (image_resize, image_resize), mode="nearest")

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

                i += 1
                if i % 10 == 0:
                    print(f"Batch {i:4} / {len(loader)} - D Loss: {d_loss.item():.4f}, G Loss: {g_loss.item():.4f}, Alpha: {gen.alpha:.6f}")

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

    gen = trainNN(160, 128, save_time=1, save_dir="PGAN13.pth")
    gen.eval()

    # slider_window(gen)
    display_random_samples(gen, 100)
