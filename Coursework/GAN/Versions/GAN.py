import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import functional as TF
import cv2


img_dir = "./Flickr.npy"

class FaceDataset(Dataset):
    def __init__(self, npy_file):
        self.data = np.load(npy_file)  #, mmap_mode='r' stays on disk, only loads slices

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = self.data[idx].astype(np.float32) / 127.5 - 1.0
        return torch.from_numpy(img)


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.noise_std = 0.1
        # start at (3, 128, 128)
        self.conv_layer1 = nn.Sequential(
            # (64, 64, 64)
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=False),
        )
        self.conv_layer2 = nn.Sequential(
            # (128, 32, 32)
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=False),
        )
        self.conv_layer3 = nn.Sequential(
            # (256, 16, 16)
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=False),
        )
        self.conv_layer4 = nn.Sequential(
            # (512, 8, 8)
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=False),
        )
        self.conv_layer5 = nn.Sequential(
            # (1024, 4, 4)
            nn.Conv2d(512, 1024, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=False),
            #nn.Dropout2d(0.3)
        )


        self.fc = nn.Sequential(
            nn.Conv2d(1024, 1, kernel_size=4, padding=0)
        )

    def forward(self, x):
        if self.training: x = x + torch.randn_like(x) * self.noise_std
        x = self.conv_layer1(x)
        if self.training: x = x + torch.randn_like(x) * self.noise_std
        x = self.conv_layer2(x)
        if self.training: x = x + torch.randn_like(x) * self.noise_std
        x = self.conv_layer3(x)
        if self.training: x = x + torch.randn_like(x) * self.noise_std
        x = self.conv_layer4(x)
        if self.training: x = x + torch.randn_like(x) * self.noise_std
        x = self.conv_layer5(x)
        return self.fc(x).view(-1)


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        ninputs = 100
        self.model = nn.Sequential(
            # Start with a 100-dim noise vector, project and reshape to (256, 6, 6)
            nn.Linear(ninputs, 1024 * 4 * 4, bias=False),
            nn.BatchNorm1d(1024 * 4 * 4),
            nn.ReLU(True),

            # start at (1024, 4, 4)
            nn.Unflatten(1, (1024, 4, 4)),

            # Upsample to (512, 8, 8)
            nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2, padding=1, bias=True),  # *2
            nn.BatchNorm2d(512),
            nn.ReLU(True),

            # Upsample to (256, 16, 16)
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=True),  # *2
            nn.BatchNorm2d(256),
            nn.ReLU(True),

            # Upsample to (128, 32, 32)
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=True),  # *2
            nn.BatchNorm2d(128),
            nn.ReLU(True),
        )
        self.model1 = nn.Sequential(
            # Upsample to (64, 64, 64)
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
        )
        self.model2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1, bias=True),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.model(x)
        x = TF.resize(x, (64, 64))
        x = self.model1(x)
        x = TF.resize(x, (128, 128))# Upsample to (3, 128, 128)
        return self.model2(x)


def show(img):
    # Display image
    img = (img + 1)*127.5
    img = img.clip(0, 255).astype(np.uint8)
    img = np.transpose(img, (1, 2, 0))
    plt.imshow(img, cmap='grey', vmin=0, vmax=255)
    plt.show()


def save_checkpoint(gen, dis, gen_opt, dis_opt, epoch, path):
    torch.save({
        'epoch': epoch,
        'gen_state_dict': gen.state_dict(),
        'dis_state_dict': dis.state_dict(),
        'gen_optimizer': gen_opt.state_dict(),
        'dis_optimizer': dis_opt.state_dict()
    }, path)


def load_checkpoint(gen, dis, gen_opt, dis_opt, path):
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


def trainNN(epochs=0, batch_size=16, lr=0.0002, save_time=1, save_dir='', slide=False, device='cuda' if torch.cuda.is_available() else 'cpu'):
    gen = Generator().to(device)
    def init_weights(m):
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
    gen.apply(init_weights)
    dis = Discriminator().to(device)
    criterion = nn.BCEWithLogitsLoss()
    dis_opt = torch.optim.Adam(dis.parameters(), lr=lr, betas=(0.5, 0.999), weight_decay=1e-4)
    gen_opt = torch.optim.Adam(gen.parameters(), lr=lr, betas=(0.5, 0.999))
    noise_dim = 100

    start_epoch = load_checkpoint(gen, dis, gen_opt, dis_opt, save_dir)

    if epochs>0:
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

                real_labels = (torch.ones_like(real_preds)*0.9).to(device)
                fake_labels = (torch.zeros_like(fake_preds)+0.1).to(device)

                real_loss = criterion(real_preds, real_labels)
                fake_loss = criterion(fake_preds, fake_labels)
                d_loss = real_loss + fake_loss
                d_loss.backward()
                dis_opt.step()

                # === Generator ===
                noise = torch.randn(batch_size, noise_dim, device=device)
                fake = gen(noise)

                fake_preds = dis(fake)
                gen_labels = torch.ones_like(fake_preds)
                g_loss = criterion(fake_preds, gen_labels)
                g_loss.backward()
                gen_opt.step()


            if (epoch + 1) % save_time == 0:
                save_checkpoint(gen, dis, gen_opt, dis_opt, epoch + 1, save_dir)
                folder_path = save_dir[:-4]
                if not os.path.exists(folder_path):
                    os.makedirs(folder_path)
                r = torch.randn(2, 100).to(device)
                im = gen(r*0.5).detach().cpu().numpy()[0]
                im = np.transpose(im, (1, 2, 0))  # shape: (218, 178, 3)
                im = ((im + 1)*127.5).clip(0, 255).astype(np.uint8)
                plt.imsave(f'{folder_path}/epoch{epoch+1}.png', im)
                print(f"Epoch {epoch+1} - D Loss: {d_loss.item():.4f}, G Loss: {g_loss.item():.4f}")

    if (not slide):
        gen.eval()
        for i in range(100):
            r = torch.randn(2, 100).to(device)
            im = gen(r).detach().cpu().numpy()[0]
            show(im)
    else:
        def nothing(x):
            pass

        cv2.namedWindow('image')
        # create trackbars for color change
        cv2.createTrackbar('im1', 'image', 0, 100, nothing)
        cv2.createTrackbar('im2', 'image', 0, 100, nothing)
        cv2.createTrackbar('im3', 'image', 0, 100, nothing)
        cv2.createTrackbar('im4', 'image', 0, 100, nothing)
        cv2.createTrackbar('im5', 'image', 0, 100, nothing)

        # create switch for ON/OFF functionality
        r1 = ((torch.randn(2, 100))).to(device)
        r2 = ((torch.randn(2, 100))).to(device)
        r3 = ((torch.randn(2, 100))).to(device)
        r4 = ((torch.randn(2, 100))).to(device)
        r5 = ((torch.randn(2, 100))).to(device)

        img = np.zeros((128, 128, 3), np.uint8)
        while (True):
            big_img = cv2.resize(img, (128 * 4, 128 * 4), interpolation=cv2.INTER_NEAREST)
            cv2.imshow('image', big_img)
            k = cv2.waitKey(1) & 0xFF
            if k == 27:  # Escape
                break

            # get current positions of four trackbars
            k = cv2.getTrackbarPos('im1', 'image') / 100000
            g = cv2.getTrackbarPos('im2', 'image') / 100000
            b = cv2.getTrackbarPos('im3', 'image') / 100000
            a = cv2.getTrackbarPos('im4', 'image') / 100000
            aa = cv2.getTrackbarPos('im5', 'image') / 100000

            img = (gen(k * r1 + g * r2 + b * r3 + a * r4 + aa * r5).detach().cpu().numpy()[1, :, :, :])
            img = ((img + 1) * 127.5).clip(0, 255).astype('uint8')
            img = np.transpose(img, (1, 2, 0))[:, :, ::-1]

        cv2.destroyAllWindows()





print("CUDA Available:", torch.cuda.is_available())
trainNN(0, 128, save_time=1, save_dir='bestGAN3.pth', slide=0)
