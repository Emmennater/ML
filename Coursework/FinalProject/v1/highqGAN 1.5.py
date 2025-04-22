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


img_dir = "./data.npy"

class FaceDataset(Dataset):
    def __init__(self, npy_file):
        self.data = np.load(npy_file)  #, mmap_mode='r' stays on disk, only loads slices

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = self.data[idx].astype(np.float32) / 255.0  # convert only the needed image
        return torch.from_numpy(img)



class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.noise_std = 0.1
        self.conv_layer1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=7, stride=2, padding=3),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(32),
        )
        self.conv_layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(64),
        )
        self.conv_layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(128),
        )
        self.conv_layer4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=5, stride=2, padding=2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(256),
            nn.Dropout2d(0.3)
        )


        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(43008, 1)
        )

    def forward(self, x):
        x = self.conv_layer1(x)
        if self.training: x += torch.randn_like(x) * self.noise_std
        x = self.conv_layer2(x)
        #if self.training: x += torch.randn_like(x) * self.noise_std
        x = self.conv_layer3(x)
        if self.training: x += torch.randn_like(x) * self.noise_std
        x = self.conv_layer4(x)
        return self.fc(x)



class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        ninputs = 100
        self.model = nn.Sequential(
            # Start with a 100-dim noise vector, project and reshape to (256, 6, 6)
            nn.Linear(ninputs, 256 * 13 * 11),
            nn.ReLU(True),
            #nn.BatchNorm1d(256 * 6 * 6),

            # Reshape to 256 x 13 x 11
            nn.Unflatten(1, (256, 13, 11)),

            # Upsample to (128, 27, 22)
            nn.ConvTranspose2d(256, 128, kernel_size=(5,4), stride=2, padding=1),  # *2
            # nn.Upsample(scale_factor=2, mode='bilinear'),
            # nn.Conv2d(256, 128, kernel_size=(4,3), stride=1, padding=(2,1)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(128),

            # Upsample to (64, 54, 44)
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # *2
            # nn.Upsample(scale_factor=2, mode='bilinear'),  # or 'bilinear'
            # nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),  # *2
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(64),

            # Upsample to (32, 109, 89)
            # nn.ConvTranspose2d(64, 32, kernel_size=5, stride=2, padding=1),
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(64, 32, kernel_size=4, stride=1, padding=2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(32),

            # Final upsample to (3, 218, 178)
            nn.Upsample(scale_factor=2, mode='bilinear'),  # or 'bilinear'
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),  # *2
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 3, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()  # Output in range [0, 1]
        )

    def forward(self, x):
        return self.model(x)


def show(img):
    # Display image
    #image = img.reshape((48, 48))
    img = np.array(img, dtype=np.uint8)
    img = np.transpose(img, (1, 2, 0))
    plt.imshow(img, cmap='grey', vmin=0, vmax=255)
    plt.show()

    # img = np.array(img, dtype=np.uint8)
    # #img_transposed = np.transpose(img, (1, 2, 0))
    # print(img.shape, img.dtype)
    # img = cv2.resize(img, (300, 300), interpolation=cv2.INTER_NEAREST)
    # cv2.imshow('Image', img[0,:,:])
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()



def get_images(n):
    ims = np.empty((n, 3, 218, 178))
    for i in range(n):
        rand_path = np.random.choice(im_files)
        ims[i,:,:,:] = cv2.imread(f"{img_dir}/{rand_path}").transpose((2, 0, 1))
    return torch.tensor(ims / 255, dtype=torch.float).view(-1, 3, 218, 178)
    # return torch.tensor(df.sample(n).to_numpy() / 255, dtype=torch.float).view(-1, 1, 48, 48)

# Save model
def save_checkpoint(gen, dis, gen_opt, dis_opt, epoch, path):
    torch.save({
        'epoch': epoch,
        'gen_state_dict': gen.state_dict(),
        'dis_state_dict': dis.state_dict(),
        'gen_optimizer': gen_opt.state_dict(),
        'dis_optimizer': dis_opt.state_dict()
    }, path)

# Load model
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


def trainNN(epochs=0, batch_size=16, lr=0.0002, save_time=500, save_dir='', device='cuda' if torch.cuda.is_available() else 'cpu'):
    gen = Generator().to(device)
    dis = Discriminator().to(device)
    criterion = nn.BCEWithLogitsLoss()
    dis_opt = torch.optim.Adam(dis.parameters(), lr=lr, betas=(0.5, 0.999), weight_decay=1e-4)
    gen_opt = torch.optim.Adam(gen.parameters(), lr=lr, betas=(0.5, 0.999))
    noise_dim = 100

    start_epoch = load_checkpoint(gen, dis, gen_opt, dis_opt, save_dir)

    if epochs>0:
        dataset = FaceDataset(img_dir)
        loader = DataLoader(dataset, batch_size=batch_size // 2, shuffle=True, num_workers=0, pin_memory=True)

        for epoch in range(start_epoch, start_epoch + epochs):
            #i = 0
            for real in loader:
                real = real.to(device, non_blocking=True)
                # print(i)
                # i += 1
                # === Discriminator ===
                noise = torch.randn(batch_size // 2, noise_dim, device=device)
                fake = gen(noise).detach()

                dis_opt.zero_grad()
                gen_opt.zero_grad()

                real_preds = dis(real)
                fake_preds = dis(fake)

                real_labels = (torch.ones_like(real_preds)*0.9).to(device)
                fake_labels = torch.zeros_like(fake_preds).to(device)

                real_loss = criterion(real_preds, real_labels)
                fake_loss = criterion(fake_preds, fake_labels)
                d_loss = real_loss + fake_loss
                d_loss.backward()
                dis_opt.step()

                # === Generator ===
                noise = torch.randn(batch_size // 2, noise_dim, device=device)
                fake = gen(noise)

                fake_preds = dis(fake)
                gen_labels = torch.ones_like(fake_preds)
                g_loss = criterion(fake_preds, gen_labels)
                g_loss.backward()
                gen_opt.step()
                #print(f"Epoch {epoch + 1} - D Loss: {d_loss.item():.4f}, G Loss: {g_loss.item():.4f}")

            if (epoch + 1) % save_time == 0:
                save_checkpoint(gen, dis, gen_opt, dis_opt, epoch + 1, save_dir)
                folder_path = save_dir[:-4]
                if not os.path.exists(folder_path):
                    os.makedirs(folder_path)
                r = torch.randn(2, 100).to(device)
                im = gen(r).detach().cpu().numpy()[0]
                im = np.transpose(im, (1, 2, 0))  # shape: (218, 178, 3)
                im = (im * 255).clip(0, 255).astype(np.uint8)
                plt.imsave(f'{folder_path}/epoch{epoch + 1}.png', im)
                print(f"Epoch {epoch + 1} - D Loss: {d_loss.item():.4f}, G Loss: {g_loss.item():.4f}")

    gen.eval()
    for i in range(100):
        r = torch.randn(2, 100).to(device)
        im = gen(r).detach().cpu().numpy()[0] * 255
        show(im)

if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()  # Optional but recommended on Windows

    print("CUDA Available:", torch.cuda.is_available())
    trainNN(1, 128, save_time=1, save_dir='old30.pth')

# print("CUDA Available:", torch.cuda.is_available())
# trainNN(30, 16, save_time=10, save_dir='checkpoint12.pth')