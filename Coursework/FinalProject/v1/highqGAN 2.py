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
        self.data = np.load(npy_file, mmap_mode='r')  #, mmap_mode='r' stays on disk, only loads slices

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = self.data[idx].astype(np.float32) / 255.0  # convert only the needed image
        return torch.from_numpy(img)



class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=7, stride=2, padding=3),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, kernel_size=5, stride=2, padding=2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.3)
        )


        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(43008, 1)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        return self.fc(x)


# class Discriminator(nn.Module):
#     def __init__(self):
#         super(Discriminator, self).__init__()
#
#         self.conv_layers = nn.Sequential(
#             nn.Conv2d(3, 32, kernel_size=7, stride=2, padding=3),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Conv2d(128, 256, kernel_size=5, stride=2, padding=2),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Dropout2d(0.3)
#         )
#
#
#         self.fc = nn.Sequential(
#             nn.Flatten(),
#             nn.Linear(21504, 1)
#         )
#
#     def forward(self, x):
#         x = self.conv_layers(x)
#         return self.fc(x)

# class Discriminator(nn.Module):
#     def __init__(self):
#         super(Discriminator, self).__init__()
#
#         ninputs = 3 * 218 * 178
#         self.model = nn.Sequential(
#             nn.Conv2d(3, 32, (5,5), padding=2), # 32 X 218 X 178
#             nn.LeakyReLU(0.2),
#             nn.Dropout2d(0.3),
#             nn.MaxPool2d((4, 4)), # 32 X 54 X 44
#             nn.Conv2d(32, 64, 5, padding=2), # 64 X 54 X 44
#             nn.LeakyReLU(0.2),
#             nn.Dropout2d(0.3),
#             nn.MaxPool2d((2,2)), # 64 X 27 X 22
#             nn.Conv2d(64, 32, (3, 3), padding=(1, 1)), # 32 X 27 X 22
#             nn.LeakyReLU(0.2),
#             nn.Dropout2d(0.3),
#             nn.MaxPool2d((2, 2)), # 32 X 13 X 11
#             nn.Conv2d(32, 16, (4, 3), padding=(1, 1)),  # 16 X 13 X 11
#             nn.LeakyReLU(0.2),
#             nn.Dropout2d(0.3),
#             nn.MaxPool2d((2, 2)),  # 16 X 6 X 5
#             nn.Flatten(1),
#             nn.Linear(16*6*5, 6*5),
#             nn.LeakyReLU(0.2),
#             nn.Dropout2d(0.3),
#             nn.Linear(6*5, 1),
#         )
#
#     def forward(self, x):
#         return self.model(x)
#
#
# class Discriminator(nn.Module):
#     def __init__(self):
#         super(Discriminator, self).__init__()
#
#         ninputs = 3 * 218 * 178
#         self.model = nn.Sequential(
#             nn.Conv2d(3, 32, (5,5), padding=2, stride=2), # 32 X 109 X 89
#             nn.LeakyReLU(0.2),
#             # nn.Dropout2d(0.3),
#             nn.Conv2d(32, 64, 5, padding=2, stride=2), # 64 X 54 X 44
#             nn.LeakyReLU(0.2),
#             # nn.Dropout2d(0.3),
#             nn.Conv2d(64, 32, (3, 3), padding=(1, 1), stride=2), # 32 X 27 X 22
#             nn.LeakyReLU(0.2),
#             # nn.Dropout2d(0.3),
#             nn.Conv2d(32, 16, (4, 3), padding=(1, 1), stride=2),  # 16 X 13 X 11
#             nn.LeakyReLU(0.2),
#             #nn.Dropout2d(0.3),
#             nn.Flatten(1),
#             nn.Linear(2688, 1),
#             # nn.LeakyReLU(0.2),
#             # nn.Dropout2d(0.3),
#             # nn.Linear(13*11, 1),
#         )
#
#     def forward(self, x):
#         return self.model(x)

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        ninputs = 100
        self.model = nn.Sequential(
            # Start with a 100-dim noise vector, project and reshape to (256, 6, 6)
            nn.Linear(100, 256 * 13 * 11),
            nn.ReLU(True),
            #nn.BatchNorm1d(256 * 6 * 6),

            # Reshape to 256 x 13 x 11
            nn.Unflatten(1, (256, 13, 11)),

            # Upsample to (128, 27, 22)
            nn.ConvTranspose2d(256, 128, kernel_size=(5,4), stride=2, padding=1),  # *2
            nn.ReLU(True),
            nn.BatchNorm2d(128),

            # Upsample to (64, 54, 44)
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # *2
            nn.ReLU(True),
            nn.BatchNorm2d(64),

            # Upsample to (32, 109, 89)
            nn.ConvTranspose2d(64, 32, kernel_size=5, stride=2, padding=1),  # *2
            nn.ReLU(True),
            nn.BatchNorm2d(32),

            # Final upsample to (3, 218, 178)
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1),  # keep size

            nn.Sigmoid()  # Output in range [0, 1]
        )

    def forward(self, x):
        return self.model(x)

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        ninputs = 100
        self.model = nn.Sequential(
            # Start with a 100-dim noise vector, project and reshape to (256, 6, 6)
            nn.Linear(100, 256 * 13 * 11),
            nn.ReLU(True),
            #nn.BatchNorm1d(256 * 6 * 6),

            # Reshape to 256 x 13 x 11
            nn.Unflatten(1, (256, 13, 11)),

            # Upsample to (128, 27, 22)
            # nn.ConvTranspose2d(256, 128, kernel_size=(5,4), stride=2, padding=1),  # *2
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(256, 128, kernel_size=(4,3), stride=1, padding=(2,1)),
            nn.ReLU(True),
            nn.BatchNorm2d(128),

            # Upsample to (64, 54, 44)
            nn.Upsample(scale_factor=2, mode='bilinear'),  # or 'bilinear'
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),  # *2
            nn.ReLU(True),
            nn.BatchNorm2d(64),

            # Upsample to (32, 109, 89)
            # nn.ConvTranspose2d(64, 32, kernel_size=5, stride=2, padding=1),
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(64, 32, kernel_size=4, stride=1, padding=2),
            nn.ReLU(True),
            nn.BatchNorm2d(32),

            # Final upsample to (3, 218, 178)
            nn.Upsample(scale_factor=2, mode='bilinear'),  # or 'bilinear'
            nn.Conv2d(32, 3, kernel_size=3, stride=1, padding=1),  # *2

            nn.Sigmoid()  # Output in range [0, 1]
        )

    def forward(self, x):
        return self.model(x)

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

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        ninputs = 100
        self.model = nn.Sequential(
            # Start with a 100-dim noise vector, project and reshape to (256, 6, 6)
            nn.Linear(ninputs, 1024 * 13 * 11),
            nn.ReLU(True),
            #nn.BatchNorm1d(256 * 6 * 6),

            # Reshape to 256 x 13 x 11
            nn.Unflatten(1, (1024, 13, 11)),

            # Upsample to (128, 27, 22)
            nn.ConvTranspose2d(1024, 512, kernel_size=(5,4), stride=2, padding=1),  # *2
            # nn.Upsample(scale_factor=2, mode='bilinear'),
            # nn.Conv2d(256, 128, kernel_size=(4,3), stride=1, padding=(2,1)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(512),

            # Upsample to (64, 54, 44)
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),  # *2
            # nn.Upsample(scale_factor=2, mode='bilinear'),  # or 'bilinear'
            # nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),  # *2
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(256),

            # Upsample to (32, 109, 89)
            # nn.ConvTranspose2d(64, 32, kernel_size=5, stride=2, padding=1),
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(256, 128, kernel_size=4, stride=1, padding=2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(128),

            # Final upsample to (3, 218, 178)
            nn.Upsample(scale_factor=2, mode='bilinear'),  # or 'bilinear'
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),  # *2
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 3, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()  # Output in range [0, 1]
        )

    def forward(self, x):
        return self.model(x)

# class Generator(nn.Module):
#     def __init__(self):
#         super(Generator, self).__init__()
#
#         ninputs = 100
#         self.model = nn.Sequential(
#             # Start with a 100-dim noise vector, project and reshape to (256, 6, 6)
#             nn.Linear(ninputs, 256 * 13 * 11),
#             nn.ReLU(True),
#             #nn.BatchNorm1d(256 * 6 * 6),
#
#             # Reshape to 256 x 13 x 11
#             nn.Unflatten(1, (256, 13, 11)),
#
#             # Upsample to (128, 27, 22)
#             nn.ConvTranspose2d(256, 128, kernel_size=(5,4), stride=2, padding=1),  # *2
#             # nn.Upsample(scale_factor=2, mode='bilinear'),
#             # nn.Conv2d(256, 128, kernel_size=(4,3), stride=1, padding=(2,1)),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.BatchNorm2d(128),
#
#             # Upsample to (64, 54, 44)
#             nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # *2
#             # nn.Upsample(scale_factor=2, mode='bilinear'),  # or 'bilinear'
#             # nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),  # *2
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.BatchNorm2d(64),
#
#             # Upsample to (32, 109, 89)
#             # nn.ConvTranspose2d(64, 32, kernel_size=5, stride=2, padding=1),
#             nn.Upsample(scale_factor=2, mode='bilinear'),
#             nn.Conv2d(64, 32, kernel_size=4, stride=1, padding=2),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.BatchNorm2d(32),
#
#             # Final upsample to (3, 218, 178)
#             nn.Upsample(scale_factor=2, mode='bilinear'),  # or 'bilinear'
#             nn.Conv2d(32, 3, kernel_size=3, stride=1, padding=1),  # *2
#             nn.Sigmoid()  # Output in range [0, 1]
#         )
#
#     def forward(self, x):
#         return self.model(x)

# class Generator(nn.Module):
#     def __init__(self):
#         super(Generator, self).__init__()
#
#         ninputs = 100
#         self.model = nn.Sequential(
#             # Start with a 100-dim noise vector, project and reshape to (256, 6, 6)
#             nn.Linear(100, 256 * 13 * 11),
#             nn.ReLU(True),
#             #nn.BatchNorm1d(256 * 6 * 6),
#
#             # Reshape to 256 x 13 x 11
#             nn.Unflatten(1, (256, 13, 11)),
#
#             # Upsample to (128, 27, 22)
#             nn.ConvTranspose2d(256, 128, kernel_size=(5,4), stride=2, padding=1),  # *2
#             # nn.Upsample(scale_factor=2, mode='bilinear'),
#             # nn.Conv2d(256, 128, kernel_size=(4,3), stride=1, padding=(2,1)),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.BatchNorm2d(128),
#
#             # Upsample to (64, 54, 44)
#             nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # *2
#             # nn.Upsample(scale_factor=2, mode='bilinear'),  # or 'bilinear'
#             # nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),  # *2
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.BatchNorm2d(64),
#
#             # Upsample to (32, 109, 89)
#             # nn.ConvTranspose2d(64, 32, kernel_size=5, stride=2, padding=1),
#             # nn.Upsample(scale_factor=2, mode='bilinear'),
#             nn.PixelShuffle(2),  # 16
#             nn.Conv2d(16, 16, kernel_size=4, stride=1, padding=2),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.BatchNorm2d(16),
#
#             # Final upsample to (3, 218, 178)
#             # nn.Upsample(scale_factor=2, mode='bilinear'),  # or 'bilinear'
#             nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1),  # *2
#             nn.PixelShuffle(2),
#             nn.Conv2d(4, 3, kernel_size=3, stride=1, padding=1),
#             nn.Sigmoid()  # Output in range [0, 1]
#         )
#
#     def forward(self, x):
#         return self.model(x)

# class Generator(nn.Module):
#     def __init__(self):
#         super(Generator, self).__init__()
#
#         ninputs = 100
#         self.model = nn.Sequential(
#             # Start with a 100-dim noise vector, project and reshape to (256, 6, 6)
#             nn.Linear(100, 256 * 13 * 11),
#             nn.ReLU(True),
#             #nn.BatchNorm1d(256 * 6 * 6),
#
#             # Reshape to 256 x 13 x 11
#             nn.Unflatten(1, (256, 13, 11)),
#
#             # Upsample to (128, 27, 22)
#             nn.ConvTranspose2d(256, 128, kernel_size=(5,4), stride=2, padding=1),  # *2
#             # nn.Upsample(scale_factor=2, mode='bilinear'),
#             # nn.Conv2d(256, 128, kernel_size=(4,3), stride=1, padding=(2,1)),
#             nn.ReLU(True),
#             nn.BatchNorm2d(128),
#
#             # Upsample to (64, 54, 44)
#             nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # *2
#             # nn.Upsample(scale_factor=2, mode='bilinear'),  # or 'bilinear'
#             # nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),  # *2
#             nn.ReLU(True),
#             nn.BatchNorm2d(64),
#
#             # Upsample to (32, 109, 89)
#             # nn.ConvTranspose2d(64, 32, kernel_size=5, stride=2, padding=1),
#             nn.Upsample(scale_factor=2, mode='bilinear'),
#             nn.Conv2d(64, 32, kernel_size=4, stride=1, padding=2),
#             nn.ReLU(True),
#             nn.BatchNorm2d(32),
#
#             # upsample to (16, 218, 178)
#             nn.Upsample(scale_factor=2, mode='bilinear'),  # or 'bilinear'
#             nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1),  # *2
#             nn.ReLU(True),
#             nn.BatchNorm2d(16),
#
#             # upsample to (8, 436, 356)
#             nn.Upsample(scale_factor=2, mode='bilinear'),  # or 'bilinear'
#             nn.Conv2d(16, 8, kernel_size=3, stride=1, padding=1),  # *2
#             nn.ReLU(True),
#             nn.BatchNorm2d(8),
#
#             # (3, 218, 178)
#             nn.Conv2d(8, 3, kernel_size=5, stride=2, padding=2),
#
#
#             nn.Sigmoid()  # Output in range [0, 1]
#         )
#
#     def forward(self, x):
#         return self.model(x)

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
    dis_opt = torch.optim.Adam(dis.parameters(), lr=lr, betas=(0.5, 0.999))
    gen_opt = torch.optim.Adam(gen.parameters(), lr=lr, betas=(0.5, 0.999))
    noise_dim = 100

    start_epoch = load_checkpoint(gen, dis, gen_opt, dis_opt, save_dir)

    if epochs>0:
        dataset = FaceDataset(img_dir)
        loader = DataLoader(dataset, batch_size=batch_size // 2, shuffle=True, num_workers=0, pin_memory=True)

        for epoch in range(start_epoch, start_epoch + epochs):
            i = 0
            for real in loader:
                real = real.to(device, non_blocking=True)
                print(i)
                i += 1
                # === Discriminator ===
                noise = torch.randn(batch_size // 2, noise_dim, device=device)
                fake = gen(noise).detach()

                dis_opt.zero_grad()
                gen_opt.zero_grad()

                real_preds = dis(real)
                fake_preds = dis(fake)

                real_labels = torch.ones_like(real_preds).to(device)
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




                # print(f"Epoch {epoch + 1} - D Loss: {d_loss.item():.4f}, G Loss: {g_loss.item():.4f}")
                # with torch.no_grad():
                #     d_lr = dis_opt.param_groups[0]['lr']
                #     g_lr = gen_opt.param_groups[0]['lr']
                #
                #     if d_loss.item() < g_loss.item():
                #         # Discriminator is winning, help generator
                #         g_lr *= 1.05
                #         d_lr *= 0.95
                #     else:
                #         # Generator is winning, help discriminator
                #         g_lr *= 0.95
                #         d_lr *= 1.05
                #
                #     # Clamp to reasonable bounds
                #     g_lr = max(min(g_lr, lr * 10), lr * 0.0001)
                #     d_lr = max(min(d_lr, lr * 10), lr * 0.0001)
                #
                #     # Update optimizers
                #     for param_group in gen_opt.param_groups:
                #         param_group['lr'] = g_lr
                #     for param_group in dis_opt.param_groups:
                #         param_group['lr'] = d_lr


            if (epoch + 1) % save_time == 0:
                save_checkpoint(gen, dis, gen_opt, dis_opt, epoch + 1, save_dir)
                print(f"Epoch {epoch+1} - D Loss: {d_loss.item():.4f}, G Loss: {g_loss.item():.4f}")

    gen.eval()
    for i in range(100):
        r = torch.randn(2, 100).to(device)
        im = gen(r).detach().cpu().numpy()[0] * 255
        show(im)

if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()  # Optional but recommended on Windows

    print("CUDA Available:", torch.cuda.is_available())
    trainNN(0, 32, save_time=1, save_dir='old29.pth')

# print("CUDA Available:", torch.cuda.is_available())
# trainNN(30, 16, save_time=10, save_dir='checkpoint12.pth')