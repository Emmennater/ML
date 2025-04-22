import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import cv2
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader

# NEED BIGGER DATASET!!!! (OVERFITTING)
# Datset: https://www.kaggle.com/datasets/thedatasith/hotdog-nothotdog/data?select=hotdog-nothotdog
# Hotdog | Not Hotdog by Francisco 'Cisco' Zambala on Kaggle
train_path = "hotdog-nothotdog/train"
test_path = "hotdog-nothotdog/test"


def save_checkpoint(model, opt, epoch, path):
    torch.save({"epoch": epoch, "model": model.state_dict(), "opt": opt.state_dict()}, path)


def load_checkpoint(model, opt, path):
    if os.path.exists(path):
        if torch.cuda.is_available():
            checkpoint = torch.load(path, weights_only=True)
        else:
            checkpoint = torch.load(path, map_location=torch.device("cpu"), weights_only=True)
        model.load_state_dict(checkpoint["model"])
        opt.load_state_dict(checkpoint["opt"])
        print(f"Checkpoint loaded from epoch {checkpoint['epoch']}")
        return checkpoint["epoch"]
    return 0


def resize(img, w, h):
    # Crop image to w x h centered at center of image
    min_dim = min(img.shape[0], img.shape[1])
    h_start = (img.shape[0] - min_dim) // 2
    w_start = (img.shape[1] - min_dim) // 2
    return cv2.resize(img[h_start : h_start + min_dim, w_start : w_start + min_dim], (w, h))


def get_file_names(directory):
    file_list = []

    for item in os.listdir(directory):
        item_path = os.path.join(directory, item)
        if os.path.isfile(item_path):
            file_list.append(item)

    return file_list


def show(tensor_img, title="Hotdog?"):
    # (3, H, W in [0, 1]) -> (H, W, 3 in [0, 255])
    img = tensor_img.permute(1, 2, 0).cpu().numpy()  # CHW -> HWC
    img = (img * 255).astype(np.uint8)  # [0, 1] -> [0, 255]
    cv2.imshow(title, cv2.resize(img, (512, 512), interpolation=cv2.INTER_NEAREST))
    key = cv2.waitKey(0)
    if key == 27:
        cv2.destroyAllWindows()


class HotdogDataset(Dataset):
    def __init__(self, directory=train_path):
        hotdog_files = get_file_names(directory + "/hotdog")
        nothotdog_files = get_file_names(directory + "/nothotdog")
        w = 128
        h = 128

        print("No. hotdog images:", len(hotdog_files))
        print("No. not hotdog images:", len(nothotdog_files))

        self.data = []
        self.labels = []
        for file in hotdog_files:
            self.data.append(resize(cv2.imread(directory + "/hotdog/" + file), w, h))
            self.labels.append(1)

        for file in nothotdog_files:
            self.data.append(resize(cv2.imread(directory + "/nothotdog/" + file), w, h))
            self.labels.append(0)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = torch.from_numpy(np.transpose(self.data[idx], (2, 0, 1))).float() / 255.0
        label = torch.tensor(self.labels[idx], dtype=torch.float32)  # also float for loss
        return img, label


class HotdogModel(nn.Module):
    def __init__(self):
        super(HotdogModel, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=7, padding=3),  # (3x128x128) -> (32x128x128)
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 64, kernel_size=7, padding=3),  # (32x128x128) -> (64x64x64)
            nn.LeakyReLU(0.2),
            nn.Dropout2d(0.1),
            nn.MaxPool2d(2, 2),  # (64x64x64) -> (64x32x32)
            nn.Conv2d(64, 32, kernel_size=5, padding=2),  # (64x32x32) -> (32x32x32)
            nn.LeakyReLU(0.2),
            nn.Dropout2d(0.1),
            nn.Conv2d(32, 16, kernel_size=3, padding=1),  # (32x32x32) -> (16x32x32)
            nn.LeakyReLU(0.2),
            nn.Dropout2d(0.1),
            nn.MaxPool2d(2, 2),  # (16x32x32) -> (16x16x16)
            nn.Flatten(1),
            nn.Linear(16 * 32 * 32, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.model(x)


def train(cp, epochs=10, batch_size=32, lr=0.0002, save_every=1, device="cuda" if torch.cuda.is_available() else "cpu"):
    model = HotdogModel().to(device)
    dataset = HotdogDataset()
    dataloader = DataLoader(dataset, batch_size, shuffle=True, num_workers=0)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCELoss()
    start_epoch = load_checkpoint(model, opt, cp)

    print(f"Training set size: {len(dataset)}")

    if dataset is not None and len(dataset) > 0:
        for epoch in range(start_epoch, start_epoch + epochs):
            running_loss = 0.0
            for i, data in enumerate(dataloader, 0):
                inputs, labels = data
                inputs = inputs.to(device)
                labels = labels.to(device)
                labels = labels.view(-1, 1)  # (32) -> (32, 1)

                opt.zero_grad()
                outputs = model(inputs)
                loss = loss_fn(outputs, labels)
                loss.backward()
                opt.step()

                running_loss += loss.item()

            print("Finished Epoch %d" % (epoch + 1))
            print(f"Loss: {(running_loss / len(dataloader)):.4f}")
            running_loss = 0.0
            if (epoch + 1) % save_every == 0:
                save_checkpoint(model, opt, epoch + 1, cp)

        print("Finished Training")

    # Testing (train is much better than test - overfitting)
    model.eval()
    test_dataset = HotdogDataset(test_path)
    # test_dataset = dataset

    for i in range(100):
        idx = np.random.randint(0, len(test_dataset) - 1)
        img, label = test_dataset[idx]
        
        img = img.to(device).unsqueeze(0)  # Add batch dimension (3, 128, 128) -> (1, 3, 128, 128)
        label = label.to(device)
        label = label.view(-1, 1)  # (32) -> (32, 1)
        
        pred = model(img).item()
        real = label.item()
        
        msg = "Hotdog" if pred > 0.5 else "Not Hotdog"
        show(img[0], f"{msg} : {pred:.2%}")


if __name__ == "__main__":
    import multiprocessing

    multiprocessing.freeze_support()

    print("CUDA Available:", torch.cuda.is_available())
    train('checkpoint1.pt', 0, 32, 0.0002)
