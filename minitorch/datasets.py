import torch
import numpy as np


class ImageDataset1D(torch.utils.data.Dataset):

    def __init__(self, x, y, device, transform=None):
        self.x = x
        self.y = y
        self.device = device
        self.transform = transform
      
    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        x = torch.tensor(self.x[idx][None, :]).float().to(self.device)
        y = torch.tensor(np.argmax(self.y[idx])).to(self.device).unsqueeze(0)

        if self.transform:
            x = self.transform(x)

        return x, y