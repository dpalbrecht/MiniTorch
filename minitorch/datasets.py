import torch
import numpy as np


class MatrixDataset(torch.utils.data.Dataset):

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
        x = self.x[idx]

        if isinstance(x, str):
            x = Image.open(x)

        if self.transform is not None:
            x = self.transform(image=np.array(x))['image']

        x = np.expand_dims(x, axis=[n for n in range(3-len(x.shape))])

        x = torch.tensor(x).float().to(self.device)
        y = torch.tensor(np.argmax(self.y[idx])).to(self.device).unsqueeze(0)

        return x, y