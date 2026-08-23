import torch
from torch.utils.data import Dataset
from PIL import Image

class MedicalDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        # `images` can be a list of PIL Images (synthetic data) or a list of
        # file paths / str (real dataset) — loaded lazily on __getitem__ either way.
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        label = self.labels[idx]
        if isinstance(img, str):
            img = Image.open(img).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, torch.tensor(label, dtype=torch.long)
