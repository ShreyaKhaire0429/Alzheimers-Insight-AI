# src/dataset.py
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
import torch
import os

class MultimodalDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.df = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.root_dir, row['image'])
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        clinical = torch.tensor([
            row['age'], row['gender'], row['education'], row['mmse'], row['cdr']
        ], dtype=torch.float32)

        label = ['CN', 'MCI', 'AD'].index(row['label'])
        return image, clinical, label