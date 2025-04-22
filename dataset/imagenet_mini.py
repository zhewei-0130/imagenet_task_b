import os
from PIL import Image
from torch.utils.data import Dataset

class ImageNetMiniDataset(Dataset):
    def __init__(self, root_dir, txt_file, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []
        with open(txt_file, 'r') as f:
            for line in f:
                path, label = line.strip().split()
                self.samples.append((path, int(label)))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        full_path = os.path.join(self.root_dir, path)
        img = Image.open(full_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label
