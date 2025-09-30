import os
import pandas as pd
from torchvision.io import decode_image
from torchvision.transforms import v2, transforms
from torch.utils.data import Dataset

class FuelFlowDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = self.img_dir + str(self.img_labels.iloc[idx, 0])+ ".jpg"
        image = decode_image(img_path)
        label = self.img_labels.iloc[idx, 37]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label