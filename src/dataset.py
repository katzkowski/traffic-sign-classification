import os

import pandas as pd
import torchvision
from torch.utils.data import Dataset
from torchvision.io import read_image


class TrafficSignDataset(Dataset):
    def __init__(
        self, annotations_file, img_dir, transform=None, target_transform=None
    ) -> None:
        self.targets = pd.read_csv(annotations_file, header=0)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        # 7th col is relative image path
        img_path = os.path.join(self.img_dir, self.targets.iloc[idx, 7])
        image = read_image(img_path).float()

        # bbox
        width = self.targets.iloc[idx, 0]
        height = self.targets.iloc[idx, 1]
        roi_x1 = self.targets.iloc[idx, 2]
        roi_y1 = self.targets.iloc[idx, 3]
        roi_x2 = self.targets.iloc[idx, 4]
        roi_y2 = self.targets.iloc[idx, 5]

        bbox = [width, height, roi_x1, roi_y1, roi_x2, roi_y2]

        # label: 6th col
        label = self.targets.iloc[idx, 6]

        if self.transform:
            image = self.transform(image)

        if self.target_transform:
            bbox = self.target_transform(bbox)

        return image, {"label": label, "bbox": bbox}
