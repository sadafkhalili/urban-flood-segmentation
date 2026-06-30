import pandas as pd
import torch
from torch.utils.data import Dataset
import rasterio
import numpy as np


class UrbanFloodDataset(Dataset):
    def __init__(self, csv_path, transforms=None):
        self.data = pd.read_csv(csv_path)
        self.transforms = transforms

    def __len__(self):
        return len(self.data)

    def normalize_image(self, image):
        image = np.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0)
        image = np.clip(image, -60, 10)
        image = (image + 60) / 70
        return image.astype(np.float32)

    def __getitem__(self, index):
        row = self.data.iloc[index]

        image_path = row["image_path"]
        mask_path = row["mask_path"]

        with rasterio.open(image_path) as src:
            image = src.read()

        with rasterio.open(mask_path) as src:
            mask = src.read(1)

        image = self.normalize_image(image)
        image = np.transpose(image, (1, 2, 0))

        mask = mask.astype(np.int64)

        if self.transforms:
            transformed = self.transforms(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"].long()
        else:
            image = torch.tensor(image).permute(2, 0, 1).float()
            mask = torch.tensor(mask).long()

        return image, mask


if __name__ == "__main__":
    from src.config import PROCESSED_DATA_DIR
    from src.transforms import get_validation_transforms

    csv_path = PROCESSED_DATA_DIR / "train.csv"

    dataset = UrbanFloodDataset(
        csv_path=csv_path,
        transforms=get_validation_transforms(),
    )

    image, mask = dataset[0]

    print("Number of samples:", len(dataset))
    print("Image shape:", image.shape)
    print("Mask shape:", mask.shape)
    print("Image dtype:", image.dtype)
    print("Mask dtype:", mask.dtype)
    print("Mask unique values:", torch.unique(mask))
