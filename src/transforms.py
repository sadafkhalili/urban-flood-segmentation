import albumentations as A
from albumentations.pytorch import ToTensorV2

from src.config import IMAGE_SIZE


def get_train_transforms():
    return A.Compose([
        A.Resize(IMAGE_SIZE, IMAGE_SIZE),
        A.HorizontalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Normalize(),
        ToTensorV2(),
    ])


def get_validation_transforms():
    return A.Compose([
        A.Resize(IMAGE_SIZE, IMAGE_SIZE),
        A.Normalize(),
        ToTensorV2(),
    ])


if __name__ == "__main__":
    print("Segmentation transforms are ready.")
