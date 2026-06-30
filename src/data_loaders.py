from torch.utils.data import DataLoader

from src.config import BATCH_SIZE, PROCESSED_DATA_DIR
from src.dataset import UrbanFloodDataset
from src.transforms import (
    get_train_transforms,
    get_validation_transforms,
)


def get_dataloaders():

    train_dataset = UrbanFloodDataset(
        csv_path=PROCESSED_DATA_DIR / "train.csv",
        transforms=get_train_transforms(),
    )

    validation_dataset = UrbanFloodDataset(
        csv_path=PROCESSED_DATA_DIR / "validation.csv",
        transforms=get_validation_transforms(),
    )

    test_dataset = UrbanFloodDataset(
        csv_path=PROCESSED_DATA_DIR / "test.csv",
        transforms=get_validation_transforms(),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    return train_loader, validation_loader, test_loader


if __name__ == "__main__":

    train_loader, validation_loader, test_loader = get_dataloaders()

    images, masks = next(iter(train_loader))

    print("Train batches:", len(train_loader))
    print("Validation batches:", len(validation_loader))
    print("Test batches:", len(test_loader))

    print()

    print("Image batch shape:", images.shape)
    print("Mask batch shape :", masks.shape)

    print()

    print("Image dtype:", images.dtype)
    print("Mask dtype :", masks.dtype)
