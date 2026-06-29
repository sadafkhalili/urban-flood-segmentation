import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.config import DEVICE
from src.model import build_model
from src.train_utils import train_one_epoch


def main():

    images = torch.randn(4, 3, 256, 256)

    masks = torch.randint(0, 3, (4, 256, 256))

    dataset = TensorDataset(images, masks)

    dataloader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=False,
    )

    model = build_model()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-4,
    )

    criterion = nn.CrossEntropyLoss()

    train_loss, train_accuracy = train_one_epoch(
        model,
        dataloader,
        optimizer,
        criterion,
        DEVICE,
    )

    print(f"Device: {DEVICE}")
    print(f"Train loss: {train_loss:.4f}")
    print(f"Train accuracy: {train_accuracy:.4f}")
    print("Pipeline test completed successfully.")


if __name__ == "__main__":
    main()
