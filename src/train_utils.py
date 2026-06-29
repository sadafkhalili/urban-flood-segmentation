import torch

from src.metrics import pixel_accuracy


def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()

    total_loss = 0.0
    total_accuracy = 0.0

    for images, masks in dataloader:
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, masks)

        loss.backward()

        optimizer.step()

        total_loss += loss.item()
        total_accuracy += pixel_accuracy(outputs, masks)

    return (
        total_loss / len(dataloader),
        total_accuracy / len(dataloader),
    )


@torch.no_grad()
def validate_one_epoch(model, dataloader, criterion, device):
    model.eval()

    total_loss = 0.0
    total_accuracy = 0.0

    for images, masks in dataloader:
        images = images.to(device)
        masks = masks.to(device)

        outputs = model(images)

        loss = criterion(outputs, masks)

        total_loss += loss.item()
        total_accuracy += pixel_accuracy(outputs, masks)

    return (
        total_loss / len(dataloader),
        total_accuracy / len(dataloader),
    )


if __name__ == "__main__":
    print("Training utilities are ready.")
