import torch

from src.config import DEVICE, NUM_CLASSES, BEST_MODEL_NAME
from src.checkpoint import load_model
from src.data_loaders import get_dataloaders
from src.metrics import pixel_accuracy, mean_iou, mean_dice


@torch.no_grad()
def main():
    print("Evaluating model...")
    print("Device:", DEVICE)

    _, _, test_loader = get_dataloaders()

    model = load_model(BEST_MODEL_NAME)

    total_accuracy = 0.0
    total_iou = 0.0
    total_dice = 0.0

    for images, masks in test_loader:
        images = images.to(DEVICE)
        masks = masks.to(DEVICE)

        outputs = model(images)

        total_accuracy += pixel_accuracy(outputs, masks)
        total_iou += mean_iou(outputs, masks, NUM_CLASSES)
        total_dice += mean_dice(outputs, masks, NUM_CLASSES)

    num_batches = len(test_loader)

    print("\nTest Results")
    print(f"Pixel Accuracy: {total_accuracy / num_batches:.4f}")
    print(f"Mean IoU      : {total_iou / num_batches:.4f}")
    print(f"Mean Dice     : {total_dice / num_batches:.4f}")


if __name__ == "__main__":
    main()
