import torch
import matplotlib.pyplot as plt

from src.config import DEVICE, BEST_MODEL_NAME, PREDICTIONS_DIR
from src.checkpoint import load_model
from src.data_loaders import get_dataloaders


def main():
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)

    _, _, test_loader = get_dataloaders()
    model = load_model(BEST_MODEL_NAME)

    images, masks = next(iter(test_loader))

    images = images.to(DEVICE)

    with torch.no_grad():
        outputs = model(images)
        predictions = torch.argmax(outputs, dim=1).cpu()

    images = images.cpu()

    num_samples = min(5, images.shape[0])

    for i in range(num_samples):
        sar_image = images[i, 0].numpy()
        true_mask = masks[i].numpy()
        pred_mask = predictions[i].numpy()

        plt.figure(figsize=(12, 4))

        plt.subplot(1, 3, 1)
        plt.imshow(sar_image, cmap="gray")
        plt.title("SAR Image")
        plt.axis("off")

        plt.subplot(1, 3, 2)
        plt.imshow(true_mask, vmin=0, vmax=2)
        plt.title("Ground Truth")
        plt.axis("off")

        plt.subplot(1, 3, 3)
        plt.imshow(pred_mask, vmin=0, vmax=2)
        plt.title("Prediction")
        plt.axis("off")

        output_path = PREDICTIONS_DIR / f"prediction_sample_{i+1}.png"
        plt.savefig(output_path, bbox_inches="tight")
        plt.close()

        print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
