import torch
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from src.config import DEVICE, BEST_MODEL_NAME, PREDICTIONS_DIR
from src.checkpoint import load_model
from src.data_loaders import get_dataloaders


def main():
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)

    _, _, test_loader = get_dataloaders()
    model = load_model(BEST_MODEL_NAME)

    cmap = ListedColormap(["black", "blue", "red"])

    saved_count = 0
    max_samples = 5

    for images, masks in test_loader:
        images = images.to(DEVICE)

        with torch.no_grad():
            outputs = model(images)
            predictions = torch.argmax(outputs, dim=1).cpu()

        images = images.cpu()

        for i in range(images.shape[0]):
            true_mask = masks[i]

            if not torch.any(true_mask > 0):
                continue

            sar_image = images[i, 0].numpy()
            true_mask = true_mask.numpy()
            pred_mask = predictions[i].numpy()

            plt.figure(figsize=(12, 4))

            plt.subplot(1, 3, 1)
            plt.imshow(sar_image, cmap="gray")
            plt.title("SAR Image")
            plt.axis("off")

            plt.subplot(1, 3, 2)
            plt.imshow(true_mask, cmap=cmap, vmin=0, vmax=2)
            plt.title("Ground Truth")
            plt.axis("off")

            plt.subplot(1, 3, 3)
            plt.imshow(pred_mask, cmap=cmap, vmin=0, vmax=2)
            plt.title("Prediction")
            plt.axis("off")

            output_path = PREDICTIONS_DIR / f"flood_prediction_sample_{saved_count + 1}.png"
            plt.savefig(output_path, bbox_inches="tight")
            plt.close()

            print(f"Saved: {output_path}")

            saved_count += 1

            if saved_count >= max_samples:
                return


if __name__ == "__main__":
    main()
