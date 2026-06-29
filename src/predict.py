from pathlib import Path

import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt

from src.config import DEVICE, MODELS_DIR, PREDICTIONS_DIR, IMAGE_SIZE
from src.model import build_model


def predict_image(image_path, model_path):
    image_path = Path(image_path)
    model_path = Path(model_path)

    if not image_path.exists():
        print(f"Image not found: {image_path}")
        return

    if not model_path.exists():
        print(f"Model not found: {model_path}")
        print("Train the model first.")
        return

    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)

    model = build_model()
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()

    image = cv2.imread(str(image_path))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))

    image_tensor = image_resized.astype(np.float32) / 255.0
    image_tensor = torch.tensor(image_tensor).permute(2, 0, 1).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        output = model(image_tensor)
        prediction = torch.argmax(output, dim=1).squeeze().cpu().numpy()

    output_path = PREDICTIONS_DIR / "prediction_mask.png"

    plt.imsave(output_path, prediction)

    print(f"Prediction saved to: {output_path}")


if __name__ == "__main__":
    print("Prediction script is ready.")
    print("A trained model and input image are required.")
