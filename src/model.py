import torch
import segmentation_models_pytorch as smp

from src.config import (
    DEVICE,
    MODEL_NAME,
    ENCODER_NAME,
    ENCODER_WEIGHTS,
    IN_CHANNELS,
    NUM_CLASSES,
    IMAGE_SIZE,
)


def build_model():
    if MODEL_NAME.lower() != "unet":
        raise ValueError(f"Unsupported model: {MODEL_NAME}")

    model = smp.Unet(
        encoder_name=ENCODER_NAME,
        encoder_weights=ENCODER_WEIGHTS,
        in_channels=IN_CHANNELS,
        classes=NUM_CLASSES,
    )

    return model.to(DEVICE)


if __name__ == "__main__":

    model = build_model()

    x = torch.randn(2, IN_CHANNELS, IMAGE_SIZE, IMAGE_SIZE).to(DEVICE)

    with torch.no_grad():
        y = model(x)

    print("Model :", MODEL_NAME)
    print("Encoder:", ENCODER_NAME)
    print("Device :", DEVICE)
    print("Input shape :", x.shape)
    print("Output shape:", y.shape)
