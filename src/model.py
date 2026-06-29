import torch
import segmentation_models_pytorch as smp

from src.config import DEVICE, NUM_CLASSES


def build_model():

    model = smp.Unet(
        encoder_name="resnet18",
        encoder_weights="imagenet",
        in_channels=3,
        classes=NUM_CLASSES,
    )

    return model.to(DEVICE)


if __name__ == "__main__":

    model = build_model()

    x = torch.randn(2, 3, 256, 256).to(DEVICE)

    y = model(x)

    print(f"Device: {DEVICE}")
    print(f"Input shape : {x.shape}")
    print(f"Output shape: {y.shape}")
