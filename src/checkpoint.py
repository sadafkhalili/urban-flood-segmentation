import torch

from src.config import MODELS_DIR, DEVICE
from src.model import build_model


def save_model(model, filename="best_model.pth"):
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    path = MODELS_DIR / filename
    torch.save(model.state_dict(), path)
    print(f"Model saved to: {path}")


def load_model(filename="best_model.pth"):
    path = MODELS_DIR / filename

    model = build_model()
    model.load_state_dict(torch.load(path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    print(f"Model loaded from: {path}")
    return model


if __name__ == "__main__":
    model = build_model()

    save_model(model, "test_model.pth")
    loaded_model = load_model("test_model.pth")

    print("Save and load test completed successfully.")
