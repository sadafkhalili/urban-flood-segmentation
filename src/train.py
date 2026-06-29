from src.config import DEVICE, NUM_EPOCHS, LEARNING_RATE, NUM_CLASSES
from src.model import build_model
from src.losses import CombinedLoss
from src.checkpoint import save_model
from src.seed import set_seed


def main():
    set_seed()

    print("Training script is ready.")
    print("Device:", DEVICE)

    print("\nDataset is not available yet.")
    print("After preparing the dataset, this script will train the U-Net model.")

    model = build_model()
    criterion = CombinedLoss(num_classes=NUM_CLASSES)

    print("\nModel and loss function are ready.")
    print("Epochs:", NUM_EPOCHS)
    print("Learning rate:", LEARNING_RATE)


if __name__ == "__main__":
    main()
