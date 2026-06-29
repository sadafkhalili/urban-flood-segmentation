from src.config import DEVICE
from src.model import build_model


def main():

    print("Evaluation script is ready.")
    print("Device:", DEVICE)

    print("\nTest dataset is not available yet.")
    print("After preparing the dataset, this script will evaluate the trained model.")

    _ = build_model()

    print("\nModel is ready for evaluation.")


if __name__ == "__main__":
    main()
