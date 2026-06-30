import pandas as pd
import matplotlib.pyplot as plt

from src.config import LOGS_DIR, FIGURES_DIR


def main():
    history_path = LOGS_DIR / "training_history.csv"

    if not history_path.exists():
        print("Training history does not exist yet.")
        return

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    history = pd.read_csv(history_path)

    plt.figure()
    plt.plot(history["epoch"], history["train_loss"], label="Train Loss")
    plt.plot(history["epoch"], history["validation_loss"], label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.savefig(FIGURES_DIR / "loss_history.png", bbox_inches="tight")
    plt.close()

    plt.figure()
    plt.plot(history["epoch"], history["train_accuracy"], label="Train Accuracy")
    plt.plot(history["epoch"], history["validation_accuracy"], label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training and Validation Accuracy")
    plt.legend()
    plt.savefig(FIGURES_DIR / "accuracy_history.png", bbox_inches="tight")
    plt.close()

    print("History plots saved to:")
    print(FIGURES_DIR / "loss_history.png")
    print(FIGURES_DIR / "accuracy_history.png")


if __name__ == "__main__":
    main()
