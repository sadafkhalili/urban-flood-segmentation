import pandas as pd
import torch

from src.config import (
    DEVICE,
    NUM_EPOCHS,
    LEARNING_RATE,
    NUM_CLASSES,
    BEST_MODEL_NAME,
    LOGS_DIR,
    EARLY_STOPPING_PATIENCE,
)
from src.model import build_model
from src.losses import CombinedLoss
from src.data_loaders import get_dataloaders
from src.train_utils import train_one_epoch, validate_one_epoch
from src.checkpoint import save_model
from src.early_stopping import EarlyStopping
from src.seed import set_seed


def main():
    set_seed()

    print("Starting training...")
    print("Device:", DEVICE)

    train_loader, validation_loader, _ = get_dataloaders()

    model = build_model()
    criterion = CombinedLoss(num_classes=NUM_CLASSES)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    early_stopping = EarlyStopping(patience=EARLY_STOPPING_PATIENCE)

    history = []
    best_validation_loss = float("inf")

    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss, train_accuracy = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            DEVICE,
        )

        validation_loss, validation_accuracy = validate_one_epoch(
            model,
            validation_loader,
            criterion,
            DEVICE,
        )

        print(
            f"Epoch {epoch}/{NUM_EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_accuracy:.4f} | "
            f"Val Loss: {validation_loss:.4f} | "
            f"Val Acc: {validation_accuracy:.4f}"
        )

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "validation_loss": validation_loss,
            "validation_accuracy": validation_accuracy,
        })

        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            save_model(model, BEST_MODEL_NAME)

        early_stopping(validation_loss)

        if early_stopping.early_stop:
            print("Early stopping triggered.")
            break

    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    history_path = LOGS_DIR / "training_history.csv"
    pd.DataFrame(history).to_csv(history_path, index=False)

    print("\nTraining completed.")
    print(f"Training history saved to: {history_path}")


if __name__ == "__main__":
    main()
