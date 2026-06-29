import torch


class EarlyStopping:
    def __init__(self, patience=8, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta

        self.best_loss = float("inf")
        self.counter = 0
        self.early_stop = False

    def __call__(self, validation_loss):

        if validation_loss < self.best_loss - self.min_delta:
            self.best_loss = validation_loss
            self.counter = 0

        else:
            self.counter += 1

            if self.counter >= self.patience:
                self.early_stop = True


if __name__ == "__main__":

    early_stopping = EarlyStopping(patience=3)

    losses = [0.90, 0.80, 0.78, 0.78, 0.79, 0.80, 0.81]

    for epoch, loss in enumerate(losses, start=1):

        early_stopping(loss)

        print(
            f"Epoch {epoch} | "
            f"Validation Loss = {loss:.2f} | "
            f"Counter = {early_stopping.counter}"
        )

        if early_stopping.early_stop:
            print("Early stopping triggered.")
            break
