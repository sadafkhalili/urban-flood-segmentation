from pathlib import Path

from src.config import MODELS_DIR, OUTPUTS_DIR


def create_project_directories():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)


def count_trainable_parameters(model):
    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )


def print_model_summary(model):
    parameters = count_trainable_parameters(model)

    print("=" * 50)
    print("Model Summary")
    print("=" * 50)
    print(f"Trainable Parameters: {parameters:,}")
    print("=" * 50)


if __name__ == "__main__":

    from src.model import build_model

    create_project_directories()

    model = build_model()

    print_model_summary(model)

    print("Utility functions are ready.")
