from pathlib import Path
import torch

# ============================
# Project paths
# ============================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

MODELS_DIR = PROJECT_ROOT / "models"

OUTPUTS_DIR = PROJECT_ROOT / "outputs"
PREDICTIONS_DIR = OUTPUTS_DIR / "predictions"
FIGURES_DIR = OUTPUTS_DIR / "figures"

# ============================
# Dataset
# ============================

IMAGE_SIZE = 256

NUM_CLASSES = 3

CLASS_NAMES = [
    "Non-Flood",
    "Flood Open Area",
    "Flood Urban"
]

# ============================
# Training
# ============================

BATCH_SIZE = 8

NUM_EPOCHS = 30

LEARNING_RATE = 1e-4

RANDOM_SEED = 42

# ============================
# Device
# ============================

DEVICE = (
    "mps"
    if torch.backends.mps.is_available()
    else "cuda"
    if torch.cuda.is_available()
    else "cpu"
)
