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
LOGS_DIR = OUTPUTS_DIR / "logs"

# ============================
# Dataset
# ============================

IMAGE_SIZE = 256
NUM_CLASSES = 3

CLASS_NAMES = [
    "Non-Flood",
    "Flood Open Area",
    "Flood Urban",
]

TRAIN_RATIO = 0.70
VALID_RATIO = 0.15
TEST_RATIO = 0.15

# ============================
# Model
# ============================

MODEL_NAME = "unet"
ENCODER_NAME = "resnet18"
ENCODER_WEIGHTS = "imagenet"
IN_CHANNELS = 8

# ============================
# Training
# ============================

BATCH_SIZE = 8
NUM_EPOCHS = 30
LEARNING_RATE = 1e-4
RANDOM_SEED = 42

EARLY_STOPPING_PATIENCE = 8
SAVE_BEST_MODEL = True
BEST_MODEL_NAME = "best_model.pth"

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
