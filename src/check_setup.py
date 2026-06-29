import platform
import torch

from src.config import PROJECT_ROOT, DEVICE

print("=" * 50)
print("Urban Flood Segmentation")
print("=" * 50)

print(f"Project root : {PROJECT_ROOT}")
print(f"Python       : {platform.python_version()}")
print(f"PyTorch      : {torch.__version__}")
print(f"Device       : {DEVICE}")
print(f"MPS available: {torch.backends.mps.is_available()}")
print(f"CUDA available: {torch.cuda.is_available()}")

print("\nProject environment is ready.")
