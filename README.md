# Urban Flood Segmentation using Deep Learning

Semantic segmentation of flooded regions in multi-channel SAR satellite images using U-Net with a ResNet18 encoder.

---

## Project Overview

This project detects flooded regions from Synthetic Aperture Radar (SAR) satellite imagery.

Unlike image classification, semantic segmentation predicts a class for every pixel in the image.

The model classifies each pixel into one of three classes:

| Class | Description |
|--------|-------------|
| 0 | Non-Flood |
| 1 | Flood Open Area |
| 2 | Flood Urban |

---

## Dataset

Dataset: Urban SAR Floods

Characteristics:

- 6471 SAR images
- 6471 Ground Truth masks
- Image size: 512 × 512
- 8 SAR channels
- Three segmentation classes

Dataset split:

| Split | Samples |
|--------|---------|
| Train | 4529 |
| Validation | 971 |
| Test | 971 |

---

## Model

Architecture:

- U-Net
- ResNet18 Encoder
- PyTorch

Input:

- 8-channel SAR image

Output:

- Pixel-wise segmentation map
- Three classes

---

## Training Configuration

| Parameter | Value |
|-----------|------|
| Epochs | 30 |
| Batch Size | 8 |
| Learning Rate | 0.0001 |
| Optimizer | Adam |
| Early Stopping | Enabled |

---

## Evaluation Results

Test Results

| Metric | Score |
|---------|-------|
| Pixel Accuracy | 0.9943 |
| Mean IoU | 0.6681 |
| Mean Dice | 0.7101 |

---

## Project Structure

```
Urban_Flood_Segmentation/

src/
models/
data/
outputs/
notebooks/

README.md
requirements.txt
```

---

## Generated Outputs

Training:

- Best trained model
- Training history
- Loss curve
- Accuracy curve

Prediction:

- SAR image
- Ground Truth mask
- Predicted mask

---

## Technologies

- Python
- PyTorch
- segmentation-models-pytorch
- Albumentations
- Rasterio
- NumPy
- Pandas
- Matplotlib

---

## Author

Sadaf Khalili

---

## License

This project is intended for educational and research purposes.
