# Urban Flood Segmentation

Deep learning project for semantic segmentation of flooded areas in SAR satellite images.

## Objective

The model predicts the class of every pixel in a satellite image to identify flooded regions.

## Classes

- Non-Flood
- Flood Open Area
- Flood Urban

## Model

- U-Net
- ResNet18 Encoder
- PyTorch

## Project Structure

data/
models/
outputs/
notebooks/
src/

## Installation

python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

## Run

python -m src.check_setup

python -m src.smoke_test
