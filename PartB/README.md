# Part B

This directory contains scripts necessary for fine-tuning pretrained Vision models from PyTorch using the iNaturalist dataset.

## Files

### `main.py`
This script includes the logic to load and fine-tune various models. It allows customization by adding additional DenseNet layers and freezing certain layers before training.

### `pretrained_model.csv`
A reference list of pretrained models sourced from the PyTorch library. These models are sorted in descending order based on their performance on the ImageNet dataset.

### `train.py`
This is a command-line training script. Currently, it may not support all models due to differences in input formats and final layer configurations.

