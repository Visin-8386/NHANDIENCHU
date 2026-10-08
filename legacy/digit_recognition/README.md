# Legacy: MNIST Digit Recognition (TensorFlow)

This folder contains the **previous version** of the project - a CNN-based handwritten
digit recognition system built with TensorFlow/Keras. It is kept for reference only.

The current project (see root [README.md](../../README.md)) replaced this with a
PyTorch Transformer Encoder-Decoder for full handwriting text recognition.

## Contents

| File | Description |
|---|---|
| `model.py` | CNN model definition (Conv2D + BatchNorm, ~6M params) |
| `train.py` | Training script on MNIST |
| `inference.py` | Inference on MNIST test samples |
| `preprocessing.py` | Image preprocessing utilities |
| `preprocess_images.py` | Batch preprocessing for custom images |
| `visualize.py` | Training history plotting |
| `mnist_exploration.ipynb` | Dataset exploration notebook |
| `mnist_samples.png` | Sample grid from the notebook |
| `loss_chart.jpg` | Training loss chart |
| `accuracy_chart.jpg` | Training accuracy chart |
| `digit_recognition_optimized_colab.h5` | Trained weights (~30MB) |

## Why it was replaced

- Only recognized single digits (0-9), not words
- CNN + softmax cannot model variable-length text
- The new Transformer Encoder-Decoder achieves CER 3.60% on IAM words
