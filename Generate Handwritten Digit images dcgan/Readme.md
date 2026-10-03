# Generate Handwritten Digit Images using DCGAN

A deep learning project that trains a **Deep Convolutional Generative Adversarial Network (DCGAN)** on the **MNIST handwritten digits dataset** to generate new, realistic-looking digit images from random noise.

- **Notebook:** `Generate_handwritten_digit_images_DCGAN.ipynb`
- **Framework:** PyTorch
- **Dataset:** MNIST (60,000 training images, 28x28 grayscale, digits 0-9)

---

## Table of Contents
1. [Overview](#overview)
2. [How a DCGAN Works](#how-a-dcgan-works)
3. [Project Structure](#project-structure)
4. [Requirements](#requirements)
5. [Installation](#installation)
6. [How to Run](#how-to-run)
7. [Model Architecture](#model-architecture)
8. [Hyperparameters](#hyperparameters)
9. [Results](#results)
10. [Using the Trained Generator](#using-the-trained-generator)
11. [Possible Improvements](#possible-improvements)
12. [References](#references)

---

## Overview

The goal is to learn the distribution of real handwritten digits so that a model can create *new* digit images that never existed in the dataset. Training uses two competing networks, a Generator and a Discriminator, until the generated images become hard to tell apart from real ones.

## How a DCGAN Works

| Component | Role |
|-----------|------|
| **Generator (G)** | Takes a random noise vector `z` and upsamples it with transposed convolutions into a 28x28 image. |
| **Discriminator (D)** | A CNN classifier that predicts whether an image is real (from MNIST) or fake (from G). |

Both are trained together on the minimax objective:

```
min_G max_D  E[log D(x)] + E[log(1 - D(G(z)))]
```

- D is trained to correctly label real and fake images.
- G is trained to fool D into labelling its fake images as real.

## Project Structure

```
.
├── Generate_handwritten_digit_images_DCGAN.ipynb   # Main notebook
├── README.md                                       # This file
├── data/                                           # MNIST (auto-downloaded)
└── dcgan_outputs/                                  # Created when you run the notebook
    ├── epoch_001.png ... epoch_030.png             # Generated samples per epoch
    ├── generator.pth                               # Trained generator weights
    ├── discriminator.pth                           # Trained discriminator weights
    └── dcgan_training.gif                          # Training progress animation
```

## Requirements

- Python 3.8+
- PyTorch
- torchvision
- numpy
- matplotlib
- jupyter (Notebook / Lab) or Google Colab
- imageio (optional, only for the training GIF)

A **GPU is recommended** (training takes a few minutes on GPU), but the notebook also runs on CPU, just slower.

## Installation

```bash
# 1. (Optional) create a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install torch torchvision numpy matplotlib jupyter imageio
```

## How to Run

**Locally**
```bash
jupyter notebook Generate_handwritten_digit_images_DCGAN.ipynb
```
Then choose **Kernel -> Restart & Run All**.

**Google Colab**
1. Upload the notebook to [colab.research.google.com](https://colab.research.google.com).
2. Go to **Runtime -> Change runtime type -> GPU**.
3. Choose **Runtime -> Run all**.

The MNIST dataset is downloaded automatically on the first run.

## Model Architecture

### Generator
| Layer | Output Shape |
|-------|--------------|
| Input noise `z` | (100, 1, 1) |
| ConvTranspose2d + BatchNorm + ReLU | (256, 7, 7) |
| ConvTranspose2d + BatchNorm + ReLU | (128, 14, 14) |
| ConvTranspose2d + BatchNorm + ReLU | (64, 28, 28) |
| Conv2d + Tanh | (1, 28, 28) |

### Discriminator
| Layer | Output Shape |
|-------|--------------|
| Input image | (1, 28, 28) |
| Conv2d + LeakyReLU(0.2) | (64, 14, 14) |
| Conv2d + BatchNorm + LeakyReLU | (128, 7, 7) |
| Conv2d + BatchNorm + LeakyReLU | (256, 3, 3) |
| Conv2d + Sigmoid | (1, 1, 1) |

Weights are initialized from N(0, 0.02), as recommended in the original DCGAN paper.

## Hyperparameters

| Parameter | Value |
|-----------|-------|
| Batch size | 128 |
| Epochs | 30 |
| Latent vector size | 100 |
| Learning rate | 0.0002 |
| Adam beta1 / beta2 | 0.5 / 0.999 |
| Loss function | Binary Cross-Entropy (BCELoss) |
| Image normalization | [-1, 1] |

You can change these in the "Hyperparameters" cell of the notebook. Training for 50-100 epochs usually gives sharper digits.

## Results

After training, the notebook produces:

- **Loss curves** for the Generator and Discriminator.
- **Real vs generated** image grids side by side.
- **Epoch-by-epoch progress**: noisy blobs in epoch 1 become clear digits within a few epochs.
- **New digit samples** generated from random noise.
- **Latent space interpolation**: smooth morphing between two generated digits.

Generated images from every epoch are saved in `dcgan_outputs/`.

> Tip: add your own generated image (for example `dcgan_outputs/epoch_030.png`) here once you have trained the model:
> `![Generated digits](dcgan_outputs/epoch_030.png)`

## Using the Trained Generator

After training, you can generate digits without retraining:

```python
import torch

# Generator class must be defined (copy from the notebook)
G = Generator().to(device)
G.load_state_dict(torch.load("dcgan_outputs/generator.pth", map_location=device))
G.eval()

z = torch.randn(16, 100, 1, 1, device=device)
with torch.no_grad():
    images = G(z).cpu()      # shape: (16, 1, 28, 28), values in [-1, 1]
```


## References

- Radford, A., Metz, L., Chintala, S. (2015). *Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks.* [arXiv:1511.06434](https://arxiv.org/abs/1511.06434)
- Goodfellow, I. et al. (2014). *Generative Adversarial Networks.* [arXiv:1406.2661](https://arxiv.org/abs/1406.2661)
- LeCun, Y., Cortes, C., Burges, C. *The MNIST Database of Handwritten Digits.* http://yann.lecun.com/exdb/mnist/
- PyTorch DCGAN Tutorial: https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
