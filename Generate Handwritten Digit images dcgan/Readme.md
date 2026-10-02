# Generate Handwritten Digit Images using DCGAN

A deep learning project that trains a **Deep Convolutional Generative Adversarial Network (DCGAN)** on the **MNIST handwritten digits dataset** to generate new, realistic-looking digit images from random noise.

- **Notebook:** `Generate_handwritten_digit_images_DCGAN.ipynb`
- **Framework:** PyTorch
- **Dataset:** MNIST (60,000 training images, 28x28 grayscale, digits 0-9)

---
## Overview

The goal is to learn the distribution of real handwritten digits so that a model can create *new* digit images that never existed in the dataset. Training uses two competing networks, a Generator and a Discriminator, until the generated images become hard to tell apart from real ones.
