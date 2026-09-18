# MNIST Digit Classification with Neural Network

A deep learning project that classifies handwritten digits (0–9) from the
**MNIST dataset** using a fully-connected (dense) neural network built with
**TensorFlow / Keras**. The dataset is loaded directly via
`keras.datasets.mnist`, so no manual download is required.

---

## Project Structure

```
mnist_project/
├── mnist_classifier.py     # Main script: load → preprocess → build → train → evaluate → visualize
├── requirements.txt        # Python dependencies
├── README.md                # This file
└── outputs/                 # Generated after running the script
    ├── mnist_model.keras            # Saved trained model
    ├── training_history.png         # Accuracy / loss curves
    ├── confusion_matrix.png         # Confusion matrix heatmap
    └── sample_predictions.png       # Grid of sample test predictions
```
 Outputs will be generated after running the main file.

 ---
## Workflow

The pipeline follows six clear stages, each implemented as its own function
in `mnist_classifier.py`:

```
┌─────────────────┐   ┌───────────────────┐   ┌──────────────────┐
│ 1. Load Data     │──▶│ 2. Preprocess     │──▶│ 3. Build Model    │
│ keras.datasets   │   │ normalize, flatten │   │ Dense NN, Dropout │
│ .mnist.load_data │   │ one-hot, val split │   │ softmax output    │
└─────────────────┘   └───────────────────┘   └──────────────────┘
                                                          │
                                                          ▼
┌─────────────────┐   ┌───────────────────┐   ┌──────────────────┐
│ 6. Visualize &   │◀──│ 5. Evaluate        │◀──│ 4. Train          │
│    Save Model    │   │ test accuracy/loss │   │ EarlyStopping,    │
│ curves, confusion │   │ classification rpt │   │ ReduceLROnPlateau │
│ matrix, samples   │   │                    │   │                    │
└─────────────────┘   └───────────────────┘   └──────────────────┘
```
A complete, self-contained deep learning pipeline that:
  1. Loads the MNIST dataset from keras.datasets
  2. Preprocesses (normalizes + reshapes) the data
  3. Builds a fully-connected neural network
  4. Trains the model with validation monitoring
  5. Evaluates on the test set
  6. Visualizes training curves, a confusion matrix, and sample predictions
  7. Saves the trained model to disk
     
### 1. Load Data
`keras.datasets.mnist.load_data()` returns 60,000 training images and
10,000 test images, each a 28×28 grayscale image labeled 0–9.

### 2. Preprocess
- Pixel values scaled from `[0, 255]` → `[0, 1]`.
- Images flattened from `28×28` → `784`-length vectors (dense network input).
- Labels one-hot encoded (10 classes).
- 10% of the training set is held out as a validation split.

### 3. Build Model
A `Sequential` dense network:

| Layer | Units | Activation | Notes |
|---|---|---|---|
| Input | 784 | – | Flattened pixel vector |
| Dense | 256 | ReLU | + Dropout (0.3) |
| Dense | 128 | ReLU | + Dropout (0.3) |
| Dense | 64  | ReLU | |
| Dense (output) | 10 | Softmax | One probability per digit |

Compiled with the **Adam** optimizer and **categorical cross-entropy** loss.

### 4. Train
Trained for up to 20 epochs (batch size 128) with two callbacks:
- **EarlyStopping** — stops training when validation loss stops improving.
- **ReduceLROnPlateau** — halves the learning rate on a validation loss plateau.

### 5. Evaluate
Reports test loss, test accuracy, and a full **classification report**
(precision/recall/F1 per digit) using scikit-learn.

### 6. Visualize & Save
Generates and saves to `outputs/`:
- Training/validation accuracy & loss curves
- A confusion matrix heatmap
- A grid of sample predictions (correct in green, incorrect in red)
- The trained model itself (`mnist_model.keras`)

---

## Setup

```bash
# 1. Clone / download this project, then move into it
cd mnist_project

# 2. (Recommended) create a virtual environment
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

## Usage

```bash
python mnist_classifier.py
```

This will download MNIST automatically (cached by Keras under
`~/.keras/datasets/`), train the model, print progress and metrics to the
console, and write all plots plus the saved model to `outputs/`.

## Expected Results

With the default architecture, you should typically see:
- **Test accuracy:** ~97–98%
- **Training time:** a few minutes on CPU, seconds per epoch on GPU

Exact numbers will vary slightly run to run due to random weight
initialization, even with the fixed seed, across different hardware/backends.

## Using the Saved Model

```python
from tensorflow import keras
import numpy as np

model = keras.models.load_model("outputs/mnist_model.keras")

# image: a 28x28 grayscale array, values in [0, 255]
def predict_digit(image):
    x = image.astype("float32").reshape(1, 784) / 255.0
    probs = model.predict(x)
    return int(np.argmax(probs))
```
## Requirements

See `requirements.txt`:
- `tensorflow`
- `numpy`
- `matplotlib`
- `scikit-learn`

## License

Free to use and modify for learning and educational purposes.
