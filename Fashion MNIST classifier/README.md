# Fashion MNIST Classifier

A convolutional neural network (CNN) built with TensorFlow/Keras that classifies grayscale images of clothing into 10 categories using the [Fashion MNIST](https://github.com/zalandoresearch/fashion-mnist) dataset. Includes both a training notebook and an interactive Streamlit app.

## Contents

- `fashion_mnist_classifier.ipynb` — the full notebook: data loading, preprocessing, model building, training, evaluation, and visualization.
- `app.py` — a Streamlit app for interactively classifying uploaded images or random test samples.
- `requirements.txt` — dependencies for the Streamlit app.

## Requirements

- Python 3.8+
- TensorFlow 2.x
- NumPy
- Pandas
- Pillow
- Matplotlib (notebook only)
- Jupyter Notebook or JupyterLab (notebook only)
- Streamlit (app only)

Install everything:

```bash
pip install -r requirements.txt jupyter matplotlib
```

## Usage

### Notebook

1. Launch Jupyter:
   ```bash
   jupyter notebook fashion_mnist_classifier.ipynb
   ```
2. Run all cells in order (**Cell → Run All**). This trains the model and saves it to `fashion_mnist_model.keras`.

### Streamlit App

1. Run:
   ```bash
   streamlit run app.py
   ```
2. On first launch, if `fashion_mnist_model.keras` isn't present, the app trains a model automatically (takes about a minute) and caches it; if the notebook already produced `fashion_mnist_model.keras`, the app loads it directly.
3. Use the **Upload an image** tab to classify your own clothing photo, or the **Random test sample** tab to try examples from the test set.
4. Adjust the training epoch slider or click **Retrain model** in the sidebar to force a fresh training run.

The Fashion MNIST dataset (60,000 training images, 10,000 test images, 28x28 grayscale) downloads automatically on first run via `tensorflow.keras.datasets.fashion_mnist`.

## Notebook Walkthrough

| Section | Description |
|---|---|
| 1. Imports | Loads TensorFlow, NumPy, and Matplotlib |
| 2. Load data | Loads the Fashion MNIST train/test splits |
| 3. Explore data | Displays a 5x5 grid of sample images with labels |
| 4. Preprocess | Normalizes pixels to [0, 1] and reshapes for CNN input |
| 5. Build model | Defines a 3-layer Conv2D CNN with dropout |
| 6. Compile | Sets optimizer, loss, and metrics |
| 7. Train | Fits the model for 10 epochs with a validation split |
| 8. Evaluate | Reports test accuracy and loss |
| 9. Plot history | Plots training vs. validation accuracy/loss curves |
| 10. Predictions | Visualizes predictions vs. true labels on sample test images |
| 11. Save model | Saves the trained model to `fashion_mnist_model.keras` |

## Streamlit App Features

| Feature | Description |
|---|---|
| Auto train / load | Loads `fashion_mnist_model.keras` if present, otherwise trains and caches a new model |
| Epoch slider | Controls training length when a model is trained from scratch |
| Retrain button | Clears the cache and forces a fresh training run |
| Upload tab | Classifies a user-uploaded image, auto-converts to grayscale, auto-inverts light backgrounds, and resizes to 28x28 |
| Random sample tab | Picks a random image from the test set and shows the true label alongside the prediction |
| Confidence chart | Bar chart of the model's confidence across all 10 classes |
| Training curves | Expandable view of accuracy/loss curves when a model is freshly trained |

## Model Architecture

```
Conv2D(32, 3x3, relu) → MaxPooling2D
Conv2D(64, 3x3, relu) → MaxPooling2D
Conv2D(64, 3x3, relu)
Flatten
Dense(64, relu) → Dropout(0.3)
Dense(10, softmax)
```

## Expected Results

With the default settings (10 epochs, batch size 64), the model typically reaches **~91-92% test accuracy**. Adjust epochs, batch size, or the architecture to experiment with performance.

## Class Labels

| Label | Class |
|---|---|
| 0 | T-shirt/top |
| 1 | Trouser |
| 2 | Pullover |
| 3 | Dress |
| 4 | Coat |
| 5 | Sandal |
| 6 | Shirt |
| 7 | Sneaker |
| 8 | Bag |
| 9 | Ankle boot |
