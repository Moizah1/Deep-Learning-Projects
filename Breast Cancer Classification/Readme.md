# Breast Cancer Classification with a Neural Network

Feedforward neural networks that classify breast tumor samples as **malignant** or **benign** using the Wisconsin Breast Cancer diagnostic dataset. The notebook builds and trains two versions — a Keras/TensorFlow `Sequential` model and a scikit-learn `MLPClassifier` — and ends with a small predictive system that scores a single new sample.

## Results

| Model | Test Accuracy | Test ROC AUC |
|---|---|---|
| Keras `Sequential` NN | 0.9737 | — |
| scikit-learn `MLPClassifier` | 0.9474 | 0.993 |

Confusion matrix, ROC curve, and training loss curve (for the MLP) are included in `results.png` and inline in the notebook.

## Files

| File | Description |
|---|---|
| `breast_cancer.ipynb` | Jupyter notebook — full walkthrough, both models, and the prediction demo |
| `data.csv` | Wisconsin Breast Cancer diagnostic dataset (569 samples, 30 features) |
| `results.png` | Loss curve, confusion matrix, and ROC curve, saved as one figure |

## Dataset

Each row is a digitized image of a breast mass, described by 30 real-valued features (mean, standard error, and "worst" value for 10 measurements: radius, texture, perimeter, area, smoothness, compactness, concavity, concave points, symmetry, fractal dimension).

- **Target**: `diagnosis` — `M` (malignant) or `B` (benign)
- **Samples**: 569 (357 benign, 212 malignant)
- This is the same dataset as `sklearn.datasets.load_breast_cancer()`, provided here as a raw CSV.

## Models

The notebook trains two neural networks on the same preprocessed data, back to back:

**1. Keras `Sequential`**
- `Flatten` input layer → `Dense(20, activation="relu")` hidden layer → `Dense(2, activation="sigmoid")` output layer
- Compiled with `optimizer="adam"`, `loss="sparse_categorical_crossentropy"`
- Trained for 10 epochs with a 10% validation split

**2. scikit-learn `MLPClassifier`**
- 2 hidden layers: 32 → 16 neurons
- ReLU activation, Adam optimizer
- L2 regularization (`alpha=1e-4`)
- Early stopping on a held-out validation split

The evaluation, plots, and the predictive system below all use the scikit-learn MLP (`model` is reassigned to the `MLPClassifier` after the Keras section).

## Pipeline

1. Load `data.csv`, drop the `id` column
2. Encode `diagnosis` (M → 1, B → 0)
3. Stratified 80/20 train/test split
4. Standardize features with `StandardScaler` (fit on train only)
5. Train the Keras `Sequential` model and evaluate it with `model.evaluate()`
6. Train the scikit-learn `MLPClassifier` with early stopping
7. Evaluate the MLP: accuracy, ROC AUC, classification report, confusion matrix
8. Plot training loss curve, confusion matrix, and ROC curve
9. Run a single custom sample through the trained MLP as a predictive-system demo

## Usage

### Notebook
Open `breast_cancer.ipynb` in Jupyter and run all cells (already executed — outputs are visible without re-running).

Both expect `data.csv` to be in the same directory.

## Predictive System

The last section of the notebook shows how to score a single new sample:

1. Provide the 30 feature values as a tuple (in the same column order as the training data)
2. Reshape to a single row and apply the fitted `scaler`
3. Call `model.predict()` / `model.predict_proba()` on the scaled sample

This is a template for scoring new patients — swap in real feature values for `input_data`.

## Requirements

- Python 3.8+
- `pandas`, `numpy`, `scikit-learn`, `matplotlib`
- `tensorflow` (for the Keras model section)
