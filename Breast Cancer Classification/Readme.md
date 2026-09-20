# Breast Cancer Classification with a Neural Network

A feedforward neural network (multilayer perceptron) that classifies breast tumor samples as **malignant** or **benign** using the Wisconsin Breast Cancer diagnostic dataset.

## Results

| Metric | Score |
|---|---|
| Test Accuracy | 94.7% |
| Test ROC AUC | 0.993 |

Confusion matrix, ROC curve, and training loss curve are included in `results.png` and inline in the notebook.

## Files

| File | Description |
|---|---|
| `breast_cancer_nn.ipynb` | Jupyter notebook — full walkthrough, already executed with outputs |
| `breast_cancer_nn.py` | Same pipeline as a standalone script |
| `data.csv` | Wisconsin Breast Cancer diagnostic dataset (569 samples, 30 features) |
| `results.png` | Loss curve, confusion matrix, and ROC curve, saved as one figure |

## Dataset

Each row is a digitized image of a breast mass, described by 30 real-valued features (mean, standard error, and "worst" value for 10 measurements: radius, texture, perimeter, area, smoothness, compactness, concavity, concave points, symmetry, fractal dimension).

- **Target**: `diagnosis` — `M` (malignant) or `B` (benign)
- **Samples**: 569 (357 benign, 212 malignant)
- This is the same dataset as `sklearn.datasets.load_breast_cancer()`, provided here as a raw CSV.

## Model

A `scikit-learn` `MLPClassifier`:

- 2 hidden layers: 32 → 16 neurons
- ReLU activation, Adam optimizer
- L2 regularization (`alpha=1e-4`)
- Early stopping on a held-out validation split

> TensorFlow/Keras is not installed in this environment by default, so scikit-learn's MLP is used. The same architecture can be swapped into a `keras.Sequential` model if you want finer control (dropout, batch norm, custom callbacks).

## Pipeline

1. Load `data.csv`, drop the `id` column
2. Encode `diagnosis` (M → 1, B → 0)
3. Stratified 80/20 train/test split
4. Standardize features with `StandardScaler` (fit on train only)
5. Train the MLP with early stopping
6. Evaluate: accuracy, ROC AUC, classification report, confusion matrix
7. Plot training loss curve, confusion matrix, and ROC curve

## Usage

### Notebook
Open `breast_cancer_nn.ipynb` in Jupyter and run all cells (already executed — outputs are visible without re-running).

### Script
```bash
pip install pandas numpy scikit-learn matplotlib
python breast_cancer_nn.py
```

Both expect `data.csv` to be in the same directory.

## Requirements

- Python 3.8+
- `pandas`, `numpy`, `scikit-learn`, `matplotlib`

## Possible Extensions

- K-fold cross-validation
- Grid/random search over `hidden_layer_sizes` and `alpha`
- Learning curve to check for over/underfitting
- Swap in a Keras/PyTorch model for more architectural control