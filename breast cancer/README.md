# Breast Cancer Prediction using Deep Learning

A neural network built with TensorFlow/Keras that classifies breast tumors as malignant or benign from five cell-nucleus measurements.

> **Disclaimer:** This project is for learning and research only. It is not a medical diagnosis tool.

## Results

Evaluated on a held-out test set of 86 samples:

| Metric | Value |
|---|---|
| Accuracy | 87.2% |
| ROC-AUC | 0.966 |

| Class | Precision | Recall | F1-score | Support |
|---|---|---|---|---|
| Malignant (0) | 0.77 | 0.94 | 0.85 | 32 |
| Benign (1) | 0.96 | 0.83 | 0.89 | 54 |

The model catches 94% of malignant cases, at the cost of wrongly flagging some benign ones.

## Dataset

`Breast_cancer_data.csv` has 569 samples, 5 numeric features and a binary target, with no missing values.

| Column | Description |
|---|---|
| `mean_radius` | Mean distance from the center to points on the perimeter |
| `mean_texture` | Standard deviation of gray-scale values |
| `mean_perimeter` | Mean size of the core tumor perimeter |
| `mean_area` | Mean area of the tumor |
| `mean_smoothness` | Mean local variation in radius lengths |
| `diagnosis` | Target: 0 = malignant (212 samples), 1 = benign (357 samples) |

The label mapping follows the standard Wisconsin Breast Cancer dataset. Check that it matches your data source.

## Project structure

```
breast_cancer_project/
├── Breast_cancer_data.csv         # dataset
├── Breast_Cancer_Prediction.ipynb # notebook
├── requirements.txt
├── README.md
└── outputs/                       # generated when you run the code
    ├── breast_cancer_model.keras  # trained model
    ├── best_model.keras           # best checkpoint (lowest validation loss)
    ├── scaler.pkl                 # fitted StandardScaler
    ├── metrics.json               # accuracy, AUC, confusion matrix
    └── *.png                      # plots (training curves, ROC, confusion matrix, etc.)
```

## Method

1. **Split:** stratified 70/15/15 train/validation/test split (398/85/86 samples).
2. **Scaling:** `StandardScaler` fit on the training set only, to avoid data leakage.
3. **Class imbalance:** balanced class weights during training.
4. **Model:** Dense(64) → Dense(32) → Dense(16) → sigmoid output, with batch normalization, dropout (0.3) and L2 regularization. About 3,400 parameters.
5. **Training:** Adam optimizer, binary cross-entropy loss, batch size 32, up to 200 epochs. Early stopping (patience 20) and learning-rate reduction on plateau.
6. **Evaluation:** accuracy, precision, recall, F1, ROC-AUC, confusion matrix and permutation feature importance.

## Getting started

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

For the notebook you also need Jupyter: `pip install jupyter`.

### 2. Train and evaluate

Run the notebook step by step (the CSV must be in the same folder):

```bash
jupyter notebook Breast_Cancer_Prediction.ipynb
```

Save the model, scaler, metrics and plots to `outputs/`.

## Tech stack

Python, TensorFlow/Keras, scikit-learn, pandas, NumPy, Matplotlib, Seaborn, Joblib
