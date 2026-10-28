# Review Sentiment Analyzer

A Jupyter notebook that classifies product reviews as **negative**, **neutral** or **positive** using a pre-trained transformer, then summarises the results by category and over time. It is set up for `amazon_reviews_labeled.csv`, a labelled dataset of 1,500 synthetic Amazon-style reviews.

It uses [`cardiffnlp/twitter-roberta-base-sentiment-latest`](https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest), a RoBERTa model trained on tweets. It handles short, informal text with emoji, slang and ALL-CAPS emphasis, and its neutral class means lukewarm reviews like "it's fine, nothing special" aren't forced into positive or negative.

## Contents

```
.
├── review_sentiment_analyzer.ipynb   # the notebook
├── amazon_reviews_labeled.csv        # 1,500 labelled synthetic reviews (the dataset)
├── sentiment_results.csv             # created when you run the notebook
└── README.md
```

## The dataset

`amazon_reviews_labeled.csv` is **synthetic**. It was generated from templates for **fictional products and brands**. It is not real Amazon data and contains no real customer reviews.

| | |
|---|---|
| Reviews | 1,500 |
| Products | 120 (fictional), across 12 categories such as Headphones & Earbuds, Kitchen Appliances, Beauty & Skincare, Clothing and Pet Supplies |
| Date range | 2025-01-01 to 2026-10-26 |
| Labels | positive 43% (649), negative 35% (532), neutral 21% (319) |
| Average length | about 29 words |

**Columns**

| Column | Description |
|---|---|
| `review_id` | Unique review ID (`R00001`, ...) |
| `product_id`, `product_name`, `category` | The product being reviewed |
| `platform` | Always `Amazon` |
| `date` | Review date (`YYYY-MM-DD`) |
| `rating` | Star rating, 1 to 5 |
| `title` | Short review headline |
| `comment` | The review text that the notebook analyses |
| `label` | Ground-truth sentiment: `negative`, `neutral` or `positive` |
| `verified_purchase` | `True` or `False` |
| `helpful_votes` | Number of "helpful" votes |

**How labels relate to ratings:** 1 to 2 stars are `negative`, 3 stars is `neutral`, 4 to 5 stars are `positive`. The dataset also includes harder cases: 4-star reviews with a minor complaint (labelled positive), 2-star reviews with some praise (labelled negative), balanced pros-and-cons reviews (neutral), and a small number of sarcastic 1-star reviews.

## What the notebook does

| Section | Description |
|---|---|
| 1. Setup | Imports, configuration, GPU/CPU detection |
| 2. Load and explore | Row counts, missing values, comments per platform, comment length |
| 3. Text normalisation | Replaces `@handles` and URLs with placeholders; keeps emoji and capitalisation |
| 4. Model and prediction | Runs the transformer and returns a probability for each class |
| 5. Analysis | Overall distribution, per-group mix, daily trend, most negative reviews, low-confidence predictions |
| 6. Evaluation | Compares predictions with the `label` column (accuracy, per-class metrics, confusion matrix) |
| 7. Try your own | `analyze("...")` scores any review you type |
| 8. Export | Saves everything to `sentiment_results.csv` |
| 9. Fine-tuning (optional) | Trains the model on your labelled data |

## Requirements

- Python 3.9 or newer
- About 1 GB of free disk space (model weights are about 500 MB and are cached after the first download)
- Internet access on the first run, to download the model from the Hugging Face Hub
- A GPU is optional. CPU is fine for 1,500 reviews.

## Setup

```bash
# 1. (Recommended) create a virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install transformers torch pandas numpy matplotlib seaborn scikit-learn accelerate jupyter

# 3. Launch the notebook
jupyter notebook review_sentiment_analyzer.ipynb
```

Before running, point the notebook at the new dataset. In the Setup cell, change:

```python
DATA_PATH = "amazon_reviews_labeled.csv"
```

## Output

`sentiment_results.csv` contains the original columns plus:

| Column | Meaning |
|---|---|
| `sentiment` | Predicted class: `negative`, `neutral` or `positive` |
| `confidence` | Probability of the predicted class (0 to 1) |
| `negative`, `neutral`, `positive` | Probability of each class |
| `net_score` | `positive − negative`, from −1 (very negative) to +1 (very positive) |
| `needs_review` | `True` if `confidence` is below `LOW_CONF` |

## Evaluating accuracy

The dataset includes ground-truth labels, so the sanity check against hand labels compares the model's predictions against the `label` column automatically and reports accuracy, precision, recall, F1 and a confusion matrix. Neutral reviews are likely to be the hardest class, since they sit between the other two.

## Credits

Model: Camacho-Collados et al., *TimeLMs: Diachronic Language Models from Twitter* (2022) and the [Cardiff NLP](https://huggingface.co/cardiffnlp) team. Check the model card on Hugging Face for its licence and terms of use.
