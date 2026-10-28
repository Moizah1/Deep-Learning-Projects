# Review Sentiment Analyzer

A Jupyter notebook that classifies product reviews as **negative**, **neutral** or **positive** using a pre-trained transformer, then summarises the results by category and over time. It is set up for `amazon_reviews_labeled.csv`, a labelled dataset of 1,500 synthetic Amazon-style reviews.

It uses [`cardiffnlp/twitter-roberta-base-sentiment-latest`](https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest), a RoBERTa model trained on tweets. It handles short, informal text with emoji, slang and ALL-CAPS emphasis, and its neutral class means lukewarm reviews like "it's fine, nothing special" aren't forced into positive or negative.

## Contents

```
.
├── review_sentiment_analyzer.ipynb   # the notebook
├── amazon_reviews_labeled.csv        # 1,500 labelled synthetic reviews (the dataset)
├── generate_amazon_reviews.py        # script that creates the dataset (standard library only)
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

**Regenerating or resizing the dataset**

```bash
python generate_amazon_reviews.py                                # defaults: 1,500 reviews, 120 products
python generate_amazon_reviews.py --n-reviews 2000 --n-products 150 --seed 7 --out my_reviews.csv
```

The default seed (`42`) reproduces the included file exactly.

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

(The notebook's default is still `sample_data.csv`, the original 20 social-media comments.) Then run the cells from top to bottom (**Kernel → Restart & Run All**).

## Input data format

The notebook needs a CSV with these columns:

| Column | Required | Used for |
|---|---|---|
| `comment` | Yes | The text to analyse (change `TEXT_COL` to use another column) |
| `date` | Yes | Daily trend chart |
| `platform` | Yes | Per-group chart in Section 5.2 |
| `label` | No | Evaluation in Section 6 and fine-tuning in Section 9 |

**Tip:** every row in the Amazon dataset has `platform = "Amazon"`, so the per-platform chart in Section 5.2 shows a single bar. For a more useful breakdown, group by product category instead. In that cell, replace `"platform"` with `"category"` in the `groupby` and `value_counts` calls.

## Configuration

All settings are in the Setup cell:

| Variable | Default | Meaning |
|---|---|---|
| `DATA_PATH` | `sample_data.csv` | Input CSV. Set to `amazon_reviews_labeled.csv`. |
| `TEXT_COL` | `comment` | Column containing the text |
| `MODEL_ID` | `cardiffnlp/twitter-roberta-base-sentiment-latest` | Hugging Face model name or local path |
| `BATCH_SIZE` | `16` | Reviews scored per batch |
| `MAX_LEN` | `128` | Maximum tokens per review (longer text is truncated) |
| `LOW_CONF` | `0.60` | Predictions below this confidence are flagged for human review |

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

The dataset includes ground-truth labels, so Section 6 compares the model's predictions against the `label` column automatically and reports accuracy, precision, recall, F1 and a confusion matrix. Neutral reviews are likely to be the hardest class, since they sit between the other two.

**Read the scores with care.** The reviews are built from templates, so the wording is more repetitive and regular than real customer text. Results on this dataset will probably look better than on real Amazon reviews. Use it to check that the pipeline works end to end, and test on real reviews before drawing conclusions about real-world accuracy.

## Fine-tuning on the labelled data

Section 9 contains a `fine_tune()` function, switched off by default. The dataset now has a `label` column and 1,500 rows, so it is large enough to try.

1. Set `RUN_FINETUNE = True` and run the cell.
2. The function holds out 20% of the data for validation. The best model is saved to `./sentiment-finetuned`.
3. To use it, set `MODEL_ID = "./sentiment-finetuned"` in the Setup cell and re-run.

Because the reviews are templated, a fine-tuned model can score very high here by learning the templates rather than general sentiment. Don't expect the same performance on real reviews. Fine-tuning on real labelled reviews will give a more honest result.

## Known limitations

- **Synthetic data:** see above. The dataset is for prototyping and testing, not for benchmarking.
- **Sarcasm** ("Oh great, the battery life is terrible. Thanks a lot.") is often misread as positive. The dataset contains a few such cases.
- **Mixed opinions** usually come out as neutral or low-confidence.
- **Language:** the model is trained on English text. Other languages need a multilingual model.
- **Domain terms:** product-specific jargon may be misjudged until you fine-tune.

## Troubleshooting

| Problem | Fix |
|---|---|
| `ModuleNotFoundError` | Run the `pip install` command from Setup in the same environment as your Jupyter kernel |
| Model download fails or hangs | Check your internet connection or proxy. Hugging Face must be reachable on the first run. |
| "Unauthenticated requests to the HF Hub" warning | Harmless. Set an `HF_TOKEN` or run `huggingface-cli login` to silence it. |
| `CUDA out of memory` | Lower `BATCH_SIZE` (for example to 4) |
| Slow on CPU | Expected for large files. Increase `BATCH_SIZE` if you have RAM to spare, or use a GPU. |
| `FileNotFoundError` for the CSV | Keep the CSV in the same folder as the notebook, or set the full path in `DATA_PATH` |
| `KeyError: 'platform'` or `'date'` | Your CSV is missing a required column. Add it or adapt the cells that use it. |
| Section 6 reports no labels | The CSV has no `label` column, or its values aren't `negative`/`neutral`/`positive` |
| `TrainingArguments` error about `eval_strategy` | Upgrade with `pip install -U transformers` (older versions call it `evaluation_strategy`) |

## Credits

Model: Camacho-Collados et al., *TimeLMs: Diachronic Language Models from Twitter* (2022) and the [Cardiff NLP](https://huggingface.co/cardiffnlp) team. Check the model card on Hugging Face for its licence and terms of use.