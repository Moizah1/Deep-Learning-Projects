# Review Sentiment Analyzer

A Jupyter notebook that classifies social-media reviews and comments as **negative**, **neutral** or **positive** using a pre-trained transformer, then summarises the results by platform and over time.

It uses [`cardiffnlp/twitter-roberta-base-sentiment-latest`](https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest), a RoBERTa model trained on tweets. It handles short, informal text with emoji, slang and ALL-CAPS emphasis, and its neutral class means comments like "meh" or "it's fine" aren't forced into positive or negative.

## Contents

```
.
├── review_sentiment_analyzer.ipynb   # the notebook
├── sample_data.csv                   # 20 example comments (no labels)
├── sentiment_results.csv             # created when you run the notebook
└── README.md
```

## What the notebook does

| Section | Description |
|---|---|
| 1. Setup | Imports, configuration, GPU/CPU detection |
| 2. Load and explore | Row counts, missing values, comments per platform, comment length |
| 3. Text normalisation | Replaces `@handles` and URLs with placeholders; keeps emoji and capitalisation |
| 4. Model and prediction | Runs the transformer and returns a probability for each class |
| 5. Analysis | Overall distribution, per-platform mix, daily trend, most negative comments, low-confidence predictions |
| 6. Sanity check | Compares predictions with labels (accuracy, per-class metrics, confusion matrix) |
| 7. Try your own | `analyze("...")` scores any review you type |
| 8. Export | Saves everything to `sentiment_results.csv` |
| 9. Fine-tuning (optional) | Trains the model on your own labelled data |

## Requirements

- Python 3.9 or newer
- About 1 GB of free disk space (model weights are about 500 MB and are cached after the first download)
- Internet access on the first run, to download the model from the Hugging Face Hub
- A GPU is optional. CPU is fine for datasets of this size.

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

Then run the cells from top to bottom (**Kernel → Restart & Run All**).

## Input data

A CSV file with one comment per row. The default file uses these columns:

| Column | Description |
|---|---|
| `platform` | Source of the comment (e.g. Twitter, Reddit) |
| `date` | Date of the comment (`YYYY-MM-DD`) |
| `comment` | The text to analyse |

To use your own file, change the settings in the Setup section of the notebook:

```python
DATA_PATH = "your_file.csv"
TEXT_COL  = "your_text_column"
```

The per-platform and trend sections expect `platform` and `date` columns. If your file doesn't have them, skip or adapt those cells.

## Configuration

All settings are in the Setup cell:

| Variable | Default | Meaning |
|---|---|---|
| `DATA_PATH` | `sample_data.csv` | Input CSV |
| `TEXT_COL` | `comment` | Column containing the text |
| `MODEL_ID` | `cardiffnlp/twitter-roberta-base-sentiment-latest` | Hugging Face model name or local path |
| `BATCH_SIZE` | `16` | Comments scored per batch |
| `MAX_LEN` | `128` | Maximum tokens per comment (longer text is truncated) |
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

`sample_data.csv` has no ground-truth labels, so Section 6 uses quick hand labels for the 20 sample rows, written by reading each comment. A few are ambiguous (for example "Honestly not bad"), and 20 rows is too few for a real benchmark, so treat the result as a rough check only.

To evaluate on your own data, add a `label` column with the values `negative`, `neutral` or `positive`. The notebook uses it automatically.

## Fine-tuning on your own data

Section 9 contains a `fine_tune()` function, switched off by default.

1. Add a `label` column to your CSV (`negative`, `neutral` or `positive`). A few hundred labelled rows or more is recommended. A handful of rows will overfit.
2. Set `RUN_FINETUNE = True` and run the cell.
3. The best model is saved to `./sentiment-finetuned`.
4. To use it, set `MODEL_ID = "./sentiment-finetuned"` in the Setup cell and re-run.

## Known limitations

- **Sarcasm** ("Great, another crash") is often misread as positive.
- **Mixed opinions** ("some good, some bad") usually come out as neutral or low-confidence.
- **Language:** the model is trained on English text. Other languages need a multilingual model.
- **Small samples:** per-platform and daily trend charts are noisy with only a few comments per group. Treat them as directional.
- **Domain terms:** product-specific jargon may be misjudged until you fine-tune.

## Troubleshooting

| Problem | Fix |
|---|---|
| `ModuleNotFoundError` | Run the `pip install` command from Setup in the same environment as your Jupyter kernel |
| Model download fails or hangs | Check your internet connection or proxy. Hugging Face must be reachable on the first run. |
| `CUDA out of memory` | Lower `BATCH_SIZE` (for example to 4) |
| Slow on CPU | Expected for large files. Increase `BATCH_SIZE` if you have RAM to spare, or use a GPU. |
| `FileNotFoundError` for the CSV | Keep the CSV in the same folder as the notebook, or set the full path in `DATA_PATH` |
| `TrainingArguments` error about `eval_strategy` | Upgrade with `pip install -U transformers` (older versions call it `evaluation_strategy`) |

## Credits

Model: Camacho-Collados et al., *TimeLMs: Diachronic Language Models from Twitter* (2022) and the [Cardiff NLP](https://huggingface.co/cardiffnlp) team. Check the model card on Hugging Face for its licence and terms of use.
