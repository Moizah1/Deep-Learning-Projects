# ⌨️ Predictive Keyboard: Next-Word Suggestion with LSTM / Transformer

A deep learning project that takes a sentence prefix (for example `"the president of the"`) and suggests the **next 1–3 words**, like a smartphone keyboard. It also handles a *partially typed* word (`"the presid"` → `president`).

## Features

- **Two interchangeable language models**: a 2-layer word-level **LSTM** and a 4-layer causal **Transformer** (GPT-style). Switch with one config value.
- **Top-k next-word prediction** over a 10k-word vocabulary.
- **Partial-word autocomplete**: suggestions are restricted to words starting with what the user has typed so far.
- **Beam-search phrase completion**: the k most probable multi-word continuations.
- **Confidence-based 1–3 word suggestions**: each top word is extended to 2–3 words only while the model is confident (default: conditional probability ≥ 0.5).
- **Keyboard-style evaluation**: perplexity plus top-1 / top-3 / top-5 next-word accuracy.
- **Interactive demo**: a text box with three suggestion buttons (ipywidgets).
- **Checkpointing**: save and reload weights, config and vocabulary.

## Project structure

| File | Description |
|---|---|
| `predictive_keyboard.ipynb` | Data pipeline, models, training, evaluation, keyboard engine and demo |
| `README.md` | This file |
| `checkpoints/` | Created at runtime; holds the saved model (`keyboard_<model>.pt`) |

## Requirements

- Python 3.9+
- `torch`, `datasets`, `ipywidgets`, `matplotlib` (the first notebook cell installs them with `%pip install`)
- A GPU is strongly recommended. The LSTM should train in roughly 20–40 s per epoch on a T4 (for example on the free Google Colab tier). On CPU, training is much slower, so reduce `EPOCHS`.

## Quick start

Open Or upload the notebook to [Google Colab](https://colab.research.google.com), choose a GPU runtime, and run all cells.

Then run all cells top to bottom. The last interactive cell shows a text box: start typing and click a suggestion to insert it.

## Configuration

All settings are in the `CFG` dictionary near the top of the notebook:

| Key | Default | Meaning |
|---|---|---|
| `MODEL_TYPE` | `"lstm"` | `"lstm"` or `"transformer"` |
| `VOCAB_SIZE` | `10000` | Most frequent words kept; the rest become `<unk>` |
| `SEQ_LEN` | `32` | Context window (words) the model sees |
| `BATCH_SIZE` | `128` | Training batch size |
| `EPOCHS` | `12` | Maximum epochs (early stopping with patience 3) |
| `CORPUS_PATH` | `None` | Path to your own `.txt` file (one paragraph or sentence per line) |
| `LSTM` / `TRANSFORMER` | see notebook | Architecture hyperparameters and learning rate |

## How it works

1. **Data.** WikiText-2 is loaded via Hugging Face `datasets`. Text is lowercased, wiki headings are dropped, digits become `<num>`, and `<eos>` is appended per line so the model learns how text starts. If the download fails, a tiny toy corpus is used as a smoke-test fallback only.
2. **Model.** Both models map `(batch, time)` word ids to next-word logits and share (tie) input and output embeddings.
3. **Training.** Cross-entropy on every position, AdamW, gradient clipping, OneCycle learning-rate schedule, early stopping on validation loss.
4. **Evaluation.** Perplexity and top-k accuracy, scored only where the true next token is a real word and ranking only real words, matching what the keyboard actually shows.
5. **Inference.** `PredictiveKeyboard` masks out special tokens and punctuation, applies the partial-word prefix filter, and exposes the API below.

## Usage

```python
kb = PredictiveKeyboard(model, vocab, CFG["SEQ_LEN"], device)

kb.next_words("the president of the", k=3)
# [(word, probability), ...]

kb.complete_phrases("in the year", n_words=3, k=3)
# beam search: [("phrase of three words", probability), ...]

kb.suggest("he said that the", k=3, max_words=3)
# keyboard output: [("1-3 word suggestion", first_word_probability), ...]

kb.suggest("the unit")           # partial word handled automatically
apply_suggestion("the presid", "president of")   # -> "the president of "
```

Reload a saved model later without retraining:

```python
kb = load_keyboard("checkpoints/keyboard_lstm.pt")
```

## Notes and limitations

- The default model is trained on Wikipedia text, so suggestions sound encyclopedic. For a messaging-style keyboard, train on chat or email text via `CORPUS_PATH`.
- The tokenizer is word-level: rare words become `<unk>` and typos are not corrected. Subword tokenization (BPE) would handle these better.
- Results (perplexity and accuracy) depend on hardware, seed and configuration; the notebook prints your actual numbers at the end of training.

## Ideas for extension

- Personalization: fine-tune on a user's own text or blend in a user-history cache.
- Subword tokenization (BPE / SentencePiece).
- Fine-tune a pretrained model such as `distilgpt2` and reuse the same `suggest()` logic.
- On-device deployment: dynamic quantization, then export to ONNX, TorchScript, Core ML or TFLite.
- Fuzzy matching of the partial word for typo tolerance.
- Metrics closer to real usage: keystroke savings and suggestion acceptance rate.

## Dataset

[WikiText-2](https://huggingface.co/datasets/wikitext) (Merity et al., "Pointer Sentinel Mixture Models"), licensed CC BY-SA 3.0.
