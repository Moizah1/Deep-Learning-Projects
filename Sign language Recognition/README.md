# Sign Language Recognition with Deep Learning

Recognize hand gestures from **images** and **video** and convert them into **text**, using a **CNN** for static signs and a **CNN-LSTM** for signs that involve motion. Everything runs in a single Jupyter notebook.

---

## Overview

| Part | Model | Input | Output | Dataset |
|------|-------|-------|--------|---------|
| A | CNN | One image | Letter (A–Z, `space`, `del`, `nothing`) | [ASL Alphabet](https://www.kaggle.com/datasets/grassknoted/asl-alphabet) |
| B | CNN-LSTM | Sequence of 16 video frames | Word (gloss) | [WLASL](https://www.kaggle.com/datasets/risangbaskoro/wlasl-processed) |
| C | Inference | Image, video file or webcam | Text | Both |

**Why two models?**
A fingerspelled letter is mostly static, so one frame is enough and a CNN works well. Most words are signed with motion. For those, a CNN extracts features from each frame and an LSTM learns how those features change over time.

```
Part A:  image ──► CNN ──► letter

Part B:  video (T frames) ──► CNN per frame (TimeDistributed) ──► (T, 1280)
                          ──► BiLSTM ──► LSTM ──► Dense ──► word
```

---

## Features

- CNN with built-in rescaling and augmentation (rotation, zoom, shift, contrast; no horizontal flip, since a flip turns a right hand into a left hand)
- CNN-LSTM using a frozen, pretrained MobileNetV2 as the per-frame feature extractor, with optional fine-tuning
- WLASL loader that uses the official train/val/test split, trims clips to the annotated frame range and crops to the signer's bounding box
- Evaluation with accuracy curves, classification report and confusion matrix
- Inference from a single image, a video file, or a live webcam (letters to sentence, or a rolling window for words)
- Automatic fallback to pseudo-sequences from ASL Alphabet if WLASL can't be downloaded. This only checks that the architecture runs and is not a real motion benchmark.

---

## Project structure

```
.
├── sign_language_recognition.ipynb   # the full pipeline
├── README.md
├── asl_cnn_best.keras                # created after training Part A
└── cnn_lstm_best.keras               # created after training Part B
```

---

## Requirements

- Python 3.9–3.11
- TensorFlow 2.12 or newer
- A GPU is strongly recommended (a free Colab or Kaggle GPU is fine)
- A Kaggle account and API key, for downloading the datasets

---

## Usage

Run the notebook top to bottom. The datasets download automatically on first use.

1. **Part A** downloads ASL Alphabet, trains the CNN, evaluates it, and defines `predict_letter(img)`.
2. **Part B** downloads WLASL (or builds the fallback), trains the CNN-LSTM in two phases and evaluates it.
3. **Part C** gives you the inference tools:

```python
# Single image -> letter
predict_letter(cv2.imread("my_hand.jpg"))

# Video file -> word
predict_video("clip.mp4")

# Live webcam -> sentence of letters (run locally)
text = run_webcam()            # q = quit, c = clear, backspace = delete

# Live webcam -> words using a sliding window (run locally)
sentence = run_webcam_words()
```

**Webcam tips:** hold the sign inside the green box, use a plain background and even lighting, and keep your hand steady. A letter is added after about 15 consecutive confident frames.

---

## Results

| Model | Dataset | Classes | Test accuracy |
|-------|---------|---------|---------------|
| CNN | ASL Alphabet | 29 | _your result_ |
| CNN-LSTM | WLASL | 20 | _your result_ |

---

## Acknowledgements

- **ASL Alphabet** dataset by Akash Nagaraj ([Kaggle](https://www.kaggle.com/datasets/grassknoted/asl-alphabet))
- **WLASL**: Li et al., *Word-level Deep Sign Language Recognition from Video: A New Large-scale Dataset and Methods Comparison*, WACV 2020. Processed copy by risangbaskoro on Kaggle.
- **MobileNetV2**: Sandler et al., 2018 (ImageNet-pretrained weights via Keras)
