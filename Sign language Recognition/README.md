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

`requirements.txt`:

```
tensorflow>=2.12
opencv-python
kagglehub
scikit-learn
matplotlib
seaborn
numpy
jupyter
```

---

## Setup

```bash
# 1. Create an environment (optional but recommended)
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Add Kaggle credentials (pick one)
#    a) Put kaggle.json in ~/.kaggle/  (Windows: C:\Users\<you>\.kaggle\)
#    b) Or set environment variables:
export KAGGLE_USERNAME="your_username"
export KAGGLE_KEY="your_api_key"

# 4. Launch
jupyter notebook sign_language_recognition.ipynb
```

You can get `kaggle.json` from Kaggle: **Settings → API → Create New Token**.

**Google Colab:** upload the notebook, enable **Runtime → Change runtime type → GPU**, and run the first cell with `%pip install ...` uncommented. The webcam cells (C2 and C3) won't work on Colab. Run those locally.

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

## Configuration

Edit these near the top of the relevant cells:

| Variable | Default | What it controls |
|----------|---------|------------------|
| `IMG_SIZE` | `64` | Input size for the CNN. Raise to 96 or 128 for more accuracy at the cost of speed. |
| `BATCH` | `64` | CNN batch size. |
| `EPOCHS` | `20` | Maximum CNN epochs (early stopping is on). |
| `NUM_WORDS` | `20` | Number of WLASL glosses to use (top-N by sample count). |
| `SEQ_LEN` | `16` | Frames sampled per clip. |
| `SEQ_SIZE` | `96` | Frame resolution fed to MobileNetV2. |
| `USE_WLASL` | `True` | Set to `False` to force the pseudo-sequence fallback. |
| `conf_thr`, `stable_frames` | `0.90`, `15` | Webcam confidence and stability thresholds. |

Start with the defaults, confirm everything runs, then scale up (more words, higher resolution).

---

## Results

Fill this in after training on your machine:

| Model | Dataset | Classes | Test accuracy |
|-------|---------|---------|---------------|
| CNN | ASL Alphabet | 29 | _your result_ |
| CNN-LSTM | WLASL | 20 | _your result_ |

**What to expect**
- ASL Alphabet scores are typically very high on the held-out split. This overstates real-world performance, because every image shares the same background, lighting and signer.
- WLASL is a hard benchmark with few videos per word, many signers and noisy labels. A small CNN-LSTM will give modest accuracy even on 20 words, and much lower as the vocabulary grows.

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `401` / auth error from `kagglehub` | Check `kaggle.json` or the `KAGGLE_USERNAME` / `KAGGLE_KEY` variables. |
| Train folder not found | Print the downloaded path and check the folder layout. The notebook searches for `asl_alphabet_train` automatically, but versions vary. |
| Out-of-memory when loading WLASL | Lower `NUM_WORDS`, `SEQ_LEN` or `SEQ_SIZE`. All clips are loaded into RAM as `uint8`. |
| Out-of-memory on the GPU | Reduce `BATCH` (CNN) or the batch size in `make_ds` (CNN-LSTM). |
| Webcam window doesn't open | Run locally rather than on Colab or a remote server, and check the camera index in `cv2.VideoCapture(0)`. |
| Good test accuracy but poor webcam predictions | Domain shift. Use a plain background, crop tightly around the hand, or fine-tune on your own captured frames. |

---

## Limitations

- Fingerspelling covers only the alphabet. It is not full sign language, which also uses facial expression, body posture and grammar.
- The models use raw pixels, so they are sensitive to background, lighting and skin tone.
- The CNN-LSTM recognizes isolated words, not continuous signing.
- The ASL Alphabet dataset has a single signer and background, so it generalizes poorly.

---

## Ideas for improvement

1. **MediaPipe Hands/Holistic landmarks** as input instead of pixels. They are far more robust to background and lighting. Feed 21×3 landmarks per hand per frame into the LSTM.
2. Fine-tune on your own webcam data (a few hundred frames per letter helps a lot).
3. Try a 3D CNN (I3D) or a video transformer for the temporal part.
4. Add a language model or dictionary to turn letter streams into words.
5. Export to TFLite or ONNX for mobile and edge deployment.
6. Wrap inference in a small Streamlit or Gradio app.

---

## Acknowledgements

- **ASL Alphabet** dataset by Akash Nagaraj ([Kaggle](https://www.kaggle.com/datasets/grassknoted/asl-alphabet))
- **WLASL**: Li et al., *Word-level Deep Sign Language Recognition from Video: A New Large-scale Dataset and Methods Comparison*, WACV 2020. Processed copy by risangbaskoro on Kaggle.
- **MobileNetV2**: Sandler et al., 2018 (ImageNet-pretrained weights via Keras)

Check each dataset's license and terms of use before redistributing data or models trained on it.
