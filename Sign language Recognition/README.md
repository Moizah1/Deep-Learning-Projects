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
