# Real-Time Facial Emotion Recognition

Classify human facial expressions from a live webcam feed using a Convolutional Neural Network (CNN) trained on the **FER2013** dataset. Built with **OpenCV**, **TensorFlow**, and **Keras**, and run entirely from a Jupyter notebook.

The model recognizes 7 emotions: `angry`, `disgust`, `fear`, `happy`, `neutral`, `sad`, `surprise`.

## How it works

1. **Face detection:** OpenCV's Haar cascade finds faces in each webcam frame.
2. **Preprocessing:** each face is cropped, converted to grayscale, and resized to 48x48.
3. **Classification:** a CNN predicts a probability for each emotion.
4. **Display:** a colored box and label (emotion + confidence) are drawn on the frame. Predictions are averaged over the last 8 frames to reduce flicker.

## Project structure

```
.
├── emotion_recognition.ipynb   # Data loading, training, evaluation, live demo
├── emotion_cnn.keras           # Trained model (created after training)
├── README.md
└── fer2013/                    # Dataset (download separately)
    ├── train/
    │   ├── angry/ disgust/ fear/ happy/ neutral/ sad/ surprise/
    └── test/
        └── angry/ disgust/ fear/ happy/ neutral/ sad/ surprise/
```

## Requirements

- Python 3.9 - 3.11
- A webcam (for the live demo; training and image testing work without one)
- A GPU is optional but speeds up training considerably

Install dependencies:

```bash
pip install tensorflow opencv-python numpy matplotlib seaborn scikit-learn jupyter
```

Add `pandas` if you want to use the optional CSV loader.

## Getting started

1. **Download the dataset** from Kaggle (for example [`msambare/fer2013`](https://www.kaggle.com/datasets/msambare/fer2013)) and extract it so the folder layout matches the structure above. Update `DATA_DIR` in the notebook if you put it elsewhere.
   - If you only have the original `fer2013.csv`, set `USE_CSV = True` in the optional loader cell.
2. **Launch Jupyter:**
   ```bash
   jupyter notebook emotion_recognition.ipynb
   ```
3. **Run the notebook top to bottom.** Training saves the best model to `emotion_cnn.keras`.
4. **Start the live demo** in section 7.

## Notebook sections

| Section | Description |
|---|---|
| 0-1 | Imports and configuration (image size, batch size, epochs, paths) |
| 2 | Load FER2013, preview samples, plot class balance, compute class weights |
| 3 | Build the CNN (4 conv blocks, batch norm, dropout, global average pooling) |
| 4 | Train with checkpointing, early stopping, and learning-rate reduction |
| 5 | Evaluate: accuracy/loss curves, classification report, confusion matrix |
| 6 | Real-time inference helpers (face detection, preprocessing, annotation) |
| 7a | Live webcam in an OpenCV window |
| 7b | Live webcam rendered inline in the notebook |
| 7c | Predict on a single image (no webcam needed) |

## Running the live demo

**OpenCV window (7a):** run the cell and press `q` in the video window to quit. If the window freezes, run the cleanup cell right after it.

**Inline (7b):** uncomment `run_webcam_inline()` and run it. Stop it with the notebook's **Interrupt kernel** button; the camera is released automatically.

**Single image (7c):** uncomment `predict_image("my_photo.jpg")` and point it at any image containing a face.

To skip training and use an existing model, run the imports and section 6, and load the model first:

```python
model = keras.models.load_model("emotion_cnn.keras")
class_names = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
```

## Model details

- **Input:** 48x48x1 grayscale image (rescaled to 0-1 inside the model)
- **Augmentation (training only):** horizontal flip, rotation, zoom, translation
- **Architecture:** 4 blocks of two 3x3 convolutions with batch norm, ReLU, max pooling, and dropout (32 -> 64 -> 128 -> 256 filters), then global average pooling, a 256-unit dense layer, and a 7-way softmax
- **Loss / optimizer:** categorical cross-entropy, Adam (lr 1e-3, reduced on plateau)
- **Class imbalance:** handled with balanced class weights

### Expected performance

A custom CNN on FER2013 typically reaches about **60-68% test accuracy**. This is normal: the dataset has noisy labels and human agreement is only around 65%. Exact results vary by run and hardware.

## Troubleshooting

- **Webcam won't open:** try `camera_index=1`, close other apps using the camera, and on macOS grant camera permission to your terminal or Jupyter. Cloud notebooks (Colab), remote servers, and WSL generally cannot access a local webcam.
- **Wrong or odd labels:** `class_names` must match the training order (alphabetical for the folder dataset).
- **Too many "neutral" or "sad" predictions:** use good frontal lighting, keep the face reasonably large in frame, and keep class weighting enabled.
- **Out-of-memory during training:** lower `BATCH_SIZE`.
- **No faces detected:** face the camera directly; you can adjust `scaleFactor` and `minNeighbors` in `annotate_frame`.

## Ideas for improvement

- Transfer learning (MobileNetV2 / EfficientNet) with upscaled faces
- Label smoothing, stronger augmentation, or model ensembling
- A better face detector (OpenCV DNN or MediaPipe) for difficult angles and lighting
- Tracking faces across frames for more stable per-person smoothing

## Notes and limitations

Emotion recognition from facial expressions is an imperfect proxy for how a person actually feels, and models can perform unevenly across demographics, lighting, and camera quality. Use this project for learning and experimentation, not for decisions that affect people. Get consent before processing anyone's face.

## Acknowledgments

- **FER2013** dataset: Goodfellow et al., "Challenges in Representation Learning: A report on three machine learning contests" (ICML 2013 workshop)
- [TensorFlow / Keras](https://www.tensorflow.org/) and [OpenCV](https://opencv.org/)
