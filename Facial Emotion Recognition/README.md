# Real-Time Facial Emotion Recognition

Classify human facial expressions from a live webcam feed using a Convolutional Neural Network (CNN) trained on the **FER2013** dataset. Built with **OpenCV**, **TensorFlow**, and **Keras**, and run entirely from a Jupyter notebook.

The model recognizes 7 emotions: `angry`, `disgust`, `fear`, `happy`, `neutral`, `sad`, `surprise`.

## How it works

1. **Face detection:** OpenCV's Haar cascade finds faces in each webcam frame.
2. **Preprocessing:** each face is cropped, converted to grayscale, and resized to 48x48.
3. **Classification:** a CNN predicts a probability for each emotion.
4. **Display:** a colored box and label (emotion + confidence) are drawn on the frame. Predictions are averaged over the last 8 frames to reduce flicker.
