# 🥚 Egg Detection System using Deep Learning and YOLOv11n

A deep-learning object detection system that **locates and counts eggs** in images, videos and live camera streams. A COCO-pretrained **YOLOv11n** convolutional neural network is fine-tuned on a Kaggle egg dataset (transfer learning), then used to draw bounding boxes, confidence scores and the total egg count.

---

## 1. Project Introduction

Counting eggs by hand on farms, in packing lines and in grocery inspection is slow and error-prone. This project trains a lightweight detector, **YOLOv11n** (the "nano" model of Ultralytics YOLO11), that runs fast enough for real-time use, even on a CPU or a modest GPU.

## 2. Problem Statement

Given an image or video frame, automatically find every egg, mark it with a bounding box and report how many eggs are present, robustly across lighting, camera angle, egg colour and partial occlusion.

## 3. Objectives

- Build a single-class (`0: egg`) object detector with YOLOv11n.
- Download and prepare a Kaggle dataset programmatically in YOLO format.
- Fine-tune pretrained weights, with GPU auto-detection, early stopping and checkpointing.
- Evaluate with Precision, Recall, mAP@50, mAP@50-95, confusion matrix, PR and F1 curves, and egg-count accuracy.
- Provide prediction for a single image, many images, video and webcam, with an adjustable confidence threshold.

## 4. Technologies Used

Python · PyTorch · Ultralytics YOLOv11n · OpenCV · NumPy · Matplotlib · Pandas · Kaggle API · Jupyter Notebook / VS Code

## 5. Dataset Source

The data comes from **Kaggle**, downloaded via the official Kaggle API. The dataset is selected by one setting, `KAGGLE_DATASET` in `src/utils.py` (or the `EGG_KAGGLE_DATASET` environment variable):

```bash
python src/utils.py search --term "egg detection"      # list candidate datasets
export EGG_KAGGLE_DATASET="owner/dataset-slug"          # choose one (Windows: set ...)
```

Choose a dataset that has **YOLO-format `.txt` annotations** (or YOLO-segmentation polygons, which are converted to boxes). Always check the dataset's licence before reuse and cite it in your report.

### Configure `kaggle.json` securely

1. Kaggle → profile picture → **Settings → API → Create New Token** (downloads `kaggle.json`).
2. Move it to `~/.kaggle/kaggle.json` and restrict access:
   ```bash
   mkdir -p ~/.kaggle && mv ~/Downloads/kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json
   ```
   (Windows: `C:\Users\<you>\.kaggle\kaggle.json`.)
3. Or use environment variables: `KAGGLE_USERNAME` and `KAGGLE_KEY`.


