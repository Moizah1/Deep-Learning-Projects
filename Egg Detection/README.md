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

## 7. YOLOv11n Architecture / Model Explanation

YOLO ("You Only Look Once") predicts all boxes and classes in **one forward pass** through a CNN. YOLOv11n has three parts:

| Part | Role |
|---|---|
| **Backbone** | Stacked convolutional blocks (Conv, C3k2) with a spatial-pyramid-pooling layer (SPPF) and a self-attention block (C2PSA). It turns pixels into feature maps, from edges and curves up to egg-shaped parts. |
| **Neck** | A feature-pyramid / path-aggregation structure that merges feature maps from three scales, so small and large eggs are both detected. |
| **Head** | An anchor-free, decoupled head that predicts the box and the class score separately at each location. Overlapping duplicates are removed with Non-Maximum Suppression (NMS). |

Training loss = **box regression loss (CIoU) + classification loss (BCE) + distribution focal loss (DFL)**. The "n" (nano) variant has about 2.6 M parameters, which is why it is fast and suited to edge devices. Check the Ultralytics docs for the exact current figures.

