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

## 8. Training Procedure

1. Load pretrained `yolo11n.pt` (COCO). Its convolutional filters already encode general visual features.
2. Fine-tune on the egg dataset (**transfer learning**) with the Ultralytics training pipeline, which includes built-in augmentation (mosaic, flips, HSV colour jitter, scaling).
3. Monitor validation metrics every epoch; **early stopping** ends training when they stop improving.
4. `best.pt` and `last.pt` checkpoints are saved to `runs/egg_yolo11n/weights/`; the best model is copied to `models/egg_yolo11n_best.pt`.

| Parameter | Default | CLI flag |
|---|---|---|
| Image size | 640 | `--imgsz` |
| Batch size | 16 | `--batch` |
| Epochs | 100 | `--epochs` |
| Initial learning rate | 0.01 | `--lr0` |
| Early-stopping patience | 20 | `--patience` |
| Device | CUDA if available, else CPU | automatic |

## 9. Evaluation Metrics

- **Precision**: of the eggs the model reported, how many were real.
- **Recall**: of the real eggs, how many were found.
- **mAP@50**: mean average precision at IoU 0.50.
- **mAP@50-95**: mAP averaged over IoU 0.50-0.95 (stricter, localisation quality).
- **Validation loss**: box / cls / DFL loss per epoch, read from the training log.
- **Egg-count accuracy**: predicted vs ground-truth eggs per image (MAE and exact-match %).
- **Plots**: confusion matrix, Precision-Recall curve, F1-confidence curve, training/validation loss curves, sample predictions.

## 10. Results

> Results depend on the Kaggle dataset you choose and your hardware, so **run the project and fill in your own numbers** (printed by `python src/evaluate.py`, saved in `runs/eval_test/` and `outputs/`).

| Metric (test set) | Value |
|---|---|
| Precision | 0.9492 |
| Recall | 0.9695 |
| mAP@50 | 0.9788 |
| mAP@50-95 | 0.7194 |
| Count MAE (eggs / image) | 11.0000 |
| Count exact-match | 0.0000 |

<img width="3000" height="2250" alt="confusion_matrix" src="https://github.com/user-attachments/assets/2ab95554-31e7-416d-a139-43859007a542" />

