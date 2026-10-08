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
4. **Never commit it or paste it in a notebook.** `.gitignore` already excludes `kaggle.json`. If a key leaks, revoke it in Kaggle settings.

## 6. Dataset Preparation

`utils.prepare_dataset()` converts whatever layout the download has into:

```text
dataset/
├── images/{train,val,test}/
├── labels/{train,val,test}/
└── data.yaml
```

What it does automatically:

- Respects existing `train / valid / test` folders; randomly splits unsplit data (70/20/10).
- Creates a test split if the source has none.
- Maps all annotated objects (or only `--class-ids` you choose) to **class 0 = egg**.
- Converts polygon labels to bounding boxes, clips boxes to the image and drops invalid ones.
- Skips images without a label file and reports it.

`data.yaml`:

```yaml
path: /absolute/path/to/dataset     # written automatically
train: images/train
val: images/val
test: images/test
nc: 1
names:
  0: egg
```

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
| Precision | _fill in_ |
| Recall | _fill in_ |
| mAP@50 | _fill in_ |
| mAP@50-95 | _fill in_ |
| Count MAE (eggs / image) | _fill in_ |
| Count exact-match | _fill in_ |

Add your figures here, e.g. `![Confusion matrix](runs/eval_test/confusion_matrix.png)`.

## 11. How to Run the Project

```bash
# 1. Environment
python -m venv .venv && source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
# (for an NVIDIA GPU, install the CUDA build of PyTorch from https://pytorch.org first)

# 2. Dataset
export EGG_KAGGLE_DATASET="owner/dataset-slug"
python src/utils.py download
python src/utils.py prepare            # add --class-ids 0 to keep only specific source classes
python src/utils.py check

# 3. Train, then evaluate
python src/train.py --epochs 100 --batch 16 --imgsz 640
python src/evaluate.py --split test --conf 0.5
```

Or open `notebooks/egg_detection.ipynb` and run all cells in order. It covers every step, including plots.

## 12. Image / Video / Webcam Prediction

```bash
python src/predict.py image  --source path/to/egg.jpg   --conf 0.5 --show
python src/predict.py images --source path/to/folder    --conf 0.4
python src/predict.py video  --source path/to/video.mp4 --conf 0.5
python src/predict.py webcam --camera 0                  # q = quit, s = snapshot
```

Change the threshold with `--conf`, or edit `CONFIDENCE_THRESHOLD = 0.5` in `src/predict.py`. Only detections at or above it are drawn. Results are saved under `outputs/images/`, `outputs/videos/` and `outputs/webcam/`.

## 13. Example Output

Each result shows the original and the annotated image side by side. Every egg gets a box labelled `egg 0.87`, and the frame shows the banner:

```text
Total Eggs Detected: 12
```

Add your own screenshots from `outputs/images/` here.

## 14. Future Improvements

- Train on more varied data (different breeds, backgrounds, lighting, packed trays).
- Compare against larger models (YOLO11s/m) or other detectors.
- Add egg-quality classes (cracked, dirty, size grade) as extra classes.
- Add object tracking so that eggs on a moving conveyor are counted once.
- Export to ONNX/TensorRT/TFLite for Raspberry Pi or Jetson deployment.
- Build a Streamlit or Gradio web demo.

---

## Why this is a Deep Learning Project

- **CNN-based feature extraction.** YOLOv11n's backbone is a deep stack of convolutional layers. Their weights are *learned from data* by back-propagation, not hand-designed. Early layers learn edges and textures, deeper layers learn egg shape, shading and context.
- **YOLO-based object detection.** The network regresses bounding boxes and class probabilities directly from the feature maps in a single pass, trained end to end with a multi-part loss (CIoU + BCE + DFL).
- **Transfer learning.** COCO-pretrained weights are fine-tuned on egg images with a gradient-based optimiser on GPU/CPU.
- **Representation learning, not rules.** No colour thresholds or Hough circles are used. The network learns to detect eggs from labelled examples and generalises to new images.

## Project Structure

```text
Egg-Detection-YOLOv11n/
├── dataset/            # prepared YOLO dataset (generated)
├── models/             # final weights (egg_yolo11n_best.pt)
├── outputs/            # predictions, plots, count CSV
├── runs/               # Ultralytics training / evaluation runs
├── notebooks/egg_detection.ipynb
├── src/{train,evaluate,predict,utils}.py
├── requirements.txt
├── README.md
└── data.yaml
```

## Troubleshooting

| Problem | Fix |
|---|---|
| `Kaggle credentials not found` | Follow the `kaggle.json` steps in section 5. |
| `No Kaggle dataset selected` | Set `EGG_KAGGLE_DATASET` / `KAGGLE_DATASET`. |
| `No YOLO .txt label files found` | The dataset is COCO/XML. Convert it to YOLO format first. |
| CUDA out of memory | Lower `--batch` (e.g. 8) or `--imgsz` (e.g. 512). |
| Training very slow | You are on CPU. Use a GPU (Kaggle/Colab) or fewer epochs. |
| `Model weights not found` | Run `python src/train.py` or pass `--weights path/to/best.pt`. |
| Webcam window will not open | Needs a local machine with a display (not a remote notebook). |

## Acknowledgements

Ultralytics YOLO · the Kaggle dataset authors (cite your chosen dataset and its licence).
