
# Fine-tune a pretrained YOLOv11n model on the egg dataset

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import utils  # noqa: E402


def train(epochs=100, imgsz=640, batch=16, lr0=0.01, patience=20,
          pretrained_weights="yolo11n.pt", data_yaml=utils.DATA_YAML, run_name=utils.RUN_NAME):
    """
    Fine-tune YOLOv11n and return (best_weights_path, run_directory).

    epochs   : maximum number of training epochs
    imgsz    : input image size (images are resized to imgsz x imgsz)
    batch    : batch size (reduce it if you get out-of-memory errors)
    lr0      : initial learning rate (final LR = lr0 * lrf, default lrf=0.01)
    patience : early stopping - stop if validation fitness does not improve for N epochs
    """
    # ---- sanity checks -----------------------------------------------------
    if not Path(data_yaml).exists():
        raise FileNotFoundError(f"{data_yaml} not found. Prepare the dataset first (see README).")
    utils.check_dataset_structure()               # raises a clear error for bad paths / empty data

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise ImportError("Ultralytics is not installed. Run: pip install -r requirements.txt") from exc

    device = utils.get_device()                   # GPU if available, otherwise CPU
    if device == "cpu" and batch > 8:
        print("[INFO] Running on CPU - consider --batch 8 and fewer epochs for a quicker run.")

    # ---- load pretrained YOLOv11n (downloads yolo11n.pt on first use) -------
    model = YOLO(pretrained_weights)

    # ---- fine-tune ---------------------------------------------------------
    model.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        lr0=lr0,
        patience=patience,                        # early stopping
        device=device,
        workers=0 if sys.platform.startswith("win") else 4,
        project=str(utils.RUNS_DIR),
        name=run_name,
        exist_ok=True,                            # reuse the same run folder
        save=True,                                # checkpoints: weights/best.pt and last.pt
        save_period=-1,                           # set e.g. 10 to also keep every 10th epoch
        plots=True,                               # training curves, confusion matrix ...
        seed=42,
        pretrained=True,
    )

    run_dir = Path(model.trainer.save_dir)
    best = run_dir / "weights" / "best.pt"
    if not best.exists():
        raise FileNotFoundError(f"Training finished but best.pt was not found in {run_dir / 'weights'}")

    utils.MODELS_DIR.mkdir(exist_ok=True)
    shutil.copy2(best, utils.DEFAULT_WEIGHTS)     # convenient copy for prediction
    print(f"[INFO] Best weights saved to: {utils.DEFAULT_WEIGHTS}")
    return utils.DEFAULT_WEIGHTS, run_dir


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Train YOLOv11n for egg detection")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr0", type=float, default=0.01)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--weights", default="yolo11n.pt", help="pretrained weights to start from")
    args = ap.parse_args()
    try:
        train(args.epochs, args.imgsz, args.batch, args.lr0, args.patience, args.weights)
    except (FileNotFoundError, ImportError, RuntimeError) as err:
        print(f"[ERROR] {err}")
        sys.exit(1)
