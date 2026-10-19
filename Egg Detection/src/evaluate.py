"""
evaluate.py - Evaluate the trained model and generate evaluation plots.

Reports Precision, Recall, mAP@50, mAP@50-95, validation loss (from training log)
and the number of detected eggs versus the ground-truth egg count.

Usage:
    python src/evaluate.py --split test --conf 0.5
"""

import argparse
import csv
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent))
import utils  # noqa: E402


def evaluate(weights=utils.DEFAULT_WEIGHTS, split="test", conf=0.5, imgsz=640, batch=16):
    """Run Ultralytics validation + egg counting. Returns a dict of results."""
    utils.check_dataset_structure()
    model = utils.load_model(weights)
    device = utils.get_device()

    # ---- 1. standard detection metrics (conf=0.001 is the correct setting for mAP) ----
    out_name = f"eval_{split}"
    metrics = model.val(
        data=str(utils.DATA_YAML), split=split, imgsz=imgsz, batch=batch, device=device,
        project=str(utils.RUNS_DIR), name=out_name, exist_ok=True, plots=True, verbose=False,
    )
    results = {
        "split": split,
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
        "mAP50": float(metrics.box.map50),
        "mAP50-95": float(metrics.box.map),
    }
    eval_dir = Path(metrics.save_dir)

    # ---- 2. validation loss (only logged during training, so read results.csv) ----
    csv_path = utils.RUNS_DIR / utils.RUN_NAME / "results.csv"
    if csv_path.exists():
        import pandas as pd
        df = pd.read_csv(csv_path)
        df.columns = [c.strip() for c in df.columns]
        last = df.iloc[-1]
        results["val_box_loss"] = float(last["val/box_loss"])
        results["val_cls_loss"] = float(last["val/cls_loss"])
        results["val_dfl_loss"] = float(last["val/dfl_loss"])
    else:
        print("[INFO] Training log not found - validation loss not reported.")

    # ---- 3. egg counting: predicted vs ground truth on the chosen split ----
    img_dir = utils.DATASET_DIR / "images" / split
    lab_dir = utils.DATASET_DIR / "labels" / split
    images = utils.list_images(img_dir)
    rows, abs_err, exact = [], 0, 0
    for p in images:
        img = cv2.imread(str(p))
        if img is None:
            print(f"[WARN] Could not read {p.name}, skipping.")
            continue
        _, confs = utils.run_inference(model, img, conf=conf, imgsz=imgsz, device=device)
        gt = len(utils.read_yolo_labels(lab_dir / f"{p.stem}.txt"))
        pred = len(confs)
        abs_err += abs(pred - gt)
        exact += int(pred == gt)
        rows.append((p.name, gt, pred))
    n = max(len(rows), 1)
    results.update({
        "images_evaluated": len(rows),
        "eggs_ground_truth": sum(r[1] for r in rows),
        "eggs_detected": sum(r[2] for r in rows),
        "count_MAE_per_image": abs_err / n,
        "count_exact_match_%": 100 * exact / n,
        "conf_threshold_for_counting": conf,
    })

    utils.OUTPUTS_DIR.mkdir(exist_ok=True)
    with open(utils.OUTPUTS_DIR / f"egg_counts_{split}.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image", "ground_truth_eggs", "detected_eggs"])
        w.writerows(rows)

    results["plots_dir"] = str(eval_dir)
    print("\n================ EVALUATION RESULTS ================")
    for k, v in results.items():
        print(f"{k:<30}: {v:.4f}" if isinstance(v, float) else f"{k:<30}: {v}")
    return results


def save_sample_predictions(weights=utils.DEFAULT_WEIGHTS, split="test", n=6, conf=0.5, imgsz=640, show=True):
    """Save and show a grid of prediction examples from the chosen split."""
    model = utils.load_model(weights)
    device = utils.get_device()
    images = utils.list_images(utils.DATASET_DIR / "images" / split)[:n]
    if not images:
        raise FileNotFoundError(f"No images found for split '{split}'.")
    out_dir = utils.OUTPUTS_DIR / f"sample_predictions_{split}"
    out_dir.mkdir(parents=True, exist_ok=True)
    for p in images:
        img = cv2.imread(str(p))
        boxes, confs = utils.run_inference(model, img, conf, imgsz, device)
        annotated = utils.draw_detections(img, boxes, confs)
        utils.show_result(img, annotated, len(boxes), title=p.name,
                          save_path=out_dir / f"{p.stem}_result.png", show=show)
    print(f"[INFO] Sample predictions saved to {out_dir}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Evaluate the egg detector")
    ap.add_argument("--weights", default=str(utils.DEFAULT_WEIGHTS))
    ap.add_argument("--split", default="test", choices=["val", "test"])
    ap.add_argument("--conf", type=float, default=0.5, help="confidence threshold used for egg counting")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    args = ap.parse_args()
    try:
        evaluate(args.weights, args.split, args.conf, args.imgsz, args.batch)
        save_sample_predictions(args.weights, args.split, conf=args.conf, imgsz=args.imgsz, show=False)
    except (FileNotFoundError, ImportError, RuntimeError) as err:
        print(f"[ERROR] {err}")
        sys.exit(1)
