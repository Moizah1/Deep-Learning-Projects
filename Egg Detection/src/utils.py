"""
utils.py - Shared helpers for the Egg Detection (YOLOv11n) project.

Contents
    1. Project paths and constants
    2. Device selection (CUDA / CPU)
    3. Kaggle download (programmatic) + secure credential check
    4. Dataset preparation into YOLO format  (images/ labels/ data.yaml)
    5. Dataset checks and sample visualisation
    6. Drawing / inference helpers shared by train, evaluate and predict
    7. Training-curve plotting
"""


from __future__ import annotations

import argparse
import os
import random
import shutil
import sys
import zipfile
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import yaml

# --------------------------------------------------------------------------
# 1. Project paths and constants
# --------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = PROJECT_ROOT / "dataset"          # final YOLO-format dataset
RAW_DIR = PROJECT_ROOT / "dataset_raw"          # raw Kaggle download (git-ignored)
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
RUNS_DIR = PROJECT_ROOT / "runs"
DATA_YAML = PROJECT_ROOT / "data.yaml"
DEFAULT_WEIGHTS = MODELS_DIR / "egg_yolo11n_best.pt"
RUN_NAME = "egg_yolo11n"

CLASS_NAMES = ["egg"]                            # 0: egg
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".wmv"}

# >>> Set this to the Kaggle dataset you choose, e.g. "someuser/egg-detection".
# Use `python src/utils.py search --term egg` (or search_kaggle()) to find one.
KAGGLE_DATASET = os.getenv("EGG_KAGGLE_DATASET", "uzairahmad1434/eggs-detection")

KAGGLE_HELP = """
Kaggle credentials not found. Configure them securely:
  1. kaggle.com -> your profile picture -> Settings -> API -> "Create New Token"
     (downloads kaggle.json).
  2. Linux/macOS:  mkdir -p ~/.kaggle && mv ~/Downloads/kaggle.json ~/.kaggle/ \\
                   && chmod 600 ~/.kaggle/kaggle.json
     Windows:      move it to C:\\Users\\<you>\\.kaggle\\kaggle.json
  3. Alternatively set the environment variables KAGGLE_USERNAME and KAGGLE_KEY.
  NEVER commit kaggle.json to GitHub or paste the key into a notebook cell.
"""


# --------------------------------------------------------------------------
# 2. Device selection
# --------------------------------------------------------------------------
def get_device():
    """Return 0 (first CUDA GPU) when available, otherwise 'cpu'."""
    try:
        import torch
    except ImportError as exc:
        raise ImportError("PyTorch is not installed. Run: pip install -r requirements.txt") from exc

    if torch.cuda.is_available():
        print(f"[INFO] CUDA available -> using GPU: {torch.cuda.get_device_name(0)}")
        return 0
    print("[WARN] CUDA not available -> falling back to CPU (training will be slow).")
    return "cpu"


# --------------------------------------------------------------------------
# 3. Kaggle download
# --------------------------------------------------------------------------
def check_kaggle_credentials() -> None:
    """Make sure Kaggle credentials exist and warn if the file is world-readable."""
    if os.getenv("KAGGLE_API_TOKEN") or (os.getenv("KAGGLE_USERNAME") and os.getenv("KAGGLE_KEY")):
        return
    cred = Path.home() / ".kaggle" / "kaggle.json"
    token = Path.home() / ".kaggle" / "access_token"
    if cred.exists():
        if os.name != "nt" and (cred.stat().st_mode & 0o077):
            try:
                cred.chmod(0o600)
                print("[INFO] Fixed permissions of kaggle.json (chmod 600).")
            except OSError:
                print("[WARN] kaggle.json is readable by other users. Run: chmod 600 ~/.kaggle/kaggle.json")
        return
    if token.exists():
        return
    raise FileNotFoundError(KAGGLE_HELP)


def _kaggle_api():
    """Create an authenticated Kaggle API client."""
    check_kaggle_credentials()
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError as exc:
        raise ImportError("The 'kaggle' package is missing. Run: pip install kaggle") from exc
    api = KaggleApi()
    api.authenticate()
    return api


def search_kaggle(term: str = "egg detection", max_results: int = 15) -> None:
    """Print Kaggle datasets matching `term` so you can pick a slug for KAGGLE_DATASET."""
    api = _kaggle_api()
    results = api.dataset_list(search=term)
    for ds in list(results)[:max_results]:
        print(f"{getattr(ds, 'ref', ds)}  |  {getattr(ds, 'title', '')}")


def extract_nested_zips(folder: Path) -> None:
    """Some datasets contain extra .zip files (e.g. images.zip, labels.zip). Unpack them."""
    for _ in range(3):                           # handle a few levels of nesting
        zips = list(folder.rglob("*.zip"))
        if not zips:
            return
        for z in zips:
            try:
                with zipfile.ZipFile(z) as zf:
                    zf.extractall(z.with_suffix(""))
                z.unlink()
            except zipfile.BadZipFile:
                print(f"[WARN] Skipping corrupt archive: {z}")
                z.unlink()


def download_kaggle_dataset(slug: str = KAGGLE_DATASET, dest: Path = RAW_DIR) -> Path:
    """Download and unzip a Kaggle dataset into `dest`."""
    if "/" not in slug or slug.startswith("REPLACE_WITH"):
        raise ValueError(
            "No Kaggle dataset selected. Set KAGGLE_DATASET in src/utils.py (or the "
            "EGG_KAGGLE_DATASET environment variable) to 'owner/dataset-slug'. "
            "Tip: run  python src/utils.py search --term egg"
        )
    api = _kaggle_api()
    dest.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Downloading '{slug}' to {dest} ...")
    api.dataset_download_files(slug, path=str(dest), unzip=True, quiet=False)
    extract_nested_zips(dest)
    print("[INFO] Download complete.")
    return dest


# --------------------------------------------------------------------------
# 4. Dataset preparation (any common Kaggle YOLO layout -> project layout)
# --------------------------------------------------------------------------
_SPLIT_ALIASES = {"train": "train", "training": "train",
                  "val": "val", "valid": "val", "validation": "val",
                  "test": "test", "testing": "test"}


def _detect_split(path: Path, root: Path):
    """Look at folder names (nearest first) to decide train/val/test, else None."""
    for part in reversed(path.relative_to(root).parts[:-1]):
        if part.lower() in _SPLIT_ALIASES:
            return _SPLIT_ALIASES[part.lower()]
    return None


def _convert_label_file(src: Path, keep_ids=None):
    """
    Read a YOLO label file and return lines in the form '0 cx cy w h'.
    - Every kept object is mapped to class 0 (egg).
    - Segmentation polygons are converted to bounding boxes.
    - Invalid boxes are dropped.
    """
    out = []
    for line in src.read_text().splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        try:
            cls = int(float(parts[0]))
            vals = [float(v) for v in parts[1:]]
        except ValueError:
            continue
        if keep_ids is not None and cls not in keep_ids:
            continue
        if len(vals) == 4:
            cx, cy, w, h = vals
        else:                                    # polygon: x1 y1 x2 y2 ...
            xs, ys = vals[0::2], vals[1::2]
            x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
            cx, cy, w, h = (x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0
        # clip to the image and drop degenerate boxes
        x0, x1 = max(0.0, cx - w / 2), min(1.0, cx + w / 2)
        y0, y1 = max(0.0, cy - h / 2), min(1.0, cy + h / 2)
        w, h = x1 - x0, y1 - y0
        if w <= 0 or h <= 0:
            continue
        out.append(f"0 {(x0 + x1) / 2:.6f} {(y0 + y1) / 2:.6f} {w:.6f} {h:.6f}")
    return out


def prepare_dataset(raw_dir: Path = RAW_DIR, out_dir: Path = DATASET_DIR,
                    val_ratio: float = 0.2, test_ratio: float = 0.1,
                    keep_class_ids=None, seed: int = 42) -> dict:
    """
    Convert a downloaded YOLO-style dataset into:
        dataset/images/{train,val,test}  dataset/labels/{train,val,test}  + data.yaml

    * Existing train/valid/test folders are respected; unsplit data is split randomly.
    * `keep_class_ids`: list of source class ids that mean "egg" (None = keep all as egg).
    """
    raw_dir, out_dir = Path(raw_dir), Path(out_dir)
    if not raw_dir.exists() or not any(raw_dir.iterdir()):
        raise FileNotFoundError(
            f"Raw dataset folder is missing or empty: {raw_dir}\n"
            "Run download_kaggle_dataset() first (or copy a YOLO-format dataset there)."
        )
    extract_nested_zips(raw_dir)

    images = [p for p in raw_dir.rglob("*") if p.suffix.lower() in IMAGE_EXTS]
    skip_names = {"classes.txt", "labels.txt", "readme.txt", "readme.dataset.txt", "readme.roboflow.txt"}
    labels = {}
    for p in raw_dir.rglob("*.txt"):
        if p.name.lower() not in skip_names:
            labels.setdefault((_detect_split(p, raw_dir), p.stem), p)

    if not images:
        raise FileNotFoundError(f"No images found under {raw_dir}.")
    if not labels:
        raise FileNotFoundError(
            f"No YOLO .txt label files found under {raw_dir}. If the dataset uses COCO/XML "
            "annotations, convert it to YOLO format first."
        )

    # pair each image with its label file
    pairs, unlabeled = [], 0
    for img in sorted(images):
        split = _detect_split(img, raw_dir)
        lab = labels.get((split, img.stem)) or labels.get((None, img.stem))
        if lab is None:
            unlabeled += 1
            continue
        pairs.append((img, lab, split))
    if not pairs:
        raise RuntimeError("No image/label pairs matched. Check that filenames match (a.jpg <-> a.txt).")

    # assign unsplit samples randomly, then make sure val/test are not empty
    rng = random.Random(seed)
    buckets = {"train": [], "val": [], "test": []}
    unsplit = []
    for item in pairs:
        (buckets[item[2]] if item[2] else unsplit).append(item)
    rng.shuffle(unsplit)
    n_val, n_test = int(len(unsplit) * val_ratio), int(len(unsplit) * test_ratio)
    buckets["val"] += unsplit[:n_val]
    buckets["test"] += unsplit[n_val:n_val + n_test]
    buckets["train"] += unsplit[n_val + n_test:]
    for name, ratio in (("val", val_ratio), ("test", test_ratio)):
        if not buckets[name] and len(buckets["train"]) >= 3:
            k = max(1, int(len(buckets["train"]) * ratio))
            rng.shuffle(buckets["train"])
            buckets[name], buckets["train"] = buckets["train"][:k], buckets["train"][k:]

    # write the new dataset (start clean so re-running is safe)
    for sub in ("images", "labels"):
        shutil.rmtree(out_dir / sub, ignore_errors=True)
    counts = {}
    for split, items in buckets.items():
        (out_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_dir / "labels" / split).mkdir(parents=True, exist_ok=True)
        n_boxes = 0
        for i, (img, lab, _) in enumerate(items):
            new_stem = f"{img.stem}_{i:05d}"     # avoids name clashes between sources
            shutil.copy2(img, out_dir / "images" / split / f"{new_stem}{img.suffix.lower()}")
            lines = _convert_label_file(lab, keep_class_ids)
            n_boxes += len(lines)
            (out_dir / "labels" / split / f"{new_stem}.txt").write_text("\n".join(lines))
        counts[split] = {"images": len(items), "boxes": n_boxes}

    write_data_yaml(out_dir)
    print(f"[INFO] Dataset prepared at {out_dir}  (skipped {unlabeled} images without labels)")
    for split, c in counts.items():
        print(f"       {split:<5}: {c['images']:>5} images, {c['boxes']:>6} eggs")
    return counts


def write_data_yaml(out_dir: Path = DATASET_DIR) -> Path:
    """Create data.yaml (in dataset/ and in the project root)."""
    cfg = {
        "path": Path(out_dir).resolve().as_posix(),   # absolute dataset root
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": len(CLASS_NAMES),
        "names": {i: n for i, n in enumerate(CLASS_NAMES)},   # 0: egg
    }
    for target in (Path(out_dir) / "data.yaml", DATA_YAML):
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, "w") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)
    return DATA_YAML


# --------------------------------------------------------------------------
# 5. Dataset checks and sample visualisation
# --------------------------------------------------------------------------
def list_images(folder: Path):
    return sorted(p for p in Path(folder).glob("*") if p.suffix.lower() in IMAGE_EXTS)


def read_yolo_labels(label_path: Path) -> np.ndarray:
    """Return an (N, 5) array [cls, cx, cy, w, h] (empty array if no eggs)."""
    if not Path(label_path).exists():
        return np.zeros((0, 5))
    rows = [l.split() for l in Path(label_path).read_text().splitlines() if l.strip()]
    return np.array(rows, dtype=float).reshape(-1, 5) if rows else np.zeros((0, 5))


def check_dataset_structure(dataset_dir: Path = DATASET_DIR, data_yaml: Path = DATA_YAML) -> dict:
    """Verify folders, yaml paths and image/label pairing. Raises a clear error if broken."""
    dataset_dir = Path(dataset_dir)
    if not dataset_dir.exists() or not (dataset_dir / "images").exists():
        raise FileNotFoundError(f"Dataset not found at {dataset_dir}. Run the download/prepare steps first.")
    if not Path(data_yaml).exists():
        raise FileNotFoundError(f"{data_yaml} not found. Run write_data_yaml() or prepare_dataset().")

    cfg = yaml.safe_load(open(data_yaml))
    root = Path(cfg["path"])
    if not root.exists():
        raise FileNotFoundError(
            f"'path' in {data_yaml} points to a missing folder: {root}\n"
            "Re-run prepare_dataset() / write_data_yaml() to regenerate it."
        )

    summary = {}
    print(f"{'split':<6}{'images':>8}{'labels':>8}{'eggs':>8}")
    for split in ("train", "val", "test"):
        img_dir, lab_dir = root / cfg[split], root / cfg[split].replace("images", "labels")
        if not img_dir.exists():
            raise FileNotFoundError(f"Missing split folder: {img_dir}")
        imgs = list_images(img_dir)
        labs = list(lab_dir.glob("*.txt")) if lab_dir.exists() else []
        n_eggs = sum(len(read_yolo_labels(l)) for l in labs)
        missing = [i.name for i in imgs if not (lab_dir / f"{i.stem}.txt").exists()]
        summary[split] = {"images": len(imgs), "labels": len(labs), "eggs": n_eggs}
        print(f"{split:<6}{len(imgs):>8}{len(labs):>8}{n_eggs:>8}" + (f"   [WARN] {len(missing)} images lack labels" if missing else ""))
    if summary["train"]["images"] == 0:
        raise RuntimeError("Training split is empty - check the dataset preparation step.")
    print(f"classes: {cfg['names']}")
    return summary


def show_samples(split: str = "train", n: int = 6, dataset_dir: Path = DATASET_DIR, seed: int = 0, save: bool = True):
    """Show random images with their ground-truth egg boxes."""
    img_dir = Path(dataset_dir) / "images" / split
    images = list_images(img_dir)
    if not images:
        raise FileNotFoundError(f"No images in {img_dir}")
    random.Random(seed).shuffle(images)
    chosen = images[:n]
    cols = min(3, len(chosen))
    rows = int(np.ceil(len(chosen) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, p in zip(axes.ravel(), chosen):
        img = cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        labs = read_yolo_labels(Path(dataset_dir) / "labels" / split / f"{p.stem}.txt")
        for _, cx, cy, bw, bh in labs:
            x1, y1, x2, y2 = int((cx - bw / 2) * w), int((cy - bh / 2) * h), int((cx + bw / 2) * w), int((cy + bh / 2) * h)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), max(2, w // 300))
        ax.imshow(img)
        ax.set_title(f"{p.name[:28]}  |  eggs: {len(labs)}", fontsize=9)
    fig.suptitle(f"Ground-truth annotations ({split})", fontsize=14)
    plt.tight_layout()
    if save:
        OUTPUTS_DIR.mkdir(exist_ok=True)
        fig.savefig(OUTPUTS_DIR / f"dataset_samples_{split}.png", dpi=120)
    plt.show()


# --------------------------------------------------------------------------
# 6. Model loading, inference and drawing helpers
# --------------------------------------------------------------------------
def load_model(weights=DEFAULT_WEIGHTS):
    """Load a YOLO model, with a clear error if the weights are missing."""
    weights = Path(weights)
    if not weights.exists():
        raise FileNotFoundError(
            f"Model weights not found: {weights}\n"
            "Train first (python src/train.py) or pass --weights path/to/best.pt"
        )
    from ultralytics import YOLO
    return YOLO(str(weights))


def run_inference(model, image_bgr: np.ndarray, conf: float = 0.5, imgsz: int = 640, device=None):
    """Return (boxes_xyxy [N,4], confidences [N]) for detections with confidence >= conf."""
    result = model.predict(image_bgr, conf=conf, imgsz=imgsz, device=device, verbose=False)[0]
    boxes = result.boxes.xyxy.cpu().numpy()
    confs = result.boxes.conf.cpu().numpy()
    return boxes, confs


def draw_detections(image_bgr: np.ndarray, boxes: np.ndarray, confs: np.ndarray,
                    class_name: str = "egg", color=(0, 200, 0)) -> np.ndarray:
    """Draw 'egg 0.91' boxes and a 'Total Eggs Detected: X' banner. Returns a new image."""
    img = image_bgr.copy()
    h, w = img.shape[:2]
    thick = max(2, round(min(h, w) / 400))
    scale = max(0.5, min(h, w) / 900)

    for (x1, y1, x2, y2), c in zip(boxes.astype(int), confs):
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thick)
        label = f"{class_name} {c:.2f}"
        (tw, th), base = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, scale * 0.7, 1)
        top = max(y1 - th - base, 0)
        cv2.rectangle(img, (x1, top), (x1 + tw + 4, top + th + base), color, -1)
        cv2.putText(img, label, (x1 + 2, top + th), cv2.FONT_HERSHEY_SIMPLEX, scale * 0.7, (0, 0, 0), 1, cv2.LINE_AA)

    banner = f"Total Eggs Detected: {len(boxes)}"
    (tw, th), base = cv2.getTextSize(banner, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
    cv2.rectangle(img, (0, 0), (tw + 20, th + base + 16), (30, 30, 30), -1)
    cv2.putText(img, banner, (10, th + 8), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), thick, cv2.LINE_AA)
    return img


def show_result(original_bgr: np.ndarray, annotated_bgr: np.ndarray, count: int,
                title: str = "", save_path: Path | None = None, show: bool = True):
    """Side-by-side figure: original image | detected eggs with count."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    axes[0].imshow(cv2.cvtColor(original_bgr, cv2.COLOR_BGR2RGB))
    axes[0].set_title("Original image")
    axes[1].imshow(cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB))
    axes[1].set_title(f"Detected eggs: {count}")
    for ax in axes:
        ax.axis("off")
    if title:
        fig.suptitle(title, fontsize=13)
    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=120)
    if show:
        plt.show()
    plt.close(fig)


def show_images(paths, titles=None, cols: int = 2, figsize_per=(7, 6)):
    """Display saved image files (confusion matrix, curves ...). Missing files are skipped."""
    paths = [Path(p) for p in paths if Path(p).exists()]
    if not paths:
        print("[WARN] No image files found to display.")
        return
    rows = int(np.ceil(len(paths) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(figsize_per[0] * cols, figsize_per[1] * rows), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for i, (ax, p) in enumerate(zip(axes.ravel(), paths)):
        ax.imshow(cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB))
        ax.set_title(titles[i] if titles else p.stem)
    plt.tight_layout()
    plt.show()


# --------------------------------------------------------------------------
# 7. Training curves
# --------------------------------------------------------------------------
def plot_training_curves(results_csv: Path, save_path: Path | None = None):
    """Plot train/val losses and detection metrics from Ultralytics' results.csv."""
    import pandas as pd

    results_csv = Path(results_csv)
    if not results_csv.exists():
        raise FileNotFoundError(f"results.csv not found: {results_csv} (has training finished?)")
    df = pd.read_csv(results_csv)
    df.columns = [c.strip() for c in df.columns]

    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    for ax, name in zip(axes[0], ["box_loss", "cls_loss", "dfl_loss"]):
        ax.plot(df["epoch"], df[f"train/{name}"], label="train")
        ax.plot(df["epoch"], df[f"val/{name}"], label="validation")
        ax.set_title(name.replace("_", " ").title())
        ax.set_xlabel("epoch")
        ax.legend()
        ax.grid(alpha=0.3)
    for ax, (col, ttl) in zip(axes[1], [("metrics/precision(B)", "Precision"),
                                         ("metrics/recall(B)", "Recall"),
                                         (None, "mAP@50 and mAP@50-95")]):
        if col:
            ax.plot(df["epoch"], df[col])
        else:
            ax.plot(df["epoch"], df["metrics/mAP50(B)"], label="mAP@50")
            ax.plot(df["epoch"], df["metrics/mAP50-95(B)"], label="mAP@50-95")
            ax.legend()
        ax.set_title(ttl)
        ax.set_xlabel("epoch")
        ax.grid(alpha=0.3)
    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=120)
    plt.show()
    return df


# --------------------------------------------------------------------------
# Optional command line interface
# --------------------------------------------------------------------------
def _cli():
    ap = argparse.ArgumentParser(description="Dataset helpers for the egg detection project")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("search"); s.add_argument("--term", default="egg detection")
    d = sub.add_parser("download"); d.add_argument("--kaggle", default=KAGGLE_DATASET)
    p = sub.add_parser("prepare")
    p.add_argument("--raw", default=str(RAW_DIR))
    p.add_argument("--class-ids", type=int, nargs="*", default=None, help="source class ids meaning 'egg'")
    sub.add_parser("check")
    a = ap.parse_args()
    try:
        if a.cmd == "search":
            search_kaggle(a.term)
        elif a.cmd == "download":
            download_kaggle_dataset(a.kaggle)
        elif a.cmd == "prepare":
            prepare_dataset(Path(a.raw), keep_class_ids=a.class_ids)
        elif a.cmd == "check":
            check_dataset_structure()
    except (FileNotFoundError, ValueError, RuntimeError, ImportError) as err:
        print(f"[ERROR] {err}")
        sys.exit(1)


if __name__ == "__main__":
    _cli()
