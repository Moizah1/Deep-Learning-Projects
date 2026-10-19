"""
predict.py - Egg detection on a single image, many images, a video, or a webcam.

Every detection is drawn with: bounding box + "egg <confidence>",
and the frame shows "Total Eggs Detected: X". Results are saved to outputs/.

Usage:
    python src/predict.py image  --source path/to/egg.jpg --conf 0.5
    python src/predict.py images --source path/to/folder  --conf 0.4
    python src/predict.py video  --source path/to/video.mp4
    python src/predict.py webcam --camera 0          (press 'q' to quit)
"""

import argparse
import sys
import time
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent))
import utils  # noqa: E402

# Only detections with confidence >= this value are shown (change freely).
CONFIDENCE_THRESHOLD = 0.5


def predict_image(model, image_path, conf=CONFIDENCE_THRESHOLD, imgsz=640, device=None,
                  out_dir=utils.OUTPUTS_DIR / "images", show=True):
    """Detect eggs in one image, save the figure + annotated image, return the egg count."""
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"OpenCV could not read the file as an image: {image_path}")

    boxes, confs = utils.run_inference(model, img, conf, imgsz, device)
    annotated = utils.draw_detections(img, boxes, confs)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_dir / f"{image_path.stem}_detected.jpg"), annotated)
    utils.show_result(img, annotated, len(boxes), title=f"{image_path.name}  (conf >= {conf})",
                      save_path=out_dir / f"{image_path.stem}_report.png", show=show)
    print(f"[RESULT] {image_path.name}: Total Eggs Detected: {len(boxes)}")
    return len(boxes)


def predict_images(model, folder, conf=CONFIDENCE_THRESHOLD, imgsz=640, device=None,
                   out_dir=utils.OUTPUTS_DIR / "images", show=False):
    """Detect eggs in every image of a folder (or in a list of image paths)."""
    if isinstance(folder, (list, tuple)):
        paths = [Path(p) for p in folder]
    else:
        folder = Path(folder)
        if not folder.is_dir():
            raise FileNotFoundError(f"Image folder not found: {folder}")
        paths = utils.list_images(folder)
    if not paths:
        raise FileNotFoundError("No images found to process.")

    counts = {}
    for p in paths:
        try:
            counts[p.name] = predict_image(model, p, conf, imgsz, device, out_dir, show)
        except (FileNotFoundError, ValueError) as err:
            print(f"[WARN] {err}")
    print(f"[SUMMARY] {len(counts)} images processed, {sum(counts.values())} eggs detected in total.")
    return counts


def predict_video(model, video_path, conf=CONFIDENCE_THRESHOLD, imgsz=640, device=None,
                  out_dir=utils.OUTPUTS_DIR / "videos", display=False):
    """Detect eggs in every frame of a video and save the annotated video."""
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"OpenCV could not open the video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{video_path.stem}_detected.mp4"
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    frame_counts = []
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            boxes, confs = utils.run_inference(model, frame, conf, imgsz, device)
            writer.write(utils.draw_detections(frame, boxes, confs))
            frame_counts.append(len(boxes))
            if display:
                cv2.imshow("Egg detection (press q to quit)", utils.draw_detections(frame, boxes, confs))
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        cap.release()
        writer.release()
        if display:
            cv2.destroyAllWindows()

    if not frame_counts:
        raise ValueError("The video contained no readable frames.")
    print(f"[RESULT] {len(frame_counts)} frames | max eggs in a frame: {max(frame_counts)} "
          f"| average: {sum(frame_counts) / len(frame_counts):.1f}")
    print(f"[INFO] Saved: {out_path}")
    return out_path, frame_counts


def predict_webcam(model, camera=0, conf=CONFIDENCE_THRESHOLD, imgsz=640, device=None,
                   out_dir=utils.OUTPUTS_DIR / "webcam", save=False):
    """Live detection from a camera. Press 'q' to quit, 's' to save a snapshot."""
    cap = cv2.VideoCapture(camera)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera {camera}. Check that it is connected and not in use.")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    writer = None
    prev = time.time()
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("[WARN] Camera returned no frame - stopping.")
                break
            boxes, confs = utils.run_inference(model, frame, conf, imgsz, device)
            annotated = utils.draw_detections(frame, boxes, confs)

            now = time.time()
            cv2.putText(annotated, f"FPS: {1 / max(now - prev, 1e-6):.1f}", (10, annotated.shape[0] - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            prev = now

            if save:
                if writer is None:
                    h, w = annotated.shape[:2]
                    writer = cv2.VideoWriter(str(out_dir / "webcam_session.mp4"),
                                             cv2.VideoWriter_fourcc(*"mp4v"), 20, (w, h))
                writer.write(annotated)
            try:
                cv2.imshow("Egg detection - q: quit, s: snapshot", annotated)
            except cv2.error as err:
                raise RuntimeError("OpenCV cannot open a window here (headless install or remote "
                                   "notebook). Run this on a local machine with a display.") from err
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("s"):
                snap = out_dir / f"snapshot_{int(now)}.jpg"
                cv2.imwrite(str(snap), annotated)
                print(f"[INFO] Snapshot saved: {snap}")
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Egg detection with YOLOv11n")
    ap.add_argument("mode", choices=["image", "images", "video", "webcam"])
    ap.add_argument("--source", help="image / folder / video path")
    ap.add_argument("--camera", type=int, default=0, help="webcam index")
    ap.add_argument("--weights", default=str(utils.DEFAULT_WEIGHTS))
    ap.add_argument("--conf", type=float, default=CONFIDENCE_THRESHOLD, help="confidence threshold (0-1)")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--show", action="store_true", help="open result windows/figures")
    ap.add_argument("--save-webcam", action="store_true", help="record the webcam session")
    a = ap.parse_args()

    try:
        if not 0 <= a.conf <= 1:
            raise ValueError("--conf must be between 0 and 1")
        if a.mode != "webcam" and not a.source:
            raise ValueError(f"--source is required for mode '{a.mode}'")
        mdl = utils.load_model(a.weights)
        dev = utils.get_device()
        if a.mode == "image":
            predict_image(mdl, a.source, a.conf, a.imgsz, dev, show=a.show)
        elif a.mode == "images":
            predict_images(mdl, a.source, a.conf, a.imgsz, dev, show=a.show)
        elif a.mode == "video":
            predict_video(mdl, a.source, a.conf, a.imgsz, dev, display=a.show)
        else:
            predict_webcam(mdl, a.camera, a.conf, a.imgsz, dev, save=a.save_webcam)
    except (FileNotFoundError, ValueError, RuntimeError) as err:
        print(f"[ERROR] {err}")
        sys.exit(1)
