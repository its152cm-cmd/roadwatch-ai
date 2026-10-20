"""
Train the SEGMENTATION model: pixel outlines of potholes (Roboflow) and cracks (CRACK500).

Example:
    python scripts/train_segment.py --model yolov8n-seg.pt --epochs 60

The best weights are copied to weights/roadwatch_seg.pt when training ends.
"""
import argparse
import shutil
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="yolov8n-seg.pt", help="yolov8n-seg.pt / yolov8s-seg.pt")
    ap.add_argument("--data", default=str(ROOT / "configs" / "roadwatch_seg.yaml"))
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default="0")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--name", default="seg_pothole_crack")
    args = ap.parse_args()

    model = YOLO(args.model)
    model.train(
        data=args.data, epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
        device=args.device, workers=args.workers,
        project=str(ROOT / "runs" / "segment"), name=args.name, exist_ok=True,
        patience=20, cos_lr=True, hsv_v=0.5, degrees=5.0, plots=True,
    )
    best = ROOT / "runs" / "segment" / args.name / "weights" / "best.pt"
    if best.exists():
        (ROOT / "weights").mkdir(exist_ok=True)
        shutil.copy2(best, ROOT / "weights" / "roadwatch_seg.pt")
        print("\nSaved best model to weights/roadwatch_seg.pt")


if __name__ == "__main__":
    main()
