"""
Train the 4-class road damage DETECTION model (boxes).

Example (GPU):
    python scripts/train_detect.py --model yolov8s.pt --epochs 40 --batch 16
With a hard time limit (stops after 2.5 hours, keeps the best model so far):
    python scripts/train_detect.py --model yolov8s.pt --epochs 60 --hours 2.5
Quick check that everything works (2 epochs, small images):
    python scripts/train_detect.py --model yolov8n.pt --epochs 2 --imgsz 320 --fraction 0.1

The best weights are copied to weights/roadwatch_det.pt when training ends.
"""
import argparse
import shutil
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="yolov8s.pt", help="yolov8n.pt (fastest) / yolov8s.pt / yolo11s.pt")
    ap.add_argument("--data", default=str(ROOT / "configs" / "roadwatch_det.yaml"))
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16, help="Lower to 8 if you get CUDA out-of-memory")
    ap.add_argument("--device", default="0", help="'0' = first GPU, 'cpu' = CPU")
    ap.add_argument("--workers", type=int, default=4, help="Use 0-2 on Windows if data loading hangs")
    ap.add_argument("--fraction", type=float, default=1.0, help="Use part of the train set (quick tests)")
    ap.add_argument("--hours", type=float, default=None,
                    help="Hard time limit in hours (overrides --epochs), e.g. 2.5 to protect the deadline")
    ap.add_argument("--name", default="det_4class")
    args = ap.parse_args()

    model = YOLO(args.model)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        fraction=args.fraction,
        project=str(ROOT / "runs" / "detect"),
        name=args.name,
        exist_ok=True,
        patience=15,          # stop early if validation stops improving
        cos_lr=True,
        close_mosaic=5,
        # robustness-oriented augmentation (lighting, viewpoint)
        hsv_v=0.5, hsv_s=0.6, degrees=5.0, translate=0.1, scale=0.5, fliplr=0.5,
        plots=True,
        **({"time": args.hours} if args.hours else {}),
    )
    best = ROOT / "runs" / "detect" / args.name / "weights" / "best.pt"
    if best.exists():
        (ROOT / "weights").mkdir(exist_ok=True)
        shutil.copy2(best, ROOT / "weights" / "roadwatch_det.pt")
        print(f"\nSaved best model to weights/roadwatch_det.pt")


if __name__ == "__main__":
    main()
