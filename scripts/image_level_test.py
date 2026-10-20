"""
Image-level test on an independent dataset with folder labels only (no boxes needed),
e.g. Road_Condition/Alert (Kaggle): potholes/ vs clean_roads/ + smooth_roads/.

  * Pothole images:  does the model find at least one pothole?   -> detection rate
  * Clean roads:     does the model stay quiet (no damage at all)? -> false-alarm rate

Images used as training negatives (results/extra_negatives_used.txt) are skipped,
so every test image is unseen.

Outputs: results/image_level_test.csv, results/image_level_test.md

Example:
    python scripts/image_level_test.py --pothole-dir data/RoadAlert/potholes \
        --clean-dirs data/RoadAlert/clean_roads data/RoadAlert/smooth_roads --max-images 300
"""
import argparse
import csv
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ultralytics import YOLO  # noqa: E402

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}
RES = ROOT / "results"


def list_images(dirs, skip, max_images, seed=0):
    files = [p for d in dirs for p in Path(d).rglob("*") if p.suffix.lower() in IMG_EXT
             and str(p.resolve()) not in skip]
    random.Random(seed).shuffle(files)
    return files[:max_images] if max_images else files


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--det", default=str(ROOT / "weights" / "roadwatch_det.pt"))
    ap.add_argument("--pothole-dir", required=True)
    ap.add_argument("--clean-dirs", nargs="+", required=True)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--max-images", type=int, default=300, help="Per group (0 = all)")
    ap.add_argument("--device", default="0")
    args = ap.parse_args()

    used = RES / "extra_negatives_used.txt"
    skip = set(used.read_text().split("\n")) if used.exists() else set()
    model = YOLO(args.det)
    pothole_id = [k for k, v in model.names.items() if v == "pothole"][0]

    pos = list_images([args.pothole_dir], skip, args.max_images)
    neg = list_images(args.clean_dirs, skip, args.max_images)
    found = sum(any(int(c) == pothole_id for c in model.predict(str(p), conf=args.conf, device=args.device,
                                                                verbose=False)[0].boxes.cls) for p in pos)
    alarms = sum(len(model.predict(str(p), conf=args.conf, device=args.device, verbose=False)[0].boxes) > 0
                 for p in neg)
    rows = [
        dict(group="Pothole images", images=len(pos), metric="pothole detected", count=found,
             rate=found / max(1, len(pos))),
        dict(group="Clean / smooth roads", images=len(neg), metric="false alarm (any damage)", count=alarms,
             rate=alarms / max(1, len(neg))),
    ]
    RES.mkdir(exist_ok=True)
    with open(RES / "image_level_test.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    md = [f"### Image-level test on an independent dataset (confidence >= {args.conf})", "",
          "| Group | Images | Result | Rate |", "|---|---|---|---|"]
    md += [f"| {r['group']} | {r['images']} | {r['count']} {r['metric']} | {r['rate']:.0%} |" for r in rows]
    (RES / "image_level_test.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
