"""
Error analysis: where and why does the model go wrong?

Part 1 - Test set: matches predictions to ground truth (IoU >= 0.5, same class) and
  counts, per class: correct detections, false alarms (FP), misses (FN), and
  wrong-class confusions. Saves galleries of the most confident false alarms and of
  missed hazards so causes can be inspected (shadows, puddles, markings, distance...).

Part 2 - Look-alike test (optional): a folder of images that contain NO road damage
  but things that look like it, one subfolder per type, e.g.
      lookalikes/puddle/  lookalikes/shadow/  lookalikes/manhole/  lookalikes/cone/
  Reports how often each type fools the model.

Outputs (results/): error_summary.csv, error_analysis.md,
  errors_false_alarms.jpg, errors_misses.jpg, lookalike_test.csv (optional)

Example:
    python scripts/error_analysis.py --lookalike-dir data/lookalikes
"""
import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ultralytics import YOLO  # noqa: E402

from roadwatch import CLASS_NAMES  # noqa: E402

RES = ROOT / "results"
GREEN, YELLOW, RED = (80, 200, 80), (63, 210, 255), (60, 60, 230)


def iou(a, b):
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def read_gt(lbl, w, h):
    out = []
    if lbl.exists():
        for row in lbl.read_text().split("\n"):
            p = row.split()
            if len(p) == 5:
                c, cx, cy, bw, bh = int(p[0]), *map(float, p[1:])
                out.append((c, [(cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h]))
    return out


def crop(img, box, color, text, pad=40, size=260):
    h, w = img.shape[:2]
    x1, y1, x2, y2 = map(int, box)
    v = img.copy()
    cv2.rectangle(v, (x1, y1), (x2, y2), color, 3)
    c = v[max(0, y1 - pad):min(h, y2 + pad), max(0, x1 - pad):min(w, x2 + pad)]
    c = cv2.resize(c, (size, size))
    cv2.rectangle(c, (0, size - 26), (size, size), (25, 23, 21), -1)
    cv2.putText(c, text, (6, size - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (243, 247, 247), 1, cv2.LINE_AA)
    return c


def gallery(tiles, path, cols=6):
    if not tiles:
        return
    while len(tiles) % cols:
        tiles.append(np.full_like(tiles[0], 30))
    cv2.imwrite(str(path), np.vstack([np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--det", default=str(ROOT / "weights" / "roadwatch_det.pt"))
    ap.add_argument("--test-dir", default=str(ROOT / "datasets" / "roadwatch_det" / "test"))
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--lookalike-dir", default=None)
    ap.add_argument("--device", default="0")
    args = ap.parse_args()
    RES.mkdir(exist_ok=True)
    model = YOLO(args.det)
    test = Path(args.test_dir)
    images = sorted(p for p in (test / "images").iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})

    stats = defaultdict(lambda: dict(correct=0, false_alarm=0, missed=0, wrong_class=0))
    fps, fns = [], []
    for p in images:
        img = cv2.imread(str(p))
        h, w = img.shape[:2]
        gt = read_gt(test / "labels" / f"{p.stem}.txt", w, h)
        r = model.predict(img, conf=args.conf, device=args.device, verbose=False)[0]
        preds = [(int(c), b.tolist(), float(s)) for c, b, s in
                 zip(r.boxes.cls.cpu(), r.boxes.xyxy.cpu(), r.boxes.conf.cpu())]
        used = set()
        for pc, pb, ps in sorted(preds, key=lambda x: -x[2]):
            best, bi = 0, None
            for i, (gc, gb) in enumerate(gt):
                if i not in used and (o := iou(pb, gb)) > best:
                    best, bi = o, i
            if best >= 0.5 and gt[bi][0] == pc:
                used.add(bi)
                stats[CLASS_NAMES[pc]]["correct"] += 1
            elif best >= 0.5:
                used.add(bi)
                stats[CLASS_NAMES[gt[bi][0]]]["wrong_class"] += 1
                fps.append((ps, img, pb, f"{CLASS_NAMES[pc]} {ps:.2f} (is {CLASS_NAMES[gt[bi][0]]})"))
            else:
                stats[CLASS_NAMES[pc]]["false_alarm"] += 1
                fps.append((ps, img, pb, f"FALSE {CLASS_NAMES[pc]} {ps:.2f}"))
        for i, (gc, gb) in enumerate(gt):
            if i not in used:
                stats[CLASS_NAMES[gc]]["missed"] += 1
                fns.append(((gb[2] - gb[0]) * (gb[3] - gb[1]), img, gb, f"MISSED {CLASS_NAMES[gc]}"))

    rows = [dict(cls=c, **stats[c]) for c in CLASS_NAMES]
    with open(RES / "error_summary.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)
    md = [f"### Error analysis (test set, confidence >= {args.conf})", "",
          "| Class | Correct | False alarms | Missed | Wrong class |", "|---|---|---|---|---|"]
    md += [f"| {r['cls']} | {r['correct']} | {r['false_alarm']} | {r['missed']} | {r['wrong_class']} |" for r in rows]

    fps.sort(key=lambda x: -x[0])          # most confident false alarms first
    fns.sort(key=lambda x: -x[0])          # biggest missed hazards first
    gallery([crop(i, b, RED, t) for _, i, b, t in fps[:18]], RES / "errors_false_alarms.jpg")
    gallery([crop(i, b, GREEN, t) for _, i, b, t in fns[:18]], RES / "errors_misses.jpg")

    if args.lookalike_dir:
        la_rows = []
        for sub in sorted(d for d in Path(args.lookalike_dir).iterdir() if d.is_dir()):
            ims = [p for p in sub.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
            fooled = sum(len(model.predict(str(p), conf=args.conf, device=args.device, verbose=False)[0].boxes) > 0
                         for p in ims)
            la_rows.append(dict(lookalike=sub.name, images=len(ims), fooled=fooled,
                                false_alarm_rate=fooled / max(1, len(ims))))
        if la_rows:
            with open(RES / "lookalike_test.csv", "w", newline="") as f:
                wr = csv.DictWriter(f, fieldnames=list(la_rows[0].keys()))
                wr.writeheader()
                wr.writerows(la_rows)
            md += ["", "### Look-alike test (images with no road damage)", "",
                   "| Look-alike | Images | Fooled the model | False-alarm rate |", "|---|---|---|---|"]
            md += [f"| {r['lookalike']} | {r['images']} | {r['fooled']} | {r['false_alarm_rate']:.0%} |"
                   for r in la_rows]

    (RES / "error_analysis.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
    print("\nSaved results/error_analysis.md, errors_false_alarms.jpg, errors_misses.jpg")


if __name__ == "__main__":
    main()
