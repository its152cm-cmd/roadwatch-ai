"""
Robustness (stress) test: how much does accuracy drop under bad conditions?

Takes the test set, makes a copy of every image under each condition
(low light, motion blur, rain, glare, fog, heavy JPEG compression), runs the
same evaluation on each copy, and reports the change versus clean images.
It also reports how often the image-quality check flags each condition.

Outputs (results/):
  robustness.csv, robustness.md, robustness_chart.png, robustness_examples.jpg

Example:
    python scripts/robustness.py --max-images 400
"""
import argparse
import csv
import random
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ultralytics import YOLO  # noqa: E402

from roadwatch import CLASS_NAMES  # noqa: E402
from roadwatch.perturb import CONDITIONS  # noqa: E402
from roadwatch.quality import assess  # noqa: E402

RES = ROOT / "results"


def build_condition_set(images, labels_dir, out_dir, fn):
    (out_dir / "images").mkdir(parents=True, exist_ok=True)
    (out_dir / "labels").mkdir(parents=True, exist_ok=True)
    flagged = 0
    for p in images:
        img = fn(cv2.imread(str(p)))
        flagged += not assess(img).ok
        cv2.imwrite(str(out_dir / "images" / p.name), img)
        lbl = labels_dir / f"{p.stem}.txt"
        if lbl.exists():
            shutil.copy2(lbl, out_dir / "labels" / lbl.name)
    return flagged / max(1, len(images))


def chart(rows, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names = [r["condition"].replace("_", " ") for r in rows]
    vals = [r["mAP50"] for r in rows]
    colors = ["#2b2f33" if r["condition"] == "clean" else "#e0a800" for r in rows]
    fig, ax = plt.subplots(figsize=(9, 4.2), dpi=150)
    bars = ax.bar(names, vals, color=colors)
    ax.axhline(rows[0]["mAP50"], color="#2b2f33", lw=1, ls="--")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.2f}", ha="center", fontsize=9)
    ax.set_ylabel("mAP@0.5 (higher is better)")
    ax.set_title("RoadWatch AI under different road conditions (test set)")
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_ylim(0, max(vals) * 1.2 + 0.01)
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(path)
    plt.close(fig)


def examples(model, img_path, path):
    """One test image under every condition, with detections drawn."""
    tiles = []
    base = cv2.imread(str(img_path))
    for name, fn in CONDITIONS.items():
        x = fn(base.copy())
        r = model.predict(x, conf=0.25, verbose=False)[0]
        v = r.plot(line_width=2)
        v = cv2.resize(v, (400, int(400 * v.shape[0] / v.shape[1])))
        q = assess(x)
        cv2.rectangle(v, (0, 0), (400, 30), (25, 23, 21), -1)
        label = f"{name}: {len(r.boxes)} found" + ("  [quality warning]" if not q.ok else "")
        cv2.putText(v, label, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (63, 210, 255), 1, cv2.LINE_AA)
        tiles.append(v)
    h = max(t.shape[0] for t in tiles)
    tiles = [cv2.copyMakeBorder(t, 0, h - t.shape[0], 0, 0, cv2.BORDER_CONSTANT) for t in tiles]
    while len(tiles) % 4:
        tiles.append(np.zeros_like(tiles[0]))
    grid = np.vstack([np.hstack(tiles[i:i + 4]) for i in range(0, len(tiles), 4)])
    cv2.imwrite(str(path), grid)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--det", default=str(ROOT / "weights" / "roadwatch_det.pt"))
    ap.add_argument("--test-dir", default=str(ROOT / "datasets" / "roadwatch_det" / "test"))
    ap.add_argument("--max-images", type=int, default=400, help="Subsample for speed (0 = all)")
    ap.add_argument("--device", default="0")
    ap.add_argument("--imgsz", type=int, default=640)
    args = ap.parse_args()

    test = Path(args.test_dir)
    images = sorted(p for p in (test / "images").iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    random.Random(0).shuffle(images)
    if args.max_images:
        images = images[: args.max_images]
    work = ROOT / "datasets" / "robustness"
    if work.exists():
        shutil.rmtree(work)
    model = YOLO(args.det)
    RES.mkdir(exist_ok=True)

    rows = []
    for name, fn in CONDITIONS.items():
        print(f"== {name} ({len(images)} images) ==")
        cdir = work / name
        flagged = build_condition_set(images, test / "labels", cdir, fn)
        yaml = work / f"{name}.yaml"
        yaml.write_text(f"path: {cdir.resolve().as_posix()}\ntrain: images\nval: images\ntest: images\nnames:\n"
                        + "".join(f"  {i}: {n}\n" for i, n in enumerate(CLASS_NAMES)))
        m = model.val(data=str(yaml), split="test", device=args.device, imgsz=args.imgsz, plots=False,
                      verbose=False, project=str(ROOT / "runs" / "robustness"), name=name, exist_ok=True)
        rows.append(dict(condition=name, precision=m.box.mp, recall=m.box.mr, mAP50=m.box.map50,
                         mAP50_95=m.box.map, flagged_by_quality_check=flagged))

    clean = rows[0]["mAP50"] or 1e-9
    for r in rows:
        r["mAP50_change_vs_clean"] = (r["mAP50"] - clean) / clean
    with open(RES / "robustness.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    md = ["### Robustness: performance under different conditions", "",
          "| Condition | Precision | Recall | mAP@0.5 | Change vs clean | Flagged by quality check |",
          "|---|---|---|---|---|---|"]
    for r in rows:
        md.append(f"| {r['condition']} | {r['precision']:.3f} | {r['recall']:.3f} | {r['mAP50']:.3f} | "
                  f"{r['mAP50_change_vs_clean']:+.0%} | {r['flagged_by_quality_check']:.0%} |")
    (RES / "robustness.md").write_text("\n".join(md) + "\n")
    print("\n" + "\n".join(md))
    chart(rows, RES / "robustness_chart.png")
    with_labels = [p for p in images if (test / "labels" / f"{p.stem}.txt").exists()
                   and (test / "labels" / f"{p.stem}.txt").read_text().strip()]
    if with_labels:
        examples(model, with_labels[0], RES / "robustness_examples.jpg")
    print("\nSaved results/robustness.csv, robustness.md, robustness_chart.png, robustness_examples.jpg")


if __name__ == "__main__":
    main()
