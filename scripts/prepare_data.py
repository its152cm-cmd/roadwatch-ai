"""
Prepare all datasets for RoadWatch AI.

What it does
------------
1. DETECTION dataset (4 classes, boxes):
     0 longitudinal_crack (RDD D00)   1 transverse_crack (D10)
     2 alligator_crack   (D20)        3 pothole          (D40)
   * RDD2022 Pascal VOC .xml -> YOLO boxes. Images with none of these 4 damage
     types are kept (up to --neg-ratio) as negatives: normal road, no damage.
   * Roboflow pothole polygons -> pothole boxes (class 3).
   * Extra damage-free road images (e.g. Road_Condition/Alert clean_roads,
     smooth_roads) -> extra negatives (--extra-negatives).
2. SEGMENTATION dataset (2 classes, pixel outlines):
     0 pothole  (Roboflow polygons)
     1 crack    (CRACK500 binary masks -> polygons, converted automatically)
3. Splits into train / val / test (80 / 10 / 10):
   * Augmented copies of the same Roboflow photo always stay in the same split
     (prevents test leakage).
   * One RDD2022 country can be held out completely (--holdout) to test the model
     on roads it has never seen.
   * CRACK500's own train/val/test folders are respected when present.

Output
------
datasets/roadwatch_det/{train,val,test,unseen_<country>,pothole_test}/{images,labels}
datasets/roadwatch_seg/{train,val,test,pothole_test}/{images,labels}
configs/roadwatch_det.yaml  configs/unseen_<country>.yaml  configs/roadwatch_seg.yaml
configs/pothole_test_det.yaml  configs/pothole_test_seg.yaml
results/dataset_summary.json  results/extra_negatives_used.txt

Example
-------
python scripts/prepare_data.py --rdd-root data/RDD2022 \
    --countries Japan India United_States Czech --holdout Czech \
    --pothole-root data/pothole_dataset --crack500-root data/CRACK500 \
    --extra-negatives data/RoadAlert/clean_roads data/RoadAlert/smooth_roads \
    --max-per-country 2500
"""
import argparse
import json
import random
import re
import shutil
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

NAMES = ["longitudinal_crack", "transverse_crack", "alligator_crack", "pothole"]
SEG_NAMES = ["pothole", "crack"]
RDD_MAP = {"D00": 0, "D10": 1, "D20": 2, "D40": 3}
POTHOLE_ID = 3
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}
ROOT = Path(__file__).resolve().parents[1]


# ----------------------------------------------------------------- RDD2022
def find_country_dir(rdd_root: Path, country: str) -> Path:
    """Accept RDD2022/<Country> or RDD2022/RDD2022_<Country> (any capitalisation)."""
    wanted = country.lower().replace(" ", "_")
    candidates = [rdd_root] + [d for d in rdd_root.iterdir() if d.is_dir()]
    candidates += [d for d in rdd_root.glob("*/*") if d.is_dir()]
    for p in candidates:
        if p.name.lower() in (wanted, f"rdd2022_{wanted}"):
            return p
    raise FileNotFoundError(f"Country folder for '{country}' not found under {rdd_root}")


def parse_voc(xml_path: Path, img_path: Path):
    """Return YOLO label lines for the 4 classes (empty list = negative image)."""
    root = ET.parse(xml_path).getroot()
    try:
        w = float(root.find("size/width").text)
        h = float(root.find("size/height").text)
    except (AttributeError, TypeError, ValueError):
        w = h = 0
    if w <= 0 or h <= 0:  # some files have missing/zero size -> read the image
        with Image.open(img_path) as im:
            w, h = im.size
    lines = []
    for obj in root.iter("object"):
        name = (obj.findtext("name") or "").strip()
        if name not in RDD_MAP:
            continue
        bb = obj.find("bndbox")
        x1, y1, x2, y2 = (float(bb.findtext(k)) for k in ("xmin", "ymin", "xmax", "ymax"))
        x1, x2 = sorted((max(0, min(w, x1)), max(0, min(w, x2))))
        y1, y2 = sorted((max(0, min(h, y1)), max(0, min(h, y2))))
        if (x2 - x1) < 2 or (y2 - y1) < 2:
            continue
        cx, cy, bw, bh = (x1 + x2) / 2 / w, (y1 + y2) / 2 / h, (x2 - x1) / w, (y2 - y1) / h
        lines.append(f"{RDD_MAP[name]} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
    return lines


def collect_rdd(country_dir: Path, country: str, max_images: int, neg_ratio: float, rng):
    images = {p.stem: p for p in country_dir.rglob("*") if p.suffix.lower() in IMG_EXT}
    pos, neg = [], []
    for xml in sorted(country_dir.rglob("*.xml")):
        img = images.get(xml.stem)
        if img is None:
            continue
        lines = parse_voc(xml, img)
        item = dict(img=img, lines=lines, group=f"{country}/{xml.stem}", source=country)
        (pos if lines else neg).append(item)
    rng.shuffle(pos)
    rng.shuffle(neg)
    if max_images:
        pos = pos[:max_images]
    neg = neg[: int(len(pos) * neg_ratio)]
    print(f"  {country:<18} {len(pos):>6} images with damage, {len(neg):>5} negatives")
    return pos + neg


# ---------------------------------------------------------- Roboflow potholes
def polygon_to_box(vals):
    xs, ys = vals[0::2], vals[1::2]
    x1, x2, y1, y2 = max(0, min(xs)), min(1, max(xs)), max(0, min(ys)), min(1, max(ys))
    return (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1


def collect_potholes(pothole_root: Path):
    items = []
    for lbl in pothole_root.rglob("labels/*.txt"):
        img = next((lbl.parent.parent / "images" / (lbl.stem + e) for e in (".jpg", ".jpeg", ".png")
                    if (lbl.parent.parent / "images" / (lbl.stem + e)).exists()), None)
        if img is None:
            continue
        seg_lines, det_lines = [], []
        for row in lbl.read_text().split("\n"):
            parts = row.split()
            if len(parts) < 5:
                continue
            vals = [float(v) for v in parts[1:]]
            if len(vals) == 4:
                cx, cy, bw, bh = vals
            else:
                cx, cy, bw, bh = polygon_to_box(vals)
                seg_lines.append("0 " + " ".join(f"{v:.6f}" for v in vals))
            det_lines.append(f"{POTHOLE_ID} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
        group = "pothole/" + re.split(r"_jpg\.rf\.|\.rf\.", lbl.stem)[0]
        items.append(dict(img=img, lines=det_lines, seg=seg_lines, group=group, source="roboflow_pothole"))
    print(f"  {'roboflow_pothole':<18} {len(items):>6} images "
          f"({len({i['group'] for i in items})} unique photos)")
    return items


# ------------------------------------------------------------------ CRACK500
def mask_to_polygons(mask_path: Path, cls=1, min_len=12, eps=1.5):
    """Binary crack mask -> YOLO polygon lines (one per connected crack region)."""
    m = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if m is None:
        return []
    h, w = m.shape
    _, m = cv2.threshold(m, 127, 255, cv2.THRESH_BINARY)
    m = cv2.dilate(m, np.ones((3, 3), np.uint8))          # join broken hairline pixels
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    lines = []
    for c in contours:
        x, y, bw, bh = cv2.boundingRect(c)
        if max(bw, bh) < min_len:
            continue
        c = cv2.approxPolyDP(c, eps, True).reshape(-1, 2)
        if len(c) < 3:
            continue
        lines.append(f"{cls} " + " ".join(f"{px / w:.6f} {py / h:.6f}" for px, py in c))
    return lines


def find_mask(img: Path):
    """Ground-truth mask in the SAME folder: <name>.png or <name>_mask/_gt/_label.png."""
    for suffix in ("", "_mask", "_gt", "_label"):
        m = img.with_name(img.stem + suffix + ".png")
        if m.exists():
            return m
    return None


def collect_crack500(root: Path, max_images: int, rng):
    """CRACK500 layout: traincrop/ valcrop/ testcrop/ (640x360 crops, image .jpg + mask .png),
    traindata/ valdata/ testdata/ (full-size originals of the same photos) and
    fphb_testresult/ (the authors' model PREDICTIONS, not ground truth).
    Uses only the *crop folders when present (no duplicates), never the results folder."""
    folders = [d for d in root.rglob("*") if d.is_dir() and "result" not in d.name.lower()]
    crops = [d for d in folders if d.name.lower().endswith("crop")]
    use = crops or [root] + folders
    items, seen = [], set()
    for folder in use:
        for img in folder.iterdir():
            if img.suffix.lower() not in {".jpg", ".jpeg"} or img in seen:
                continue
            seen.add(img)
            mask = find_mask(img)
            if mask is None:
                continue
            lines = mask_to_polygons(mask)
            if not lines:
                continue
            name = img.parent.name.lower()   # traincrop -> train, valcrop -> val, testcrop -> test
            split = "test" if "test" in name else "val" if "val" in name else "train" if "train" in name else None
            items.append(dict(img=img, seg=lines, split=split, source="crack500"))
    rng.shuffle(items)
    if max_images:
        items = items[:max_images]
    for it in items:                         # no split folders -> 80/10/10 at random
        if it["split"] is None:
            r = rng.random()
            it["split"] = "test" if r < 0.1 else "val" if r < 0.2 else "train"
    print(f"  {'crack500':<18} {len(items):>6} images with crack masks")
    return items


# ----------------------------------------------------------- extra negatives
def collect_extra_negatives(dirs, max_images, rng):
    files = []
    for d in dirs:
        files += [p for p in Path(d).rglob("*") if p.suffix.lower() in IMG_EXT]
    rng.shuffle(files)
    files = files[:max_images]
    print(f"  {'extra_negatives':<18} {len(files):>6} damage-free road images")
    return [dict(img=p, lines=[], group=f"neg/{p.parent.name}/{p.stem}", source="extra_negative") for p in files]


# --------------------------------------------------------------- splitting
def split_by_group(items, rng, val=0.1, test=0.1):
    by_source = defaultdict(list)
    for it in items:
        by_source[it["source"]].append(it)
    out = {"train": [], "val": [], "test": []}
    for its in by_source.values():
        groups = sorted({i["group"] for i in its})
        rng.shuffle(groups)
        n = len(groups)
        n_test, n_val = max(1, round(n * test)), max(1, round(n * val))
        test_g, val_g = set(groups[:n_test]), set(groups[n_test:n_test + n_val])
        for it in its:
            out["test" if it["group"] in test_g else "val" if it["group"] in val_g else "train"].append(it)
    return out


def write_split(items, dest: Path, key="lines"):
    (dest / "images").mkdir(parents=True, exist_ok=True)
    (dest / "labels").mkdir(parents=True, exist_ok=True)
    for it in items:
        stem = f"{it['source']}__{it['img'].parent.name}__{it['img'].stem}"
        shutil.copy2(it["img"], dest / "images" / f"{stem}{it['img'].suffix.lower()}")
        (dest / "labels" / f"{stem}.txt").write_text("\n".join(it[key]))


def write_yaml(path: Path, base: Path, splits: dict, names):
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"path: {base.resolve().as_posix()}"] + [f"{k}: {v}" for k, v in splits.items()]
    lines += ["names:"] + [f"  {i}: {n}" for i, n in enumerate(names)]
    path.write_text("\n".join(lines) + "\n")
    print(f"  wrote {path.relative_to(ROOT)}")


def class_counts(items, key="lines", names=NAMES):
    c = Counter()
    for it in items:
        for line in it.get(key, []):
            c[names[int(line.split()[0])]] += 1
    return dict(c)


# --------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rdd-root", type=Path, help="Folder containing the unzipped RDD2022 country folders")
    ap.add_argument("--countries", nargs="*", default=["Japan", "India", "United_States", "Czech"])
    ap.add_argument("--holdout", default=None, help="Country kept out of training (unseen-country test)")
    ap.add_argument("--pothole-root", type=Path, help="Roboflow pothole dataset folder (has train/ valid/)")
    ap.add_argument("--crack500-root", type=Path, help="CRACK500 folder (images + binary mask .png files)")
    ap.add_argument("--crack500-max", type=int, default=1500, help="Max CRACK500 images (0 = all)")
    ap.add_argument("--extra-negatives", nargs="*", default=[], help="Folders of damage-free road images")
    ap.add_argument("--extra-neg-max", type=int, default=400, help="Max extra negative images")
    ap.add_argument("--max-per-country", type=int, default=2500, help="Max damage images per country (0 = all)")
    ap.add_argument("--neg-ratio", type=float, default=0.15, help="RDD negatives per damage image")
    ap.add_argument("--out", type=Path, default=ROOT / "datasets")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    det_root, seg_root = args.out / "roadwatch_det", args.out / "roadwatch_seg"
    for d in (det_root, seg_root):
        if d.exists():
            shutil.rmtree(d)

    print("Collecting images...")
    det_items, unseen, potholes, cracks = [], [], [], []
    if args.rdd_root:
        for c in args.countries:
            got = collect_rdd(find_country_dir(args.rdd_root, c), c, args.max_per_country, args.neg_ratio, rng)
            (unseen if args.holdout and c.lower() == args.holdout.lower() else det_items).extend(got)
    if args.pothole_root:
        potholes = collect_potholes(args.pothole_root)
        det_items.extend(potholes)
    if args.extra_negatives:
        extra = collect_extra_negatives(args.extra_negatives, args.extra_neg_max, rng)
        det_items.extend(extra)
        (ROOT / "results").mkdir(exist_ok=True)
        (ROOT / "results" / "extra_negatives_used.txt").write_text(
            "\n".join(str(i["img"].resolve()) for i in extra))
    if args.crack500_root:
        cracks = collect_crack500(args.crack500_root, args.crack500_max, rng)
    if not det_items and not cracks:
        raise SystemExit("No data found. Pass --rdd-root / --pothole-root / --crack500-root.")

    splits = split_by_group(det_items, rng) if det_items else {"train": [], "val": [], "test": []}
    yaml_splits = {"train": "train/images", "val": "val/images", "test": "test/images"}
    summary = {}

    if det_items:
        print("\nWriting detection dataset...")
        for name, its in splits.items():
            write_split(its, det_root / name)
            summary[f"det_{name}"] = {"images": len(its), "boxes_per_class": class_counts(its),
                                      "negatives": sum(1 for i in its if not i["lines"])}
        write_yaml(ROOT / "configs" / "roadwatch_det.yaml", det_root, yaml_splits, NAMES)
    if unseen:
        uname = f"unseen_{args.holdout.lower()}"
        write_split(unseen, det_root / uname)
        write_yaml(ROOT / "configs" / f"{uname}.yaml", det_root,
                   {"train": "train/images", "val": f"{uname}/images", "test": f"{uname}/images"}, NAMES)
        summary[uname] = {"images": len(unseen), "boxes_per_class": class_counts(unseen)}

    if potholes or cracks:
        print("\nWriting segmentation dataset (pothole + crack outlines)...")
        for name in ("train", "val", "test"):
            seg_items = [i for i in splits[name] if i["source"] == "roboflow_pothole" and i["seg"]]
            seg_items += [i for i in cracks if i["split"] == name]
            write_split(seg_items, seg_root / name, key="seg")
            summary[f"seg_{name}"] = {"images": len(seg_items),
                                      "outlines_per_class": class_counts(seg_items, "seg", SEG_NAMES)}
        write_yaml(ROOT / "configs" / "roadwatch_seg.yaml", seg_root, yaml_splits, SEG_NAMES)
    if potholes:
        # identical pothole-only test images for a fair detection vs segmentation comparison
        pt = [i for i in splits["test"] if i["source"] == "roboflow_pothole"]
        write_split(pt, det_root / "pothole_test")
        write_split([i for i in pt if i["seg"]], seg_root / "pothole_test", key="seg")
        write_yaml(ROOT / "configs" / "pothole_test_det.yaml", det_root,
                   {"train": "train/images", "val": "pothole_test/images", "test": "pothole_test/images"}, NAMES)
        write_yaml(ROOT / "configs" / "pothole_test_seg.yaml", seg_root,
                   {"train": "train/images", "val": "pothole_test/images", "test": "pothole_test/images"}, SEG_NAMES)

    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "dataset_summary.json").write_text(json.dumps(summary, indent=2))
    print("\nDataset summary:")
    for k, v in summary.items():
        print(f"  {k:<16} {v['images']:>6} images  {v.get('boxes_per_class', v.get('outlines_per_class'))}")
    print("\nDone. Next: python scripts/train_detect.py")


if __name__ == "__main__":
    main()
