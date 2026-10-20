"""
Download and unzip RDD2022 country subsets (official CRDDC2022 links).

Sizes: Japan ~1.0 GB, India ~0.5 GB, United_States ~0.42 GB, Czech ~0.25 GB,
       China_MotorBike ~0.18 GB.  (Norway is ~10 GB, so skip it.)

Example:
    python scripts/download_rdd.py --countries Japan India United_States Czech --out data/RDD2022
"""
import argparse
import sys
import urllib.request
import zipfile
from pathlib import Path

BASE = "https://bigdatacup.s3.ap-northeast-1.amazonaws.com/2022/CRDDC2022/RDD2022/Country_Specific_Data_CRDDC2022/"


def progress(count, block, total):
    done = count * block
    pct = min(100, done * 100 / total) if total > 0 else 0
    sys.stdout.write(f"\r  {done / 1e6:8.1f} MB  {pct:5.1f}%")
    sys.stdout.flush()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--countries", nargs="+", default=["Japan", "India", "United_States", "Czech"])
    ap.add_argument("--out", type=Path, default=Path("data/RDD2022"))
    ap.add_argument("--keep-zip", action="store_true")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    for c in args.countries:
        zpath = args.out / f"RDD2022_{c}.zip"
        if not zpath.exists():
            print(f"Downloading {c} ...")
            urllib.request.urlretrieve(BASE + f"RDD2022_{c}.zip", zpath, progress)
            print()
        print(f"Unzipping {zpath.name} ...")
        with zipfile.ZipFile(zpath) as z:
            z.extractall(args.out / f"RDD2022_{c}")
        if not args.keep_zip:
            zpath.unlink()
    print(f"Done. Use: python scripts/prepare_data.py --rdd-root {args.out} --countries {' '.join(args.countries)}")


if __name__ == "__main__":
    main()
