# RoadWatch AI: Hackathon Day Runbook

Development window: 10:30 AM – 6:00 PM IST = **1:00 PM – 8:30 PM Malaysia time (MYT)**.
Hard deadline: **8:30 PM MYT**. All times below are MYT.

Commands are for the PyCharm terminal on Windows (run from the `roadwatch-ai` folder).

---

## Before 1:00 PM (setup only, if the rules allow)

**1. Python environment**
```bash
python -m venv .venv
.venv\Scripts\activate
```
Install PyTorch with CUDA (pick the command for your CUDA version at https://pytorch.org/get-started/locally/), e.g.:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

**2. Check the GPU is visible**
```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```
It must print `True` and your GPU name. If it prints `False`, reinstall torch with the CUDA command above.

**3. Start the RDD2022 download (about 2.2 GB, can take a while)**
```bash
python scripts/download_rdd.py --countries Japan India United_States Czech --out data/RDD2022
```
If the download is slow, start with `--countries Czech India` (smallest) and add Japan / United_States later.

**4. Copy the pothole dataset** (the `dataset` folder you already have, with `train/` and `valid/`) to `data/pothole_dataset`.

**5. Kaggle datasets** (download in the browser, unzip):
- CRACK500 (about 5.8 GB, so **start this early**) → unzip into `data/CRACK500`
- Road_Condition/Alert (about 320 MB) → unzip so the folders are `data/RoadAlert/potholes`, `data/RoadAlert/clean_roads`, `data/RoadAlert/smooth_roads`, …

Nothing needs labelling: CRACK500 masks are converted automatically, and Road_Condition/Alert uses its folder names.

---

## 1:00 – 1:30 PM · Prepare data and run a quick test

```bash
python scripts/prepare_data.py --rdd-root data/RDD2022 --countries Japan India United_States Czech --holdout Czech --pothole-root data/pothole_dataset --crack500-root data/CRACK500 --extra-negatives data/RoadAlert/clean_roads data/RoadAlert/smooth_roads --max-per-country 2500
```
Check the printed summary: train/val/test counts, boxes per class (detection) and outlines per class (segmentation: pothole + crack). Saved in `results/dataset_summary.json`.
- `crack500 0 images` → the mask files weren't found; check that the CRACK500 folder has `.jpg` images with matching `.png` masks, and send the folder layout to Claude
- Missing a dataset? Just leave out its option. Every dataset is optional except at least one of RDD2022 / pothole.

Quick 2-epoch test (about 5 min) to catch errors before the long run:
```bash
python scripts/train_detect.py --model yolov8n.pt --epochs 2 --imgsz 416 --fraction 0.1 --workers 2 --name quicktest
```

---

## 1:30 – 4:15 PM · Main training (runs by itself)

```bash
python scripts/train_detect.py --model yolov8s.pt --epochs 60 --batch 16 --workers 2 --hours 2.5
```
- `--hours 2.5` guarantees training stops by about 4:00 PM, keeping the best model so far.
- **CUDA out of memory?** Use `--batch 8`. Still failing? Use `--model yolov8n.pt`.
- **GPU with 4 GB or less:** `--model yolov8n.pt --batch 8`.
- Progress: `runs/detect/det_4class/results.png` updates every epoch.
- Result: `weights/roadwatch_det.pt`

**While it trains (other person or other laptop):**
- Fill in the README team/intro sections and paste the dataset summary numbers
- Optional look-alike test: put 10–20 photos per type with **no damage** into
  `data/lookalikes/puddle`, `data/lookalikes/shadow`, `data/lookalikes/manhole`, `data/lookalikes/cone`
  (your own phone photos are best, which avoids licensing issues)
- Create the public GitHub repo (see the end of this file)
- Prepare the 1–2 minute demo script

## 4:15 – 4:45 PM · Segmentation model (potholes + cracks)

```bash
python scripts/train_segment.py --model yolov8n-seg.pt --epochs 60 --workers 2
```
About 20–35 min on a GPU (more images now, with CRACK500). Add `--epochs 40` if short on time. Result: `weights/roadwatch_seg.pt`.
*Short on time? Skip this step. The app and evaluation still work without it.*

## 4:45 – 5:45 PM · Evaluation (all results into results/)

```bash
python scripts/evaluate.py
python scripts/robustness.py --max-images 400
python scripts/error_analysis.py
python scripts/image_level_test.py --pothole-dir data/RoadAlert/potholes --clean-dirs data/RoadAlert/clean_roads data/RoadAlert/smooth_roads
```
Add `--lookalike-dir data/lookalikes` to the last command if you collected look-alike photos.

Then **look at** `results/errors_false_alarms.jpg` and `results/errors_misses.jpg` and write 3–4 bullet
points on the main causes into the README "Error analysis" section.

## 5:45 – 6:30 PM · Demo

```bash
python app.py
```
Open http://127.0.0.1:7860. Try an example image, then a video (the sample video or your own road clip).
- Take a screenshot → save as `assets/demo_screenshot.png` and add it to the top of the README
- Record a 1–2 minute screen video of the demo (Windows: Win+Alt+R with Xbox Game Bar, or OBS)
- `python app.py --share` gives a temporary public link if judges want to try it live

## 6:30 – 7:30 PM · README results and GitHub push

1. Copy the tables from `results/summary.md`, `results/robustness.md`, `results/error_analysis.md` and `results/image_level_test.md` into the README placeholders
2. Push:
```bash
git init
git add .
git commit -m "RoadWatch AI: road damage detection"
git branch -M main
git remote add origin https://github.com/<your-username>/roadwatch-ai.git
git push -u origin main
```
`.gitignore` already excludes datasets, runs and large files; `weights/*.pt` (about 6–25 MB) are included.
3. Open the repo in a private/incognito window to check it is public and the README images show.

## 7:30 – 8:30 PM · Buffer and submission

Submit the GitHub link **before 8:30 PM MYT**. Don't start new features after 7:30 PM.

---

## If something goes wrong

| Problem | Fix |
|---|---|
| `torch.cuda.is_available()` is False | Reinstall torch with the CUDA command from pytorch.org |
| CUDA out of memory | `--batch 8`, or `--model yolov8n.pt`, or `--imgsz 512` |
| Training hangs at start on Windows | `--workers 0` |
| Download too slow | Use fewer countries (`Czech India`), lower `--max-per-country 1500` |
| Behind schedule at 4:00 PM | Skip segmentation; run `robustness.py --max-images 150` |
| `No module named lap` (video) | `pip install lap` |
| App video won't play | `pip install imageio-ffmpeg` |

## GitHub repo settings
- Public, name `roadwatch-ai`
- License: **AGPL-3.0** (Ultralytics YOLO is AGPL-3.0, so a matching license is the safe choice)
- Add description: "AI road surface damage detection: potholes and cracks, with robustness testing"
