"""Image quality check: decide whether an image is good enough to trust.

The detector still runs on every image, but poor-quality images are flagged so
road teams know the result may be unreliable (instead of the AI silently guessing).

Measures (computed on a 640-px-wide grayscale copy so thresholds are size-independent):
  sharpness  - variance of the Laplacian (low = blurry)
  brightness - mean pixel value (low = too dark, high = washed out)
  glare      - share of almost-white pixels (high = sun glare / overexposure)
  contrast   - standard deviation of pixel values (low = fog / haze)

Thresholds were calibrated on the pothole validation images and their synthetic
low-light / blur / rain / glare / fog versions (see scripts/robustness.py). They are
a first heuristic, not a learned model.
"""
from dataclasses import dataclass, field

import cv2
import numpy as np

THRESHOLDS = dict(
    min_sharpness=70.0,
    max_sharpness=12000.0,   # extreme high-frequency noise, e.g. heavy rain streaks
    min_brightness=45.0,
    max_brightness=215.0,
    max_glare=0.20,
    min_contrast=24.0,
)


@dataclass
class QualityReport:
    ok: bool
    warnings: list = field(default_factory=list)
    metrics: dict = field(default_factory=dict)

    def as_markdown(self):
        if self.ok:
            return "✅ **Image quality OK.** Results can be trusted as normal."
        items = "\n".join(f"- {w}" for w in self.warnings)
        return f"⚠️ **Low image quality: results may be unreliable.**\n{items}"


def assess(img_bgr, t=THRESHOLDS) -> QualityReport:
    h, w = img_bgr.shape[:2]
    small = cv2.resize(img_bgr, (640, max(1, int(h * 640 / w))), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    m = dict(
        sharpness=float(cv2.Laplacian(gray, cv2.CV_64F).var()),
        brightness=float(gray.mean()),
        glare=float((gray >= 250).mean()),
        contrast=float(gray.std()),
    )
    warn = []
    if m["sharpness"] < t["min_sharpness"]:
        warn.append(f"Blurry image (sharpness {m['sharpness']:.0f} < {t['min_sharpness']:.0f})")
    if m["brightness"] < t["min_brightness"]:
        warn.append(f"Too dark (brightness {m['brightness']:.0f} < {t['min_brightness']:.0f})")
    if m["brightness"] > t["max_brightness"]:
        warn.append(f"Washed out (brightness {m['brightness']:.0f} > {t['max_brightness']:.0f})")
    if m["glare"] > t["max_glare"]:
        warn.append(f"Strong glare ({m['glare']:.0%} of image near-white)")
    if m["sharpness"] > t["max_sharpness"]:
        warn.append("Heavy noise / rain streaks detected")
    if m["contrast"] < t["min_contrast"] or (m["contrast"] < 32 and m["brightness"] > 150):
        warn.append(f"Low contrast / haze (contrast {m['contrast']:.0f} < {t['min_contrast']:.0f})")
    return QualityReport(ok=not warn, warnings=warn, metrics=m)
