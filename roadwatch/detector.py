"""RoadWatch AI pipeline: quality check -> damage detection -> (segmentation) -> privacy blur -> output.

    from roadwatch.detector import RoadWatch
    rw = RoadWatch("weights/roadwatch_det.pt", "weights/roadwatch_seg.pt")
    result = rw.analyze_image(cv2.imread("road.jpg"))
    rw.analyze_video("road.mp4", "out.mp4")
"""
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from . import CLASS_NAMES
from .privacy import PrivacyFilter
from .quality import assess

# BGR colours per class (yellow = pothole to match the project theme)
COLORS = {
    "longitudinal_crack": (255, 170, 60),
    "transverse_crack": (200, 90, 255),
    "alligator_crack": (60, 120, 255),
    "pothole": (63, 210, 255),
}
SEG_COLORS = {"pothole": (63, 210, 255), "crack": (255, 80, 200)}
LABELS = {
    "longitudinal_crack": "long. crack",
    "transverse_crack": "trans. crack",
    "alligator_crack": "alligator crack",
    "pothole": "pothole",
}


def draw_box(img, box, name, conf, track_id=None):
    x1, y1, x2, y2 = map(int, box)
    color = COLORS.get(name, (255, 255, 255))
    t = max(2, img.shape[1] // 400)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, t)
    text = f"{LABELS.get(name, name)} {conf:.2f}" + (f" #{track_id}" if track_id is not None else "")
    fs = max(0.5, img.shape[1] / 1600)
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fs, 1)
    y = max(th + 6, y1)
    cv2.rectangle(img, (x1, y - th - 6), (x1 + tw + 6, y), color, -1)
    cv2.putText(img, text, (x1 + 3, y - 4), cv2.FONT_HERSHEY_SIMPLEX, fs, (20, 20, 20), 1, cv2.LINE_AA)


class RoadWatch:
    def __init__(self, det_path, seg_path=None, privacy_model="yolov8n.pt", device=None):
        if not Path(det_path).exists():
            raise FileNotFoundError(f"Detection model not found: {det_path} (train it or copy best.pt there)")
        self.det = YOLO(det_path)
        self.seg = YOLO(seg_path) if seg_path and Path(seg_path).exists() else None
        self.device = device
        self._privacy_model = privacy_model
        self._privacy = None
        self.names = self.det.names  # {id: name}

    @property
    def privacy(self):
        if self._privacy is None:
            self._privacy = PrivacyFilter(self._privacy_model, self.device)
        return self._privacy

    # ------------------------------------------------------------- images
    def analyze_image(self, img_bgr, conf=0.25, use_seg=True, privacy=True):
        quality = assess(img_bgr)
        r = self.det.predict(img_bgr, conf=conf, device=self.device, verbose=False)[0]
        dets = [dict(hazard=self.names[int(c)], confidence=round(float(s), 3),
                     box=[round(v) for v in b])
                for c, s, b in zip(r.boxes.cls.tolist(), r.boxes.conf.tolist(), r.boxes.xyxy.tolist())]

        out = self.privacy.blur(img_bgr, self.privacy.regions(img_bgr)) if privacy else img_bgr.copy()
        if use_seg and self.seg is not None:   # pixel outlines of potholes and cracks
            s = self.seg.predict(img_bgr, conf=conf, device=self.device, verbose=False, retina_masks=True)[0]
            if s.masks is not None:
                overlay = out.copy()
                for poly, c in zip(s.masks.xy, s.boxes.cls.tolist()):
                    pts = poly.astype(np.int32)
                    color = SEG_COLORS.get(self.seg.names[int(c)], (255, 255, 255))
                    if len(pts) >= 3:
                        cv2.fillPoly(overlay, [pts], color)
                        cv2.polylines(out, [pts], True, color, 2)
                out = cv2.addWeighted(overlay, 0.4, out, 0.6, 0)
        for d in dets:
            draw_box(out, d["box"], d["hazard"], d["confidence"])
        counts = Counter(d["hazard"] for d in dets)
        return dict(image=out, detections=dets, counts=dict(counts), quality=quality)

    # ------------------------------------------------------------- videos
    def analyze_video(self, src, dst, conf=0.25, privacy=True, max_seconds=None,
                      min_track_frames=3, progress=None):
        """Track hazards across frames so each one is counted once (ByteTrack)."""
        import imageio.v2 as imageio

        cap = cv2.VideoCapture(str(src))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        limit = int(fps * max_seconds) if max_seconds else total
        writer = imageio.get_writer(str(dst), fps=fps, codec="libx264", quality=7,
                                    macro_block_size=1, ffmpeg_log_level="error")
        track_frames = defaultdict(int)      # track id -> frames seen
        track_class = {}                     # track id -> class name
        flagged = n = 0
        self.det.predictor = None            # reset tracker state between videos
        while True:
            ok, frame = cap.read()
            if not ok or (limit and n >= limit):
                break
            n += 1
            flagged += not assess(frame).ok
            r = self.det.track(frame, conf=conf, persist=True, tracker="bytetrack.yaml",
                               device=self.device, verbose=False)[0]
            out = self.privacy.blur(frame, self.privacy.regions(frame)) if privacy else frame.copy()
            ids = r.boxes.id.int().tolist() if r.boxes.id is not None else [None] * len(r.boxes)
            for c, s, b, tid in zip(r.boxes.cls.tolist(), r.boxes.conf.tolist(), r.boxes.xyxy.tolist(), ids):
                name = self.names[int(c)]
                if tid is not None:
                    track_frames[tid] += 1
                    track_class[tid] = name
                draw_box(out, b, name, s, tid)
            writer.append_data(cv2.cvtColor(out, cv2.COLOR_BGR2RGB))
            if progress and limit:
                progress(n / limit)
        cap.release()
        writer.close()
        unique = Counter(track_class[t] for t, k in track_frames.items() if k >= min_track_frames)
        return dict(video=str(dst), frames=n, unique_counts=dict(unique),
                    quality_flagged=flagged / max(1, n))
