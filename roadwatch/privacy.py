"""Privacy filter: blur people and vehicle number-plate areas in OUTPUT images.

Uses a standard COCO-pretrained YOLO model (yolov8n.pt, downloaded automatically
the first time) to find people and vehicles.
  * people      -> whole body blurred
  * vehicles    -> lower part of the vehicle blurred (where number plates are)

Road-damage detection runs on the original frame in memory; only the anonymised
version is displayed or saved, so blurring never hides damage from the detector.
"""
import cv2
from ultralytics import YOLO

PERSON = 0
VEHICLES = {1, 2, 3, 5, 7}  # bicycle, car, motorcycle, bus, truck


class PrivacyFilter:
    def __init__(self, model_path="yolov8n.pt", device=None, conf=0.3):
        self.model = YOLO(model_path)
        self.device = device
        self.conf = conf

    def regions(self, img_bgr):
        r = self.model.predict(img_bgr, conf=self.conf, classes=[PERSON, *VEHICLES],
                               device=self.device, verbose=False)[0]
        out = []
        for c, b in zip(r.boxes.cls.tolist(), r.boxes.xyxy.tolist()):
            x1, y1, x2, y2 = map(int, b)
            if int(c) == PERSON:
                out.append((x1, y1, x2, y2))
            else:  # number plates sit in the lower part of a vehicle
                out.append((x1, int(y1 + 0.55 * (y2 - y1)), x2, y2))
        return out

    @staticmethod
    def blur(img_bgr, regions):
        out = img_bgr.copy()
        h, w = out.shape[:2]
        for x1, y1, x2, y2 in regions:
            x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
            if x2 - x1 < 4 or y2 - y1 < 4:
                continue
            roi = out[y1:y2, x1:x2]
            k = max(15, (min(roi.shape[:2]) // 3) | 1)
            out[y1:y2, x1:x2] = cv2.GaussianBlur(roi, (k, k), 0)
        return out
