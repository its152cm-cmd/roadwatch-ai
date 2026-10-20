"""Synthetic road conditions used for robustness (stress) testing.

Each function takes and returns a BGR uint8 image of the same size, so box
labels stay valid. A fixed random seed keeps results reproducible.
"""
import cv2
import numpy as np


def low_light(img, factor=0.3):
    return np.clip(img.astype(np.float32) * factor, 0, 255).astype(np.uint8)


def motion_blur(img, k=None):
    k = k or max(9, int(img.shape[1] / 25) | 1)
    kernel = np.zeros((k, k), np.float32)
    kernel[k // 2, :] = 1.0 / k
    return cv2.filter2D(img, -1, kernel)


def rain(img, drops=None, seed=0):
    rng = np.random.default_rng(seed)
    h, w = img.shape[:2]
    drops = drops or int(h * w / 350)
    base = cv2.GaussianBlur((img * 0.75).astype(np.uint8), (5, 5), 0)
    layer = np.zeros_like(img)
    length = max(10, h // 22)
    for _ in range(drops):
        x, y = int(rng.integers(0, w)), int(rng.integers(0, h))
        cv2.line(layer, (x, y), (x - length // 4, y + length), (210, 210, 210), 1)
    return cv2.addWeighted(base, 1.0, layer, 0.7, 0)


def glare(img, seed=0):
    rng = np.random.default_rng(seed)
    h, w = img.shape[:2]
    cx, cy = rng.uniform(0.3, 0.7) * w, rng.uniform(0.2, 0.5) * h
    yy, xx = np.mgrid[0:h, 0:w]
    g = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * (0.27 * w) ** 2))
    return np.clip(img.astype(np.float32) + 255 * g[..., None], 0, 255).astype(np.uint8)


def fog(img, strength=0.55):
    return cv2.addWeighted(img, 1 - strength, np.full_like(img, 215), strength, 0)


def jpeg(img, quality=10):
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


CONDITIONS = {
    "clean": lambda x: x,
    "low_light": low_light,
    "motion_blur": motion_blur,
    "rain": rain,
    "glare": glare,
    "fog": fog,
    "jpeg_compression": jpeg,
}
