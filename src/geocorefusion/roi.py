"""ROI selection constrained by RGB texture and sensor coverage."""

from __future__ import annotations

import cv2
import numpy as np

from .config import RoiConfig
from .registration import RegistrationBundle


def _clip_roi(x: int, y: int, width: int, height: int, rgb_shape: tuple[int, int]) -> dict[str, int]:
    width = min(int(width), rgb_shape[1])
    height = min(int(height), rgb_shape[0])
    x = int(np.clip(x, 0, rgb_shape[1] - width))
    y = int(np.clip(y, 0, rgb_shape[0] - height))
    return {"x": x, "y": y, "width": width, "height": height}


def choose_roi(config: RoiConfig, registration: RegistrationBundle, rgb_shape: tuple[int, int]) -> dict[str, int]:
    if config.mode.lower() in {"manual", "fixed"}:
        if config.x is None or config.y is None:
            raise ValueError("Manual ROI requires x and y")
        return _clip_roi(config.x, config.y, config.width, config.height, rgb_shape)

    preview = registration.preview_rgb
    ph, pw = preview.shape
    win_w = max(8, int(round(config.width * pw / rgb_shape[1])))
    win_h = max(8, int(round(config.height * ph / rgb_shape[0])))
    gx = cv2.Sobel(preview, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(preview, cv2.CV_32F, 0, 1, ksize=3)
    texture = cv2.GaussianBlur(cv2.magnitude(gx, gy), (0, 0), 2.0)
    integral = cv2.integral(texture)

    def window_mean(x0: int, y0: int) -> float:
        x1, y1 = x0 + win_w, y0 + win_h
        total = integral[y1, x1] - integral[y0, x1] - integral[y1, x0] + integral[y0, x0]
        return float(total / max(1, win_w * win_h))

    rng = np.random.default_rng(20260717)
    candidates: list[tuple[float, int, int]] = []
    for _ in range(max(config.auto_candidates, 16)):
        px = int(rng.integers(0, max(1, pw - win_w + 1)))
        py = int(rng.integers(0, max(1, ph - win_h + 1)))
        raw_x = int(round(px * rgb_shape[1] / pw))
        raw_y = int(round(py * rgb_shape[0] / ph))
        roi = _clip_roi(raw_x, raw_y, config.width, config.height, rgb_shape)
        ys = np.array([roi["y"], roi["y"], roi["y"] + roi["height"] - 1, roi["y"] + roi["height"] - 1], dtype=np.float32)
        xs = np.array([roi["x"], roi["x"] + roi["width"] - 1, roi["x"], roi["x"] + roi["width"] - 1], dtype=np.float32)
        valid = min(registration.nir.valid_fraction(ys, xs, margin=2), registration.swir.valid_fraction(ys, xs, margin=2))
        if valid < 0.99:
            continue
        patch = preview[py : py + win_h, px : px + win_w]
        dynamic = float(np.percentile(patch, 95) - np.percentile(patch, 5))
        score = window_mean(px, py) + 0.25 * dynamic
        candidates.append((score, raw_x, raw_y))
    if not candidates:
        return _clip_roi((rgb_shape[1] - config.width) // 2, (rgb_shape[0] - config.height) // 2, config.width, config.height, rgb_shape)
    _, x, y = max(candidates, key=lambda item: item[0])
    return _clip_roi(x, y, config.width, config.height, rgb_shape)

