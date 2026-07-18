"""Anisotropic blur/downsampling model used for coefficient consistency."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from .config import DegradationConfig
from .dataset import normalize_image, normalize_rgb


@dataclass(slots=True)
class PsfModel:
    sigma_x_highres: float
    sigma_y_highres: float
    score: float
    low_shape: tuple[int, int]
    high_shape: tuple[int, int]
    method: str = "anisotropic_gaussian_grid_search"

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "sigma_x_highres": self.sigma_x_highres,
            "sigma_y_highres": self.sigma_y_highres,
            "score": self.score,
            "low_shape": list(self.low_shape),
            "high_shape": list(self.high_shape),
        }


def _edge(image: np.ndarray) -> np.ndarray:
    img = normalize_image(image)
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
    return normalize_image(cv2.magnitude(gx, gy))


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    aa = a.reshape(-1).astype(np.float32)
    bb = b.reshape(-1).astype(np.float32)
    valid = np.isfinite(aa) & np.isfinite(bb)
    if valid.sum() < 32:
        return -1.0
    aa = aa[valid] - aa[valid].mean()
    bb = bb[valid] - bb[valid].mean()
    denom = float(np.sqrt(np.sum(aa * aa) * np.sum(bb * bb)))
    return float(np.sum(aa * bb) / denom) if denom > 1e-9 else -1.0


def estimate_psf(rgb: np.ndarray, hsi_structure: np.ndarray, config: DegradationConfig) -> PsfModel:
    high_shape = rgb.shape[:2]
    low_shape = hsi_structure.shape[:2]
    lum = cv2.cvtColor((normalize_rgb(rgb) * 255.0).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    target_edge = _edge(hsi_structure)
    scale_x = high_shape[1] / float(low_shape[1])
    scale_y = high_shape[0] / float(low_shape[0])
    best = (-np.inf, config.default_sigma_x * scale_x, config.default_sigma_y * scale_y)
    if config.estimate_psf:
        for sigma_y_low in config.psf_sigma_y_candidates:
            for sigma_x_low in config.psf_sigma_x_candidates:
                sigma_x = max(0.0, float(sigma_x_low) * scale_x)
                sigma_y = max(0.0, float(sigma_y_low) * scale_y)
                blurred = cv2.GaussianBlur(lum, (0, 0), sigmaX=max(sigma_x, 1e-6), sigmaY=max(sigma_y, 1e-6)) if sigma_x > 0 or sigma_y > 0 else lum
                low = cv2.resize(blurred, (low_shape[1], low_shape[0]), interpolation=cv2.INTER_AREA)
                score = _corr(_edge(low), target_edge)
                if score > best[0]:
                    best = (score, sigma_x, sigma_y)
    method = "anisotropic_gaussian_grid_search"
    if not np.isfinite(best[0]) or best[0] < config.minimum_identifiable_score:
        best = (
            float(best[0]),
            float(config.default_sigma_x) * scale_x,
            float(config.default_sigma_y) * scale_y,
        )
        method = "physical_default_due_to_low_cross_modal_psf_identifiability"
    return PsfModel(
        sigma_x_highres=float(best[1]),
        sigma_y_highres=float(best[2]),
        score=float(best[0]),
        low_shape=low_shape,
        high_shape=high_shape,
        method=method,
    )


def degrade_coefficients(coeff: np.ndarray, model: PsfModel) -> np.ndarray:
    arr = np.asarray(coeff, dtype=np.float32)
    out = np.empty(model.low_shape + (arr.shape[2],), dtype=np.float32)
    for k in range(arr.shape[2]):
        band = arr[:, :, k]
        if model.sigma_x_highres > 0 or model.sigma_y_highres > 0:
            band = cv2.GaussianBlur(
                band,
                (0, 0),
                sigmaX=max(model.sigma_x_highres, 1e-6),
                sigmaY=max(model.sigma_y_highres, 1e-6),
                borderType=cv2.BORDER_REFLECT101,
            )
        out[:, :, k] = cv2.resize(band, (model.low_shape[1], model.low_shape[0]), interpolation=cv2.INTER_AREA)
    return out


def degrade_spatial_map(image: np.ndarray, model: PsfModel) -> np.ndarray:
    """Apply the sensor PSF/downsampling model to a single high-resolution map."""

    band = np.asarray(image, dtype=np.float32)
    if model.sigma_x_highres > 0 or model.sigma_y_highres > 0:
        band = cv2.GaussianBlur(
            band,
            (0, 0),
            sigmaX=max(model.sigma_x_highres, 1e-6),
            sigmaY=max(model.sigma_y_highres, 1e-6),
            borderType=cv2.BORDER_REFLECT101,
        )
    return cv2.resize(band, (model.low_shape[1], model.low_shape[0]), interpolation=cv2.INTER_AREA).astype(np.float32)


def upsample_coefficients(coeff: np.ndarray, high_shape: tuple[int, int]) -> np.ndarray:
    arr = np.asarray(coeff, dtype=np.float32)
    out = np.empty(high_shape + (arr.shape[2],), dtype=np.float32)
    for k in range(arr.shape[2]):
        out[:, :, k] = cv2.resize(arr[:, :, k], (high_shape[1], high_shape[0]), interpolation=cv2.INTER_CUBIC)
    return out
