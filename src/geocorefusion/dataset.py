"""Dataset discovery and modality-stable previews."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .envi import EnviMetadata, open_cube, parse_header


@dataclass(slots=True)
class SensorData:
    name: str
    meta: EnviMetadata
    cube: np.ndarray


@dataclass(slots=True)
class DatasetTriplet:
    root: Path
    rgb: SensorData
    nir: SensorData
    swir: SensorData


def discover_triplet(root: str | Path) -> DatasetTriplet:
    directory = Path(root)
    headers = sorted(directory.glob("*.hdr"))
    selected: dict[str, Path] = {}
    for path in headers:
        upper = path.name.upper()
        for sensor in ("RGB", "NIR", "SWIR"):
            if upper.startswith(sensor + "-"):
                selected[sensor.lower()] = path
    missing = [name for name in ("rgb", "nir", "swir") if name not in selected]
    if missing:
        raise FileNotFoundError(f"Missing sensor headers {missing} under {directory}")

    sensors: dict[str, SensorData] = {}
    for name, hdr in selected.items():
        meta = parse_header(hdr)
        cube, _ = open_cube(meta)
        sensors[name] = SensorData(name=name.upper(), meta=meta, cube=cube)
    return DatasetTriplet(root=directory, rgb=sensors["rgb"], nir=sensors["nir"], swir=sensors["swir"])


def normalize_image(image: np.ndarray, low: float = 2.0, high: float = 98.0) -> np.ndarray:
    arr = np.asarray(image, dtype=np.float32)
    valid = np.isfinite(arr)
    if not valid.any():
        return np.zeros(arr.shape, dtype=np.float32)
    lo, hi = np.percentile(arr[valid], [low, high])
    if hi <= lo:
        hi = lo + 1e-6
    return np.clip((arr - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def normalize_rgb(rgb: np.ndarray) -> np.ndarray:
    arr = np.asarray(rgb[:, :, :3])
    if np.issubdtype(arr.dtype, np.integer):
        scale = float(np.iinfo(arr.dtype).max)
        return arr.astype(np.float32) / scale
    out = arr.astype(np.float32)
    finite = out[np.isfinite(out)]
    if finite.size and np.percentile(finite, 99.9) > 2.0:
        out /= 255.0
    return np.clip(out, 0.0, 1.0)


def select_band_indices(meta: EnviMetadata, wavelengths_nm: list[float]) -> list[int]:
    if not meta.wavelengths.size:
        return list(range(min(meta.bands, len(wavelengths_nm))))
    return [int(np.argmin(np.abs(meta.wavelengths - wavelength))) for wavelength in wavelengths_nm]


def hsi_structure(
    cube: np.ndarray,
    meta: EnviMetadata,
    sensor: str,
    *,
    max_height: int | None = None,
    target_shape: tuple[int, int] | None = None,
) -> np.ndarray:
    targets = [720, 820, 950, 1100, 1300, 1450] if sensor.upper() == "NIR" else [1050, 1250, 1650, 2200, 2350]
    indices = sorted(set(select_band_indices(meta, targets)))
    if target_shape is not None:
        yi = np.linspace(0, cube.shape[0] - 1, target_shape[0]).round().astype(np.int64)
        xi = np.linspace(0, cube.shape[1] - 1, target_shape[1]).round().astype(np.int64)
        sampled = np.asarray(cube[np.ix_(yi, xi, np.asarray(indices, dtype=np.int64))], dtype=np.float32)
    else:
        step = max(1, int(np.ceil(cube.shape[0] / max_height))) if max_height else 1
        sampled = np.asarray(cube[::step, :, indices], dtype=np.float32)
    sampled = np.nan_to_num(sampled, nan=0.0, posinf=0.0, neginf=0.0)
    channels = [normalize_image(sampled[:, :, i]) for i in range(sampled.shape[2])]
    mean = np.mean(channels, axis=0)
    spectral_contrast = np.std(np.stack(channels, axis=2), axis=2)
    grad_x = cv2.Sobel(mean, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(mean, cv2.CV_32F, 0, 1, ksize=3)
    edges = normalize_image(cv2.magnitude(grad_x, grad_y))
    return normalize_image(0.45 * mean + 0.25 * normalize_image(spectral_contrast) + 0.30 * edges)


def rgb_structure(
    rgb: np.ndarray,
    *,
    max_height: int | None = None,
    target_shape: tuple[int, int] | None = None,
) -> np.ndarray:
    if target_shape is not None:
        yi = np.linspace(0, rgb.shape[0] - 1, target_shape[0]).round().astype(np.int64)
        xi = np.linspace(0, rgb.shape[1] - 1, target_shape[1]).round().astype(np.int64)
        sampled = np.asarray(rgb[np.ix_(yi, xi, np.arange(3, dtype=np.int64))])
    else:
        step = max(1, int(np.ceil(rgb.shape[0] / max_height))) if max_height else 1
        sampled = np.asarray(rgb[::step, :, :3])
    rgb01 = normalize_rgb(sampled)
    lab = cv2.cvtColor((rgb01 * 255.0).astype(np.uint8), cv2.COLOR_RGB2LAB).astype(np.float32)
    lum = normalize_image(lab[:, :, 0])
    chroma = normalize_image(np.sqrt((lab[:, :, 1] - 128.0) ** 2 + (lab[:, :, 2] - 128.0) ** 2))
    grad_x = cv2.Sobel(lum, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(lum, cv2.CV_32F, 0, 1, ksize=3)
    edges = normalize_image(cv2.magnitude(grad_x, grad_y))
    return normalize_image(0.45 * lum + 0.15 * chroma + 0.40 * edges)


def resize_preview(image: np.ndarray, width: int, max_height: int) -> tuple[np.ndarray, float, float]:
    h, w = image.shape[:2]
    scale = width / float(w)
    target_h = min(max_height, max(64, int(round(h * scale))))
    preview = cv2.resize(image, (width, target_h), interpolation=cv2.INTER_AREA if target_h < h else cv2.INTER_LINEAR)
    return preview.astype(np.float32), h / float(target_h), w / float(width)
