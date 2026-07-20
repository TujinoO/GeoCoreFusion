"""Candidate-independent RGB-to-NIR/SWIR identifiability audit.

This script deliberately stops at the aligned, harmonized low-resolution
observation cube.  It never reads a fused cube, material coefficients, gain,
additive detail, or fusion uncertainty.  RGB and every spectral band are put
through the same low-resolution log/DoG analysis passband, then evaluated by
3 x 3 spatial blocked cross-validation with a guard equal to the filter
support.  The guard prevents pixels contributing to a held-out filtered
sample from also appearing in its training set.

The output is an identifiability diagnostic, not HR-NIR/SWIR truth.  The
registration and scene-level PSF are frozen from the existing preprocessing
run and were estimated on the full ROI, so the result remains conditional on
that geometry.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np


REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from geocorefusion.envi import open_cube, parse_header  # noqa: E402


SCENES = {
    "3DSSZ": REPO / "runs" / "3dssz_roi_fusion_v7_final_ampfix",
    "ZKH3": REPO / "runs" / "zkh3_roi_fusion_v7_final_ampfix",
}

REPRESENTATIVE_CACHES = {
    "3DSSZ": (
        "3dssz_roi_fusion_v5_matched",
        "3dssz_roi_fusion_v6",
        "3dssz_roi_fusion_v7_final_ampfix",
        "3dssz_roi_fusion_v8_sweep_bridge_r2_s050",
    ),
    "ZKH3": (
        "zkh3_roi_fusion_v5_matched",
        "zkh3_roi_fusion_v6",
        "zkh3_roi_fusion_v7_final_ampfix",
        "zkh3_roi_fusion_v8_sweep_bridge_r2_s050",
    ),
}

SELECTED_WAVELENGTHS = (901.0, 1651.0, 2201.0, 2351.0)
MODEL_NAMES = ("luma_rank1", "rgb3_ridge")
SCOPE_NAMES = ("all_valid", "rgb_low_contrast")


@dataclass(slots=True)
class Fold:
    fold_id: int
    train: np.ndarray
    test: np.ndarray
    block: tuple[int, int, int, int]
    test_core: tuple[int, int, int, int]


@dataclass(slots=True)
class FitResult:
    prediction: np.ndarray
    baseline_prediction: np.ndarray
    fold_id: np.ndarray
    native_coefficients: np.ndarray
    train_counts: np.ndarray
    test_counts: np.ndarray


@dataclass(slots=True)
class MetricResult:
    pooled_rho: np.ndarray
    pooled_r2: np.ndarray
    fold_rho_median: np.ndarray
    fold_rho_q10: np.ndarray
    fold_rho_q90: np.ndarray
    fold_r2_median: np.ndarray
    fold_r2_q10: np.ndarray
    fold_r2_q90: np.ndarray
    fold_positive_r2_fraction: np.ndarray
    evaluated_pixels: int
    evaluated_folds: int


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _gaussian(image: np.ndarray, sigma: float) -> np.ndarray:
    radius = max(1, int(math.ceil(4.0 * float(sigma))))
    kernel = 2 * radius + 1
    return cv2.GaussianBlur(
        np.asarray(image, dtype=np.float32),
        (kernel, kernel),
        sigmaX=float(sigma),
        sigmaY=float(sigma),
        borderType=cv2.BORDER_REFLECT101,
    )


def _bandpass(image: np.ndarray, sigma_small: float, sigma_large: float) -> np.ndarray:
    arr = np.asarray(image, dtype=np.float32)
    if arr.ndim == 2:
        return (_gaussian(arr, sigma_small) - _gaussian(arr, sigma_large)).astype(
            np.float32
        )
    out = np.empty_like(arr, dtype=np.float32)
    # Batching avoids relying on OpenCV builds accepting hundreds of channels.
    for start in range(0, arr.shape[2], 24):
        stop = min(arr.shape[2], start + 24)
        block = np.ascontiguousarray(arr[:, :, start:stop])
        out[:, :, start:stop] = _gaussian(block, sigma_small) - _gaussian(
            block, sigma_large
        )
    return out


def _degrade_rgb(
    rgb: np.ndarray,
    low_shape: tuple[int, int],
    sigma_x_highres: float,
    sigma_y_highres: float,
) -> np.ndarray:
    arr = np.asarray(rgb[:, :, :3], dtype=np.float32) / 255.0
    if sigma_x_highres > 0.0 or sigma_y_highres > 0.0:
        arr = cv2.GaussianBlur(
            arr,
            (0, 0),
            sigmaX=max(float(sigma_x_highres), 1e-6),
            sigmaY=max(float(sigma_y_highres), 1e-6),
            borderType=cv2.BORDER_REFLECT101,
        )
    return cv2.resize(
        arr,
        (int(low_shape[1]), int(low_shape[0])),
        interpolation=cv2.INTER_AREA,
    ).astype(np.float32)


def _make_folds(
    shape: tuple[int, int],
    base_valid: np.ndarray,
    guard: int,
    rows: int = 3,
    cols: int = 3,
) -> list[Fold]:
    height, width = shape
    y_edges = np.linspace(0, height, rows + 1).round().astype(int)
    x_edges = np.linspace(0, width, cols + 1).round().astype(int)
    folds: list[Fold] = []
    fold_id = 0
    for row in range(rows):
        for col in range(cols):
            y0, y1 = int(y_edges[row]), int(y_edges[row + 1])
            x0, x1 = int(x_edges[col]), int(x_edges[col + 1])
            ty0, ty1 = y0 + guard, y1 - guard
            tx0, tx1 = x0 + guard, x1 - guard
            if ty1 <= ty0 or tx1 <= tx0:
                continue
            test = np.zeros(shape, dtype=bool)
            test[ty0:ty1, tx0:tx1] = True
            test &= base_valid

            # The full test block is dilated by the analysis-filter support.
            # No training filter window can therefore include a raw test pixel.
            dy0, dy1 = max(0, y0 - guard), min(height, y1 + guard)
            dx0, dx1 = max(0, x0 - guard), min(width, x1 + guard)
            train = base_valid.copy()
            train[dy0:dy1, dx0:dx1] = False
            if int(train.sum()) < 1024 or int(test.sum()) < 256:
                continue
            folds.append(
                Fold(
                    fold_id=fold_id,
                    train=train.reshape(-1),
                    test=test.reshape(-1),
                    block=(y0, y1, x0, x1),
                    test_core=(ty0, ty1, tx0, tx1),
                )
            )
            fold_id += 1
    return folds


def _fit_blocked_ridge(
    features: np.ndarray,
    targets: np.ndarray,
    folds: list[Fold],
    ridge: float,
    winsor_sigma: float,
) -> FitResult:
    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(targets, dtype=np.float64)
    prediction = np.full(y.shape, np.nan, dtype=np.float32)
    baseline_prediction = np.full(y.shape, np.nan, dtype=np.float32)
    fold_id_map = np.full(y.shape[0], -1, dtype=np.int16)
    mappings: list[np.ndarray] = []
    train_counts: list[int] = []
    test_counts: list[int] = []

    for fold in folds:
        train = fold.train & np.isfinite(x).all(axis=1) & np.isfinite(y).all(axis=1)
        test = fold.test & np.isfinite(x).all(axis=1) & np.isfinite(y).all(axis=1)
        x_train = x[train]
        y_train = y[train]
        x_mean = np.mean(x_train, axis=0)
        x_scale = np.maximum(np.std(x_train, axis=0), 1e-8)
        y_mean = np.mean(y_train, axis=0)
        y_scale = np.maximum(np.std(y_train, axis=0), 1e-8)

        xs = np.clip(
            (x_train - x_mean[None, :]) / x_scale[None, :],
            -winsor_sigma,
            winsor_sigma,
        )
        ys = np.clip(
            (y_train - y_mean[None, :]) / y_scale[None, :],
            -winsor_sigma,
            winsor_sigma,
        )
        normal = (xs.T @ xs) / float(xs.shape[0])
        normal += max(float(ridge), 1e-8) * np.eye(xs.shape[1], dtype=np.float64)
        rhs = (xs.T @ ys) / float(xs.shape[0])
        mapping_standardized = np.linalg.solve(normal, rhs)

        xt = np.clip(
            (x[test] - x_mean[None, :]) / x_scale[None, :],
            -winsor_sigma,
            winsor_sigma,
        )
        prediction[test] = (
            (xt @ mapping_standardized) * y_scale[None, :] + y_mean[None, :]
        ).astype(np.float32)
        baseline_prediction[test] = y_mean[None, :].astype(np.float32)
        fold_id_map[test] = int(fold.fold_id)
        mappings.append(
            (
                mapping_standardized
                * y_scale[None, :]
                / x_scale[:, None]
            ).astype(np.float64)
        )
        train_counts.append(int(train.sum()))
        test_counts.append(int(test.sum()))

    return FitResult(
        prediction=prediction,
        baseline_prediction=baseline_prediction,
        fold_id=fold_id_map,
        native_coefficients=np.stack(mappings, axis=0),
        train_counts=np.asarray(train_counts, dtype=np.int64),
        test_counts=np.asarray(test_counts, dtype=np.int64),
    )


def _column_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    baseline: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    pred = np.asarray(prediction, dtype=np.float64)
    truth = np.asarray(target, dtype=np.float64)
    pred_centered = pred - np.mean(pred, axis=0, keepdims=True)
    truth_centered = truth - np.mean(truth, axis=0, keepdims=True)
    numerator = np.sum(pred_centered * truth_centered, axis=0)
    denominator = np.sqrt(
        np.sum(pred_centered * pred_centered, axis=0)
        * np.sum(truth_centered * truth_centered, axis=0)
    )
    rho = np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator),
        where=denominator > 1e-12,
    )
    residual = pred - truth
    sse = np.sum(residual * residual, axis=0)
    reference = np.asarray(baseline, dtype=np.float64)
    baseline_error = truth - reference
    sst = np.sum(baseline_error * baseline_error, axis=0)
    r2 = 1.0 - np.divide(
        sse,
        sst,
        out=np.full_like(sse, np.inf),
        where=sst > 1e-12,
    )
    return rho, r2


def _evaluate_fit(
    fit: FitResult,
    target: np.ndarray,
    scope: np.ndarray,
) -> MetricResult:
    valid = (
        np.asarray(scope, dtype=bool).reshape(-1)
        & (fit.fold_id >= 0)
        & np.isfinite(fit.prediction).all(axis=1)
        & np.isfinite(target).all(axis=1)
    )
    pooled_rho, pooled_r2 = _column_metrics(
        fit.prediction[valid],
        target[valid],
        fit.baseline_prediction[valid],
    )
    fold_rho: list[np.ndarray] = []
    fold_r2: list[np.ndarray] = []
    for fold_value in sorted(int(value) for value in np.unique(fit.fold_id) if value >= 0):
        selected = valid & (fit.fold_id == fold_value)
        if int(selected.sum()) < 64:
            continue
        rho, r2 = _column_metrics(
            fit.prediction[selected],
            target[selected],
            fit.baseline_prediction[selected],
        )
        fold_rho.append(rho)
        fold_r2.append(r2)
    fold_rho_array = np.stack(fold_rho, axis=0)
    fold_r2_array = np.stack(fold_r2, axis=0)
    return MetricResult(
        pooled_rho=pooled_rho,
        pooled_r2=pooled_r2,
        fold_rho_median=np.median(fold_rho_array, axis=0),
        fold_rho_q10=np.quantile(fold_rho_array, 0.10, axis=0),
        fold_rho_q90=np.quantile(fold_rho_array, 0.90, axis=0),
        fold_r2_median=np.median(fold_r2_array, axis=0),
        fold_r2_q10=np.quantile(fold_r2_array, 0.10, axis=0),
        fold_r2_q90=np.quantile(fold_r2_array, 0.90, axis=0),
        fold_positive_r2_fraction=np.mean(fold_r2_array > 0.0, axis=0),
        evaluated_pixels=int(valid.sum()),
        evaluated_folds=int(fold_rho_array.shape[0]),
    )


def _mapping_statistics(
    coefficients: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mapping = np.asarray(coefficients, dtype=np.float64)
    norm = np.linalg.norm(mapping, axis=1)
    norm_median = np.median(norm, axis=0)
    if mapping.shape[1] == 1:
        slope = mapping[:, 0, :]
        positive = np.mean(slope > 0.0, axis=0)
        sign_stability = np.maximum(positive, 1.0 - positive)
        direction_stability = sign_stability.copy()
        return (
            np.median(slope, axis=0),
            np.quantile(slope, 0.10, axis=0),
            np.quantile(slope, 0.90, axis=0),
            direction_stability,
            norm_median,
        )

    unit = mapping / np.maximum(norm[:, None, :], 1e-12)
    similarities: list[np.ndarray] = []
    for first in range(unit.shape[0]):
        for second in range(first + 1, unit.shape[0]):
            similarities.append(np.sum(unit[first] * unit[second], axis=0))
    direction_stability = np.median(np.stack(similarities, axis=0), axis=0)
    nan = np.full(mapping.shape[2], np.nan, dtype=np.float64)
    return nan, nan.copy(), nan.copy(), direction_stability, norm_median


def _raw_feature_correlation(features: np.ndarray, target: np.ndarray, scope: np.ndarray) -> np.ndarray:
    mask = np.asarray(scope, dtype=bool).reshape(-1)
    x = np.asarray(features, dtype=np.float64)[mask]
    y = np.asarray(target, dtype=np.float64)[mask]
    values: list[np.ndarray] = []
    for channel in range(x.shape[1]):
        xx = x[:, channel : channel + 1]
        xx = xx - np.mean(xx, axis=0, keepdims=True)
        yy = y - np.mean(y, axis=0, keepdims=True)
        numerator = np.sum(xx * yy, axis=0)
        denominator = np.sqrt(np.sum(xx * xx) * np.sum(yy * yy, axis=0))
        values.append(
            np.divide(
                numerator,
                denominator,
                out=np.zeros_like(numerator),
                where=denominator > 1e-12,
            )
        )
    correlation = np.stack(values, axis=0)
    winner = np.argmax(np.abs(correlation), axis=0)
    return correlation[winner, np.arange(correlation.shape[1])]


def _classification(
    rho: float,
    r2: float,
    fold_rho: float,
    positive_r2_fraction: float,
    stability: float,
    delta_r2_null: float,
) -> str:
    if (
        rho >= 0.50
        and r2 >= 0.20
        and fold_rho >= 0.35
        and positive_r2_fraction >= 0.67
        and stability >= 0.70
        and delta_r2_null >= 0.05
    ):
        return "identifiable"
    if (
        rho >= 0.35
        and r2 >= 0.05
        and fold_rho >= 0.20
        and positive_r2_fraction >= 0.56
        and stability >= 0.45
        and delta_r2_null >= 0.01
    ):
        return "weakly_identifiable"
    return "unidentifiable"


def _load_band_metadata(path: Path, band_count: int) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != band_count:
        raise ValueError(f"Band metadata length mismatch: {len(rows)} != {band_count}")
    return rows


def _cache_audit(scene: str, canonical: Path) -> dict[str, Any]:
    paths: list[Path] = []
    for name in REPRESENTATIVE_CACHES[scene]:
        candidate = REPO / "runs" / name / "analysis" / "harmonized_lowres.dat"
        if candidate.exists():
            paths.append(candidate)
    hashes = {str(path.relative_to(REPO)): _sha256(path) for path in paths}
    canonical_hash = _sha256(canonical)
    return {
        "canonical_sha256": canonical_hash,
        "representative_count": len(paths),
        "matching_count": sum(value == canonical_hash for value in hashes.values()),
        "representative_hashes": hashes,
    }


def _null_shifts(shape: tuple[int, int]) -> tuple[tuple[int, int], ...]:
    height, width = shape
    return (
        (height // 2, 0),
        (0, width // 2),
        (height // 3, width // 3),
        (height // 2, width // 2),
    )


def _multi_otsu_three_thresholds(image: np.ndarray) -> tuple[float, float]:
    """Return deterministic three-class Otsu thresholds on a robust range."""

    values = np.asarray(image, dtype=np.float64)
    finite = values[np.isfinite(values)]
    lower, upper = np.quantile(finite, (0.005, 0.995))
    if upper <= lower:
        return float(lower), float(upper)
    quantized = np.clip(
        np.round((values - lower) / (upper - lower) * 255.0), 0, 255
    ).astype(np.uint8)
    histogram = np.bincount(quantized.reshape(-1), minlength=256).astype(np.float64)
    probability = histogram / max(float(np.sum(histogram)), 1.0)
    cumulative_weight = np.cumsum(probability)
    cumulative_mean = np.cumsum(probability * np.arange(256, dtype=np.float64))
    total_mean = float(cumulative_mean[-1])
    best_score = -np.inf
    best = (1, 254)
    for first in range(1, 254):
        for second in range(first + 1, 255):
            weights = (
                float(cumulative_weight[first]),
                float(cumulative_weight[second] - cumulative_weight[first]),
                float(1.0 - cumulative_weight[second]),
            )
            if min(weights) <= 1e-10:
                continue
            means = (
                float(cumulative_mean[first] / weights[0]),
                float(
                    (cumulative_mean[second] - cumulative_mean[first])
                    / weights[1]
                ),
                float((total_mean - cumulative_mean[second]) / weights[2]),
            )
            score = sum(
                weight * (mean - total_mean) ** 2
                for weight, mean in zip(weights, means)
            )
            if score > best_score:
                best_score = score
                best = (first, second)
    scale = (upper - lower) / 255.0
    return float(lower + best[0] * scale), float(lower + best[1] * scale)


def _rock_proxy_masks(low_rgb: np.ndarray) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Build a transparent RGB-chromatic rock-interior sensitivity proxy.

    This is deliberately labelled a proxy rather than an independent mask.
    In 3DSSZ the rocks are green/neutral while the tray is orange and the
    background is nearly neutral-dark.  The upper class of a three-level Otsu
    split on green excess isolates the visible rock columns without consulting
    any spectral target band or fused candidate.  Opening and one-pixel erosion
    keep interiors and reduce tray/rock boundary leakage.
    """

    rgb = np.asarray(low_rgb, dtype=np.float32)
    score = rgb[:, :, 1] - 0.5 * (rgb[:, :, 0] + rgb[:, :, 2])
    first, second = _multi_otsu_three_thresholds(score)
    perturbation = 0.15 * max(second - first, 1e-6)
    thresholds = {
        "loose": second - perturbation,
        "nominal": second,
        "strict": second + perturbation,
    }
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    masks: dict[str, np.ndarray] = {}
    details: dict[str, Any] = {
        "score": "G - 0.5*(R+B) on PSF-degraded LR RGB",
        "three_class_otsu_threshold_low": first,
        "three_class_otsu_threshold_high": second,
        "threshold_perturbation": perturbation,
        "morphology": "3x3 elliptical opening followed by 3x3 erosion",
        "variants": {},
    }
    for name, threshold in thresholds.items():
        raw = (score > threshold).astype(np.uint8)
        opened = cv2.morphologyEx(raw, cv2.MORPH_OPEN, kernel)
        mask = cv2.erode(opened, kernel).astype(bool)
        component_count, _, stats, _ = cv2.connectedComponentsWithStats(
            mask.astype(np.uint8), 8
        )
        kept_components = int(
            sum(int(stats[index, cv2.CC_STAT_AREA]) >= 10 for index in range(1, component_count))
        )
        masks[name] = mask
        details["variants"][name] = {
            "threshold": float(threshold),
            "pixel_count_before_filter_border": int(mask.sum()),
            "area_fraction_before_filter_border": float(np.mean(mask)),
            "connected_components_area_ge_10": kept_components,
        }
    return masks, details


def _process_scene(
    scene: str,
    run: Path,
    sigma_small: float,
    sigma_large: float,
    ridge: float,
    winsor_sigma: float,
    run_null: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    processing = json.loads(
        (run / "metadata" / "processing_config.json").read_text(encoding="utf-8")
    )
    input_metadata = json.loads(
        (run / "metadata" / "input_metadata.json").read_text(encoding="utf-8")
    )
    psf = json.loads((run / "metadata" / "psf_model.json").read_text(encoding="utf-8"))
    low_header = run / "analysis" / "harmonized_lowres.hdr"
    low_cube_map, low_meta = open_cube(low_header)
    low_cube = np.asarray(low_cube_map, dtype=np.float32)
    low_shape = low_cube.shape[:2]

    rgb_header = Path(input_metadata["rgb"]["hdr_path"])
    rgb_map, _ = open_cube(rgb_header)
    roi = processing["roi"]
    rgb = np.asarray(
        rgb_map[
            int(roi["y"]) : int(roi["y"] + roi["height"]),
            int(roi["x"]) : int(roi["x"] + roi["width"]),
            :3,
        ],
        dtype=np.uint8,
    )
    if tuple(rgb.shape[:2]) != tuple(int(value) for value in psf["high_shape"]):
        raise ValueError(f"{scene}: raw RGB ROI and PSF high_shape do not agree")

    low_rgb = _degrade_rgb(
        rgb,
        low_shape,
        float(psf["sigma_x_highres"]),
        float(psf["sigma_y_highres"]),
    )
    rgb_epsilon = 1.0 / 255.0
    rgb_log = np.log(np.maximum(low_rgb, rgb_epsilon)).astype(np.float32)
    rgb_bandpass = _bandpass(rgb_log, sigma_small, sigma_large)
    luminance = (
        0.299 * low_rgb[:, :, 0]
        + 0.587 * low_rgb[:, :, 1]
        + 0.114 * low_rgb[:, :, 2]
    )
    luma_bandpass = _bandpass(
        np.log(np.maximum(luminance, rgb_epsilon)).astype(np.float32),
        sigma_small,
        sigma_large,
    )

    # All current harmonized observations are strictly positive.  The tiny
    # floor prevents log failure but is far below every value in both scenes;
    # it is not presented as a dark-frame noise estimate.
    spectral_epsilon = 1e-4
    spectral_log = np.log(np.maximum(low_cube, spectral_epsilon)).astype(np.float32)
    target_bandpass = _bandpass(spectral_log, sigma_small, sigma_large)

    guard = int(math.ceil(4.0 * sigma_large))
    base_valid = (
        np.isfinite(luma_bandpass)
        & np.isfinite(rgb_bandpass).all(axis=2)
        & np.isfinite(target_bandpass).all(axis=2)
    )
    base_valid[:guard, :] = False
    base_valid[-guard:, :] = False
    base_valid[:, :guard] = False
    base_valid[:, -guard:] = False
    folds = _make_folds(low_shape, base_valid, guard)
    if len(folds) != 9:
        raise RuntimeError(f"{scene}: expected 9 valid folds, received {len(folds)}")

    valid_abs = np.abs(luma_bandpass[base_valid])
    low_q20, low_q80 = np.quantile(valid_abs, (0.20, 0.80))
    low_contrast = (
        base_valid
        & (np.abs(luma_bandpass) >= low_q20)
        & (np.abs(luma_bandpass) <= low_q80)
    )
    scopes = {"all_valid": base_valid, "rgb_low_contrast": low_contrast}

    y = target_bandpass.reshape(-1, target_bandpass.shape[2])
    feature_maps = {
        "luma_rank1": luma_bandpass[:, :, None],
        "rgb3_ridge": rgb_bandpass,
    }
    band_metadata = _load_band_metadata(
        run / "metadata" / "band_metadata.csv", target_bandpass.shape[2]
    )
    rows: list[dict[str, Any]] = []

    fit_cache: dict[str, FitResult] = {}
    metric_cache: dict[tuple[str, str], MetricResult] = {}
    raw_rho_cache: dict[tuple[str, str], np.ndarray] = {}
    mapping_cache: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
    null_cache: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]] = {}

    for model_name, feature_map in feature_maps.items():
        features = feature_map.reshape(-1, feature_map.shape[2])
        fit = _fit_blocked_ridge(features, y, folds, ridge, winsor_sigma)
        fit_cache[model_name] = fit
        mapping_cache[model_name] = _mapping_statistics(fit.native_coefficients)
        for scope_name, scope in scopes.items():
            metric_cache[(model_name, scope_name)] = _evaluate_fit(fit, y, scope)
            raw_rho_cache[(model_name, scope_name)] = _raw_feature_correlation(
                features, y, scope
            )

        null_r2 = {
            scope_name: np.full(y.shape[1], -np.inf, dtype=np.float64)
            for scope_name in scopes
        }
        null_rho = {
            scope_name: np.zeros(y.shape[1], dtype=np.float64)
            for scope_name in scopes
        }
        if run_null:
            for shift_y, shift_x in _null_shifts(low_shape):
                shifted = np.roll(feature_map, (shift_y, shift_x), axis=(0, 1))
                null_fit = _fit_blocked_ridge(
                    shifted.reshape(-1, shifted.shape[2]),
                    y,
                    folds,
                    ridge,
                    winsor_sigma,
                )
                for scope_name, scope in scopes.items():
                    null_metric = _evaluate_fit(null_fit, y, scope)
                    better = null_metric.pooled_r2 > null_r2[scope_name]
                    null_r2[scope_name] = np.maximum(
                        null_r2[scope_name], null_metric.pooled_r2
                    )
                    null_rho[scope_name][better] = null_metric.pooled_rho[better]
        else:
            for scope_name in scopes:
                null_r2[scope_name].fill(np.nan)
                null_rho[scope_name].fill(np.nan)
        for scope_name in scopes:
            null_cache[(model_name, scope_name)] = (
                null_rho[scope_name],
                null_r2[scope_name],
            )

    for model_name in MODEL_NAMES:
        fit = fit_cache[model_name]
        alpha_median, alpha_q10, alpha_q90, stability, coefficient_norm = mapping_cache[
            model_name
        ]
        for scope_name in SCOPE_NAMES:
            metrics = metric_cache[(model_name, scope_name)]
            raw_rho = raw_rho_cache[(model_name, scope_name)]
            null_rho, null_r2 = null_cache[(model_name, scope_name)]
            for band_index, metadata in enumerate(band_metadata):
                delta_null = float(metrics.pooled_r2[band_index] - null_r2[band_index])
                status = _classification(
                    float(metrics.pooled_rho[band_index]),
                    float(metrics.pooled_r2[band_index]),
                    float(metrics.fold_rho_median[band_index]),
                    float(metrics.fold_positive_r2_fraction[band_index]),
                    float(stability[band_index]),
                    delta_null,
                )
                rows.append(
                    {
                        "scene": scene,
                        "model": model_name,
                        "scope": scope_name,
                        "band_index": band_index,
                        "wavelength_nm": float(low_meta.wavelengths[band_index]),
                        "nir_weight": float(metadata["nir_weight"]),
                        "swir_weight": float(metadata["swir_weight"]),
                        "band_low_confidence": metadata["is_low_confidence"],
                        "band_low_confidence_reason": metadata["reason"],
                        "low_height": int(low_shape[0]),
                        "low_width": int(low_shape[1]),
                        "psf_method": str(psf["method"]),
                        "psf_sigma_x_highres": float(psf["sigma_x_highres"]),
                        "psf_sigma_y_highres": float(psf["sigma_y_highres"]),
                        "bandpass_sigma_small_lowres": sigma_small,
                        "bandpass_sigma_large_lowres": sigma_large,
                        "filter_guard_lowres_px": guard,
                        "fold_count": metrics.evaluated_folds,
                        "evaluated_pixels": metrics.evaluated_pixels,
                        "minimum_train_pixels": int(np.min(fit.train_counts)),
                        "minimum_test_pixels": int(np.min(fit.test_counts)),
                        "raw_best_feature_rho": float(raw_rho[band_index]),
                        "cv_pooled_rho": float(metrics.pooled_rho[band_index]),
                        "cv_pooled_r2": float(metrics.pooled_r2[band_index]),
                        "cv_fold_rho_median": float(metrics.fold_rho_median[band_index]),
                        "cv_fold_rho_q10": float(metrics.fold_rho_q10[band_index]),
                        "cv_fold_rho_q90": float(metrics.fold_rho_q90[band_index]),
                        "cv_fold_r2_median": float(metrics.fold_r2_median[band_index]),
                        "cv_fold_r2_q10": float(metrics.fold_r2_q10[band_index]),
                        "cv_fold_r2_q90": float(metrics.fold_r2_q90[band_index]),
                        "cv_fold_positive_r2_fraction": float(
                            metrics.fold_positive_r2_fraction[band_index]
                        ),
                        "luma_alpha_median": float(alpha_median[band_index]),
                        "luma_alpha_q10": float(alpha_q10[band_index]),
                        "luma_alpha_q90": float(alpha_q90[band_index]),
                        "mapping_direction_stability": float(stability[band_index]),
                        "mapping_coefficient_norm_median": float(
                            coefficient_norm[band_index]
                        ),
                        "shift_null_rho_at_max_r2": float(null_rho[band_index]),
                        "shift_null_r2_max": float(null_r2[band_index]),
                        "delta_r2_vs_shift_null": delta_null,
                        "identifiability": status,
                    }
                )

    raw_headers = (Path(input_metadata["nir"]["hdr_path"]), Path(input_metadata["swir"]["hdr_path"]))
    extra_info = " ".join(
        str(parse_header(header).raw.get("extra info", "")) for header in raw_headers
    ).lower()
    cache_audit = _cache_audit(scene, low_meta.data_path)
    audit = {
        "scene": scene,
        "run": str(run.relative_to(REPO)),
        "rgb_header": str(rgb_header),
        "rgb_roi": {key: int(roi[key]) for key in ("x", "y", "width", "height")},
        "rgb_shape": list(rgb.shape),
        "low_header": str(low_header.relative_to(REPO)),
        "low_shape": list(low_cube.shape),
        "low_min": float(np.min(low_cube)),
        "low_max": float(np.max(low_cube)),
        "low_finite_fraction": float(np.mean(np.isfinite(low_cube))),
        "psf": psf,
        "filter_guard_lowres_px": guard,
        "fold_count": len(folds),
        "all_valid_pixels": int(base_valid.sum()),
        "low_contrast_pixels": int(low_contrast.sum()),
        "low_contrast_abs_luma_bandpass_q20": float(low_q20),
        "low_contrast_abs_luma_bandpass_q80": float(low_q80),
        "configured_independent_material_mask": False,
        "configured_dark_flat_frames": False,
        "raw_headers_reference_external_calibration_files": (
            "dark file" in extra_info or "reflect file" in extra_info
        ),
        "candidate_independence": cache_audit,
    }
    return rows, audit


def _process_rock_proxy_sensitivity(
    run: Path,
    sigma_small: float,
    sigma_large: float,
    ridge: float,
    winsor_sigma: float,
    run_null: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Refit 3DSSZ relations inside three frozen rock-proxy variants."""

    processing = json.loads(
        (run / "metadata" / "processing_config.json").read_text(encoding="utf-8")
    )
    input_metadata = json.loads(
        (run / "metadata" / "input_metadata.json").read_text(encoding="utf-8")
    )
    psf_payload = json.loads(
        (run / "metadata" / "psf_model.json").read_text(encoding="utf-8")
    )
    low_cube_map, low_meta = open_cube(run / "analysis" / "harmonized_lowres.hdr")
    selected_indices = np.asarray(
        [
            int(np.argmin(np.abs(low_meta.wavelengths - wavelength)))
            for wavelength in (901.0, 1651.0, 2201.0)
        ],
        dtype=np.int64,
    )
    low_selected = np.asarray(low_cube_map[:, :, selected_indices], dtype=np.float32)
    low_shape = low_selected.shape[:2]

    rgb_map, _ = open_cube(Path(input_metadata["rgb"]["hdr_path"]))
    roi = processing["roi"]
    rgb = np.asarray(
        rgb_map[
            int(roi["y"]) : int(roi["y"] + roi["height"]),
            int(roi["x"]) : int(roi["x"] + roi["width"]),
            :3,
        ],
        dtype=np.uint8,
    )
    low_rgb = _degrade_rgb(
        rgb,
        low_shape,
        float(psf_payload["sigma_x_highres"]),
        float(psf_payload["sigma_y_highres"]),
    )
    rock_masks, proxy_details = _rock_proxy_masks(low_rgb)
    rgb_epsilon = 1.0 / 255.0
    rgb_bandpass = _bandpass(
        np.log(np.maximum(low_rgb, rgb_epsilon)).astype(np.float32),
        sigma_small,
        sigma_large,
    )
    luminance = (
        0.299 * low_rgb[:, :, 0]
        + 0.587 * low_rgb[:, :, 1]
        + 0.114 * low_rgb[:, :, 2]
    )
    luma_bandpass = _bandpass(
        np.log(np.maximum(luminance, rgb_epsilon)).astype(np.float32),
        sigma_small,
        sigma_large,
    )
    target_bandpass = _bandpass(
        np.log(np.maximum(low_selected, 1e-4)).astype(np.float32),
        sigma_small,
        sigma_large,
    )
    guard = int(math.ceil(4.0 * sigma_large))
    finite = (
        np.isfinite(luma_bandpass)
        & np.isfinite(rgb_bandpass).all(axis=2)
        & np.isfinite(target_bandpass).all(axis=2)
    )
    finite[:guard, :] = False
    finite[-guard:, :] = False
    finite[:, :guard] = False
    finite[:, -guard:] = False
    y = target_bandpass.reshape(-1, target_bandpass.shape[2])
    feature_maps = {
        "luma_rank1": luma_bandpass[:, :, None],
        "rgb3_ridge": rgb_bandpass,
    }
    rows: list[dict[str, Any]] = []
    for variant, raw_mask in rock_masks.items():
        support = finite & raw_mask
        folds = _make_folds(low_shape, support, guard)
        if len(folds) < 4:
            raise RuntimeError(
                f"3DSSZ rock proxy {variant!r} has only {len(folds)} usable folds"
            )
        for model_name, feature_map in feature_maps.items():
            features = feature_map.reshape(-1, feature_map.shape[2])
            fit = _fit_blocked_ridge(features, y, folds, ridge, winsor_sigma)
            metrics = _evaluate_fit(fit, y, support)
            raw_rho = _raw_feature_correlation(features, y, support)
            alpha_median, alpha_q10, alpha_q90, stability, coefficient_norm = (
                _mapping_statistics(fit.native_coefficients)
            )
            null_r2 = np.full(y.shape[1], -np.inf, dtype=np.float64)
            null_rho = np.zeros(y.shape[1], dtype=np.float64)
            if run_null:
                for shift_y, shift_x in _null_shifts(low_shape):
                    shifted = np.roll(feature_map, (shift_y, shift_x), axis=(0, 1))
                    null_fit = _fit_blocked_ridge(
                        shifted.reshape(-1, shifted.shape[2]),
                        y,
                        folds,
                        ridge,
                        winsor_sigma,
                    )
                    null_metric = _evaluate_fit(null_fit, y, support)
                    better = null_metric.pooled_r2 > null_r2
                    null_r2 = np.maximum(null_r2, null_metric.pooled_r2)
                    null_rho[better] = null_metric.pooled_rho[better]
            else:
                null_r2.fill(np.nan)
                null_rho.fill(np.nan)
            detail = proxy_details["variants"][variant]
            for position, band_index in enumerate(selected_indices):
                delta_null = float(metrics.pooled_r2[position] - null_r2[position])
                status = _classification(
                    float(metrics.pooled_rho[position]),
                    float(metrics.pooled_r2[position]),
                    float(metrics.fold_rho_median[position]),
                    float(metrics.fold_positive_r2_fraction[position]),
                    float(stability[position]),
                    delta_null,
                )
                rows.append(
                    {
                        "scene": "3DSSZ",
                        "proxy_variant": variant,
                        "model": model_name,
                        "band_index": int(band_index),
                        "wavelength_nm": float(low_meta.wavelengths[band_index]),
                        "proxy_threshold": float(detail["threshold"]),
                        "proxy_area_fraction_before_filter_border": float(
                            detail["area_fraction_before_filter_border"]
                        ),
                        "proxy_components_area_ge_10": int(
                            detail["connected_components_area_ge_10"]
                        ),
                        "support_pixels_after_filter_border": int(support.sum()),
                        "fold_count": int(metrics.evaluated_folds),
                        "minimum_train_pixels": int(np.min(fit.train_counts)),
                        "minimum_test_pixels": int(np.min(fit.test_counts)),
                        "raw_best_feature_rho": float(raw_rho[position]),
                        "cv_pooled_rho": float(metrics.pooled_rho[position]),
                        "cv_predictive_r2": float(metrics.pooled_r2[position]),
                        "cv_fold_rho_median": float(metrics.fold_rho_median[position]),
                        "cv_fold_r2_median": float(metrics.fold_r2_median[position]),
                        "cv_fold_positive_r2_fraction": float(
                            metrics.fold_positive_r2_fraction[position]
                        ),
                        "luma_alpha_median": float(alpha_median[position]),
                        "luma_alpha_q10": float(alpha_q10[position]),
                        "luma_alpha_q90": float(alpha_q90[position]),
                        "mapping_direction_stability": float(stability[position]),
                        "mapping_coefficient_norm_median": float(
                            coefficient_norm[position]
                        ),
                        "shift_null_rho_at_max_r2": float(null_rho[position]),
                        "shift_null_r2_max": float(null_r2[position]),
                        "delta_r2_vs_shift_null": delta_null,
                        "identifiability": status,
                    }
                )
    audit = {
        **proxy_details,
        "candidate_independent": True,
        "independent_rock_truth_mask": False,
        "interpretation": (
            "RGB-chromatic rock-interior proxy for sensitivity only; not an "
            "independent geological segmentation or HR spectral truth"
        ),
        "filter_guard_lowres_px": guard,
    }
    return rows, audit


def _format_value(value: Any, digits: int = 3) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not np.isfinite(number):
        return "NA"
    return f"{number:.{digits}f}"


def _write_markdown(
    output: Path,
    rows: list[dict[str, Any]],
    audits: list[dict[str, Any]],
    rock_rows: list[dict[str, Any]],
    rock_audit: dict[str, Any],
    sigma_small: float,
    sigma_large: float,
    ridge: float,
    run_null: bool,
) -> None:
    selected = [
        row
        for row in rows
        if row["model"] == "rgb3_ridge"
        and any(abs(float(row["wavelength_nm"]) - wave) < 1e-6 for wave in SELECTED_WAVELENGTHS)
    ]
    low_selected = [row for row in selected if row["scope"] == "rgb_low_contrast"]
    counts: dict[tuple[str, str, str], dict[str, int]] = {}
    for row in rows:
        key = (str(row["scene"]), str(row["model"]), str(row["scope"]))
        counts.setdefault(
            key,
            {"identifiable": 0, "weakly_identifiable": 0, "unidentifiable": 0},
        )[str(row["identifiability"])] += 1

    selected_supports_transfer = any(
        row["identifiability"] in {"identifiable", "weakly_identifiable"}
        for row in low_selected
    )
    if selected_supports_transfer:
        lead = (
            "严格留块结果只在部分代表波段显示可迁移关系；RGB 细节必须按波段和区域门控，"
            "不能用一个公共增益写入全部 NIR/SWIR 波段。"
        )
    else:
        lead = (
            "在当前真实 ROI 的低对比 RGB 子集上，代表波段均未形成稳定、超越空间错位零模型的"
            "可预测关系；现有数据不支持继续增强无条件 RGB 细节迁移。"
        )

    lines = [
        "# V8 候选无关的 RGB–NIR/SWIR 可辨识性实验",
        "",
        "> 实验日期：2026-07-20  ",
        "> 结论性质：真实观测上的条件可辨识性诊断；不是 HR-NIR/HR-SWIR 真值验证。",
        "",
        "## 结论先行",
        "",
        lead,
        "",
        "本实验没有读取任何融合立方体、PCA/simplex 系数、空间增益、additive detail 或融合不确定度。"
        "输入只包括原始 RGB ROI 与配准、光谱谐调后但尚未融合的低分辨率观测。",
        "",
        "## 数据与泄漏审计",
        "",
        "| 场景 | 原始 RGB ROI | LR 光谱立方体 | PSF（HR px） | 候选缓存一致性 | 独立岩性 mask | dark/flat |",
        "|---|---:|---:|---|---:|---|---|",
    ]
    for audit in audits:
        psf = audit["psf"]
        cache = audit["candidate_independence"]
        lines.append(
            "| {scene} | {rgb} | {low} | {sx:.3f}, {sy:.3f} ({method}) | {match}/{count} SHA-256 相同 | 无 | 未随数据提供 |".format(
                scene=audit["scene"],
                rgb="×".join(str(value) for value in audit["rgb_shape"][:2]),
                low="×".join(str(value) for value in audit["low_shape"]),
                sx=float(psf["sigma_x_highres"]),
                sy=float(psf["sigma_y_highres"]),
                method=psf["method"],
                match=cache["matching_count"],
                count=cache["representative_count"],
            )
        )
    lines.extend(["", "复现所用的精确入口：", ""])
    for audit in audits:
        cache = audit["candidate_independence"]
        lines.append(
            f"- **{audit['scene']}**：RGB=`{audit['rgb_header']}`；"
            f"LR=`{audit['low_header']}`；PSF=`{audit['run']}/metadata/psf_model.json`；"
            f"LR SHA-256=`{cache['canonical_sha256']}`。"
        )
    lines.extend(
        [
            "",
            "本实验没有读取上采样 NIR/SWIR、预览 PNG 或任何高分辨率融合候选；"
            "光谱 target 始终保留在 293×160 / 297×157 的实际观测网格。",
            "- `harmonized_lowres.dat` 在 V5-matched、V6、V7 与代表性 V8 bridge 运行中逐字节一致；"
            "因此本实验不会因所评价的融合候选而改变。早期 V3/V4 使用过不同配准版本，本次没有混用。",
            "- 空间配准、scene-level PSF 和光谱谐调曾在完整 ROI 上估计，因此 CV 只隔离 RGB→波段关系的拟合，"
            "不能视为独立的配准盲测。尤其 3DSSZ 的最优 PSF 为 0 px，说明当前数据没有可靠的 band-wise MTF 标定。",
            "- 工程目录没有配置到共同 RGB 网格的独立岩性/岩心 mask。3DSSZ 原始目录虽有 ENVI 历史处理产物，"
            "其几何与 ZKH3 不对称且未接入当前配准链，故没有把它当真值 mask。",
            "- 原始 NIR/SWIR 头文件引用采集机上的 dark/reflect 标定路径，但对应标定帧不在项目数据中；"
            "本次 `log` 下限仅是数值保护，不是噪声标定。",
            "",
            "## 协议",
            "",
            f"1. 用冻结 PSF 将原始 RGB 面阵退化到各场景 LR 网格；对三通道 RGB 与每个光谱波段使用相同的 "
            f"log-DoG 通带（σ={sigma_small:.1f}/{sigma_large:.1f} LR px）。",
            f"2. 采用 3×3 留一空间块 CV；测试块向内腐蚀、训练区向外排除 {int(math.ceil(4*sigma_large))} LR px，"
            "等于大尺度高斯的 4σ 支撑。这样同一原始像元不会同时进入训练和测试的滤波窗口。",
            f"3. 每个 fold 内独立中心化/标准化，使用 winsorized ridge（λ={ridge:.3f}）。"
            "比较亮度单因子与三通道 RGB ridge。",
            "4. `rgb_low_contrast` 仅在 RGB 亮度 band-pass 绝对幅值的 20%–80% 区间评价；模型仍只在训练块拟合。"
            "它用于降低大托盘边缘主导结论的风险，不是岩性 mask。",
            (
                "5. 用 4 个大幅环移 RGB 作为错位零模型，并要求真实关系的 held-out R² 明显超过最强零模型。"
                if run_null
                else "5. 本次通过 `--no-null` 跳过错位零模型；状态只能作调试输出。"
            ),
            "",
            "状态阈值预先固定为：`identifiable` 要求 pooled ρ≥0.50、R²≥0.20、fold 中位 ρ≥0.35、"
            "至少 67% folds 的 R²>0、映射稳定度≥0.70，且 R² 比最强 shift-null 高≥0.05；"
            "`weakly_identifiable` 使用 ρ≥0.35、R²≥0.05、fold 中位 ρ≥0.20、至少 56% 正 R²、"
            "稳定度≥0.45、null 增量≥0.01。其余为 `unidentifiable`。阈值不是从融合结果反向调参。",
            "",
            "## 代表波段结果：三通道 RGB ridge",
            "",
            "| 场景 | 波长 nm | 区域 | pooled ρ | pooled R² | fold 中位 ρ | 正 R² folds | mapping 稳定度 | shift-null 最大 R² | ΔR² | 状态 |",
            "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for row in selected:
        lines.append(
            "| {scene} | {wave:.0f} | {scope} | {rho} | {r2} | {frho} | {positive} | {stability} | {null} | {delta} | {status} |".format(
                scene=row["scene"],
                wave=float(row["wavelength_nm"]),
                scope=row["scope"],
                rho=_format_value(row["cv_pooled_rho"]),
                r2=_format_value(row["cv_pooled_r2"]),
                frho=_format_value(row["cv_fold_rho_median"]),
                positive=_format_value(row["cv_fold_positive_r2_fraction"]),
                stability=_format_value(row["mapping_direction_stability"]),
                null=_format_value(row["shift_null_r2_max"]),
                delta=_format_value(row["delta_r2_vs_shift_null"]),
                status=row["identifiability"],
            )
        )
    rgb_rock_rows = [row for row in rock_rows if row["model"] == "rgb3_ridge"]
    rock_all_unidentifiable = all(
        row["identifiability"] == "unidentifiable" for row in rgb_rock_rows
    )
    lines.extend(
        [
            "",
            "## 3DSSZ rock-like interior proxy 敏感性",
            "",
            "项目没有独立岩心 mask，因此这里没有把自动分割包装成 `rock-only` 真值。"
            "敏感性代理完全由退化到 LR 网格的原始 RGB 构造："
            f"`{rock_audit['score']}`，做三类 Otsu 后取最高绿色过量类，再做 "
            "3×3 椭圆 opening 与一次 3×3 erosion。阈值同时取 nominal 及相邻 Otsu 类间距的 ±15%；"
            "每个变体都重新拟合 blocked-CV，而不是只在全图模型上挑像元。",
            "",
            (
                "三个阈值下 901/1651/2201 nm 均保持 `unidentifiable`；在该代理显式压低黑背景与"
                "橙色托盘主色之后，3DSSZ 的关闭结论仍不改变。这个结果仍只是代理敏感性，"
                "不替代人工/独立岩心 mask。"
                if rock_all_unidentifiable
                else "至少一个阈值/波段改变了状态，因此自动代理不够稳定，不能把它作为 rock-only 论文证据。"
            ),
            "",
            "| proxy | 面积占比 | folds | 波长 nm | pooled ρ | predictive R² | fold 中位 ρ | shift-null 最大 R² | ΔR² | 状态 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for row in rgb_rock_rows:
        lines.append(
            "| {proxy} | {area} | {folds} | {wave:.0f} | {rho} | {r2} | {frho} | {null} | {delta} | {status} |".format(
                proxy=row["proxy_variant"],
                area=_format_value(row["proxy_area_fraction_before_filter_border"]),
                folds=row["fold_count"],
                wave=float(row["wavelength_nm"]),
                rho=_format_value(row["cv_pooled_rho"]),
                r2=_format_value(row["cv_predictive_r2"]),
                frho=_format_value(row["cv_fold_rho_median"]),
                null=_format_value(row["shift_null_r2_max"]),
                delta=_format_value(row["delta_r2_vs_shift_null"]),
                status=row["identifiability"],
            )
        )
    lines.extend(
        [
            "",
            "## 全谱状态计数（相邻 5 nm 行不是独立样本）",
            "",
            "| 场景 | 模型 | 区域 | identifiable | weak | unidentifiable |",
            "|---|---|---|---:|---:|---:|",
        ]
    )
    for key in sorted(counts):
        count = counts[key]
        lines.append(
            f"| {key[0]} | {key[1]} | {key[2]} | {count['identifiable']} | "
            f"{count['weakly_identifiable']} | {count['unidentifiable']} |"
        )
    lines.extend(
        [
            "",
            "计数只用于查看波长连续区间，不能把 5 nm 插值网格的相邻波段当作数百次独立重复。"
            "各输出波段仍由原生 NIR/SWIR 光谱插值得到；空间像元也存在自相关。",
            "",
            "## 对 RGB 细节迁移的含义",
            "",
            "- `identifiable` 只表示在当前 LR 可观测通带、冻结配准条件下，RGB 能跨空间块预测该波段的部分结构；"
            "它最多支持冻结一个 band-specific 载荷和置信度，不证明 LR Nyquist 以上的 RGB 高频在 SWIR 中真实存在。",
            "- `weakly_identifiable` 只允许低强度、低秩、可关闭的共享因子，并必须继续通过 warp/shuffle/forward-cycle。",
            "- `unidentifiable` 波段的 scientific fusion 应将 RGB 注入置零；若为展示复制 RGB 纹理，必须另存为"
            " `RGB-textured visualization`，不得进入定量光谱产品。",
            "- 当前最关键的新数据不是继续扫 strength，而是共同 RGB 网格的岩心 mask、dark/flat、band-wise MTF/PSF、"
            "配准 covariance，以及至少一种独立 HR-NIR/SWIR 或严格 Wald 真值。",
            "",
            "完整逐波段数值见 `v8_identifiability.csv`；3DSSZ 代理敏感性见 "
            "`v8_identifiability_rock_proxy_sensitivity.csv`。",
        ]
    )
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sigma-small", type=float, default=0.8)
    parser.add_argument("--sigma-large", type=float, default=2.4)
    parser.add_argument("--ridge", type=float, default=0.03)
    parser.add_argument("--winsor-sigma", type=float, default=4.0)
    parser.add_argument(
        "--no-null",
        action="store_true",
        help="Skip four spatial-shift null fits (debug only).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO / "artifacts" / "v7_research" / "evidence",
    )
    args = parser.parse_args()
    if not 0.0 < args.sigma_small < args.sigma_large:
        raise ValueError("Require 0 < sigma-small < sigma-large")

    all_rows: list[dict[str, Any]] = []
    audits: list[dict[str, Any]] = []
    for scene, run in SCENES.items():
        rows, audit = _process_scene(
            scene,
            run,
            args.sigma_small,
            args.sigma_large,
            args.ridge,
            args.winsor_sigma,
            not args.no_null,
        )
        all_rows.extend(rows)
        audits.append(audit)

    rock_rows, rock_audit = _process_rock_proxy_sensitivity(
        SCENES["3DSSZ"],
        args.sigma_small,
        args.sigma_large,
        args.ridge,
        args.winsor_sigma,
        not args.no_null,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "v8_identifiability.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)
    rock_csv_path = args.output_dir / "v8_identifiability_rock_proxy_sensitivity.csv"
    with rock_csv_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rock_rows[0].keys()))
        writer.writeheader()
        writer.writerows(rock_rows)
    markdown_path = args.output_dir / "v8_identifiability.md"
    _write_markdown(
        markdown_path,
        all_rows,
        audits,
        rock_rows,
        rock_audit,
        args.sigma_small,
        args.sigma_large,
        args.ridge,
        not args.no_null,
    )
    print(f"Wrote {len(all_rows)} rows to {csv_path}")
    print(f"Wrote {len(rock_rows)} rows to {rock_csv_path}")
    print(f"Wrote interpretation to {markdown_path}")


if __name__ == "__main__":
    main()
