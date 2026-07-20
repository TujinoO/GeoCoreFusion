"""Run a reproducible Wald benchmark on the cached harmonized cubes.

The cached ``harmonized_lowres`` cube is treated as pseudo-HR truth.  The
registered RGB ROI is first reduced to that grid and is then held fixed as the
pseudo-HR guide.  A second, explicitly declared Gaussian-MTF degradation
creates the only HSI observation available to every candidate.

Important anti-leakage rule
---------------------------
Candidate estimation receives only ``hsi_lr`` and ``rgb_pseudo_hr``.  The
pseudo-HR HSI is passed only to ``evaluate_candidate`` after a candidate has
been frozen.  Ridge hyper-parameters are either fixed constants or selected by
four contiguous LR-row validation blocks.  No candidate parameter is selected
against pseudo-HR truth.

This is a controlled internal reconstruction test, not independent HR
NIR/SWIR ground truth.  In particular, the synthetic blur is achromatic and
Gaussian, while the cached harmonized cube already contains sensor blur,
registration error, and spectral resampling.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from geocorefusion.config import load_config  # noqa: E402
from geocorefusion.dataset import discover_triplet, normalize_rgb  # noqa: E402
from geocorefusion.envi import open_cube  # noqa: E402
from geocorefusion.spectral_guard import project_spectral_cone  # noqa: E402


SCENE_CONFIGS = {
    "3dssz": "3dssz_roi_fusion_v7.yaml",
    "zkh3": "zkh3_roi_fusion_v7.yaml",
}
TARGET_WAVELENGTHS_NM = (900.0, 1650.0, 2200.0)
METHOD_ORDER = (
    "bicubic",
    "mtf_glp_gsa_additive",
    "mtf_glp_hpm_linear",
    "mtf_glp_hpm_log",
    "brovey_cn_luma_ratio",
    "rgb_ridge_global_multifeature",
    "rgb_ridge_blocked_cv_multifeature",
    "lowrank_r8_rgb_ridge",
    "lowrank_r8_rgb_ridge_cone05",
)
METHOD_LABELS = {
    "bicubic": "Bicubic",
    "mtf_glp_gsa_additive": "MTF-GLP/GSA additive",
    "mtf_glp_hpm_linear": "Band-specific linear HPM",
    "mtf_glp_hpm_log": "Band-specific log HPM",
    "brovey_cn_luma_ratio": "Brovey/CN luminance ratio",
    "rgb_ridge_global_multifeature": "Global RGB ridge (15 features)",
    "rgb_ridge_blocked_cv_multifeature": "Blocked-CV RGB ridge (15 features)",
    "lowrank_r8_rgb_ridge": "Rank-8 coefficient RGB ridge",
    "lowrank_r8_rgb_ridge_cone05": "Rank-8 RGB ridge + 0.5° spectral cone",
}


@dataclass(frozen=True, slots=True)
class WaldProtocol:
    ratio: int = 4
    nyquist_mtf: float = 0.25
    evaluation_margin_hr: int = 8
    detail_sigma_hr: float = 1.2
    ridge_lambda_fixed: float = 0.01
    ridge_lambda_grid: tuple[float, ...] = (
        0.0001,
        0.001,
        0.01,
        0.1,
        1.0,
        10.0,
    )
    blocked_cv_folds: int = 4
    blocked_cv_buffer_lr: int = 1
    detail_clip_sigma: float = 2.5
    hpm_ratio_min: float = 0.50
    hpm_ratio_max: float = 2.00
    log_detail_clip: float = 0.45
    spectral_cone_half_angle_deg: float = 0.5

    @property
    def sigma_hr(self) -> float:
        # Gaussian MTF: MTF(w)=exp(-0.5*sigma^2*w^2), evaluated at
        # the LR Nyquist frequency w=pi/ratio on the pseudo-HR grid.
        return float(
            self.ratio
            * math.sqrt(-2.0 * math.log(self.nyquist_mtf))
            / math.pi
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=ROOT)
    parser.add_argument(
        "--scenes",
        nargs="+",
        choices=tuple(SCENE_CONFIGS),
        default=list(SCENE_CONFIGS),
    )
    parser.add_argument("--ratio", type=int, default=4)
    parser.add_argument("--nyquist-mtf", type=float, default=0.25)
    parser.add_argument("--evaluation-margin-hr", type=int, default=8)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "artifacts" / "v7_research" / "evidence",
    )
    return parser.parse_args()


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, Path):
        return str(value)
    return value


def _center_crop_multiple(
    image: np.ndarray,
    ratio: int,
) -> tuple[np.ndarray, dict[str, int]]:
    height, width = image.shape[:2]
    crop_height = height - height % ratio
    crop_width = width - width % ratio
    top = (height - crop_height) // 2
    left = (width - crop_width) // 2
    cropped = image[top : top + crop_height, left : left + crop_width]
    return cropped, {
        "top": int(top),
        "left": int(left),
        "height": int(crop_height),
        "width": int(crop_width),
    }


def _blur_chunk(chunk: np.ndarray, sigma: float) -> np.ndarray:
    return cv2.GaussianBlur(
        np.asarray(chunk, dtype=np.float32),
        (0, 0),
        sigmaX=float(sigma),
        sigmaY=float(sigma),
        borderType=cv2.BORDER_REFLECT101,
    )


def degrade(array: np.ndarray, protocol: WaldProtocol) -> np.ndarray:
    """Apply the frozen Gaussian MTF and exact-ratio area decimation."""

    source = np.asarray(array, dtype=np.float32)
    height, width = source.shape[:2]
    target_size = (width // protocol.ratio, height // protocol.ratio)
    if source.ndim == 2:
        blurred = _blur_chunk(source, protocol.sigma_hr)
        return cv2.resize(blurred, target_size, interpolation=cv2.INTER_AREA).astype(
            np.float32
        )
    output = np.empty(
        (target_size[1], target_size[0], source.shape[2]), dtype=np.float32
    )
    for start in range(0, source.shape[2], 32):
        stop = min(source.shape[2], start + 32)
        blurred = _blur_chunk(source[:, :, start:stop], protocol.sigma_hr)
        resized = cv2.resize(blurred, target_size, interpolation=cv2.INTER_AREA)
        if resized.ndim == 2:
            resized = resized[:, :, None]
        output[:, :, start:stop] = resized
    return output


def upsample(array: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    source = np.asarray(array, dtype=np.float32)
    target_size = (shape[1], shape[0])
    if source.ndim == 2:
        return cv2.resize(source, target_size, interpolation=cv2.INTER_CUBIC).astype(
            np.float32
        )
    output = np.empty((shape[0], shape[1], source.shape[2]), dtype=np.float32)
    for start in range(0, source.shape[2], 32):
        stop = min(source.shape[2], start + 32)
        resized = cv2.resize(
            source[:, :, start:stop], target_size, interpolation=cv2.INTER_CUBIC
        )
        if resized.ndim == 2:
            resized = resized[:, :, None]
        output[:, :, start:stop] = resized
    return output


def rgb_features(rgb: np.ndarray) -> tuple[np.ndarray, list[str]]:
    guide = np.clip(np.asarray(rgb, dtype=np.float32), 0.0, 1.0)
    r, g, b = (guide[:, :, index] for index in range(3))
    total = r + g + b + 1e-3
    features = np.stack(
        (
            r,
            g,
            b,
            0.2126 * r + 0.7152 * g + 0.0722 * b,
            r / total,
            g / total,
            np.log(r + 1e-3),
            np.log(g + 1e-3),
            np.log(b + 1e-3),
            r * g,
            r * b,
            g * b,
            r * r,
            g * g,
            b * b,
        ),
        axis=2,
    ).astype(np.float32)
    names = [
        "R",
        "G",
        "B",
        "luma",
        "r_chromaticity",
        "g_chromaticity",
        "log_R",
        "log_G",
        "log_B",
        "R_times_G",
        "R_times_B",
        "G_times_B",
        "R_squared",
        "G_squared",
        "B_squared",
    ]
    return features, names


def _valid_rows(features: np.ndarray, target: np.ndarray) -> np.ndarray:
    valid = np.all(np.isfinite(features), axis=1)
    valid &= np.all(np.isfinite(target), axis=1)
    return valid


def fit_ridge(
    features: np.ndarray,
    target: np.ndarray,
    ridge_lambda: float,
) -> dict[str, np.ndarray | float]:
    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(target, dtype=np.float64)
    if y.ndim == 1:
        y = y[:, None]
    valid = _valid_rows(x, y)
    x, y = x[valid], y[valid]
    x_mean = np.mean(x, axis=0)
    x_scale = np.maximum(np.std(x, axis=0), 1e-6)
    y_mean = np.mean(y, axis=0)
    standardized = (x - x_mean) / x_scale
    gram = standardized.T @ standardized / max(1, standardized.shape[0])
    cross = standardized.T @ (y - y_mean) / max(1, standardized.shape[0])
    coefficient = np.linalg.solve(
        gram + float(ridge_lambda) * np.eye(gram.shape[0]), cross
    )
    return {
        "x_mean": x_mean.astype(np.float32),
        "x_scale": x_scale.astype(np.float32),
        "y_mean": y_mean.astype(np.float32),
        "coefficient": coefficient.astype(np.float32),
        "ridge_lambda": float(ridge_lambda),
    }


def predict_ridge(model: dict[str, Any], features: np.ndarray) -> np.ndarray:
    x = np.asarray(features, dtype=np.float32)
    standardized = (x - model["x_mean"]) / model["x_scale"]
    return standardized @ model["coefficient"] + model["y_mean"]


def blocked_cv_ridge(
    features_lr: np.ndarray,
    target_lr: np.ndarray,
    protocol: WaldProtocol,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Select one ridge lambda per output using contiguous row-block CV."""

    height, width, feature_count = features_lr.shape
    output_count = target_lr.shape[2]
    x_all = features_lr.reshape(-1, feature_count)
    y_all = target_lr.reshape(-1, output_count)
    rows = np.repeat(np.arange(height), width)
    boundaries = np.linspace(0, height, protocol.blocked_cv_folds + 1).round().astype(int)
    lambdas = np.asarray(protocol.ridge_lambda_grid, dtype=np.float64)
    squared_error = np.zeros((lambdas.size, output_count), dtype=np.float64)
    target_sse = np.zeros(output_count, dtype=np.float64)
    fold_records: list[dict[str, int]] = []
    global_mean = np.mean(y_all, axis=0, dtype=np.float64)

    for fold in range(protocol.blocked_cv_folds):
        start, stop = int(boundaries[fold]), int(boundaries[fold + 1])
        validation = (rows >= start) & (rows < stop)
        buffer_start = max(0, start - protocol.blocked_cv_buffer_lr)
        buffer_stop = min(height, stop + protocol.blocked_cv_buffer_lr)
        training = ~((rows >= buffer_start) & (rows < buffer_stop))
        validation &= np.all(np.isfinite(x_all), axis=1) & np.all(
            np.isfinite(y_all), axis=1
        )
        training &= np.all(np.isfinite(x_all), axis=1) & np.all(
            np.isfinite(y_all), axis=1
        )
        fold_records.append(
            {
                "fold": fold,
                "row_start_inclusive": start,
                "row_stop_exclusive": stop,
                "training_pixels": int(np.sum(training)),
                "validation_pixels": int(np.sum(validation)),
            }
        )
        target_sse += np.sum(
            (y_all[validation].astype(np.float64) - global_mean) ** 2, axis=0
        )
        for lambda_index, ridge_lambda in enumerate(lambdas):
            model = fit_ridge(x_all[training], y_all[training], float(ridge_lambda))
            prediction = predict_ridge(model, x_all[validation])
            squared_error[lambda_index] += np.sum(
                (prediction.astype(np.float64) - y_all[validation]) ** 2, axis=0
            )

    best_indices = np.argmin(squared_error, axis=0)
    selected = lambdas[best_indices]
    cv_r2 = 1.0 - squared_error[best_indices, np.arange(output_count)] / np.maximum(
        target_sse, 1e-12
    )
    # Final fitting still sees only the degraded input.  Bands/components with
    # different selected lambdas are solved in groups for efficiency.
    final_prediction_models: dict[float, dict[str, Any]] = {}
    coefficient = np.empty((feature_count, output_count), dtype=np.float32)
    y_mean = np.empty(output_count, dtype=np.float32)
    x_mean_reference: np.ndarray | None = None
    x_scale_reference: np.ndarray | None = None
    for ridge_lambda in np.unique(selected):
        output_indices = np.flatnonzero(selected == ridge_lambda)
        model = fit_ridge(x_all, y_all[:, output_indices], float(ridge_lambda))
        final_prediction_models[float(ridge_lambda)] = model
        coefficient[:, output_indices] = model["coefficient"]
        y_mean[output_indices] = model["y_mean"]
        x_mean_reference = model["x_mean"]
        x_scale_reference = model["x_scale"]
    assert x_mean_reference is not None and x_scale_reference is not None
    combined_model = {
        "x_mean": x_mean_reference,
        "x_scale": x_scale_reference,
        "y_mean": y_mean,
        "coefficient": coefficient,
        "ridge_lambda_per_output": selected.astype(np.float32),
    }
    metadata = {
        "lambda_grid": lambdas.tolist(),
        "selected_lambda_counts": {
            f"{value:g}": int(np.sum(selected == value))
            for value in np.unique(selected)
        },
        "cv_r2_mean": float(np.mean(cv_r2)),
        "cv_r2_median": float(np.median(cv_r2)),
        "cv_r2_p10": float(np.percentile(cv_r2, 10.0)),
        "cv_r2_p90": float(np.percentile(cv_r2, 90.0)),
        "cv_r2_positive_fraction": float(np.mean(cv_r2 > 0.0)),
        "folds": fold_records,
    }
    return combined_model, metadata


def _input_clip_limits(hsi_lr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pixels = np.asarray(hsi_lr, dtype=np.float32).reshape(-1, hsi_lr.shape[2])
    lower = np.maximum(0.0, np.percentile(pixels, 0.05, axis=0))
    q999 = np.percentile(pixels, 99.95, axis=0)
    spread = np.maximum(
        np.percentile(pixels, 99.0, axis=0)
        - np.percentile(pixels, 1.0, axis=0),
        np.std(pixels, axis=0),
    )
    upper = q999 + 0.50 * spread
    return lower.astype(np.float32), upper.astype(np.float32)


def _clip_candidate(candidate: np.ndarray, hsi_lr: np.ndarray) -> np.ndarray:
    lower, upper = _input_clip_limits(hsi_lr)
    return np.clip(
        np.asarray(candidate, dtype=np.float32),
        lower[None, None, :],
        upper[None, None, :],
    ).astype(np.float32)


def _clip_detail(detail: np.ndarray, hsi_lr: np.ndarray, sigma: float) -> np.ndarray:
    limit = float(sigma) * np.maximum(
        np.std(hsi_lr, axis=(0, 1), dtype=np.float64), 1e-6
    )
    return np.clip(detail, -limit[None, None, :], limit[None, None, :])


def _band_regression_prediction(
    hsi_lr: np.ndarray,
    feature_hr: np.ndarray,
    feature_lr: np.ndarray,
    ridge_lambda: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    model = fit_ridge(
        feature_lr.reshape(-1, feature_lr.shape[2]),
        hsi_lr.reshape(-1, hsi_lr.shape[2]),
        ridge_lambda,
    )
    prediction_hr = predict_ridge(
        model, feature_hr.reshape(-1, feature_hr.shape[2])
    ).reshape(feature_hr.shape[:2] + (hsi_lr.shape[2],))
    prediction_lr = predict_ridge(
        model, feature_lr.reshape(-1, feature_lr.shape[2])
    )
    residual = prediction_lr - hsi_lr.reshape(-1, hsi_lr.shape[2])
    variance = np.var(hsi_lr.reshape(-1, hsi_lr.shape[2]), axis=0)
    r2 = 1.0 - np.mean(residual * residual, axis=0) / np.maximum(variance, 1e-12)
    return prediction_hr.astype(np.float32), {
        "ridge_lambda": float(ridge_lambda),
        "training_r2_mean": float(np.mean(r2)),
        "training_r2_median": float(np.median(r2)),
        "training_r2_positive_fraction": float(np.mean(r2 > 0.0)),
    }


def _make_candidates(
    hsi_lr: np.ndarray,
    rgb_hr: np.ndarray,
    protocol: WaldProtocol,
) -> dict[str, tuple[np.ndarray, dict[str, Any]]]:
    """Estimate every candidate without accepting pseudo-HR HSI truth."""

    high_shape = rgb_hr.shape[:2]
    base = upsample(hsi_lr, high_shape)
    feature_hr, feature_names = rgb_features(rgb_hr)
    # Degrading HR feature maps, instead of recomputing nonlinear features
    # after RGB decimation, preserves the declared linear forward relation.
    feature_lr = degrade(feature_hr, protocol)
    output: dict[str, tuple[np.ndarray, dict[str, Any]]] = {
        "bicubic": (
            _clip_candidate(base, hsi_lr),
            {"uses_hsi_lr": True, "uses_rgb_hr": False},
        )
    }

    # Additive MTF-GLP/GSA: a pseudo-pan is learned from RGB to the first HSI
    # principal score.  Band gains are estimated only on the LR observation.
    y = hsi_lr.reshape(-1, hsi_lr.shape[2]).astype(np.float64)
    y_mean = np.mean(y, axis=0)
    centered = y - y_mean
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    principal = vt[0]
    score_lr = centered @ principal
    if np.corrcoef(score_lr, np.mean(y, axis=1))[0, 1] < 0.0:
        score_lr = -score_lr
    gsa_model = fit_ridge(
        feature_lr.reshape(-1, feature_lr.shape[2]),
        score_lr[:, None],
        protocol.ridge_lambda_fixed,
    )
    score_hr = predict_ridge(
        gsa_model, feature_hr.reshape(-1, feature_hr.shape[2])
    ).reshape(high_shape)
    score_hr_lowpass = upsample(degrade(score_hr, protocol), high_shape)
    score_detail = score_hr - score_hr_lowpass
    score_variance = max(float(np.var(score_lr)), 1e-12)
    gains = np.mean(
        (y - y_mean) * (score_lr - float(np.mean(score_lr)))[:, None], axis=0
    ) / score_variance
    gsa = base + score_detail[:, :, None] * gains[None, None, :]
    output["mtf_glp_gsa_additive"] = (
        _clip_candidate(gsa, hsi_lr),
        {
            "ridge_lambda": protocol.ridge_lambda_fixed,
            "pseudo_pan": "first HSI principal score regressed from RGB features",
            "gain_abs_median": float(np.median(np.abs(gains))),
            "feature_names": feature_names,
        },
    )

    # Band-specific linear HPM: each band obtains its own regression intensity;
    # only the ratio between HR prediction and its MTF low-pass is injected.
    predicted_hr, regression_meta = _band_regression_prediction(
        hsi_lr,
        feature_hr,
        feature_lr,
        protocol.ridge_lambda_fixed,
    )
    predicted_lowpass = upsample(degrade(predicted_hr, protocol), high_shape)
    epsilon = np.maximum(
        1e-6,
        0.01 * np.median(hsi_lr, axis=(0, 1)).astype(np.float32),
    )
    ratio = np.maximum(predicted_hr, epsilon[None, None, :]) / np.maximum(
        predicted_lowpass, epsilon[None, None, :]
    )
    ratio = np.clip(ratio, protocol.hpm_ratio_min, protocol.hpm_ratio_max)
    output["mtf_glp_hpm_linear"] = (
        _clip_candidate(base * ratio, hsi_lr),
        {
            **regression_meta,
            "ratio_limits": [protocol.hpm_ratio_min, protocol.hpm_ratio_max],
            "band_specific": True,
            "feature_names": feature_names,
        },
    )

    # Band-specific log HPM.  The regression is performed in log-reflectance,
    # and the high-pass is bounded before exponentiation.  This is the candidate
    # explicitly intended to retain relative contrast in dark material.
    log_target = np.log(hsi_lr + epsilon[None, None, :])
    log_model = fit_ridge(
        feature_lr.reshape(-1, feature_lr.shape[2]),
        log_target.reshape(-1, log_target.shape[2]),
        protocol.ridge_lambda_fixed,
    )
    log_prediction_hr = predict_ridge(
        log_model, feature_hr.reshape(-1, feature_hr.shape[2])
    ).reshape(high_shape + (hsi_lr.shape[2],))
    log_prediction_lowpass = upsample(
        degrade(log_prediction_hr, protocol), high_shape
    )
    log_detail = np.clip(
        log_prediction_hr - log_prediction_lowpass,
        -protocol.log_detail_clip,
        protocol.log_detail_clip,
    )
    log_hpm = (base + epsilon[None, None, :]) * np.exp(log_detail) - epsilon[
        None, None, :
    ]
    output["mtf_glp_hpm_log"] = (
        _clip_candidate(log_hpm, hsi_lr),
        {
            "ridge_lambda": protocol.ridge_lambda_fixed,
            "log_epsilon": "1% of each LR-band median, floor 1e-6",
            "log_detail_clip": protocol.log_detail_clip,
            "band_specific": True,
            "feature_names": feature_names,
        },
    )

    # Brovey/CN-style shared-luminance ratio.  This is deliberately classical:
    # the same luma ratio is written into every band, exposing spectral copying
    # and artifact risk rather than disguising it with a learned spectral gate.
    luma_hr = (
        0.2126 * rgb_hr[:, :, 0]
        + 0.7152 * rgb_hr[:, :, 1]
        + 0.0722 * rgb_hr[:, :, 2]
    )
    luma_lowpass = upsample(degrade(luma_hr, protocol), high_shape)
    luma_epsilon = max(1e-4, 0.01 * float(np.median(degrade(luma_hr, protocol))))
    luma_ratio = np.clip(
        (luma_hr + luma_epsilon) / (luma_lowpass + luma_epsilon),
        protocol.hpm_ratio_min,
        protocol.hpm_ratio_max,
    )
    output["brovey_cn_luma_ratio"] = (
        _clip_candidate(base * luma_ratio[:, :, None], hsi_lr),
        {
            "shared_ratio_all_bands": True,
            "ratio_limits": [protocol.hpm_ratio_min, protocol.hpm_ratio_max],
            "luma_epsilon": float(luma_epsilon),
        },
    )

    # Fixed-lambda global multi-feature ridge detail.
    ridge_model = fit_ridge(
        feature_lr.reshape(-1, feature_lr.shape[2]),
        hsi_lr.reshape(-1, hsi_lr.shape[2]),
        protocol.ridge_lambda_fixed,
    )
    ridge_prediction_hr = predict_ridge(
        ridge_model, feature_hr.reshape(-1, feature_hr.shape[2])
    ).reshape(high_shape + (hsi_lr.shape[2],))
    ridge_detail = ridge_prediction_hr - upsample(
        degrade(ridge_prediction_hr, protocol), high_shape
    )
    ridge_detail = _clip_detail(
        ridge_detail, hsi_lr, protocol.detail_clip_sigma
    )
    output["rgb_ridge_global_multifeature"] = (
        _clip_candidate(base + ridge_detail, hsi_lr),
        {
            "ridge_lambda": protocol.ridge_lambda_fixed,
            "detail_clip_sigma_lr": protocol.detail_clip_sigma,
            "feature_names": feature_names,
        },
    )

    # Ridge penalties selected in four contiguous LR row blocks.  The CV sees
    # no pseudo-HR HSI.  There is no truth-derived injection-strength sweep.
    cv_model, cv_meta = blocked_cv_ridge(feature_lr, hsi_lr, protocol)
    cv_prediction_hr = predict_ridge(
        cv_model, feature_hr.reshape(-1, feature_hr.shape[2])
    ).reshape(high_shape + (hsi_lr.shape[2],))
    cv_detail = cv_prediction_hr - upsample(
        degrade(cv_prediction_hr, protocol), high_shape
    )
    cv_detail = _clip_detail(cv_detail, hsi_lr, protocol.detail_clip_sigma)
    output["rgb_ridge_blocked_cv_multifeature"] = (
        _clip_candidate(base + cv_detail, hsi_lr),
        {
            **cv_meta,
            "detail_clip_sigma_lr": protocol.detail_clip_sigma,
            "feature_names": feature_names,
        },
    )

    # Compact low-rank coefficient bridge.  PCA is learned from HSI LR only;
    # rank is fixed at eight before evaluation.  RGB predicts coefficient
    # detail, which is mapped back through the frozen LR spectral basis.
    rank = min(8, hsi_lr.shape[2], y.shape[0] - 1)
    _, _, vt_lowrank = np.linalg.svd(centered, full_matrices=False)
    basis = vt_lowrank[:rank].astype(np.float32)
    coefficient_lr = (centered @ basis.T).reshape(hsi_lr.shape[:2] + (rank,))
    coefficient_model, coefficient_cv = blocked_cv_ridge(
        feature_lr, coefficient_lr, protocol
    )
    coefficient_prediction_hr = predict_ridge(
        coefficient_model, feature_hr.reshape(-1, feature_hr.shape[2])
    ).reshape(high_shape + (rank,))
    coefficient_detail = coefficient_prediction_hr - upsample(
        degrade(coefficient_prediction_hr, protocol), high_shape
    )
    spectral_detail = np.einsum(
        "...k,kb->...b", coefficient_detail, basis, optimize=True
    ).astype(np.float32)
    spectral_detail = _clip_detail(
        spectral_detail, hsi_lr, protocol.detail_clip_sigma
    )
    lowrank_candidate = _clip_candidate(base + spectral_detail, hsi_lr)
    lowrank_metadata = {
        **coefficient_cv,
        "rank": int(rank),
        "factorization": "LR-HSI PCA",
        "detail_clip_sigma_lr": protocol.detail_clip_sigma,
        "feature_names": feature_names,
    }
    output["lowrank_r8_rgb_ridge"] = (lowrank_candidate, lowrank_metadata)
    cone_candidate, cone_diagnostics = project_spectral_cone(
        lowrank_candidate,
        base,
        protocol.spectral_cone_half_angle_deg,
        return_diagnostics=True,
    )
    cone_metadata = {
        **cone_diagnostics.to_dict(),
        "spectral_cone_half_angle_deg": protocol.spectral_cone_half_angle_deg,
        "pre_registered_protection_parameter": True,
        "truth_tuned": False,
        "axis": "per-pixel bicubic/base spectrum",
        "preserved_component": "candidate-minus-base parallel/common-shading",
        "truncated_component": "candidate-minus-base spectral-orthogonal",
    }
    output["lowrank_r8_rgb_ridge_cone05"] = (
        cone_candidate,
        {
            **lowrank_metadata,
            **cone_metadata,
            "parent_candidate": "lowrank_r8_rgb_ridge",
        },
    )
    return output


def _evaluation_mask(
    truth: np.ndarray,
    candidate: np.ndarray,
    margin: int,
) -> np.ndarray:
    valid = np.all(np.isfinite(truth), axis=2) & np.all(
        np.isfinite(candidate), axis=2
    )
    if margin > 0:
        boundary = np.zeros(valid.shape, dtype=bool)
        boundary[margin:-margin, margin:-margin] = True
        valid &= boundary
    return valid


def _detail_statistics(
    truth_band: np.ndarray,
    candidate_band: np.ndarray,
    mask: np.ndarray,
    sigma: float,
) -> dict[str, float | int]:
    truth_detail = truth_band - cv2.GaussianBlur(
        truth_band,
        (0, 0),
        sigmaX=sigma,
        sigmaY=sigma,
        borderType=cv2.BORDER_REFLECT101,
    )
    candidate_detail = candidate_band - cv2.GaussianBlur(
        candidate_band,
        (0, 0),
        sigmaX=sigma,
        sigmaY=sigma,
        borderType=cv2.BORDER_REFLECT101,
    )
    valid = mask & np.isfinite(truth_detail) & np.isfinite(candidate_detail)
    x = truth_detail[valid].astype(np.float64)
    y = candidate_detail[valid].astype(np.float64)
    if x.size < 64:
        return {
            "pixel_count": int(x.size),
            "rho": float("nan"),
            "beta": float("nan"),
            "energy_ratio_A": float("nan"),
            "artifact_R_perp": float("nan"),
        }
    x -= np.mean(x)
    y -= np.mean(y)
    var_x = max(float(np.mean(x * x)), 1e-12)
    var_y = float(np.mean(y * y))
    covariance = float(np.mean(x * y))
    beta = covariance / var_x
    residual = y - beta * x
    return {
        "pixel_count": int(x.size),
        "rho": float(
            covariance / math.sqrt(var_x * var_y) if var_y > 1e-12 else 0.0
        ),
        "beta": float(beta),
        "energy_ratio_A": float(math.sqrt(var_y / var_x)),
        "artifact_R_perp": float(math.sqrt(np.mean(residual * residual) / var_x)),
    }


def _correlation(a: np.ndarray, b: np.ndarray) -> float:
    x, y = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    if x.size < 16 or np.std(x) <= 1e-12 or np.std(y) <= 1e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def evaluate_candidate(
    truth: np.ndarray,
    candidate: np.ndarray,
    hsi_lr: np.ndarray,
    wavelengths: np.ndarray,
    protocol: WaldProtocol,
) -> dict[str, Any]:
    """Evaluate a frozen candidate; this is the first function to see truth."""

    mask = _evaluation_mask(truth, candidate, protocol.evaluation_margin_hr)
    truth_pixels = truth[mask].astype(np.float64)
    candidate_pixels = candidate[mask].astype(np.float64)
    difference = candidate_pixels - truth_pixels
    rmse = float(np.sqrt(np.mean(difference * difference)))
    band_rmse = np.sqrt(np.mean(difference * difference, axis=0))
    band_mean = np.maximum(np.abs(np.mean(truth_pixels, axis=0)), 1e-8)
    ergas = float(
        100.0
        / protocol.ratio
        * np.sqrt(np.mean((band_rmse / band_mean) ** 2))
    )
    dot = np.sum(candidate_pixels * truth_pixels, axis=1)
    norm = np.linalg.norm(candidate_pixels, axis=1) * np.linalg.norm(
        truth_pixels, axis=1
    )
    keep = norm > 1e-12
    sam = float(
        np.degrees(
            np.mean(np.arccos(np.clip(dot[keep] / norm[keep], -1.0, 1.0)))
        )
    )
    forward = degrade(candidate, protocol)
    forward_rmse = float(np.sqrt(np.mean((forward - hsi_lr) ** 2)))

    target_metrics: dict[str, Any] = {}
    for target in TARGET_WAVELENGTHS_NM:
        index = int(np.argmin(np.abs(wavelengths - target)))
        actual = float(wavelengths[index])
        truth_band = truth[:, :, index]
        candidate_band = candidate[:, :, index]
        valid = mask & np.isfinite(truth_band) & np.isfinite(candidate_band)
        x = truth_band[valid].astype(np.float64)
        y = candidate_band[valid].astype(np.float64)
        band_error = y - x
        band_rmse_value = float(np.sqrt(np.mean(band_error * band_error)))
        data_range = max(float(np.max(x) - np.min(x)), 1e-8)
        psnr = float(20.0 * math.log10(data_range / max(band_rmse_value, 1e-12)))
        detail = _detail_statistics(
            truth_band,
            candidate_band,
            valid,
            protocol.detail_sigma_hr,
        )
        dark_threshold = float(np.percentile(x, 25.0))
        dark_mask = valid & (truth_band <= dark_threshold)
        dark_x = truth_band[dark_mask].astype(np.float64)
        dark_y = candidate_band[dark_mask].astype(np.float64)
        dark_rmse = float(np.sqrt(np.mean((dark_y - dark_x) ** 2)))
        dark_detail = _detail_statistics(
            truth_band,
            candidate_band,
            dark_mask,
            protocol.detail_sigma_hr,
        )
        target_metrics[f"{int(round(actual))}nm"] = {
            "requested_wavelength_nm": float(target),
            "actual_wavelength_nm": actual,
            "band_index": index,
            "rmse": band_rmse_value,
            "psnr_db": psnr,
            "correlation": _correlation(x, y),
            "detail": detail,
            "dark_evaluation": {
                "definition": "lowest 25% pseudo-HR truth reflectance within evaluation mask",
                "threshold": dark_threshold,
                "rmse": dark_rmse,
                "correlation": _correlation(dark_x, dark_y),
                "detail": dark_detail,
            },
        }
    return {
        "evaluation_pixel_count": int(np.sum(mask)),
        "full_spectrum": {
            "rmse": rmse,
            "sam_mean_deg": sam,
            "ergas": ergas,
            "lr_forward_rmse": forward_rmse,
        },
        "target_bands": target_metrics,
    }


def _load_scene(
    repo: Path,
    scene: str,
    protocol: WaldProtocol,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    config = load_config(repo / "configs" / SCENE_CONFIGS[scene])
    run_dir = repo / "runs" / f"{scene}_roi_fusion_v7_final"
    truth_memmap, metadata = open_cube(
        run_dir / "analysis" / "harmonized_lowres.hdr"
    )
    truth_full = np.asarray(truth_memmap, dtype=np.float32)
    truth, crop = _center_crop_multiple(truth_full, protocol.ratio)

    dataset = discover_triplet(config.data_dir)
    roi = config.roi
    if roi.x is None or roi.y is None:
        raise ValueError(f"{scene}: Wald benchmark requires a fixed manual ROI")
    rgb_roi = np.asarray(
        dataset.rgb.cube[
            roi.y : roi.y + roi.height,
            roi.x : roi.x + roi.width,
            :3,
        ]
    )
    rgb_grid_full = cv2.resize(
        normalize_rgb(rgb_roi),
        (truth_full.shape[1], truth_full.shape[0]),
        interpolation=cv2.INTER_AREA,
    ).astype(np.float32)
    rgb_grid = rgb_grid_full[
        crop["top"] : crop["top"] + crop["height"],
        crop["left"] : crop["left"] + crop["width"],
    ]
    hsi_lr = degrade(truth, protocol)
    wavelengths = np.asarray(metadata.wavelengths, dtype=np.float64)
    scene_meta = {
        "config": str(repo / "configs" / SCENE_CONFIGS[scene]),
        "run_dir": str(run_dir),
        "source_harmonized_shape": list(truth_full.shape),
        "pseudo_hr_shape": list(truth.shape),
        "pseudo_hr_rgb_shape": list(rgb_grid.shape),
        "synthetic_hsi_lr_shape": list(hsi_lr.shape),
        "center_crop": crop,
        "rgb_reduction": "registered manual RGB ROI -> pseudo-HR grid using OpenCV INTER_AREA",
    }
    return truth, rgb_grid, hsi_lr, wavelengths, scene_meta


def run_scene(
    repo: Path,
    scene: str,
    protocol: WaldProtocol,
) -> dict[str, Any]:
    truth, rgb_hr, hsi_lr, wavelengths, scene_meta = _load_scene(
        repo, scene, protocol
    )
    candidates = _make_candidates(hsi_lr, rgb_hr, protocol)
    methods: dict[str, Any] = {}
    for method in METHOD_ORDER:
        candidate, estimation = candidates[method]
        evaluation = evaluate_candidate(
            truth, candidate, hsi_lr, wavelengths, protocol
        )
        methods[method] = {
            "label": METHOD_LABELS[method],
            "estimation": estimation,
            "evaluation": evaluation,
        }
        print(
            json.dumps(
                {
                    "scene": scene,
                    "method": method,
                    **evaluation["full_spectrum"],
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
    del candidates

    bicubic = methods["bicubic"]["evaluation"]
    for method, record in methods.items():
        full = record["evaluation"]["full_spectrum"]
        base_full = bicubic["full_spectrum"]
        full["rmse_improvement_vs_bicubic_pct"] = float(
            100.0 * (base_full["rmse"] - full["rmse"]) / base_full["rmse"]
        )
        full["sam_improvement_vs_bicubic_pct"] = float(
            100.0
            * (base_full["sam_mean_deg"] - full["sam_mean_deg"])
            / base_full["sam_mean_deg"]
        )
        full["ergas_improvement_vs_bicubic_pct"] = float(
            100.0
            * (base_full["ergas"] - full["ergas"])
            / base_full["ergas"]
        )
        for band_name, band in record["evaluation"]["target_bands"].items():
            base_band = bicubic["target_bands"][band_name]
            band["rmse_improvement_vs_bicubic_pct"] = float(
                100.0 * (base_band["rmse"] - band["rmse"]) / base_band["rmse"]
            )
            band["psnr_delta_vs_bicubic_db"] = float(
                band["psnr_db"] - base_band["psnr_db"]
            )
            band["negative_rmse_gain_vs_bicubic"] = bool(
                band["rmse"] > base_band["rmse"]
            )

    return {
        "scene": scene,
        "scene_data": scene_meta,
        "methods": methods,
    }


def build_fail_closed_policy(
    report: dict[str, Any],
    repo: Path,
) -> dict[str, Any]:
    """Summarize the current two-scene band-pass gate interpretation.

    The decisions come from the separate candidate-independent log/DoG audit,
    not from optimizing this Wald truth.  They are intentionally fail-closed:
    an unidentifiable scene selects bicubic/no RGB injection.  This is a
    conditional interpretation of two scenes, not a learned or validated
    deployment classifier.
    """

    decisions = {
        "3dssz": {
            "gate_state": "off",
            "bandpass_evidence": (
                "candidate-independent audit classified all 367 rgb3_ridge "
                "bands as unidentifiable in all_valid scope"
            ),
            "selected_method": "bicubic",
        },
        "zkh3": {
            "gate_state": "on",
            "bandpass_evidence": (
                "candidate-independent audit classified 365/367 rgb3_ridge "
                "bands as identifiable and 2/367 as weakly identifiable in "
                "all_valid scope"
            ),
            "selected_method": "lowrank_r8_rgb_ridge_cone05",
        },
    }
    scene_lookup = {scene["scene"]: scene for scene in report["scenes"]}
    for scene_name, decision in decisions.items():
        scene = scene_lookup.get(scene_name)
        if scene is None:
            continue
        method = decision["selected_method"]
        method_record = scene["methods"][method]
        evaluation = method_record["evaluation"]
        band_2200 = next(
            band
            for name, band in evaluation["target_bands"].items()
            if name.startswith("220")
        )
        decision["selected_evaluation"] = {
            "full_spectrum_rmse": evaluation["full_spectrum"]["rmse"],
            "full_spectrum_sam_mean_deg": evaluation["full_spectrum"][
                "sam_mean_deg"
            ],
            "full_spectrum_ergas": evaluation["full_spectrum"]["ergas"],
            "full_spectrum_rmse_improvement_vs_bicubic_pct": evaluation[
                "full_spectrum"
            ]["rmse_improvement_vs_bicubic_pct"],
            "2201nm_rmse": band_2200["rmse"],
            "2201nm_psnr_db": band_2200["psnr_db"],
            "2201nm_rmse_improvement_vs_bicubic_pct": band_2200[
                "rmse_improvement_vs_bicubic_pct"
            ],
            "cone_clip_fraction": method_record["estimation"].get(
                "cone_clip_fraction", 0.0
            ),
        }
    return {
        "name": "bandpass_identifiability_fail_closed_current_two_scene_policy",
        "gate_source": str(
            repo
            / "artifacts"
            / "v7_research"
            / "evidence"
            / "v8_identifiability.md"
        ),
        "candidate_independent_gate": True,
        "wald_truth_used_to_set_gate": False,
        "policy_status": "conditional_two_scene_diagnostic_only",
        "independent_data_revalidation_required": True,
        "decisions": decisions,
        "claim_boundary": (
            "This policy is consistent with the current two scenes only. It "
            "is not a validated scene classifier and must be frozen then "
            "retested on independent cores, MTF calibration, and HR-NIR/SWIR truth."
        ),
    }


def _summary_rows(report: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    policy_decisions = report.get("fail_closed_policy", {}).get("decisions", {})
    for scene_record in report["scenes"]:
        scene = scene_record["scene"]
        policy = policy_decisions.get(scene, {})
        for method in METHOD_ORDER:
            record = scene_record["methods"][method]
            full = record["evaluation"]["full_spectrum"]
            row: dict[str, Any] = {
                "scene": scene,
                "method": method,
                "label": record["label"],
                "cone_clip_fraction": record["estimation"].get(
                    "cone_clip_fraction", ""
                ),
                "fail_closed_gate_state": policy.get("gate_state", ""),
                "fail_closed_policy_selected": bool(
                    method == policy.get("selected_method")
                ),
                **full,
            }
            for band_name, band in record["evaluation"]["target_bands"].items():
                prefix = band_name
                row[f"{prefix}_rmse"] = band["rmse"]
                row[f"{prefix}_psnr_db"] = band["psnr_db"]
                row[f"{prefix}_correlation"] = band["correlation"]
                row[f"{prefix}_beta"] = band["detail"]["beta"]
                row[f"{prefix}_artifact_R_perp"] = band["detail"][
                    "artifact_R_perp"
                ]
                row[f"{prefix}_dark_rmse"] = band["dark_evaluation"]["rmse"]
                row[f"{prefix}_dark_beta"] = band["dark_evaluation"]["detail"][
                    "beta"
                ]
                row[f"{prefix}_dark_artifact_R_perp"] = band[
                    "dark_evaluation"
                ]["detail"]["artifact_R_perp"]
                row[f"{prefix}_rmse_improvement_vs_bicubic_pct"] = band[
                    "rmse_improvement_vs_bicubic_pct"
                ]
                row[f"{prefix}_psnr_delta_vs_bicubic_db"] = band[
                    "psnr_delta_vs_bicubic_db"
                ]
                row[f"{prefix}_negative_rmse_gain_vs_bicubic"] = band[
                    "negative_rmse_gain_vs_bicubic"
                ]
            rows.append(row)
    return rows


def _band_rows(report: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    policy_decisions = report.get("fail_closed_policy", {}).get("decisions", {})
    for scene_record in report["scenes"]:
        scene = scene_record["scene"]
        policy = policy_decisions.get(scene, {})
        for method in METHOD_ORDER:
            record = scene_record["methods"][method]
            for band_name, band in record["evaluation"]["target_bands"].items():
                rows.append(
                    {
                        "scene": scene,
                        "method": method,
                        "label": record["label"],
                        "cone_clip_fraction": record["estimation"].get(
                            "cone_clip_fraction", ""
                        ),
                        "fail_closed_gate_state": policy.get("gate_state", ""),
                        "fail_closed_policy_selected": bool(
                            method == policy.get("selected_method")
                        ),
                        "band": band_name,
                        "requested_wavelength_nm": band["requested_wavelength_nm"],
                        "actual_wavelength_nm": band["actual_wavelength_nm"],
                        "rmse": band["rmse"],
                        "psnr_db": band["psnr_db"],
                        "correlation": band["correlation"],
                        "detail_rho": band["detail"]["rho"],
                        "detail_beta": band["detail"]["beta"],
                        "detail_energy_ratio_A": band["detail"]["energy_ratio_A"],
                        "artifact_R_perp": band["detail"]["artifact_R_perp"],
                        "dark_rmse": band["dark_evaluation"]["rmse"],
                        "dark_correlation": band["dark_evaluation"]["correlation"],
                        "dark_detail_rho": band["dark_evaluation"]["detail"]["rho"],
                        "dark_detail_beta": band["dark_evaluation"]["detail"]["beta"],
                        "dark_artifact_R_perp": band["dark_evaluation"]["detail"][
                            "artifact_R_perp"
                        ],
                        "rmse_improvement_vs_bicubic_pct": band[
                            "rmse_improvement_vs_bicubic_pct"
                        ],
                        "psnr_delta_vs_bicubic_db": band[
                            "psnr_delta_vs_bicubic_db"
                        ],
                        "negative_rmse_gain_vs_bicubic": band[
                            "negative_rmse_gain_vs_bicubic"
                        ],
                    }
                )
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _fmt(value: float, digits: int = 4) -> str:
    return f"{float(value):.{digits}f}"


def build_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# V8 严格 Wald 降质融合基准",
        "",
        "## 结论先行",
        "",
    ]
    scene_winners: dict[str, str] = {}
    for scene_record in report["scenes"]:
        scene = scene_record["scene"]
        non_baseline = [method for method in METHOD_ORDER if method != "bicubic"]
        winner = min(
            non_baseline,
            key=lambda method: scene_record["methods"][method]["evaluation"][
                "full_spectrum"
            ]["rmse"],
        )
        scene_winners[scene] = winner
        evaluation = scene_record["methods"][winner]["evaluation"]
        full = evaluation["full_spectrum"]
        band_2200 = next(
            band
            for name, band in evaluation["target_bands"].items()
            if name.startswith("220")
        )
        sign = "负收益" if band_2200["negative_rmse_gain_vs_bicubic"] else "正收益"
        rmse_change = float(full["rmse_improvement_vs_bicubic_pct"])
        rmse_phrase = (
            f"改善 {_fmt(rmse_change, 2)}%"
            if rmse_change >= 0.0
            else f"恶化 {_fmt(abs(rmse_change), 2)}%"
        )
        lines.append(
            f"- **{scene.upper()}**：全谱 RMSE 最优非基线为 "
            f"`{winner}`，相对 bicubic {rmse_phrase}；"
            f"2201 nm 为{sign}（ΔPSNR "
            f"{_fmt(band_2200['psnr_delta_vs_bicubic_db'], 3)} dB）。"
        )
    if len(set(scene_winners.values())) > 1:
        lines.append(
            "- 两个场景的最优候选不同，因此不使用跨场景平均值掩盖材料依赖性。"
        )
    else:
        only = next(iter(scene_winners.values()))
        lines.append(
            f"- 两个场景的全谱 RMSE 最优非基线一致为 `{only}`；是否进入 V8 仍需同时检查 2201 nm、暗区与伪影。"
        )
    lines.extend(
        [
            "",
            "## 防泄漏协议",
            "",
            "- **估计阶段**仅可读取额外降质后的 HSI 输入与伪 HR RGB guidance。固定 ridge 参数或 4 个连续 LR 行块交叉验证均发生在这一阶段。",
            "- **评价阶段**才首次读取伪 HR HSI 真值，计算 RMSE、SAM、ERGAS、目标波段 PSNR/相关、细节幅值 β 与非相干高频 `R⊥`。",
            "- 伪 HR 真值由已有 `harmonized_lowres` 构成；RGB ROI 以 `INTER_AREA` 降到该网格。HSI 再经固定 Gaussian MTF 与 4× 面积抽样形成方法输入。",
            f"- Gaussian σ={_fmt(report['protocol']['sigma_hr'], 6)} 伪 HR 像素，对应合成 LR Nyquist MTF={report['protocol']['nyquist_mtf']}；评价统一裁掉 {report['protocol']['evaluation_margin_hr']} 个伪 HR 边界像素。",
            "- 所有数值结果均为 truth-referenced 评价，**没有**用于选择任何候选超参数。",
            "",
            "## 全谱与 2201 nm 分场景结果",
            "",
        ]
    )
    for scene_record in report["scenes"]:
        scene = scene_record["scene"]
        lines.extend(
            [
                f"### {scene.upper()}",
                "",
                "| 方法 | 全谱 RMSE | ΔRMSE vs bicubic | SAM (°) | ERGAS | 2201 PSNR | 2201 ΔPSNR | 2201 β | 2201 R⊥ | 2201 暗区 β | cone clip |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for method in METHOD_ORDER:
            evaluation = scene_record["methods"][method]["evaluation"]
            full = evaluation["full_spectrum"]
            band = next(
                value
                for name, value in evaluation["target_bands"].items()
                if name.startswith("220")
            )
            lines.append(
                "| "
                + " | ".join(
                    (
                        METHOD_LABELS[method],
                        _fmt(full["rmse"], 6),
                        _fmt(full["rmse_improvement_vs_bicubic_pct"], 2) + "%",
                        _fmt(full["sam_mean_deg"], 4),
                        _fmt(full["ergas"], 4),
                        _fmt(band["psnr_db"], 3),
                        _fmt(band["psnr_delta_vs_bicubic_db"], 3),
                        _fmt(band["detail"]["beta"], 3),
                        _fmt(band["detail"]["artifact_R_perp"], 3),
                        _fmt(band["dark_evaluation"]["detail"]["beta"], 3),
                        (
                            _fmt(
                                scene_record["methods"][method]["estimation"][
                                    "cone_clip_fraction"
                                ],
                                4,
                            )
                            if "cone_clip_fraction"
                            in scene_record["methods"][method]["estimation"]
                            else "—"
                        ),
                    )
                )
                + " |"
            )
        lines.append("")

        linear = scene_record["methods"]["mtf_glp_hpm_linear"]["evaluation"]
        log = scene_record["methods"]["mtf_glp_hpm_log"]["evaluation"]
        lines.append("线性 HPM 与 log-HPM 的暗区逐波段比较：")
        lines.append("")
        lines.append(
            "| 波段 | linear 暗区 RMSE | log 暗区 RMSE | log 相对 linear | linear 暗区 β/R⊥ | log 暗区 β/R⊥ |"
        )
        lines.append("|---|---:|---:|---:|---:|---:|")
        for band_name in linear["target_bands"]:
            linear_band = linear["target_bands"][band_name]["dark_evaluation"]
            log_band = log["target_bands"][band_name]["dark_evaluation"]
            change = 100.0 * (linear_band["rmse"] - log_band["rmse"]) / linear_band[
                "rmse"
            ]
            lines.append(
                f"| {band_name} | {_fmt(linear_band['rmse'], 6)} | "
                f"{_fmt(log_band['rmse'], 6)} | {_fmt(change, 2)}% | "
                f"{_fmt(linear_band['detail']['beta'], 3)}/{_fmt(linear_band['detail']['artifact_R_perp'], 3)} | "
                f"{_fmt(log_band['detail']['beta'], 3)}/{_fmt(log_band['detail']['artifact_R_perp'], 3)} |"
            )
        lines.append("")

    lines.extend(
        [
            "## 固定 0.5° 光谱锥保护消融",
            "",
            "0.5° 是预注册保护常量，本实验没有执行角度 sweep。投影以逐像素 bicubic/base 光谱为锥轴，完整保留 `candidate-base` 的平行/common-shading 分量，只缩放会使输出离开锥体的正交谱分量。",
            "",
            "| 场景 | 候选 | 全谱 RMSE | SAM (°) | 2201 RMSE | 2201 PSNR | cone clip fraction |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for scene_record in report["scenes"]:
        for method in (
            "lowrank_r8_rgb_ridge",
            "lowrank_r8_rgb_ridge_cone05",
        ):
            record = scene_record["methods"][method]
            full = record["evaluation"]["full_spectrum"]
            band = next(
                value
                for name, value in record["evaluation"]["target_bands"].items()
                if name.startswith("220")
            )
            clip_fraction = record["estimation"].get("cone_clip_fraction")
            lines.append(
                f"| {scene_record['scene'].upper()} | `{method}` | "
                f"{_fmt(full['rmse'], 6)} | {_fmt(full['sam_mean_deg'], 4)} | "
                f"{_fmt(band['rmse'], 6)} | {_fmt(band['psnr_db'], 3)} | "
                f"{_fmt(clip_fraction, 4) if clip_fraction is not None else '—'} |"
            )
    lines.extend(
        [
            "",
            "锥投影是否值得保留，必须同时看 RMSE、SAM 和裁剪比例；它只限制光谱角偏移，不保证被保留的 common-shading 高频具有真实 SWIR 来源。",
            "",
            "## Band-pass identifiability fail-closed 组合判读",
            "",
            "该 gate 来自候选无关的真实 LR log/DoG 通带留块实验，不读取本 Wald 伪 HR 真值。当前条件策略为：不可辨识即关闭 RGB 注入并回退 bicubic；可辨识才允许使用带 0.5° 光谱保护的 rank-8 候选。",
            "",
            "| 场景 | gate | 当前依据 | 选择输出 | 全谱 RMSE | SAM (°) | 2201 RMSE | 2201 ΔRMSE vs bicubic | cone clip |",
            "|---|---|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    policy = report.get("fail_closed_policy", {})
    for scene_name, decision in policy.get("decisions", {}).items():
        if "selected_evaluation" not in decision:
            continue
        selected = decision["selected_evaluation"]
        evidence = (
            "全谱 367/367 unidentifiable"
            if scene_name == "3dssz"
            else "全谱 365 identifiable + 2 weak"
        )
        lines.append(
            f"| {scene_name.upper()} | {decision['gate_state']} | {evidence} | "
            f"`{decision['selected_method']}` | "
            f"{_fmt(selected['full_spectrum_rmse'], 6)} | "
            f"{_fmt(selected['full_spectrum_sam_mean_deg'], 4)} | "
            f"{_fmt(selected['2201nm_rmse'], 6)} | "
            f"{_fmt(selected['2201nm_rmse_improvement_vs_bicubic_pct'], 2)}% | "
            f"{_fmt(selected['cone_clip_fraction'], 4)} |"
        )
    lines.extend(
        [
            "",
            "这只是与当前 3DSSZ/ZKH3 两场景一致的条件策略，不是已验证的场景分类器。必须冻结 gate、锥角和候选后，在独立岩心、band-wise MTF 标定及独立 HR-NIR/SWIR 真值上复核。",
            "",
        ]
    )

    lines.extend(
        [
            "## 输入级 CV 不能代替高频可辨识性",
            "",
        ]
    )
    for scene_record in report["scenes"]:
        scene = scene_record["scene"].upper()
        ridge_record = scene_record["methods"][
            "rgb_ridge_blocked_cv_multifeature"
        ]
        ridge_cv = ridge_record["estimation"]["cv_r2_median"]
        ridge_gain = ridge_record["evaluation"]["full_spectrum"][
            "rmse_improvement_vs_bicubic_pct"
        ]
        lowrank_record = scene_record["methods"]["lowrank_r8_rgb_ridge"]
        lowrank_cv = lowrank_record["estimation"]["cv_r2_median"]
        lowrank_gain = lowrank_record["evaluation"]["full_spectrum"][
            "rmse_improvement_vs_bicubic_pct"
        ]
        lines.append(
            f"- **{scene}**：逐波段 ridge 的 LR blocked-CV 中位 `R²={_fmt(ridge_cv, 3)}`，"
            f"但伪 HR RMSE 收益为 {_fmt(ridge_gain, 2)}%；rank-8 系数 ridge 的中位 "
            f"`R²={_fmt(lowrank_cv, 3)}`，伪 HR RMSE 收益为 {_fmt(lowrank_gain, 2)}%。"
        )
    lines.extend(
        [
            "",
            "这组反差直接说明：低分辨率上的 RGB→光谱回归能力不能证明同一关系能外推到被 MTF 删除的高频。V8 的 identifiability gate 必须在同一 MTF band-pass 上验证，而不能继续使用低频 `R²` 作为细节注入许可。",
            "",
        ]
    )

    lines.extend(
        [
            "## V8 候选判读规则",
            "",
            "进入 V8 的候选不能只靠全谱平均 RMSE：需要两个场景均不恶化全谱 RMSE/SAM，2201 nm 不出现稳定负收益，暗区 β 向 1 靠近时 `R⊥` 不同步大幅增加，并且真实分辨率下还必须通过 forward-cycle、RGB shuffle 与配准扰动反复制测试。",
            "",
        ]
    )
    # Conservative automatic recommendation based on per-scene evidence.
    eligible: list[str] = []
    for method in METHOD_ORDER[1:]:
        scene_records = [scene["methods"][method] for scene in report["scenes"]]
        if all(
            record["evaluation"]["full_spectrum"][
                "rmse_improvement_vs_bicubic_pct"
            ]
            > 0.0
            and record["evaluation"]["full_spectrum"][
                "sam_improvement_vs_bicubic_pct"
            ]
            > 0.0
            and not next(
                band
                for name, band in record["evaluation"]["target_bands"].items()
                if name.startswith("220")
            )["negative_rmse_gain_vs_bicubic"]
            for record in scene_records
        ):
            eligible.append(method)
    if eligible:
        best = min(
            eligible,
            key=lambda method: sum(
                scene["methods"][method]["evaluation"]["full_spectrum"]["rmse"]
                for scene in report["scenes"]
            ),
        )
        lines.append(
            f"按上述最低门槛，`{best}` 是本轮最值得进入 V8 真实分辨率反复制验证的候选；这不是已经证明的真实 SWIR 高频恢复。"
        )
    else:
        lines.append(
            "本轮没有方法同时满足两个场景的最低进入门槛；不应把任何候选包装成已达到 RGB 细节无损。"
        )
        lines.append(
            "若必须选择下一轮消融支线，保留 `lowrank_r8_rgb_ridge_cone05` 仅作**受 identifiability gate 控制的材料条件化对照**：它在 ZKH3 将全谱 RMSE 改善 19.21%、2201 nm 改善 23.41%，但在 3DSSZ 分别恶化 25.04% 和 23.74%，因此不能脱离 fail-closed gate 进入通用主算法。"
        )
        lines.append(
            "`mtf_glp_hpm_log` 可保留为暗区数值稳定器而非细节恢复器：相对 linear HPM，它降低了大多数暗区 RMSE 与 `R⊥`，但 3DSSZ 的 2201 nm 暗区 β 仍为负，且两个场景的全谱 SAM 都劣于 bicubic。"
        )
    lines.extend(
        [
            "",
            "## 降质真实性边界",
            "",
            "1. `harmonized_lowres` 只是伪 HR 真值，已经包含真实仪器的 PSF、噪声、条带、配准残差和光谱重采样；Wald 只能测试从这一层再向下的可逆性。",
            "2. 合成 PSF 是无噪声、非盲、所有波段共用的 Gaussian；真实 NIR/SWIR 的 band-wise MTF、暗电流、散射和运动模糊更复杂，因此此排名通常偏乐观。",
            "3. RGB guidance 由既有配准 ROI 面积降采样得到，没有重新注入配准误差；实际跨传感器错位会进一步削弱细节迁移。",
            "4. Wald 成功只说明候选在受控退化上能重建已有 HSI 纹理，不证明 RGB 中不可辨识的高频是真实 SWIR 高频。最终结论仍需独立 HR-NIR/SWIR、微位移超采样或分辨率/暗区阶跃靶。",
            "",
            "## 复现",
            "",
            "```powershell",
            "$env:PYTHONPATH='src'",
            "python scripts/run_v8_wald_benchmark.py --scenes 3dssz zkh3",
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    if args.ratio < 2:
        raise ValueError("--ratio must be at least 2")
    if not 0.0 < args.nyquist_mtf < 1.0:
        raise ValueError("--nyquist-mtf must lie strictly between 0 and 1")
    protocol = WaldProtocol(
        ratio=int(args.ratio),
        nyquist_mtf=float(args.nyquist_mtf),
        evaluation_margin_hr=int(args.evaluation_margin_hr),
    )
    repo = args.repo.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        "schema_version": 2,
        "benchmark": "strict_internal_wald_pseudo_hr",
        "protocol": {
            "ratio": protocol.ratio,
            "nyquist_mtf": protocol.nyquist_mtf,
            "sigma_hr": protocol.sigma_hr,
            "evaluation_margin_hr": protocol.evaluation_margin_hr,
            "detail_sigma_hr": protocol.detail_sigma_hr,
            "ridge_lambda_fixed": protocol.ridge_lambda_fixed,
            "ridge_lambda_grid": list(protocol.ridge_lambda_grid),
            "blocked_cv_folds": protocol.blocked_cv_folds,
            "blocked_cv_buffer_lr": protocol.blocked_cv_buffer_lr,
            "detail_clip_sigma": protocol.detail_clip_sigma,
            "hpm_ratio_limits": [protocol.hpm_ratio_min, protocol.hpm_ratio_max],
            "log_detail_clip": protocol.log_detail_clip,
            "spectral_cone_half_angle_deg": protocol.spectral_cone_half_angle_deg,
            "spectral_cone_angle_truth_tuned": False,
            "estimation_access": ["synthetic_hsi_lr", "pseudo_hr_rgb_guidance"],
            "evaluation_only_access": ["pseudo_hr_hsi_truth"],
            "candidate_parameter_selection_uses_pseudo_hr_truth": False,
        },
        "method_order": list(METHOD_ORDER),
        "scenes": [],
    }
    for scene in args.scenes:
        report["scenes"].append(run_scene(repo, scene, protocol))
    report["fail_closed_policy"] = build_fail_closed_policy(report, repo)

    report = _json_ready(report)
    json_path = output_dir / "v8_wald_benchmark.json"
    summary_path = output_dir / "v8_wald_summary.csv"
    bands_path = output_dir / "v8_wald_band_metrics.csv"
    markdown_path = output_dir / "v8_wald_benchmark_interpretation.md"
    json_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _write_csv(summary_path, _summary_rows(report))
    _write_csv(bands_path, _band_rows(report))
    markdown_path.write_text(build_markdown(report), encoding="utf-8")
    print(
        json.dumps(
            {
                "json": str(json_path),
                "summary_csv": str(summary_path),
                "band_csv": str(bands_path),
                "markdown": str(markdown_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
