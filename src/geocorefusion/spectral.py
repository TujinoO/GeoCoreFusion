"""Wavelength-dependent NIR/SWIR calibration and continuous-grid harmonization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import warnings

import numpy as np
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter
from scipy.signal import savgol_filter

from .config import SpectralConfig


@dataclass(slots=True)
class SpectralHarmonizationResult:
    cube: np.ndarray
    wavelengths_nm: np.ndarray
    calibrated_swir: np.ndarray
    swir_gain: np.ndarray
    swir_offset: np.ndarray
    nir_reliability: np.ndarray
    swir_reliability: np.ndarray
    uncertainty_by_band: np.ndarray
    band_metadata: list[dict[str, Any]]
    model: dict[str, Any]


def normalize_radiometry(cube: np.ndarray) -> tuple[np.ndarray, float]:
    arr = np.asarray(cube, dtype=np.float32)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return np.nan_to_num(arr), 1.0
    p99 = float(np.percentile(finite, 99.5))
    scale = p99 if p99 > 3.0 else 1.0
    out = arr / max(scale, 1e-12)
    return np.nan_to_num(out, nan=np.nan, posinf=np.nan, neginf=np.nan), float(scale)


def _spectral_interp(cube: np.ndarray, source_w: np.ndarray, target_w: np.ndarray) -> np.ndarray:
    fn = interp1d(
        np.asarray(source_w, dtype=np.float64),
        np.asarray(cube, dtype=np.float32),
        axis=2,
        kind="linear",
        bounds_error=False,
        fill_value=np.nan,
        assume_sorted=True,
    )
    return np.asarray(fn(np.asarray(target_w, dtype=np.float64)), dtype=np.float32)


def resample_spectra(cube: np.ndarray, source_w: np.ndarray, target_w: np.ndarray) -> np.ndarray:
    return _spectral_interp(cube, source_w, target_w)


def _valid_material_mask(nir: np.ndarray, swir: np.ndarray) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        intensity = 0.5 * np.nanmedian(nir, axis=2) + 0.5 * np.nanmedian(swir, axis=2)
    valid = np.isfinite(intensity)
    if valid.sum() < 32:
        return valid
    lo, hi = np.percentile(intensity[valid], [5, 98.5])
    return valid & (intensity > lo) & (intensity < hi)


def _robust_affine_per_band(source: np.ndarray, target: np.ndarray, mask: np.ndarray, config: SpectralConfig) -> tuple[np.ndarray, np.ndarray]:
    bands = source.shape[2]
    gain = np.ones(bands, dtype=np.float64)
    offset = np.zeros(bands, dtype=np.float64)
    flat_mask = mask.reshape(-1)
    for band in range(bands):
        x = source[:, :, band].reshape(-1)
        y = target[:, :, band].reshape(-1)
        valid = flat_mask & np.isfinite(x) & np.isfinite(y)
        if valid.sum() < 64:
            continue
        x = x[valid]
        y = y[valid]
        xq = np.percentile(x, [10, 25, 50, 75, 90])
        yq = np.percentile(y, [10, 25, 50, 75, 90])
        denom = float(xq[3] - xq[1])
        band_gain = float((yq[3] - yq[1]) / denom) if abs(denom) > 1e-9 else 1.0
        band_gain = float(np.clip(band_gain, *config.gain_limits))
        band_offset = float(yq[2] - band_gain * xq[2])
        trim = (x >= xq[0]) & (x <= xq[4]) & (y >= yq[0]) & (y <= yq[4])
        x_fit, y_fit = x[trim], y[trim]
        for _ in range(4):
            residual = y_fit - (band_gain * x_fit + band_offset)
            scale = 1.4826 * float(np.median(np.abs(residual - np.median(residual)))) + 1e-6
            weights = 1.0 / (1.0 + (residual / (2.5 * scale)) ** 2)
            sw = float(np.sum(weights))
            sx = float(np.sum(weights * x_fit))
            sy = float(np.sum(weights * y_fit))
            sxx = float(np.sum(weights * x_fit * x_fit))
            sxy = float(np.sum(weights * x_fit * y_fit))
            det = sw * sxx - sx * sx
            if abs(det) <= 1e-10:
                break
            band_gain = float(np.clip((sw * sxy - sx * sy) / det, *config.gain_limits))
            band_offset = float(np.clip((sy - band_gain * sx) / max(sw, 1e-10), *config.offset_limits))
        gain[band] = band_gain
        offset[band] = float(np.clip(band_offset, *config.offset_limits))
    window = min(config.smoothing_window, bands if bands % 2 == 1 else bands - 1)
    window = max(3, window if window % 2 == 1 else window - 1)
    poly = min(config.smoothing_polyorder, window - 1)
    if bands >= window:
        gain = savgol_filter(gain, window, poly, mode="interp")
        offset = savgol_filter(offset, window, poly, mode="interp")
    return np.clip(gain, *config.gain_limits).astype(np.float32), np.clip(offset, *config.offset_limits).astype(np.float32)


def _band_noise(cube: np.ndarray) -> np.ndarray:
    arr = np.asarray(cube, dtype=np.float32)
    smooth = gaussian_filter(np.nan_to_num(arr), sigma=(0.8, 0.8, 0.0), mode="nearest")
    residual = arr - smooth
    med = np.nanmedian(residual, axis=(0, 1), keepdims=True)
    mad = np.nanmedian(np.abs(residual - med), axis=(0, 1)) * 1.4826
    mad = np.nan_to_num(mad, nan=np.nanmedian(mad[np.isfinite(mad)]) if np.isfinite(mad).any() else 1.0)
    return np.maximum(mad.astype(np.float32), 1e-6)


def _reliability(noise: np.ndarray) -> np.ndarray:
    rel = 1.0 / np.maximum(noise, 1e-6) ** 2
    finite = rel[np.isfinite(rel)]
    scale = float(np.median(finite)) if finite.size else 1.0
    return np.clip(rel / max(scale, 1e-9), 0.05, 20.0).astype(np.float32)


def harmonize_sensors(
    nir_cube: np.ndarray,
    swir_cube: np.ndarray,
    nir_wavelengths: np.ndarray,
    swir_wavelengths: np.ndarray,
    config: SpectralConfig,
) -> SpectralHarmonizationResult:
    nir, nir_scale = normalize_radiometry(nir_cube)
    swir, swir_scale = normalize_radiometry(swir_cube)
    nir_w = np.asarray(nir_wavelengths, dtype=np.float64)
    swir_w = np.asarray(swir_wavelengths, dtype=np.float64)
    if nir_w.size != nir.shape[2] or swir_w.size != swir.shape[2]:
        raise ValueError("Wavelength counts do not match sensor band counts")
    overlap_lo = float(max(nir_w.min(), swir_w.min()))
    overlap_hi = float(min(nir_w.max(), swir_w.max()))
    swir_overlap_idx = np.flatnonzero((swir_w >= overlap_lo) & (swir_w <= overlap_hi))
    if swir_overlap_idx.size < 3:
        raise ValueError("NIR and SWIR do not provide a usable overlap")
    overlap_w = swir_w[swir_overlap_idx]
    nir_at_overlap = _spectral_interp(nir, nir_w, overlap_w)
    swir_overlap = swir[:, :, swir_overlap_idx]
    mask = _valid_material_mask(nir_at_overlap, swir_overlap)
    gain_overlap, offset_overlap = _robust_affine_per_band(swir_overlap, nir_at_overlap, mask, config)
    gain_all = np.interp(swir_w, overlap_w, gain_overlap, left=gain_overlap[0], right=gain_overlap[-1]).astype(np.float32)
    offset_all = np.interp(swir_w, overlap_w, offset_overlap, left=offset_overlap[0], right=offset_overlap[-1]).astype(np.float32)
    swir_cal = swir * gain_all[None, None, :] + offset_all[None, None, :]

    nir_noise = _band_noise(nir)
    swir_noise = _band_noise(swir_cal)
    nir_rel = _reliability(nir_noise)
    swir_rel = _reliability(swir_noise)

    target_w = np.arange(config.output_start_nm, config.output_end_nm + 0.5 * config.output_step_nm, config.output_step_nm, dtype=np.float64)
    if target_w[-1] < config.output_end_nm - 1e-6:
        target_w = np.append(target_w, float(config.output_end_nm))
    nir_target = _spectral_interp(nir, nir_w, target_w)
    swir_target = _spectral_interp(swir_cal, swir_w, target_w)
    nir_rel_target = np.interp(target_w, nir_w, nir_rel, left=0.0, right=0.0).astype(np.float32)
    swir_rel_target = np.interp(target_w, swir_w, swir_rel, left=0.0, right=0.0).astype(np.float32)

    overlap_span = max(overlap_hi - overlap_lo, 1e-6)
    position = np.clip((target_w - overlap_lo) / overlap_span, 0.0, 1.0)
    cosine = 0.5 - 0.5 * np.cos(np.pi * position)
    nir_taper = np.where(target_w < overlap_lo, 1.0, np.where(target_w > overlap_hi, 0.0, 1.0 - cosine))
    swir_taper = np.where(target_w < overlap_lo, 0.0, np.where(target_w > overlap_hi, 1.0, cosine))
    nir_weight = nir_taper.astype(np.float32) * nir_rel_target
    swir_weight = swir_taper.astype(np.float32) * swir_rel_target
    nir_weight[~np.isfinite(nir_target).any(axis=(0, 1))] = 0.0
    swir_weight[~np.isfinite(swir_target).any(axis=(0, 1))] = 0.0
    denom = np.maximum(nir_weight + swir_weight, 1e-8)
    cube = (
        np.nan_to_num(nir_target) * nir_weight[None, None, :]
        + np.nan_to_num(swir_target) * swir_weight[None, None, :]
    ) / denom[None, None, :]

    disagreement = np.zeros(target_w.size, dtype=np.float32)
    both = (nir_weight > 0) & (swir_weight > 0)
    if both.any():
        diff = np.abs(nir_target[:, :, both] - swir_target[:, :, both])
        disagreement[both] = np.nanmedian(diff, axis=(0, 1)).astype(np.float32)
    base_uncertainty = 1.0 / np.maximum(nir_weight + swir_weight, 1e-6)
    base_uncertainty /= max(float(np.nanmedian(base_uncertainty)), 1e-6)
    disagreement_scale = max(float(np.nanmedian(disagreement[both])) if both.any() else 0.0, 1e-6)
    uncertainty = np.clip(0.65 * base_uncertainty + 0.35 * disagreement / disagreement_scale, 0.0, 10.0).astype(np.float32)

    pre_rmse = float(np.sqrt(np.nanmean((nir_at_overlap - swir_overlap) ** 2)))
    post_rmse = float(np.sqrt(np.nanmean((nir_at_overlap - swir_overlap * gain_overlap[None, None, :] - offset_overlap[None, None, :]) ** 2)))
    band_metadata: list[dict[str, Any]] = []
    bad_threshold = float(np.quantile(uncertainty, config.bad_band_noise_quantile))
    for idx, wavelength in enumerate(target_w):
        water_absorption = 1340.0 <= wavelength <= 1450.0 or 1800.0 <= wavelength <= 1960.0
        band_metadata.append({
            "band_index": idx,
            "wavelength_nm": float(wavelength),
            "nir_weight": float(nir_weight[idx] / denom[idx]),
            "swir_weight": float(swir_weight[idx] / denom[idx]),
            "uncertainty": float(uncertainty[idx]),
            "is_low_confidence": bool(uncertainty[idx] >= bad_threshold or water_absorption),
            "reason": "water_absorption_or_low_snr" if water_absorption else ("high_noise" if uncertainty[idx] >= bad_threshold else ""),
        })
    model = {
        "method": config.calibration_method,
        "output_grid_nm": {"start": float(target_w[0]), "end": float(target_w[-1]), "step": float(config.output_step_nm), "count": int(target_w.size)},
        "overlap_nm": [overlap_lo, overlap_hi],
        "radiometric_scales": {"nir": nir_scale, "swir": swir_scale},
        "swir_calibration": {
            "knot_wavelengths_nm": overlap_w.tolist(),
            "gain": gain_overlap.tolist(),
            "offset": offset_overlap.tolist(),
        },
        "overlap_rmse_before": pre_rmse,
        "overlap_rmse_after": post_rmse,
        "output_policy": "continuous_fixed_grid_reliability_weighted_overlap_blend",
    }
    return SpectralHarmonizationResult(
        cube=cube.astype(np.float32),
        wavelengths_nm=target_w.astype(np.float32),
        calibrated_swir=swir_cal.astype(np.float32),
        swir_gain=gain_all,
        swir_offset=offset_all,
        nir_reliability=nir_rel,
        swir_reliability=swir_rel,
        uncertainty_by_band=uncertainty,
        band_metadata=band_metadata,
        model=model,
    )
