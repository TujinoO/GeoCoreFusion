"""Sensor-consistency and spectral-safety diagnostics."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from .dataset import normalize_rgb
from .degradation import PsfModel, degrade_coefficients, degrade_spatial_map
from .lowrank import SubspaceModel, reconstruct
from .spectral import SpectralHarmonizationResult, normalize_radiometry, resample_spectra


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    diff = np.asarray(a, dtype=np.float32) - np.asarray(b, dtype=np.float32)
    return float(np.sqrt(np.nanmean(diff * diff)))


def sam_degrees(a: np.ndarray, b: np.ndarray, max_pixels: int = 40000) -> float:
    aa = np.asarray(a, dtype=np.float32).reshape(-1, a.shape[-1])
    bb = np.asarray(b, dtype=np.float32).reshape(-1, b.shape[-1])
    common = np.isfinite(aa) & np.isfinite(bb)
    valid = common.sum(axis=1) >= max(3, aa.shape[1] // 2)
    aa = np.where(common[valid], aa[valid], 0.0)
    bb = np.where(common[valid], bb[valid], 0.0)
    if aa.shape[0] > max_pixels:
        idx = np.linspace(0, aa.shape[0] - 1, max_pixels).round().astype(int)
        aa, bb = aa[idx], bb[idx]
    if aa.size == 0:
        return float("nan")
    denom = np.linalg.norm(aa, axis=1) * np.linalg.norm(bb, axis=1)
    keep = denom > 1e-10
    cosine = np.clip(np.sum(aa[keep] * bb[keep], axis=1) / denom[keep], -1.0, 1.0)
    return float(np.degrees(np.mean(np.arccos(cosine)))) if cosine.size else float("nan")


def mean_band_correlation(a: np.ndarray, b: np.ndarray) -> float:
    aa = np.asarray(a, dtype=np.float32).reshape(-1, a.shape[-1])
    bb = np.asarray(b, dtype=np.float32).reshape(-1, b.shape[-1])
    values: list[float] = []
    for band in range(aa.shape[1]):
        x, y = aa[:, band], bb[:, band]
        valid = np.isfinite(x) & np.isfinite(y)
        if valid.sum() < 16:
            continue
        x, y = x[valid], y[valid]
        sx, sy = float(np.std(x)), float(np.std(y))
        if sx > 1e-9 and sy > 1e-9:
            values.append(float(np.mean((x - x.mean()) * (y - y.mean())) / (sx * sy)))
    return float(np.mean(values)) if values else float("nan")


def _edge_correlation(coeff: np.ndarray, rgb: np.ndarray, detail_gain: np.ndarray | None = None) -> float:
    material = np.sqrt(np.sum(coeff**2, axis=2))
    if detail_gain is not None:
        material = material * np.asarray(detail_gain, dtype=np.float32)
    guide = cv2.cvtColor((normalize_rgb(rgb) * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32)
    mg = cv2.magnitude(cv2.Sobel(material, cv2.CV_32F, 1, 0), cv2.Sobel(material, cv2.CV_32F, 0, 1)).reshape(-1)
    gg = cv2.magnitude(cv2.Sobel(guide, cv2.CV_32F, 1, 0), cv2.Sobel(guide, cv2.CV_32F, 0, 1)).reshape(-1)
    valid = np.isfinite(mg) & np.isfinite(gg)
    mg, gg = mg[valid], gg[valid]
    if mg.size < 32 or np.std(mg) < 1e-9 or np.std(gg) < 1e-9:
        return float("nan")
    return float(np.corrcoef(mg, gg)[0, 1])


def build_quality_report(
    refined_coeff: np.ndarray,
    low_coeff: np.ndarray,
    subspace: SubspaceModel,
    psf: PsfModel,
    rgb: np.ndarray,
    spectral: SpectralHarmonizationResult,
    nir_aligned: np.ndarray,
    swir_aligned: np.ndarray,
    nir_wavelengths: np.ndarray,
    swir_wavelengths: np.ndarray,
    uncertainty_map: np.ndarray,
    detail_gain_map: np.ndarray | None = None,
    additive_detail_map: np.ndarray | None = None,
    additive_spectral_scale: np.ndarray | None = None,
) -> dict[str, Any]:
    low_coeff_pred = degrade_coefficients(refined_coeff, psf)
    if detail_gain_map is None:
        low_cube_pred = reconstruct(low_coeff_pred, subspace)
        low_gain = np.ones(psf.low_shape, dtype=np.float32)
    else:
        gain = np.asarray(detail_gain_map, dtype=np.float32)
        low_gain = degrade_spatial_map(gain, psf)
        low_weighted_coeff = degrade_coefficients(refined_coeff * gain[:, :, None], psf)
        low_cube_pred = (
            np.einsum("...k,kb->...b", low_weighted_coeff, subspace.basis, optimize=True)
            + low_gain[:, :, None] * subspace.mean_spectrum[None, None, :]
        ).astype(np.float32)
        low_cube_pred = np.clip(
            low_cube_pred,
            subspace.clip_min[None, None, :],
            subspace.clip_max[None, None, :],
        )
    if additive_detail_map is None or additive_spectral_scale is None:
        low_additive = np.zeros(psf.low_shape, dtype=np.float32)
    else:
        low_additive = degrade_spatial_map(np.asarray(additive_detail_map, dtype=np.float32), psf)
        low_cube_pred = low_cube_pred + (
            low_additive[:, :, None]
            * np.asarray(additive_spectral_scale, dtype=np.float32)[None, None, :]
        )
        low_cube_pred = np.clip(
            low_cube_pred,
            subspace.clip_min[None, None, :],
            subspace.clip_max[None, None, :],
        )
    nir_norm, _ = normalize_radiometry(nir_aligned)
    swir_norm, _ = normalize_radiometry(swir_aligned)
    nir_pred = resample_spectra(
        low_cube_pred,
        spectral.wavelengths_nm,
        np.clip(nir_wavelengths, spectral.wavelengths_nm[0], spectral.wavelengths_nm[-1]),
    )
    swir_pred = resample_spectra(
        low_cube_pred,
        spectral.wavelengths_nm,
        np.clip(swir_wavelengths, spectral.wavelengths_nm[0], spectral.wavelengths_nm[-1]),
    )
    swir_reference = spectral.calibrated_swir
    coefficient_rmse = rmse(low_coeff_pred, low_coeff)
    harmonized_rmse = rmse(low_cube_pred, spectral.cube)
    report = {
        "summary": {
            "status": "passed" if coefficient_rmse < 0.08 and harmonized_rmse < 0.08 else "warning",
            "interpretation": "All primary errors are obtained by degrading the high-resolution material field back to the observation grid; they verify consistency, not independent HR-HSI truth.",
        },
        "registration": {},
        "spectral_harmonization": {
            "overlap_rmse_before": spectral.model["overlap_rmse_before"],
            "overlap_rmse_after": spectral.model["overlap_rmse_after"],
            "output_band_count": int(spectral.wavelengths_nm.size),
        },
        "coefficient_observation": {
            "rmse": coefficient_rmse,
            "max_abs": float(np.max(np.abs(low_coeff_pred - low_coeff))),
        },
        "continuous_cube_observation": {
            "rmse": harmonized_rmse,
            "sam_mean_deg": sam_degrees(low_cube_pred, spectral.cube),
            "band_cc_mean": mean_band_correlation(low_cube_pred, spectral.cube),
        },
        "sensor_reprojection": {
            "nir_rmse": rmse(nir_pred, nir_norm),
            "nir_sam_mean_deg": sam_degrees(nir_pred, nir_norm),
            "swir_calibrated_rmse": rmse(swir_pred, swir_reference),
            "swir_calibrated_sam_mean_deg": sam_degrees(swir_pred, swir_reference),
            "note": "SWIR comparison uses the wavelength-dependent calibrated SWIR observation.",
        },
        "spatial": {
            "rgb_material_boundary_correlation": _edge_correlation(refined_coeff, rgb, detail_gain_map),
            "detail_gain_lowres_rmse_from_one": float(np.sqrt(np.mean((low_gain - 1.0) ** 2))),
            "detail_gain_lowres_max_abs_from_one": float(np.max(np.abs(low_gain - 1.0))),
            "additive_detail_lowres_rmse_from_zero": float(np.sqrt(np.mean(low_additive**2))),
            "additive_detail_lowres_max_abs_from_zero": float(np.max(np.abs(low_additive))),
            "note": "This is reported only as a boundary-alignment diagnostic and is not treated as proof of SWIR high-frequency truth.",
        },
        "uncertainty": {
            "mean": float(np.mean(uncertainty_map)),
            "p95": float(np.percentile(uncertainty_map, 95)),
            "max": float(np.max(uncertainty_map)),
        },
    }
    return report
