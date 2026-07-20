"""Sensor-consistency and spectral-safety diagnostics."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import cv2
import numpy as np

from .dataset import normalize_image, normalize_rgb
from .degradation import PsfModel, degrade_coefficients, degrade_spatial_map
from .lowrank import SubspaceModel
from .output import PhysicalClipLimits, reconstruct_modulated
from .spectral import SpectralHarmonizationResult, normalize_radiometry, resample_spectra


DEFAULT_DETAIL_SCALES_PX = (1.2, 2.4, 4.8)

# These are conservative engineering screening bounds, not claims of
# independently verified high-resolution NIR/SWIR truth.  Callers may override
# them in ``build_quality_report`` after preregistering validation-set values.
DEFAULT_DETAIL_TRANSFER_THRESHOLDS: dict[str, float] = {
    "dark_percentile": 20.0,
    "rgb_reliable_coherence_min": 0.60,
    "rgb_reliable_detail_percentile": 30.0,
    "rgb_flat_percentile": 30.0,
    "rho_reliable_min": 0.80,
    "rho_dark_reliable_min": 0.70,
    "beta_min": 0.80,
    "beta_max": 1.20,
    "energy_ratio_min": 0.80,
    "energy_ratio_max": 1.25,
    "orthogonal_residual_ratio_max": 0.35,
    "beta_dark_min": 0.80,
    "beta_dark_max": 1.20,
    "energy_ratio_dark_min": 0.80,
    "energy_ratio_dark_max": 1.25,
    "orthogonal_residual_ratio_dark_max": 0.35,
    "flat_energy_ratio_max": 1.10,
    "gradient_orientation_coherence_min": 0.90,
    "edge_f1_1px_min": 0.85,
    "halo_overshoot_p95_max": 0.05,
}


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


def _resolved_detail_thresholds(overrides: Mapping[str, float] | None) -> dict[str, float]:
    thresholds = DEFAULT_DETAIL_TRANSFER_THRESHOLDS.copy()
    if overrides is not None:
        unknown = set(overrides) - set(thresholds)
        if unknown:
            raise ValueError(f"Unknown detail threshold keys: {sorted(unknown)}")
        thresholds.update({str(key): float(value) for key, value in overrides.items()})
    for key in ("dark_percentile", "rgb_reliable_detail_percentile", "rgb_flat_percentile"):
        if not 0.0 < thresholds[key] < 100.0:
            raise ValueError(f"{key} must lie strictly between 0 and 100")
    for key in (
        "rgb_reliable_coherence_min",
        "rho_reliable_min",
        "rho_dark_reliable_min",
        "gradient_orientation_coherence_min",
        "edge_f1_1px_min",
    ):
        if not 0.0 <= thresholds[key] <= 1.0:
            raise ValueError(f"{key} must lie between 0 and 1")
    if thresholds["beta_min"] > thresholds["beta_max"]:
        raise ValueError("beta_min cannot exceed beta_max")
    if thresholds["energy_ratio_min"] > thresholds["energy_ratio_max"]:
        raise ValueError("energy_ratio_min cannot exceed energy_ratio_max")
    if thresholds["beta_dark_min"] > thresholds["beta_dark_max"]:
        raise ValueError("beta_dark_min cannot exceed beta_dark_max")
    if thresholds["energy_ratio_dark_min"] > thresholds["energy_ratio_dark_max"]:
        raise ValueError("energy_ratio_dark_min cannot exceed energy_ratio_dark_max")
    return thresholds


def _masked_detail_statistics(reference: np.ndarray, candidate: np.ndarray, mask: np.ndarray) -> dict[str, float | int]:
    """Return correlation plus amplitude and non-aligned high-frequency terms."""

    ref = np.asarray(reference, dtype=np.float32)
    out = np.asarray(candidate, dtype=np.float32)
    valid = np.asarray(mask, dtype=bool) & np.isfinite(ref) & np.isfinite(out)
    count = int(np.sum(valid))
    empty = {
        "pixel_count": count,
        "rho": float("nan"),
        "beta": float("nan"),
        "energy_ratio_A": float("nan"),
        "orthogonal_residual_ratio_R_perp": float("nan"),
        "aligned_variance_fraction_rho2": float("nan"),
        "reference_detail_rms": float("nan"),
        "candidate_detail_rms": float("nan"),
        "orthogonal_residual_rms": float("nan"),
    }
    if count < 64:
        return empty
    x = ref[valid].astype(np.float64)
    y = out[valid].astype(np.float64)
    x -= float(np.mean(x))
    y -= float(np.mean(y))
    var_x = float(np.mean(x * x))
    var_y = float(np.mean(y * y))
    covariance = float(np.mean(x * y))
    variance_floor = 1e-12
    if var_x <= variance_floor:
        # A perfectly flat guide with non-flat output is an artifact warning,
        # not an unevaluated ratio.  Keep the ratio finite so screening fails.
        energy_ratio = float(np.sqrt(var_y / variance_floor))
        residual_ratio = energy_ratio
        empty.update(
            {
                "beta": 0.0,
                "energy_ratio_A": energy_ratio,
                "orthogonal_residual_ratio_R_perp": residual_ratio,
                "reference_detail_rms": 0.0,
                "candidate_detail_rms": float(np.sqrt(var_y)),
                "orthogonal_residual_rms": float(np.sqrt(var_y)),
            }
        )
        return empty
    beta = covariance / var_x
    energy_ratio = float(np.sqrt(var_y / var_x))
    residual = y - beta * x
    residual_ratio = float(np.sqrt(np.mean(residual * residual) / var_x))
    if var_y <= variance_floor:
        rho = float("nan")
    else:
        rho = float(np.clip(covariance / np.sqrt(var_x * var_y), -1.0, 1.0))
    return {
        "pixel_count": count,
        "rho": rho,
        "beta": float(beta),
        "energy_ratio_A": energy_ratio,
        "orthogonal_residual_ratio_R_perp": residual_ratio,
        "reference_detail_rms": float(np.sqrt(var_x)),
        "candidate_detail_rms": float(np.sqrt(var_y)),
        "orthogonal_residual_rms": float(np.sqrt(np.mean(residual * residual))),
        "aligned_variance_fraction_rho2": (
            float(rho * rho) if np.isfinite(rho) else float("nan")
        ),
    }


def _gradient_components(image: np.ndarray, epsilon: float = 0.012) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    normalized = normalize_image(np.asarray(image, dtype=np.float32))
    log_image = np.log(normalized + max(float(epsilon), 1e-4))
    smooth = cv2.GaussianBlur(log_image, (0, 0), 0.7, borderType=cv2.BORDER_REFLECT101)
    gx = cv2.Scharr(smooth, cv2.CV_32F, 1, 0)
    gy = cv2.Scharr(smooth, cv2.CV_32F, 0, 1)
    return gx, gy, cv2.magnitude(gx, gy)


def _gradient_structure_metrics(reference: np.ndarray, candidate: np.ndarray, mask: np.ndarray) -> dict[str, float | int]:
    """Measure shared edge location and orientation without assuming polarity."""

    ref_x, ref_y, ref_mag = _gradient_components(reference)
    out_x, out_y, out_mag = _gradient_components(candidate)
    valid = (
        np.asarray(mask, dtype=bool)
        & np.isfinite(ref_mag)
        & np.isfinite(out_mag)
    )
    valid_count = int(np.sum(valid))
    empty = {
        "pixel_count": valid_count,
        "gradient_orientation_coherence_abs_cosine": float("nan"),
        "reference_edge_count": 0,
        "candidate_edge_count": 0,
        "edge_precision_1px": float("nan"),
        "edge_recall_1px": float("nan"),
        "edge_f1_1px": float("nan"),
    }
    if valid_count < 128:
        return empty
    ref_threshold = float(np.percentile(ref_mag[valid], 90.0))
    out_threshold = float(np.percentile(out_mag[valid], 90.0))
    reference_edge = valid & (ref_mag >= max(ref_threshold, 1e-8))
    candidate_edge = valid & (out_mag >= max(out_threshold, 1e-8))
    reference_count = int(np.sum(reference_edge))
    candidate_count = int(np.sum(candidate_edge))
    if reference_count < 16 or candidate_count < 16:
        empty["reference_edge_count"] = reference_count
        empty["candidate_edge_count"] = candidate_count
        return empty
    kernel = np.ones((3, 3), dtype=np.uint8)
    reference_dilated = cv2.dilate(reference_edge.astype(np.uint8), kernel) > 0
    candidate_dilated = cv2.dilate(candidate_edge.astype(np.uint8), kernel) > 0
    precision = float(np.sum(candidate_edge & reference_dilated) / candidate_count)
    recall = float(np.sum(reference_edge & candidate_dilated) / reference_count)
    f1 = float(2.0 * precision * recall / max(precision + recall, 1e-12))
    orientation_mask = reference_edge & (out_mag > max(float(np.percentile(out_mag[valid], 50.0)), 1e-8))
    denominator = ref_mag[orientation_mask] * out_mag[orientation_mask]
    if denominator.size:
        cosine = np.abs(
            (ref_x[orientation_mask] * out_x[orientation_mask]
             + ref_y[orientation_mask] * out_y[orientation_mask])
            / np.maximum(denominator, 1e-8)
        )
        orientation = float(np.mean(np.clip(cosine, 0.0, 1.0)))
    else:
        orientation = float("nan")
    return {
        "pixel_count": valid_count,
        "gradient_orientation_coherence_abs_cosine": orientation,
        "reference_edge_count": reference_count,
        "candidate_edge_count": candidate_count,
        "edge_precision_1px": precision,
        "edge_recall_1px": recall,
        "edge_f1_1px": f1,
    }


def _sample_edge_profiles(
    image: np.ndarray,
    y: np.ndarray,
    x: np.ndarray,
    normal_y: np.ndarray,
    normal_x: np.ndarray,
    offsets: np.ndarray,
) -> np.ndarray:
    map_x = x[:, None].astype(np.float32) + normal_x[:, None].astype(np.float32) * offsets[None, :]
    map_y = y[:, None].astype(np.float32) + normal_y[:, None].astype(np.float32) * offsets[None, :]
    return cv2.remap(
        np.asarray(image, dtype=np.float32),
        map_x,
        map_y,
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT101,
    )


def _edge_width_10_90(profile: np.ndarray, offsets: np.ndarray) -> float:
    values = np.asarray(profile, dtype=np.float32)
    left = float(np.mean(values[:6]))
    right = float(np.mean(values[-6:]))
    step = right - left
    if abs(step) <= 1e-8:
        return float("nan")
    normalized = (values - left) / step
    monotonic = np.maximum.accumulate(normalized)

    def crossing(level: float) -> float:
        indices = np.flatnonzero(monotonic >= level)
        if not indices.size:
            return float("nan")
        index = int(indices[0])
        if index == 0:
            return float(offsets[0])
        delta = float(monotonic[index] - monotonic[index - 1])
        if abs(delta) <= 1e-8:
            return float(offsets[index])
        fraction = float((level - monotonic[index - 1]) / delta)
        return float(offsets[index - 1] + fraction * (offsets[index] - offsets[index - 1]))

    low = crossing(0.1)
    high = crossing(0.9)
    return float(high - low) if np.isfinite(low) and np.isfinite(high) else float("nan")


def _halo_overshoot_proxy(
    reference: np.ndarray,
    candidate: np.ndarray,
    mask: np.ndarray,
    *,
    max_edges: int = 4096,
) -> dict[str, float | int | str]:
    """Natural-edge overshoot proxy; formal MTF still requires a calibrated edge target."""

    def robust_affine_without_clipping(image: np.ndarray) -> np.ndarray:
        values = np.asarray(image, dtype=np.float32)
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            return np.zeros_like(values, dtype=np.float32)
        low, high = np.percentile(finite, [2.0, 98.0])
        return ((values - low) / max(float(high - low), 1e-6)).astype(np.float32)

    # Do not clip the robust normalization: values outside the central range
    # are precisely the overshoot/undershoot signal this diagnostic measures.
    ref = robust_affine_without_clipping(reference)
    out = robust_affine_without_clipping(candidate)
    smooth = cv2.GaussianBlur(ref, (0, 0), 0.8, borderType=cv2.BORDER_REFLECT101)
    gx = cv2.Scharr(smooth, cv2.CV_32F, 1, 0)
    gy = cv2.Scharr(smooth, cv2.CV_32F, 0, 1)
    magnitude = cv2.magnitude(gx, gy)
    valid = np.asarray(mask, dtype=bool) & np.isfinite(magnitude)
    valid[:6, :] = False
    valid[-6:, :] = False
    valid[:, :6] = False
    valid[:, -6:] = False
    if int(np.sum(valid)) < 128:
        return {"edge_profile_count": 0, "status": "insufficient_support"}
    threshold = float(np.percentile(magnitude[valid], 97.0))
    y, x = np.where(valid & (magnitude >= max(threshold, 1e-8)))
    thinning = ((y % 3) == 0) & ((x % 3) == 0)
    y, x = y[thinning], x[thinning]
    if y.size > max_edges:
        indices = np.linspace(0, y.size - 1, max_edges).round().astype(np.int64)
        y, x = y[indices], x[indices]
    if y.size < 32:
        return {"edge_profile_count": int(y.size), "status": "insufficient_support"}
    normal_x = gx[y, x] / np.maximum(magnitude[y, x], 1e-8)
    normal_y = gy[y, x] / np.maximum(magnitude[y, x], 1e-8)
    offsets = np.linspace(-4.0, 4.0, 33, dtype=np.float32)
    reference_profiles = _sample_edge_profiles(ref, y, x, normal_y, normal_x, offsets)
    candidate_profiles = _sample_edge_profiles(out, y, x, normal_y, normal_x, offsets)

    def normalized_profiles(profiles: np.ndarray, minimum_step: float) -> tuple[np.ndarray, np.ndarray]:
        left = np.mean(profiles[:, :6], axis=1)
        right = np.mean(profiles[:, -6:], axis=1)
        step = right - left
        accepted = np.abs(step) >= minimum_step
        normalized = (profiles - left[:, None]) / np.where(np.abs(step) > 1e-8, step, 1.0)[:, None]
        return normalized, accepted

    normalized_reference, reference_ok = normalized_profiles(reference_profiles, 0.08)
    normalized_candidate, candidate_ok = normalized_profiles(candidate_profiles, 0.03)
    accepted = reference_ok & candidate_ok
    count = int(np.sum(accepted))
    if count < 32:
        return {"edge_profile_count": count, "status": "insufficient_support"}
    ref_profiles = normalized_reference[accepted]
    out_profiles = normalized_candidate[accepted]
    overshoot = np.maximum(0.0, np.max(out_profiles, axis=1) - 1.0)
    undershoot = np.maximum(0.0, -np.min(out_profiles, axis=1))
    halo = overshoot + undershoot
    ref_median = np.median(ref_profiles, axis=0)
    out_median = np.median(out_profiles, axis=0)
    reference_width = _edge_width_10_90(ref_median, offsets)
    candidate_width = _edge_width_10_90(out_median, offsets)
    return {
        "edge_profile_count": count,
        "status": "diagnostic_only",
        "overshoot_plus_undershoot_p50_edge_step": float(np.percentile(halo, 50.0)),
        "overshoot_plus_undershoot_p95_edge_step": float(np.percentile(halo, 95.0)),
        "reference_edge_width_10_90_px": reference_width,
        "candidate_edge_width_10_90_px": candidate_width,
        "edge_width_ratio": (
            float(candidate_width / reference_width)
            if np.isfinite(reference_width) and reference_width > 1e-8 and np.isfinite(candidate_width)
            else float("nan")
        ),
        "note": "Natural-scene, independently normalized edge-profile proxy; use a calibrated slanted edge for publishable MTF/halo claims.",
    }


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


def _masked_correlation(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float:
    aa = np.asarray(a, dtype=np.float32)
    bb = np.asarray(b, dtype=np.float32)
    valid = np.asarray(mask, dtype=bool) & np.isfinite(aa) & np.isfinite(bb)
    if int(np.sum(valid)) < 64:
        return float("nan")
    x = aa[valid] - float(np.mean(aa[valid]))
    y = bb[valid] - float(np.mean(bb[valid]))
    denominator = float(np.sqrt(np.sum(x * x) * np.sum(y * y)))
    return float(np.sum(x * y) / denominator) if denominator > 1e-9 else float("nan")


def _log_high_frequency(
    image: np.ndarray,
    epsilon: float = 0.012,
    *,
    sigma_px: float = 2.4,
    normalize: bool = False,
) -> np.ndarray:
    """Return signed log high-frequency detail without candidate rescaling.

    ``normalize_image`` performs an image-specific 2--98 % stretch.  Applying
    it independently to RGB and every fused band erases the very amplitude
    difference measured by beta/A/R_perp (a ten-times weaker candidate can
    otherwise score as unit amplitude).  The default therefore operates on
    the native normalized RGB/reflectance values.  ``normalize=True`` remains
    available only for explicitly labelled display/legacy diagnostics.
    """

    source = np.asarray(image, dtype=np.float32)
    working = normalize_image(source) if normalize else np.clip(source, 0.0, None)
    log_image = np.log(working + max(float(epsilon), 1e-4))
    base = cv2.GaussianBlur(
        log_image,
        (0, 0),
        max(float(sigma_px), 0.25),
        borderType=cv2.BORDER_REFLECT101,
    )
    return (log_image - base).astype(np.float32)


def _fit_low_resolution_log_relation(
    rgb_low: np.ndarray,
    band_low: np.ndarray,
    *,
    epsilon_rgb: float = 0.012,
    epsilon_band: float = 0.012,
    minimum_abs_correlation: float = 0.20,
    minimum_abs_slope: float = 0.05,
) -> dict[str, float | int | bool | str]:
    """Fit a candidate-independent band/RGB log-contrast relation on LR data.

    The slope is estimated only from observed low-resolution RGB/HSI values
    and is frozen before inspecting the fused HR candidate.  It supplies a
    second, band-supported amplitude reference in addition to the deliberately
    stricter unit-RGB-equivalent diagnostic.  A weak relation is reported as
    unidentifiable rather than being forced to an arbitrary target.
    """

    rgb_arr = np.asarray(rgb_low, dtype=np.float32)
    band_arr = np.asarray(band_low, dtype=np.float32)
    valid = (
        np.isfinite(rgb_arr)
        & np.isfinite(band_arr)
        & (rgb_arr >= 0.0)
        & (band_arr >= 0.0)
    )
    count = int(np.sum(valid))
    result: dict[str, float | int | bool | str] = {
        "pixel_count": count,
        "alpha_log_band_per_log_rgb": float("nan"),
        "intercept_log_band": float("nan"),
        "rho_low_resolution": float("nan"),
        "identifiable": False,
        "status": "insufficient_support",
        "epsilon_rgb": float(epsilon_rgb),
        "epsilon_band": float(epsilon_band),
    }
    if count < 64:
        return result
    x = np.log(np.clip(rgb_arr[valid], 0.0, None) + max(float(epsilon_rgb), 1e-4)).astype(
        np.float64
    )
    y = np.log(np.clip(band_arr[valid], 0.0, None) + max(float(epsilon_band), 1e-4)).astype(
        np.float64
    )
    x_mean = float(np.mean(x))
    y_mean = float(np.mean(y))
    x_centered = x - x_mean
    y_centered = y - y_mean
    var_x = float(np.mean(x_centered * x_centered))
    var_y = float(np.mean(y_centered * y_centered))
    if var_x <= 1e-12 or var_y <= 1e-12:
        result["status"] = "degenerate_low_resolution_signal"
        return result
    covariance = float(np.mean(x_centered * y_centered))
    alpha = covariance / var_x
    rho = float(np.clip(covariance / np.sqrt(var_x * var_y), -1.0, 1.0))
    identifiable = bool(
        abs(rho) >= float(minimum_abs_correlation)
        and abs(alpha) >= float(minimum_abs_slope)
    )
    result.update(
        {
            "alpha_log_band_per_log_rgb": float(alpha),
            "intercept_log_band": float(y_mean - alpha * x_mean),
            "rho_low_resolution": rho,
            "identifiable": identifiable,
            "status": "identifiable" if identifiable else "weak_cross_modal_relation",
        }
    )
    return result


def _rgb_detail_masks(
    rgb: np.ndarray,
    thresholds: Mapping[str, float],
) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, Any]] | None:
    guide = normalize_rgb(rgb)
    luminance = 0.299 * guide[:, :, 0] + 0.587 * guide[:, :, 1] + 0.114 * guide[:, :, 2]
    signal = luminance[np.isfinite(luminance) & (luminance > 0.003)]
    if signal.size < 64:
        return None
    dark_threshold = float(np.percentile(signal, thresholds["dark_percentile"]))
    black_threshold = max(float(np.percentile(signal, 1.0)), 0.003)
    valid_signal = np.isfinite(luminance) & (luminance > black_threshold)
    dark_mask = valid_signal & (luminance <= dark_threshold)

    channel_detail = np.stack(
        [
            _log_high_frequency(guide[:, :, channel], sigma_px=2.4, normalize=False)
            for channel in range(3)
        ],
        axis=2,
    )
    luma_detail = _log_high_frequency(luminance, sigma_px=2.4, normalize=False)
    disagreement = np.mean((channel_detail - luma_detail[:, :, None]) ** 2, axis=2)
    common_power = cv2.GaussianBlur(
        luma_detail * luma_detail,
        (0, 0),
        2.0,
        borderType=cv2.BORDER_REFLECT101,
    )
    disagreement_power = cv2.GaussianBlur(
        disagreement,
        (0, 0),
        2.0,
        borderType=cv2.BORDER_REFLECT101,
    )
    coherence = common_power / np.maximum(common_power + disagreement_power, 1e-10)
    detail_strength = np.sqrt(np.maximum(common_power, 0.0))
    detail_floor = float(
        np.percentile(detail_strength[valid_signal], thresholds["rgb_reliable_detail_percentile"])
    )
    reliable = (
        valid_signal
        & (coherence >= thresholds["rgb_reliable_coherence_min"])
        & (detail_strength >= detail_floor)
    )

    _, _, gradient = _gradient_components(luminance)
    gradient_floor = float(np.percentile(gradient[valid_signal], thresholds["rgb_flat_percentile"]))
    flat_detail_ceiling = float(
        np.percentile(detail_strength[valid_signal], thresholds["rgb_flat_percentile"])
    )
    flat = valid_signal & (gradient <= gradient_floor) & (detail_strength <= flat_detail_ceiling)
    masks = {
        "all_valid": valid_signal,
        "darkest_percentile": dark_mask,
        "reliable_rgb_detail": reliable,
        "dark_reliable_rgb_detail": dark_mask & reliable,
        "rgb_flat": flat,
    }
    summary = {
        "dark_percentile": float(thresholds["dark_percentile"]),
        "dark_threshold_normalized_rgb": dark_threshold,
        "black_threshold_normalized_rgb": black_threshold,
        "rgb_reliable_coherence_min": float(thresholds["rgb_reliable_coherence_min"]),
        "rgb_reliable_detail_floor": detail_floor,
        "rgb_flat_gradient_ceiling": gradient_floor,
        "rgb_flat_detail_ceiling": flat_detail_ceiling,
        "pixel_counts": {name: int(np.sum(mask)) for name, mask in masks.items()},
        "coherence_mean_all_valid": float(np.mean(coherence[valid_signal])),
        "coherence_mean_dark": (
            float(np.mean(coherence[dark_mask])) if np.any(dark_mask) else float("nan")
        ),
        "definitions": {
            "reliable_rgb_detail": "Valid RGB signal with locally coherent cross-channel log detail above the configured detail percentile.",
            "dark_reliable_rgb_detail": "Intersection of the darkest configured RGB percentile and the reliable-detail mask.",
            "rgb_flat": "Low RGB gradient and low coherent-detail magnitude; fused high frequency here is only an artifact/noise screening proxy.",
        },
    }
    return luminance.astype(np.float32), masks, summary


def _screening_check(
    name: str,
    value: float,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> dict[str, Any]:
    finite = bool(np.isfinite(value))
    passed = finite
    if finite and minimum is not None:
        passed = passed and value >= minimum
    if finite and maximum is not None:
        passed = passed and value <= maximum
    return {
        "metric": name,
        "value": float(value),
        "minimum": minimum,
        "maximum": maximum,
        "evaluated": finite,
        "within_bound": bool(passed) if finite else None,
    }


def _detail_screening(
    multiscale: Mapping[str, Mapping[str, Mapping[str, float | int]]],
    gradient: Mapping[str, Mapping[str, float | int]],
    halo: Mapping[str, float | int | str],
    thresholds: Mapping[str, float],
    selected_scale_key: str,
) -> dict[str, Any]:
    scale_metrics = multiscale[selected_scale_key]
    reliable = scale_metrics["reliable_rgb_detail"]
    dark = scale_metrics["dark_reliable_rgb_detail"]
    flat = scale_metrics["rgb_flat"]
    reliable_gradient = gradient["reliable_rgb_detail"]
    checks = [
        _screening_check(
            "rho_reliable",
            float(reliable["rho"]),
            minimum=thresholds["rho_reliable_min"],
        ),
        _screening_check(
            "rho_dark_reliable",
            float(dark["rho"]),
            minimum=thresholds["rho_dark_reliable_min"],
        ),
        _screening_check(
            "beta_reliable",
            float(reliable["beta"]),
            minimum=thresholds["beta_min"],
            maximum=thresholds["beta_max"],
        ),
        _screening_check(
            "energy_ratio_A_reliable",
            float(reliable["energy_ratio_A"]),
            minimum=thresholds["energy_ratio_min"],
            maximum=thresholds["energy_ratio_max"],
        ),
        _screening_check(
            "orthogonal_residual_ratio_R_perp_reliable",
            float(reliable["orthogonal_residual_ratio_R_perp"]),
            maximum=thresholds["orthogonal_residual_ratio_max"],
        ),
        _screening_check(
            "beta_dark_reliable",
            float(dark["beta"]),
            minimum=thresholds["beta_dark_min"],
            maximum=thresholds["beta_dark_max"],
        ),
        _screening_check(
            "energy_ratio_A_dark_reliable",
            float(dark["energy_ratio_A"]),
            minimum=thresholds["energy_ratio_dark_min"],
            maximum=thresholds["energy_ratio_dark_max"],
        ),
        _screening_check(
            "orthogonal_residual_ratio_R_perp_dark_reliable",
            float(dark["orthogonal_residual_ratio_R_perp"]),
            maximum=thresholds["orthogonal_residual_ratio_dark_max"],
        ),
        _screening_check(
            "energy_ratio_A_rgb_flat",
            float(flat["energy_ratio_A"]),
            maximum=thresholds["flat_energy_ratio_max"],
        ),
        _screening_check(
            "gradient_orientation_coherence_reliable",
            float(reliable_gradient["gradient_orientation_coherence_abs_cosine"]),
            minimum=thresholds["gradient_orientation_coherence_min"],
        ),
        _screening_check(
            "edge_f1_1px_reliable",
            float(reliable_gradient["edge_f1_1px"]),
            minimum=thresholds["edge_f1_1px_min"],
        ),
        _screening_check(
            "halo_overshoot_p95",
            (
                float(halo.get("overshoot_plus_undershoot_p95_edge_step", float("nan")))
                if halo.get("status") in {"calibrated_target", "synthetic_truth"}
                else float("nan")
            ),
            maximum=thresholds["halo_overshoot_p95_max"],
        ),
    ]
    optional_checks = {"halo_overshoot_p95"}
    required = [
        check for check in checks if str(check["metric"]) not in optional_checks
    ]
    evaluated = [check for check in required if check["evaluated"]]
    if not evaluated or len(evaluated) != len(required):
        screening_status = "insufficient_support"
    elif all(bool(check["within_bound"]) for check in evaluated):
        screening_status = "within_conservative_screening_bounds"
    else:
        screening_status = "outside_conservative_screening_bounds"
    return {
        "screening_status": screening_status,
        "selected_scale": selected_scale_key,
        "checks": {str(check["metric"]): check for check in checks},
        "claim_status": "same_data_rgb_guide_diagnostic_not_independent_swir_truth",
        "note": (
            "Passing these checks cannot establish high-resolution NIR/SWIR truth because RGB supplies both the guide and the structural reference. "
            "Natural-edge halo remains diagnostic-only and is excluded from pass/fail unless the input is explicitly labelled calibrated_target or synthetic_truth."
        ),
    }


def _subspace_band_chunk(subspace: SubspaceModel, start: int, stop: int) -> SubspaceModel:
    return SubspaceModel(
        mean_spectrum=np.asarray(subspace.mean_spectrum[start:stop], dtype=np.float32),
        basis=np.asarray(subspace.basis[:, start:stop], dtype=np.float32),
        explained_variance_ratio=np.asarray(subspace.explained_variance_ratio, dtype=np.float32),
        clip_min=np.asarray(subspace.clip_min[start:stop], dtype=np.float32),
        clip_max=np.asarray(subspace.clip_max[start:stop], dtype=np.float32),
        representation=str(subspace.representation),
        fit_metadata=dict(subspace.fit_metadata),
    )


def _degrade_final_modulated_cube(
    refined_coeff: np.ndarray,
    subspace: SubspaceModel,
    psf: PsfModel,
    detail_gain_map: np.ndarray | None,
    additive_detail_map: np.ndarray | None,
    additive_spectral_scale: np.ndarray | None,
    *,
    physical_clip_limits: PhysicalClipLimits = (0.0, None),
    chunk_bands: int = 8,
    retained_band_indices: Sequence[int] = (),
) -> tuple[np.ndarray, dict[int, np.ndarray]]:
    """Degrade the actual final HR product, including modulation and one clip.

    Bands are reconstructed in small chunks to avoid allocating the complete
    high-resolution cube.  The order is identical to ``output``:
    unbounded subspace reconstruction, gain, additive detail, one physical or
    configured clip, then the sensor PSF/downsampling operator.
    """

    coefficients = np.asarray(refined_coeff, dtype=np.float32)
    if coefficients.shape[:2] != tuple(psf.high_shape):
        raise ValueError(
            f"Coefficient grid {coefficients.shape[:2]} does not match PSF high_shape {psf.high_shape}"
        )
    band_count = int(subspace.basis.shape[1])
    chunk_size = max(1, int(chunk_bands))
    scales = None if additive_spectral_scale is None else np.asarray(additive_spectral_scale, dtype=np.float32)
    if scales is not None and scales.size != band_count:
        raise ValueError(
            f"additive_spectral_scale has {scales.size} values for {band_count} bands"
        )
    retained = {int(index) for index in retained_band_indices}
    if any(index < 0 or index >= band_count for index in retained):
        raise ValueError("retained_band_indices contains an out-of-range band")
    low_cube = np.empty(tuple(psf.low_shape) + (band_count,), dtype=np.float32)
    retained_bands: dict[int, np.ndarray] = {}
    for start in range(0, band_count, chunk_size):
        stop = min(band_count, start + chunk_size)
        chunk_subspace = _subspace_band_chunk(subspace, start, stop)
        chunk_scales = None if scales is None else scales[start:stop]
        final_hr_chunk = reconstruct_modulated(
            coefficients,
            chunk_subspace,
            detail_gain=detail_gain_map,
            additive_detail=additive_detail_map,
            additive_spectral_scale=chunk_scales,
            physical_clip_limits=physical_clip_limits,
        )
        for local_index, band_index in enumerate(range(start, stop)):
            final_band = final_hr_chunk[:, :, local_index]
            low_cube[:, :, band_index] = degrade_spatial_map(final_band, psf)
            if band_index in retained:
                retained_bands[band_index] = np.asarray(final_band, dtype=np.float32).copy()
        del final_hr_chunk
    return low_cube, retained_bands


def _selected_band_detail_metrics(
    refined_coeff: np.ndarray,
    subspace: SubspaceModel,
    rgb: np.ndarray,
    wavelengths_nm: np.ndarray,
    detail_gain_map: np.ndarray | None,
    additive_detail_map: np.ndarray | None,
    additive_spectral_scale: np.ndarray | None,
    *,
    final_hr_bands: Mapping[int, np.ndarray] | None = None,
    observed_low_cube: np.ndarray | None = None,
    psf: PsfModel | None = None,
    thresholds: Mapping[str, float] | None = None,
    scales_px: Sequence[float] = DEFAULT_DETAIL_SCALES_PX,
    physical_clip_limits: PhysicalClipLimits = (0.0, None),
) -> dict[str, Any]:
    resolved_thresholds = _resolved_detail_thresholds(thresholds)
    mask_result = _rgb_detail_masks(rgb, resolved_thresholds)
    if mask_result is None:
        return {"status": "insufficient_rgb_signal"}
    luminance, masks, mask_summary = mask_result
    requested_scales = tuple(float(scale) for scale in scales_px)
    if not requested_scales:
        raise ValueError("At least one detail scale is required")
    if any(scale <= 0.0 for scale in requested_scales):
        raise ValueError("Detail scales must be positive")
    selected_scale = min(requested_scales, key=lambda value: abs(value - 2.4))
    selected_scale_key = f"sigma_{selected_scale:g}px"
    # Native log-reflectance / log-RGB units are deliberately retained.  No
    # candidate-specific percentile stretch is allowed in amplitude metrics.
    rgb_detail_by_scale = {
        scale: _log_high_frequency(luminance, sigma_px=scale, normalize=False)
        for scale in requested_scales
    }
    rgb_detail_selected = rgb_detail_by_scale[selected_scale]
    gain = None if detail_gain_map is None else np.asarray(detail_gain_map, dtype=np.float32)
    additive = None if additive_detail_map is None else np.asarray(additive_detail_map, dtype=np.float32)
    additive_scales = None if additive_spectral_scale is None else np.asarray(additive_spectral_scale, dtype=np.float32)
    wavelengths = np.asarray(wavelengths_nm, dtype=np.float32)
    observed = None if observed_low_cube is None else np.asarray(observed_low_cube, dtype=np.float32)
    if observed is not None and observed.shape[2] != wavelengths.size:
        raise ValueError(
            f"observed_low_cube has {observed.shape[2]} bands for {wavelengths.size} wavelengths"
        )
    if (observed is None) != (psf is None):
        raise ValueError("observed_low_cube and psf must be supplied together")
    rgb_low = None if psf is None else degrade_spatial_map(luminance, psf)
    old_correlations: dict[str, Any] = {}
    multiscale_by_band: dict[str, Any] = {}
    for requested_nm in (900.0, 1650.0, 2200.0):
        index = int(np.argmin(np.abs(wavelengths - requested_nm)))
        if final_hr_bands is not None and index in final_hr_bands:
            band = np.asarray(final_hr_bands[index], dtype=np.float32)
        else:
            band_subspace = _subspace_band_chunk(subspace, index, index + 1)
            band_scale = None if additive_scales is None else additive_scales[index : index + 1]
            band = reconstruct_modulated(
                refined_coeff,
                band_subspace,
                detail_gain=gain,
                additive_detail=additive,
                additive_spectral_scale=band_scale,
                physical_clip_limits=physical_clip_limits,
            )[:, :, 0]
        label = f"{float(wavelengths[index]):.1f}nm"
        band_detail_selected = _log_high_frequency(
            band, sigma_px=selected_scale, normalize=False
        )
        old_correlations[label] = {
            "all_valid": _masked_correlation(
                rgb_detail_selected,
                band_detail_selected,
                masks["all_valid"],
            ),
            "darkest_20pct": _masked_correlation(
                rgb_detail_selected,
                band_detail_selected,
                masks["darkest_percentile"],
            ),
        }
        scale_metrics: dict[str, Any] = {}
        for scale in requested_scales:
            scale_key = f"sigma_{scale:g}px"
            band_detail = _log_high_frequency(band, sigma_px=scale, normalize=False)
            scale_metrics[scale_key] = {
                name: _masked_detail_statistics(rgb_detail_by_scale[scale], band_detail, mask)
                for name, mask in masks.items()
            }
        low_relation: dict[str, float | int | bool | str]
        calibrated_scale_metrics: dict[str, Any] = {}
        if observed is not None and rgb_low is not None:
            low_relation = _fit_low_resolution_log_relation(
                rgb_low,
                observed[:, :, index],
            )
            alpha = float(low_relation["alpha_log_band_per_log_rgb"])
            identifiable = bool(low_relation["identifiable"])
            for scale in requested_scales:
                scale_key = f"sigma_{scale:g}px"
                if identifiable and np.isfinite(alpha):
                    fixed_reference = alpha * rgb_detail_by_scale[scale]
                    band_detail = _log_high_frequency(
                        band, sigma_px=scale, normalize=False
                    )
                    calibrated_scale_metrics[scale_key] = {
                        name: _masked_detail_statistics(fixed_reference, band_detail, mask)
                        for name, mask in masks.items()
                    }
                else:
                    calibrated_scale_metrics[scale_key] = {
                        name: {
                            "pixel_count": int(np.sum(mask)),
                            "rho": float("nan"),
                            "beta": float("nan"),
                            "energy_ratio_A": float("nan"),
                            "orthogonal_residual_ratio_R_perp": float("nan"),
                            "aligned_variance_fraction_rho2": float("nan"),
                            "reference_detail_rms": float("nan"),
                            "candidate_detail_rms": float("nan"),
                            "orthogonal_residual_rms": float("nan"),
                            "status": "unidentifiable_low_resolution_relation",
                        }
                        for name, mask in masks.items()
                    }
        else:
            low_relation = {
                "status": "not_evaluated",
                "identifiable": False,
                "reason": "observed_low_cube_and_psf_not_supplied",
            }
        gradient_metrics = {
            name: _gradient_structure_metrics(luminance, band, masks[name])
            for name in ("all_valid", "reliable_rgb_detail", "dark_reliable_rgb_detail")
        }
        halo_mask = masks["reliable_rgb_detail"]
        if int(np.sum(halo_mask)) < 128:
            halo_mask = masks["all_valid"]
        halo_metrics = _halo_overshoot_proxy(luminance, band, halo_mask)
        screening = _detail_screening(
            scale_metrics,
            gradient_metrics,
            halo_metrics,
            resolved_thresholds,
            selected_scale_key,
        )
        multiscale_by_band[label] = {
            "multiscale_log_high_frequency": scale_metrics,
            "lr_calibrated_log_high_frequency": calibrated_scale_metrics,
            "low_resolution_log_relation": low_relation,
            "gradient_and_edge": gradient_metrics,
            "halo_and_edge_spread_proxy": halo_metrics,
            "conservative_screening": screening,
        }
    return {
        "status": "diagnostic_only",
        "dark_threshold_normalized_rgb": mask_summary["dark_threshold_normalized_rgb"],
        "dark_pixel_count": mask_summary["pixel_counts"]["darkest_percentile"],
        # Compatibility field retained verbatim in shape and meaning.
        "log_high_frequency_correlation": old_correlations,
        "mask_summary": mask_summary,
        "scales_sigma_px": [float(scale) for scale in scales_px],
        "selected_scale_for_screening": selected_scale_key,
        "bands": multiscale_by_band,
        "thresholds": dict(resolved_thresholds),
        "metric_definitions": {
            "detail_normalization": "Native normalized RGB and reflectance values with fixed epsilon; no per-image/candidate percentile stretch.",
            "rho": "Pearson correlation of signed native-log high-frequency detail; invariant to amplitude.",
            "beta": "Least-squares native-log detail amplitude cov(RGB, band) / var(RGB). Unit target is an RGB-equivalent relative-contrast diagnostic, not SWIR truth.",
            "energy_ratio_A": "Native-log band-detail standard deviation divided by native-log RGB-detail standard deviation.",
            "orthogonal_residual_ratio_R_perp": "Standard deviation of band detail after removing beta*RGB detail, normalized by RGB-detail standard deviation.",
            "lr_calibrated_log_high_frequency": "Uses a band-specific alpha fitted only on observed LR log band versus PSF-degraded log RGB; weak relations are marked unidentifiable.",
            "aligned_variance_fraction_rho2": "Squared correlation; a linear same-data alignment fraction, not truth recovery.",
            "gradient_orientation_coherence_abs_cosine": "Absolute gradient-direction cosine on strong RGB edges, allowing wavelength-dependent contrast reversal.",
            "edge_f1_1px": "Symmetric top-decile edge overlap after one-pixel dilation.",
            "halo_overshoot": "Natural-edge independently normalized profile proxy; formal claims require a calibrated target.",
        },
        "note": "RGB/fused-band metrics measure same-data structural transfer, amplitude, excess high frequency, and artifact risk. They are not independent SWIR high-frequency truth and cannot alone support a recovery claim.",
    }


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
    *,
    detail_thresholds: Mapping[str, float] | None = None,
    detail_scales_px: Sequence[float] = DEFAULT_DETAIL_SCALES_PX,
    physical_clip_limits: PhysicalClipLimits = (0.0, None),
    observation_chunk_bands: int = 8,
) -> dict[str, Any]:
    low_coeff_pred = degrade_coefficients(refined_coeff, psf)
    low_gain = (
        np.ones(psf.low_shape, dtype=np.float32)
        if detail_gain_map is None
        else degrade_spatial_map(np.asarray(detail_gain_map, dtype=np.float32), psf)
    )
    if additive_detail_map is None or additive_spectral_scale is None:
        low_additive = np.zeros(psf.low_shape, dtype=np.float32)
    else:
        low_additive = degrade_spatial_map(np.asarray(additive_detail_map, dtype=np.float32), psf)
    wavelengths = np.asarray(spectral.wavelengths_nm, dtype=np.float32)
    selected_indices = {
        int(np.argmin(np.abs(wavelengths - requested_nm)))
        for requested_nm in (900.0, 1650.0, 2200.0)
    }
    low_cube_pred, selected_final_hr_bands = _degrade_final_modulated_cube(
        refined_coeff,
        subspace,
        psf,
        detail_gain_map,
        additive_detail_map,
        additive_spectral_scale,
        physical_clip_limits=physical_clip_limits,
        chunk_bands=observation_chunk_bands,
        retained_band_indices=sorted(selected_indices),
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
    final_observation = {
        "rmse": harmonized_rmse,
        "sam_mean_deg": sam_degrees(low_cube_pred, spectral.cube),
        "band_cc_mean": mean_band_correlation(low_cube_pred, spectral.cube),
        "method": "final_hr_unbounded_subspace_then_gain_then_additive_then_single_clip_then_psf_downsample",
        "physical_clip_limits": (
            None if physical_clip_limits is None else list(physical_clip_limits)
        ),
        "chunk_bands": max(1, int(observation_chunk_bands)),
        "proxy_D_gain_used": False,
        "truth_scope": "low_resolution_observation_consistency_not_independent_hr_hsi_truth",
    }
    report = {
        "summary": {
            "status": "passed" if coefficient_rmse < 0.08 and harmonized_rmse < 0.08 else "warning",
            "status_scope": "low_resolution_observation_consistency_only",
            "independent_hr_hsi_truth_status": "not_evaluated",
            "interpretation": "The final high-resolution product is reconstructed without fitted-quantile clipping, fully modulated, physically/configurably clipped once, and then degraded through the sensor PSF to the observation grid. These errors verify observation consistency, not independent HR-HSI truth.",
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
        # Compatibility name retained; values now use the actual final-product
        # forward path instead of a D(gain) coefficient-domain proxy.
        "continuous_cube_observation": dict(final_observation),
        "final_hr_product_observation": dict(final_observation),
        "sensor_reprojection": {
            "nir_rmse": rmse(nir_pred, nir_norm),
            "nir_sam_mean_deg": sam_degrees(nir_pred, nir_norm),
            "swir_calibrated_rmse": rmse(swir_pred, swir_reference),
            "swir_calibrated_sam_mean_deg": sam_degrees(swir_pred, swir_reference),
            "note": "SWIR comparison uses the wavelength-dependent calibrated SWIR observation.",
        },
        "spatial": {
            "rgb_material_boundary_correlation": _edge_correlation(refined_coeff, rgb, detail_gain_map),
            "band_detail_by_brightness": _selected_band_detail_metrics(
                refined_coeff,
                subspace,
                rgb,
                spectral.wavelengths_nm,
                detail_gain_map,
                additive_detail_map,
                additive_spectral_scale,
                final_hr_bands=selected_final_hr_bands,
                observed_low_cube=spectral.cube,
                psf=psf,
                thresholds=detail_thresholds,
                scales_px=detail_scales_px,
                physical_clip_limits=physical_clip_limits,
            ),
            "detail_gain_lowres_rmse_from_one": float(np.sqrt(np.mean((low_gain - 1.0) ** 2))),
            "detail_gain_lowres_max_abs_from_one": float(np.max(np.abs(low_gain - 1.0))),
            "additive_detail_lowres_rmse_from_zero": float(np.sqrt(np.mean(low_additive**2))),
            "additive_detail_lowres_max_abs_from_zero": float(np.max(np.abs(low_additive))),
            "note": "RGB detail metrics and conservative bounds are same-data structural-transfer screens only. They cannot establish independent SWIR high-frequency truth; final-product observation consistency is reported separately.",
        },
        "uncertainty": {
            "mean": float(np.mean(uncertainty_map)),
            "p95": float(np.percentile(uncertainty_map, 95)),
            "max": float(np.max(uncertainty_map)),
        },
    }
    return report
