"""RGB-guided material-field refinement with sensor-consistency backprojection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from .config import FusionConfig
from .dataset import normalize_rgb
from .degradation import PsfModel, degrade_coefficients, degrade_spatial_map, upsample_coefficients


@dataclass(slots=True)
class CoefficientFusionResult:
    coefficients: np.ndarray
    uncertainty_map: np.ndarray
    detail_gain_map: np.ndarray
    additive_detail_map: np.ndarray
    history: list[dict[str, float]]
    details: dict[str, Any]


def _rgb_weights(rgb: np.ndarray, sigma: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    guide = normalize_rgb(rgb)
    dx = guide[:, 1:, :] - guide[:, :-1, :]
    dy = guide[1:, :, :] - guide[:-1, :, :]
    sigma2 = max(float(sigma) ** 2, 1e-6)
    wx = np.exp(-np.sum(dx * dx, axis=2) / (2.0 * sigma2)).astype(np.float32)
    wy = np.exp(-np.sum(dy * dy, axis=2) / (2.0 * sigma2)).astype(np.float32)
    brightness = np.mean(guide, axis=2)
    confidence = np.clip((brightness - 0.015) / 0.08, 0.0, 1.0) * np.clip((0.995 - brightness) / 0.10, 0.0, 1.0)
    saturation = np.max(guide, axis=2) - np.min(guide, axis=2)
    confidence *= np.clip(1.25 - saturation, 0.15, 1.0)
    wx *= np.minimum(confidence[:, 1:], confidence[:, :-1])
    wy *= np.minimum(confidence[1:, :], confidence[:-1, :])
    return wx, wy, confidence.astype(np.float32)


def _gradient_magnitude(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image, dtype=np.float32)
    gx = cv2.Sobel(arr, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(arr, cv2.CV_32F, 0, 1, ksize=3)
    return cv2.magnitude(gx, gy)


def _rgb_detail_features(
    rgb: np.ndarray,
    small_sigma: float,
    large_sigma: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return decorrelated RGB low-frequency features, detail residuals, and edges."""

    guide = normalize_rgb(rgb)
    guide = np.log1p(4.0 * guide) / np.log(5.0)
    luminance = 0.299 * guide[:, :, 0] + 0.587 * guide[:, :, 1] + 0.114 * guide[:, :, 2]
    features = np.stack(
        [
            luminance,
            guide[:, :, 0] - guide[:, :, 1],
            guide[:, :, 2] - guide[:, :, 1],
        ],
        axis=2,
    ).astype(np.float32)
    small = np.empty_like(features)
    large = np.empty_like(features)
    for channel in range(features.shape[2]):
        small[:, :, channel] = cv2.GaussianBlur(
            features[:, :, channel],
            (0, 0),
            sigmaX=small_sigma,
            sigmaY=small_sigma,
            borderType=cv2.BORDER_REFLECT101,
        )
        large[:, :, channel] = cv2.GaussianBlur(
            features[:, :, channel],
            (0, 0),
            sigmaX=large_sigma,
            sigmaY=large_sigma,
            borderType=cv2.BORDER_REFLECT101,
        )
    detail = (features - small) + 0.55 * (small - large)
    edge = _gradient_magnitude(luminance.astype(np.float32))
    edge_scale = max(float(np.percentile(edge[np.isfinite(edge)], 98.0)), 1e-6)
    return large, detail.astype(np.float32), np.clip(edge / edge_scale, 0.0, 1.0).astype(np.float32)


def _project_coefficients_to_observation_nullspace(
    detail: np.ndarray,
    psf: PsfModel,
    iterations: int,
) -> np.ndarray:
    projected = np.asarray(detail, dtype=np.float32).copy()
    for _ in range(max(0, int(iterations))):
        low = degrade_coefficients(projected, psf)
        projected -= upsample_coefficients(low, projected.shape[:2])
    return projected.astype(np.float32)


def _project_map_to_observation_nullspace(
    detail: np.ndarray,
    psf: PsfModel,
    iterations: int,
) -> np.ndarray:
    projected = np.asarray(detail, dtype=np.float32).copy()
    for _ in range(max(0, int(iterations))):
        low = degrade_spatial_map(projected, psf)
        projected -= cv2.resize(
            low,
            (projected.shape[1], projected.shape[0]),
            interpolation=cv2.INTER_CUBIC,
        )
    return projected.astype(np.float32)


def inject_coefficient_detail(
    coefficients: np.ndarray,
    low_coeff: np.ndarray,
    rgb: np.ndarray,
    psf: PsfModel,
    rgb_confidence: np.ndarray,
    config: FusionConfig,
    *,
    clip_min: np.ndarray | None = None,
    clip_max: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Inject coefficient-specific RGB detail using signed ridge-regression guides.

    RGB luminance and chromatic contrasts are regressed against each smooth
    material coefficient. Their high-frequency residuals are then transferred
    with coefficient-specific signs, projected into the sensor observation
    near-nullspace, and followed by low-resolution back-projection.
    """

    current = np.asarray(coefficients, dtype=np.float32)
    strength = max(0.0, float(config.coefficient_detail_strength))
    if strength <= 0.0:
        return current.copy(), {
            "enabled": False,
            "coefficient_detail_strength": strength,
            "accepted_components": 0,
        }

    small_sigma = max(0.5, float(config.spatial_detail_small_sigma))
    large_sigma = max(small_sigma + 0.5, float(config.spatial_detail_large_sigma))
    low_features, high_features, _ = _rgb_detail_features(rgb, small_sigma, large_sigma)
    valid = np.isfinite(low_features).all(axis=2) & (np.asarray(rgb_confidence) > 0.12)
    valid_idx = np.flatnonzero(valid)
    if valid_idx.size < 64:
        return current.copy(), {
            "enabled": False,
            "reason": "insufficient_confident_rgb_pixels",
            "coefficient_detail_strength": strength,
            "accepted_components": 0,
        }
    if valid_idx.size > 120000:
        step = max(1, valid_idx.size // 120000)
        valid_idx = valid_idx[::step][:120000]

    feature_flat = low_features.reshape(-1, low_features.shape[2])
    feature_mean = np.mean(feature_flat[valid_idx], axis=0).astype(np.float32)
    feature_scale = np.maximum(np.std(feature_flat[valid_idx], axis=0).astype(np.float32), 1e-4)
    x = (feature_flat[valid_idx] - feature_mean[None, :]) / feature_scale[None, :]
    high_standardized = high_features / feature_scale[None, None, :]
    ridge = max(1e-6, float(config.coefficient_detail_ridge))
    normal = (x.T @ x) / float(x.shape[0]) + ridge * np.eye(x.shape[1], dtype=np.float32)

    material = np.sqrt(np.sum(current * current, axis=2))
    material_edge = _gradient_magnitude(material)
    material_scale = max(float(np.percentile(material_edge[np.isfinite(material_edge)], 96.0)), 1e-6)
    material_edge = np.clip(material_edge / material_scale, 0.0, 1.0)
    support_radius = max(1, int(config.boundary_support_radius))
    support_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (2 * support_radius + 1, 2 * support_radius + 1),
    )
    material_support = cv2.dilate(material_edge.astype(np.float32), support_kernel)
    support_floor = float(np.clip(config.coefficient_detail_support_floor, 0.0, 1.0))
    spatial_gate = np.clip(rgb_confidence, 0.0, 1.0) * (
        support_floor + (1.0 - support_floor) * material_support
    )

    low_std = np.maximum(np.std(low_coeff, axis=(0, 1)).astype(np.float32), 1e-5)
    delta = np.zeros_like(current, dtype=np.float32)
    correlations: list[float] = []
    reliabilities: list[float] = []
    regression_weights: list[list[float]] = []
    min_correlation = float(np.clip(config.coefficient_detail_min_correlation, 0.0, 0.95))
    target_amplitudes: list[float] = []
    for component in range(current.shape[2]):
        target = cv2.GaussianBlur(
            current[:, :, component],
            (0, 0),
            sigmaX=large_sigma,
            sigmaY=large_sigma,
            borderType=cv2.BORDER_REFLECT101,
        ).reshape(-1)[valid_idx]
        target = target - float(np.mean(target))
        rhs = (x.T @ target) / float(x.shape[0])
        beta = np.linalg.solve(normal, rhs.astype(np.float32)).astype(np.float32)
        prediction = x @ beta
        denom = float(np.sqrt(np.sum(prediction * prediction) * np.sum(target * target)))
        correlation = float(np.sum(prediction * target) / denom) if denom > 1e-9 else 0.0
        reliability = np.sqrt(
            np.clip(
                (abs(correlation) - min_correlation) / max(0.45 - min_correlation, 0.10),
                0.0,
                1.0,
            )
        )
        raw = np.einsum("...c,c->...", high_standardized, beta, optimize=True).astype(np.float32)
        raw_scale = float(np.percentile(np.abs(raw[np.isfinite(raw)]), 98.0)) if np.isfinite(raw).any() else 0.0
        target_amplitude = strength * float(low_std[component]) * float(reliability)
        if raw_scale > 1e-8 and target_amplitude > 0.0:
            delta[:, :, component] = (
                raw / raw_scale * target_amplitude * spatial_gate
            ).astype(np.float32)
        correlations.append(correlation)
        reliabilities.append(float(reliability))
        regression_weights.append(beta.tolist())
        target_amplitudes.append(target_amplitude)

    delta = _project_coefficients_to_observation_nullspace(
        delta,
        psf,
        config.coefficient_detail_nullspace_iterations,
    )
    clip_sigma = max(0.0, float(config.coefficient_detail_clip_sigma))
    for component in range(delta.shape[2]):
        projected_scale = float(np.percentile(np.abs(delta[:, :, component]), 98.0))
        target_amplitude = target_amplitudes[component]
        if projected_scale > 1e-8 and target_amplitude > 0.0:
            delta[:, :, component] *= target_amplitude / projected_scale
        limit = clip_sigma * float(low_std[component])
        delta[:, :, component] = np.clip(delta[:, :, component], -limit, limit)

    low_detail_before = degrade_coefficients(delta, psf)
    enhanced = current + delta
    if clip_min is not None and clip_max is not None:
        enhanced = np.clip(enhanced, clip_min[None, None, :], clip_max[None, None, :])
    rmse_before = float(np.sqrt(np.mean((degrade_coefficients(enhanced, psf) - low_coeff) ** 2)))
    for _ in range(max(0, int(config.coefficient_detail_back_projection_iterations))):
        residual = low_coeff - degrade_coefficients(enhanced, psf)
        correction = upsample_coefficients(residual, enhanced.shape[:2])
        enhanced += float(config.back_projection_weight) * correction
        if clip_min is not None and clip_max is not None:
            enhanced = np.clip(enhanced, clip_min[None, None, :], clip_max[None, None, :])
    rmse_after = float(np.sqrt(np.mean((degrade_coefficients(enhanced, psf) - low_coeff) ** 2)))
    accepted = int(np.sum(np.asarray(reliabilities) > 0.0))
    return enhanced.astype(np.float32), {
        "enabled": True,
        "coefficient_detail_strength": strength,
        "ridge": ridge,
        "minimum_correlation": min_correlation,
        "support_floor": support_floor,
        "clip_sigma": clip_sigma,
        "nullspace_iterations": int(config.coefficient_detail_nullspace_iterations),
        "back_projection_iterations": int(config.coefficient_detail_back_projection_iterations),
        "component_correlations": correlations,
        "component_reliabilities": reliabilities,
        "component_regression_weights_luminance_rg_bg": regression_weights,
        "accepted_components": accepted,
        "detail_p98_by_component": [
            float(np.percentile(np.abs(delta[:, :, component]), 98.0))
            for component in range(delta.shape[2])
        ],
        "detail_lowres_rmse": float(np.sqrt(np.mean(low_detail_before * low_detail_before))),
        "coefficient_rmse_before_detail_back_projection": rmse_before,
        "coefficient_rmse_after_detail_back_projection": rmse_after,
    }


def _material_guided_anchor(initial: np.ndarray, rgb: np.ndarray, config: FusionConfig) -> tuple[np.ndarray, np.ndarray]:
    guide = normalize_rgb(rgb)
    guide_u8 = (guide * 255.0).round().astype(np.uint8)
    guide_smooth_u8 = cv2.bilateralFilter(guide_u8, d=9, sigmaColor=28, sigmaSpace=7)
    guide_smooth = guide_smooth_u8.astype(np.float32) / 255.0
    gray = cv2.cvtColor(guide_smooth_u8, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    rgb_edge = _gradient_magnitude(gray)
    material_norm = np.sqrt(np.sum(initial**2, axis=2))
    hsi_edge = _gradient_magnitude(material_norm)
    rgb_scale = max(float(np.percentile(rgb_edge, 98)), 1e-6)
    hsi_scale = max(float(np.percentile(hsi_edge, 96)), 1e-6)
    rgb_edge = np.clip(rgb_edge / rgb_scale, 0.0, 1.0)
    hsi_edge = np.clip(hsi_edge / hsi_scale, 0.0, 1.0)
    support_radius = max(1, int(config.boundary_support_radius))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * support_radius + 1, 2 * support_radius + 1))
    hsi_support = cv2.dilate(hsi_edge, kernel)
    gate = np.clip(rgb_edge * (0.10 + 0.90 * hsi_support), 0.0, 1.0).astype(np.float32)
    anchor = initial.copy()
    for k in range(initial.shape[2]):
        if hasattr(cv2, "ximgproc"):
            guided = cv2.ximgproc.guidedFilter(
                guide=guide_smooth,
                src=initial[:, :, k],
                radius=max(1, int(config.guided_radius)),
                eps=float(config.guided_epsilon),
            )
        else:
            guided = cv2.bilateralFilter(
                initial[:, :, k],
                d=7,
                sigmaColor=0.08,
                sigmaSpace=max(1, int(config.guided_radius)),
            )
        anchor[:, :, k] = initial[:, :, k] + float(config.boundary_injection_strength) * gate * (guided - initial[:, :, k])
    return anchor.astype(np.float32), gate


def _diffusion_step(
    current: np.ndarray,
    anchor: np.ndarray,
    wx: np.ndarray,
    wy: np.ndarray,
    strength: float,
    anchor_weight: float,
) -> np.ndarray:
    h, w, rank = current.shape
    updated = np.empty_like(current)
    for k in range(rank):
        field = current[:, :, k]
        numerator = float(anchor_weight) * anchor[:, :, k]
        denominator = np.full((h, w), float(anchor_weight), dtype=np.float32)
        numerator[:, :-1] += strength * wx * field[:, 1:]
        numerator[:, 1:] += strength * wx * field[:, :-1]
        denominator[:, :-1] += strength * wx
        denominator[:, 1:] += strength * wx
        numerator[:-1, :] += strength * wy * field[1:, :]
        numerator[1:, :] += strength * wy * field[:-1, :]
        denominator[:-1, :] += strength * wy
        denominator[1:, :] += strength * wy
        updated[:, :, k] = numerator / np.maximum(denominator, 1e-6)
    return updated


def _spectral_shape_preserving_detail_gain(
    rgb: np.ndarray,
    psf: PsfModel,
    rgb_confidence: np.ndarray,
    config: FusionConfig,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """Create a common per-pixel gain for every band from RGB high-frequency detail."""

    small_sigma = max(0.5, float(config.spatial_detail_small_sigma))
    large_sigma = max(small_sigma + 0.5, float(config.spatial_detail_large_sigma))
    _, rgb_detail, edge = _rgb_detail_features(rgb, small_sigma, large_sigma)
    detail = rgb_detail[:, :, 0]
    valid = np.isfinite(detail)
    detail_scale = float(np.percentile(np.abs(detail[valid]), 95.0)) if valid.any() else 1.0
    detail = np.tanh(detail / max(detail_scale, 1e-6)).astype(np.float32)
    texture_floor = float(np.clip(config.spatial_detail_texture_floor, 0.0, 1.0))
    detail *= (texture_floor + (1.0 - texture_floor) * edge) * np.clip(rgb_confidence, 0.0, 1.0)

    iterations = max(0, int(config.spatial_detail_nullspace_iterations))
    detail = _project_map_to_observation_nullspace(detail, psf, iterations)
    projected_scale = float(np.percentile(np.abs(detail[np.isfinite(detail)]), 98.0)) if np.isfinite(detail).any() else 1.0
    detail = np.clip(detail / max(projected_scale, 1e-6), -1.0, 1.0).astype(np.float32)
    additive_detail = detail.copy()
    low_limit, high_limit = map(float, config.spatial_detail_gain_limits)
    gain = np.clip(1.0 + float(config.spatial_detail_strength) * detail, low_limit, high_limit).astype(np.float32)
    for _ in range(iterations):
        low_gain = degrade_spatial_map(gain, psf)
        correction = cv2.resize(low_gain - 1.0, (gain.shape[1], gain.shape[0]), interpolation=cv2.INTER_CUBIC)
        gain = np.clip(gain - correction, low_limit, high_limit).astype(np.float32)
    low_gain = degrade_spatial_map(gain, psf)
    return gain, additive_detail, {
        "spatial_detail_strength": float(config.spatial_detail_strength),
        "spatial_detail_small_sigma": small_sigma,
        "spatial_detail_large_sigma": large_sigma,
        "spatial_detail_texture_floor": texture_floor,
        "detail_gain_min": float(np.min(gain)),
        "detail_gain_max": float(np.max(gain)),
        "detail_gain_mean": float(np.mean(gain)),
        "detail_gain_p05": float(np.percentile(gain, 5)),
        "detail_gain_p95": float(np.percentile(gain, 95)),
        "lowres_gain_rmse_from_one": float(np.sqrt(np.mean((low_gain - 1.0) ** 2))),
        "lowres_gain_max_abs_from_one": float(np.max(np.abs(low_gain - 1.0))),
        "spatial_detail_additive_strength": float(config.spatial_detail_additive_strength),
        "additive_detail_lowres_rmse": float(np.sqrt(np.mean(degrade_spatial_map(additive_detail, psf) ** 2))),
    }


def build_additive_spectral_scale(low_cube: np.ndarray, config: FusionConfig) -> np.ndarray:
    """Build a conservative band-adaptive scale for a shared additive detail map."""

    arr = np.asarray(low_cube, dtype=np.float32).reshape(-1, low_cube.shape[2])
    band_std = np.nanstd(arr, axis=0).astype(np.float32)
    band_mean = np.nanmean(arr, axis=0).astype(np.float32)
    scale = max(0.0, float(config.spatial_detail_additive_strength)) * np.maximum(
        max(0.0, float(config.spatial_detail_additive_std_fraction)) * band_std,
        max(0.0, float(config.spatial_detail_additive_mean_fraction)) * np.abs(band_mean),
    )
    scale[~np.isfinite(scale)] = 0.0
    return scale.astype(np.float32)


def refine_coefficients(
    low_coeff: np.ndarray,
    rgb: np.ndarray,
    psf: PsfModel,
    config: FusionConfig,
) -> CoefficientFusionResult:
    if config.refiner.lower() not in {"variational", "none", "bicubic"}:
        if config.refiner.lower() == "neural":
            from .neural import refine_coefficients_neural

            return refine_coefficients_neural(low_coeff, rgb, psf, config)
        raise ValueError(f"Unknown refiner {config.refiner}")
    initial = upsample_coefficients(low_coeff, rgb.shape[:2])
    if config.refiner.lower() in {"none", "bicubic"}:
        residual = degrade_coefficients(initial, psf) - low_coeff
        uncertainty = np.sqrt(
            np.maximum(
                np.mean(upsample_coefficients(residual**2, rgb.shape[:2]), axis=2),
                0.0,
            )
        )
        confidence = _rgb_weights(rgb, config.rgb_edge_sigma)[2]
        low_min = np.quantile(low_coeff, 0.005, axis=(0, 1)).astype(np.float32)
        low_max = np.quantile(low_coeff, 0.995, axis=(0, 1)).astype(np.float32)
        low_std = np.maximum(np.std(low_coeff, axis=(0, 1)).astype(np.float32), 1e-5)
        margin = float(config.coefficient_clip_margin) * np.maximum(low_max - low_min, low_std)
        enhanced, coefficient_detail = inject_coefficient_detail(
            initial,
            low_coeff,
            rgb,
            psf,
            confidence,
            config,
            clip_min=low_min - margin,
            clip_max=low_max + margin,
        )
        detail_gain, additive_detail, detail_details = _spectral_shape_preserving_detail_gain(rgb, psf, confidence, config)
        return CoefficientFusionResult(
            coefficients=enhanced,
            uncertainty_map=uncertainty.astype(np.float32),
            detail_gain_map=detail_gain,
            additive_detail_map=additive_detail,
            history=[],
            details={
                "method": "bicubic_coefficients",
                "coefficient_detail": coefficient_detail,
                "spatial_detail": detail_details,
            },
        )

    anchor, boundary_gate = _material_guided_anchor(initial, rgb, config)
    current = anchor.copy()
    low_min = np.quantile(low_coeff, 0.005, axis=(0, 1)).astype(np.float32)
    low_max = np.quantile(low_coeff, 0.995, axis=(0, 1)).astype(np.float32)
    low_std = np.maximum(np.std(low_coeff, axis=(0, 1)).astype(np.float32), 1e-5)
    margin = float(config.coefficient_clip_margin) * np.maximum(low_max - low_min, low_std)
    clip_min = low_min - margin
    clip_max = low_max + margin
    wx, wy, rgb_confidence = _rgb_weights(rgb, config.rgb_edge_sigma)
    history: list[dict[str, float]] = []
    for iteration in range(max(1, config.variational_iterations)):
        current = _diffusion_step(
            current,
            anchor,
            wx,
            wy,
            config.diffusion_strength,
            config.anchor_weight,
        )
        if (iteration + 1) % max(1, config.back_projection_interval) == 0:
            degraded = degrade_coefficients(current, psf)
            residual = low_coeff - degraded
            correction = upsample_coefficients(residual, rgb.shape[:2])
            correction_limit = float(config.back_projection_clip_sigma) * low_std
            correction = np.clip(
                correction,
                -correction_limit[None, None, :],
                correction_limit[None, None, :],
            )
            current += config.back_projection_weight * correction
            current = np.clip(current, clip_min[None, None, :], clip_max[None, None, :])
            history.append({
                "iteration": float(iteration + 1),
                "coefficient_rmse": float(np.sqrt(np.mean(residual**2))),
                "coefficient_max_abs": float(np.max(np.abs(residual))),
            })
    current, coefficient_detail = inject_coefficient_detail(
        current,
        low_coeff,
        rgb,
        psf,
        rgb_confidence,
        config,
        clip_min=clip_min,
        clip_max=clip_max,
    )
    final_residual = low_coeff - degrade_coefficients(current, psf)
    residual_map_low = np.sqrt(np.mean(final_residual**2, axis=2))
    residual_map = cv2.resize(residual_map_low, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_CUBIC)
    scale = float(np.percentile(residual_map[np.isfinite(residual_map)], 95)) if np.isfinite(residual_map).any() else 1.0
    uncertainty = np.clip(
        0.70 * residual_map / max(scale, 1e-6)
        + 0.20 * (1.0 - rgb_confidence)
        + 0.10 * boundary_gate,
        0.0,
        1.0,
    )
    detail_gain, additive_detail, detail_details = _spectral_shape_preserving_detail_gain(
        rgb,
        psf,
        rgb_confidence,
        config,
    )
    uncertainty = np.clip(uncertainty + 0.08 * np.abs(detail_gain - 1.0), 0.0, 1.0)
    details = {
        "method": "edge_aware_variational_coefficient_refinement",
        "principle": "RGB controls low-rank material coefficient boundaries; spectra remain in the NIR/SWIR-derived basis",
        "iterations": int(config.variational_iterations),
        "diffusion_strength": float(config.diffusion_strength),
        "anchor_weight": float(config.anchor_weight),
        "rgb_edge_sigma": float(config.rgb_edge_sigma),
        "guided_radius": int(config.guided_radius),
        "guided_epsilon": float(config.guided_epsilon),
        "boundary_injection_strength": float(config.boundary_injection_strength),
        "boundary_gate_mean": float(np.mean(boundary_gate)),
        "back_projection_weight": float(config.back_projection_weight),
        "back_projection_clip_sigma": float(config.back_projection_clip_sigma),
        "coefficient_clip_margin": float(config.coefficient_clip_margin),
        "coefficient_detail": coefficient_detail,
        "spatial_detail": detail_details,
    }
    return CoefficientFusionResult(
        coefficients=current.astype(np.float32),
        uncertainty_map=uncertainty.astype(np.float32),
        detail_gain_map=detail_gain.astype(np.float32),
        additive_detail_map=additive_detail.astype(np.float32),
        history=history,
        details=details,
    )
