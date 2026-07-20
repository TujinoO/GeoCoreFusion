"""RGB-guided material-field refinement with sensor-consistency backprojection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from .config import FusionConfig
from .dataset import normalize_rgb
from .degradation import PsfModel, degrade_coefficients, degrade_spatial_map, upsample_coefficients
from .lowrank import project_simplex


@dataclass(slots=True)
class CoefficientFusionResult:
    coefficients: np.ndarray
    uncertainty_map: np.ndarray
    detail_gain_map: np.ndarray
    additive_detail_map: np.ndarray
    history: list[dict[str, float]]
    details: dict[str, Any]


def _rgb_texture_coherence(
    guide: np.ndarray,
    config: FusionConfig,
) -> np.ndarray:
    """Return a local RGB high-frequency coherence/SNR reliability map.

    Independent read/quantisation noise is normally weakly correlated between
    RGB channels, whereas a real achromatic or chromatic edge has a locally
    low-rank (possibly signed) three-channel residual.  Absolute pairwise
    correlation therefore supplies a cheap noise gate without assuming that
    the texture must be luminance-only.
    """

    epsilon = max(float(config.intrinsic_log_epsilon), 1e-4)
    log_rgb = np.log(np.asarray(guide, dtype=np.float32) + epsilon)
    residual = np.empty_like(log_rgb, dtype=np.float32)
    for channel in range(3):
        base = cv2.GaussianBlur(
            log_rgb[:, :, channel],
            (0, 0),
            sigmaX=1.25,
            sigmaY=1.25,
            borderType=cv2.BORDER_REFLECT101,
        )
        residual[:, :, channel] = log_rgb[:, :, channel] - base

    radius = max(1, int(config.dark_texture_window_radius))
    means = [_local_mean(residual[:, :, channel], radius) for channel in range(3)]
    variances = [
        np.maximum(
            _local_mean(residual[:, :, channel] ** 2, radius) - means[channel] ** 2,
            0.0,
        )
        for channel in range(3)
    ]
    correlations: list[np.ndarray] = []
    for first, second in ((0, 1), (0, 2), (1, 2)):
        covariance = (
            _local_mean(residual[:, :, first] * residual[:, :, second], radius)
            - means[first] * means[second]
        )
        denominator = np.sqrt(np.maximum(variances[first] * variances[second], 1e-12))
        correlation = np.divide(
            covariance,
            denominator,
            out=np.zeros_like(covariance),
            where=denominator > 1e-6,
        )
        correlations.append(np.abs(correlation))
    # One stable signed channel pair is sufficient for genuine isoluminant
    # colour texture; averaging all three pairs would wrongly reject cases in
    # which the third channel is nearly constant.  Independent sensor noise
    # remains low for every pair.
    coherence = np.max(np.stack(correlations, axis=2), axis=2)
    correlation_floor = float(
        np.clip(config.dark_texture_correlation_floor, 0.0, 0.85)
    )
    coherence_reliability = np.sqrt(
        np.clip(
            (coherence - correlation_floor) / max(0.80 - correlation_floor, 0.10),
            0.0,
            1.0,
        )
    )

    noise_floor = max(float(config.dark_texture_noise_floor), 1e-4)
    residual_energy = np.sqrt(np.mean(np.stack(variances, axis=2), axis=2))
    energy_reliability = np.sqrt(
        np.clip((residual_energy - noise_floor) / (4.0 * noise_floor), 0.0, 1.0)
    )

    return (coherence_reliability * energy_reliability).astype(np.float32)


def _dark_texture_confidence(
    guide: np.ndarray,
    config: FusionConfig,
) -> np.ndarray:
    """Estimate whether dark RGB texture is coherent structure rather than noise."""

    brightness = np.mean(guide, axis=2).astype(np.float32)
    signal = brightness[np.isfinite(brightness) & (brightness > 0.003)]
    if signal.size < 64:
        return np.zeros(brightness.shape, dtype=np.float32)
    texture_reliability = _rgb_texture_coherence(guide, config)
    percentile = float(np.clip(config.dark_detail_percentile, 1.0, 49.0))
    dark_threshold = max(float(np.percentile(signal, percentile)), 0.015)
    black_floor = min(float(np.percentile(signal, 2.0)), 0.25 * dark_threshold)
    dark_weight = np.clip(
        (dark_threshold - brightness) / max(dark_threshold - black_floor, 1e-4),
        0.0,
        1.0,
    )
    signal_reliability = np.sqrt(
        np.clip(
            (brightness - black_floor)
            / max(0.25 * dark_threshold - black_floor, 1e-4),
            0.0,
            1.0,
        )
    )
    return (
        max(0.0, float(config.dark_detail_boost))
        * dark_weight
        * texture_reliability
        * signal_reliability
    ).astype(np.float32)


def _amplitude_preserving_confidence_gate(
    reliability: np.ndarray,
    low: float,
    high: float,
) -> np.ndarray:
    """Map reliability to a smooth off/full-amplitude gate.

    A continuously multiplied confidence field attenuates every reliable RGB
    detail sample by a different amount.  That weakens contrast and introduces
    an artificial high-frequency component from the confidence field itself.
    This smoothstep gate instead keeps a zero-risk zone, a short transition,
    and a unit plateau where the evidence is strong enough.
    """

    lower = float(np.clip(low, 0.0, 1.0))
    upper = float(np.clip(high, lower + 1e-4, 1.0))
    t = np.clip(
        (np.asarray(reliability, dtype=np.float32) - lower) / (upper - lower),
        0.0,
        1.0,
    )
    return (t * t * (3.0 - 2.0 * t)).astype(np.float32)


def _rgb_weights(
    rgb: np.ndarray,
    sigma: float,
    config: FusionConfig | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    guide = normalize_rgb(rgb)
    dx = guide[:, 1:, :] - guide[:, :-1, :]
    dy = guide[1:, :, :] - guide[:-1, :, :]
    sigma2 = max(float(sigma) ** 2, 1e-6)
    wx = np.exp(-np.sum(dx * dx, axis=2) / (2.0 * sigma2)).astype(np.float32)
    wy = np.exp(-np.sum(dy * dy, axis=2) / (2.0 * sigma2)).astype(np.float32)
    brightness = np.mean(guide, axis=2)
    confidence = np.clip((brightness - 0.015) / 0.08, 0.0, 1.0) * np.clip((0.995 - brightness) / 0.10, 0.0, 1.0)
    saturation = np.max(guide, axis=2) - np.min(guide, axis=2)
    saturation_reliability = np.clip(1.25 - saturation, 0.15, 1.0)
    if config is not None and float(config.dark_detail_boost) > 0.0:
        confidence = np.maximum(confidence, _dark_texture_confidence(guide, config))
    confidence *= saturation_reliability
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
    *,
    intrinsic: bool = False,
    log_epsilon: float = 0.012,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return decorrelated RGB low-frequency features, detail residuals, and edges."""

    guide = normalize_rgb(rgb)
    if intrinsic:
        epsilon = max(float(log_epsilon), 1e-4)
        luminance_linear = 0.299 * guide[:, :, 0] + 0.587 * guide[:, :, 1] + 0.114 * guide[:, :, 2]
        log_rgb = np.log(guide + epsilon)
        luminance = np.log(luminance_linear + epsilon)
        features = np.stack(
            [
                luminance,
                log_rgb[:, :, 0] - log_rgb[:, :, 1],
                log_rgb[:, :, 2] - log_rgb[:, :, 1],
            ],
            axis=2,
        ).astype(np.float32)
    else:
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
    if intrinsic:
        detail = (features - small) + 0.55 * (small - large)
        for channel in range(detail.shape[2]):
            finite = np.isfinite(detail[:, :, channel])
            scale = float(np.percentile(np.abs(detail[:, :, channel][finite]), 99.5)) if finite.any() else 1.0
            detail[:, :, channel] = np.clip(detail[:, :, channel], -scale, scale)
    else:
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


def _mtf_matched_rgb_features(
    rgb: np.ndarray,
    psf: PsfModel,
    config: FusionConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Return low-grid RGB features and their MTF-matched HR residuals.

    Unlike the legacy Gaussian high-pass, each residual is formed with the
    same degradation operator used by the HSI observation model.  The three
    channels retain luminance and isoluminant colour structure.
    """

    guide = normalize_rgb(rgb)
    if bool(config.intrinsic_detail_enabled):
        epsilon = max(float(config.intrinsic_log_epsilon), 1e-4)
        log_rgb = np.log(guide + epsilon)
        luminance = np.log(
            0.299 * guide[:, :, 0]
            + 0.587 * guide[:, :, 1]
            + 0.114 * guide[:, :, 2]
            + epsilon
        )
        features = np.stack(
            [
                luminance,
                log_rgb[:, :, 0] - log_rgb[:, :, 1],
                log_rgb[:, :, 2] - log_rgb[:, :, 1],
            ],
            axis=2,
        ).astype(np.float32)
    else:
        compressed = np.log1p(4.0 * guide) / np.log(5.0)
        luminance = (
            0.299 * compressed[:, :, 0]
            + 0.587 * compressed[:, :, 1]
            + 0.114 * compressed[:, :, 2]
        )
        features = np.stack(
            [
                luminance,
                compressed[:, :, 0] - compressed[:, :, 1],
                compressed[:, :, 2] - compressed[:, :, 1],
            ],
            axis=2,
        ).astype(np.float32)

    fine_sigma = max(0.0, float(config.coefficient_detail_mtf_fine_sigma))
    upsample_mode = str(config.coefficient_detail_mtf_upsample).strip().lower()
    if upsample_mode in {"linear", "bilinear"}:
        upsample_interpolation = cv2.INTER_LINEAR
    elif upsample_mode in {"cubic", "bicubic"}:
        upsample_interpolation = cv2.INTER_CUBIC
    else:
        raise ValueError(
            f"Unknown coefficient_detail_mtf_upsample "
            f"{config.coefficient_detail_mtf_upsample!r}"
        )
    low_features = np.empty(psf.low_shape + (features.shape[2],), dtype=np.float32)
    detail_features = np.empty_like(features, dtype=np.float32)
    for channel in range(features.shape[2]):
        feature = features[:, :, channel]
        if fine_sigma > 0.0:
            feature = cv2.GaussianBlur(
                feature,
                (0, 0),
                sigmaX=fine_sigma,
                sigmaY=fine_sigma,
                borderType=cv2.BORDER_REFLECT101,
            )
        low = degrade_spatial_map(feature, psf)
        base = cv2.resize(
            low,
            (feature.shape[1], feature.shape[0]),
            interpolation=upsample_interpolation,
        )
        detail = (feature - base).astype(np.float32)
        low_features[:, :, channel] = low
        detail_features[:, :, channel] = detail
    return low_features, detail_features


def _local_mean(image: np.ndarray, radius: int) -> np.ndarray:
    size = 2 * max(1, int(radius)) + 1
    return cv2.boxFilter(
        np.asarray(image, dtype=np.float32),
        cv2.CV_32F,
        (size, size),
        normalize=True,
        borderType=cv2.BORDER_REFLECT101,
    )


def _inject_local_mtf_coefficient_detail(
    coefficients: np.ndarray,
    low_coeff: np.ndarray,
    rgb: np.ndarray,
    psf: PsfModel,
    rgb_confidence: np.ndarray,
    config: FusionConfig,
    *,
    clip_min: np.ndarray | None,
    clip_max: np.ndarray | None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Inject locally learned, MTF-matched RGB detail in coefficient space."""

    current = np.asarray(coefficients, dtype=np.float32)
    low = np.asarray(low_coeff, dtype=np.float32)
    strength = max(0.0, float(config.coefficient_detail_strength))
    if strength <= 0.0:
        return current.copy(), {
            "enabled": False,
            "method": "local_mtf_gsa",
            "coefficient_detail_strength": strength,
            "accepted_components": 0,
        }

    low_features, high_features = _mtf_matched_rgb_features(rgb, psf, config)
    base_residual_keep = float(
        np.clip(config.coefficient_detail_base_residual_keep, 0.0, 1.0)
    )
    base_residual = current - upsample_coefficients(
        degrade_coefficients(current, psf), current.shape[:2]
    )
    detail_free_current = current - (1.0 - base_residual_keep) * base_residual
    radius = max(1, int(config.coefficient_detail_local_radius))
    ridge = max(1e-6, float(config.coefficient_detail_local_ridge))
    correlation_floor = float(
        np.clip(config.coefficient_detail_local_correlation_floor, 0.0, 0.90)
    )

    feature_global_mean = np.nanmean(low_features, axis=(0, 1)).astype(np.float32)
    feature_global_scale = np.maximum(
        np.nanstd(low_features, axis=(0, 1)).astype(np.float32),
        1e-4,
    )
    standardized_low = (
        low_features - feature_global_mean[None, None, :]
    ) / feature_global_scale[None, None, :]
    standardized_high = high_features / feature_global_scale[None, None, :]
    standardized_low = np.nan_to_num(standardized_low, nan=0.0, posinf=0.0, neginf=0.0)
    standardized_high = np.nan_to_num(standardized_high, nan=0.0, posinf=0.0, neginf=0.0)

    feature_count = standardized_low.shape[2]
    feature_means = np.stack(
        [_local_mean(standardized_low[:, :, channel], radius) for channel in range(feature_count)],
        axis=2,
    )
    feature_covariance = np.empty(
        standardized_low.shape[:2] + (feature_count, feature_count),
        dtype=np.float32,
    )
    for first in range(feature_count):
        for second in range(feature_count):
            feature_covariance[:, :, first, second] = (
                _local_mean(
                    standardized_low[:, :, first] * standardized_low[:, :, second],
                    radius,
                )
                - feature_means[:, :, first] * feature_means[:, :, second]
            )
    feature_covariance = 0.5 * (
        feature_covariance + np.swapaxes(feature_covariance, -1, -2)
    )
    ridge_matrix = ridge * np.eye(feature_count, dtype=np.float32)
    inverse_covariance = np.linalg.inv(feature_covariance + ridge_matrix)

    target_variances = np.empty(low.shape, dtype=np.float32)
    cross_covariance = np.empty(
        low.shape[:2] + (feature_count, low.shape[2]),
        dtype=np.float32,
    )
    for component in range(low.shape[2]):
        target = low[:, :, component]
        target_mean = _local_mean(target, radius)
        target_variances[:, :, component] = np.maximum(
            _local_mean(target * target, radius) - target_mean * target_mean,
            0.0,
        )
        for channel in range(feature_count):
            cross_covariance[:, :, channel, component] = (
                _local_mean(standardized_low[:, :, channel] * target, radius)
                - feature_means[:, :, channel] * target_mean
            )

    beta = np.einsum(
        "...ij,...jk->...ik",
        inverse_covariance,
        cross_covariance,
        optimize=True,
    ).astype(np.float32)
    explained_variance = np.einsum(
        "...ik,...ik->...k", beta, cross_covariance, optimize=True
    )
    multiple_correlation = np.sqrt(
        np.clip(
            explained_variance / np.maximum(target_variances, 1e-10),
            0.0,
            1.0,
        )
    ).astype(np.float32)
    reliability = np.sqrt(
        np.clip(
            (multiple_correlation - correlation_floor)
            / max(0.75 - correlation_floor, 0.10),
            0.0,
            1.0,
        )
    ).astype(np.float32)

    low_std = np.maximum(np.nanstd(low, axis=(0, 1)).astype(np.float32), 1e-5)
    beta_limit = 4.0 * low_std[None, None, None, :]
    beta = np.clip(beta, -beta_limit, beta_limit)
    support_floor = float(np.clip(config.coefficient_detail_support_floor, 0.0, 1.0))
    texture_coherence = _rgb_texture_coherence(normalize_rgb(rgb), config)
    texture_coherence_floor = float(
        np.clip(config.coefficient_detail_texture_coherence_floor, 0.0, 1.0)
    )
    texture_gate = texture_coherence_floor + (
        1.0 - texture_coherence_floor
    ) * np.clip(texture_coherence, 0.0, 1.0)
    confidence = (
        np.clip(np.asarray(rgb_confidence, dtype=np.float32), 0.0, 1.0)
        * texture_gate
    ).astype(np.float32)
    raw_delta = np.zeros_like(current, dtype=np.float32)
    correlation_medians: list[list[float]] = []
    correlation_p90: list[list[float]] = []
    multiple_correlation_medians: list[float] = []
    multiple_correlation_p90: list[float] = []
    target_detail_rms: list[float] = []
    accepted_components = 0
    for component in range(current.shape[2]):
        component_delta = np.zeros(current.shape[:2], dtype=np.float32)
        for channel in range(feature_count):
            beta_high = cv2.resize(
                beta[:, :, channel, component],
                (current.shape[1], current.shape[0]),
                interpolation=cv2.INTER_CUBIC,
            )
            component_delta += beta_high * standardized_high[:, :, channel]
        reliability_high = cv2.resize(
            reliability[:, :, component],
            (current.shape[1], current.shape[0]),
            interpolation=cv2.INTER_LINEAR,
        )
        spatial_gate = confidence * (
            support_floor + (1.0 - support_floor) * np.clip(reliability_high, 0.0, 1.0)
        )
        calibration_mask = (confidence > 0.35) & (reliability_high > 0.35)
        target_values = (strength * component_delta)[calibration_mask]
        target_detail_rms.append(
            float(np.sqrt(np.mean(target_values * target_values)))
            if target_values.size
            else 0.0
        )
        raw_delta[:, :, component] = (
            strength * component_delta * spatial_gate
        ).astype(np.float32)

        component_feature_medians: list[float] = []
        component_feature_p90: list[float] = []
        for channel in range(feature_count):
            denominator = np.sqrt(
                np.maximum(
                    feature_covariance[:, :, channel, channel]
                    * target_variances[:, :, component],
                    1e-12,
                )
            )
            feature_correlation = np.divide(
                cross_covariance[:, :, channel, component],
                denominator,
                out=np.zeros_like(denominator),
                where=denominator > 1e-6,
            )
            values = np.abs(feature_correlation[np.isfinite(feature_correlation)])
            component_feature_medians.append(float(np.median(values)) if values.size else 0.0)
            component_feature_p90.append(float(np.percentile(values, 90.0)) if values.size else 0.0)
        joint_values = multiple_correlation[:, :, component]
        joint_values = joint_values[np.isfinite(joint_values)]
        joint_median = float(np.median(joint_values)) if joint_values.size else 0.0
        joint_p90 = float(np.percentile(joint_values, 90.0)) if joint_values.size else 0.0
        correlation_medians.append(component_feature_medians)
        correlation_p90.append(component_feature_p90)
        multiple_correlation_medians.append(joint_median)
        multiple_correlation_p90.append(joint_p90)
        accepted_components += int(
            joint_values.size > 0
            and float(np.percentile(joint_values, 75.0)) > correlation_floor
        )

    delta = _project_coefficients_to_observation_nullspace(
        raw_delta,
        psf,
        config.coefficient_detail_nullspace_iterations,
    )
    recovery_limit = max(
        1.0, float(config.coefficient_detail_amplitude_recovery_limit)
    )
    raw_detail_rms: list[float] = []
    projected_detail_rms: list[float] = []
    amplitude_recovery: list[float] = []
    reliable_pixels = confidence > 0.20
    for component in range(delta.shape[2]):
        raw_values = raw_delta[:, :, component][reliable_pixels]
        projected_values = delta[:, :, component][reliable_pixels]
        raw_rms = float(np.sqrt(np.mean(raw_values * raw_values))) if raw_values.size else 0.0
        projected_rms = (
            float(np.sqrt(np.mean(projected_values * projected_values)))
            if projected_values.size
            else 0.0
        )
        target_rms = target_detail_rms[component]
        recovery = (
            min(recovery_limit, target_rms / max(projected_rms, 1e-9))
            if target_rms > 0.0
            else 1.0
        )
        delta[:, :, component] *= recovery
        raw_detail_rms.append(raw_rms)
        projected_detail_rms.append(projected_rms)
        amplitude_recovery.append(float(recovery))
    clip_sigma = max(0.0, float(config.coefficient_detail_clip_sigma))
    detail_p98: list[float] = []
    for component in range(delta.shape[2]):
        p98 = float(np.percentile(np.abs(delta[:, :, component]), 98.0))
        limit = clip_sigma * float(low_std[component])
        if p98 > max(limit, 1e-8) and limit > 0.0:
            delta[:, :, component] *= limit / p98
            p98 = limit
        detail_p98.append(p98)

    low_detail = degrade_coefficients(delta, psf)
    enhanced = detail_free_current + delta
    clip_coefficients = bool(config.coefficient_detail_clip_coefficients)
    if clip_coefficients and clip_min is not None and clip_max is not None:
        enhanced = np.clip(enhanced, clip_min[None, None, :], clip_max[None, None, :])
    rmse_before = float(
        np.sqrt(np.mean((degrade_coefficients(enhanced, psf) - low) ** 2))
    )
    for _ in range(max(0, int(config.coefficient_detail_back_projection_iterations))):
        residual = low - degrade_coefficients(enhanced, psf)
        enhanced += float(config.back_projection_weight) * upsample_coefficients(
            residual, enhanced.shape[:2]
        )
        if clip_coefficients and clip_min is not None and clip_max is not None:
            enhanced = np.clip(
                enhanced,
                clip_min[None, None, :],
                clip_max[None, None, :],
            )
    rmse_after = float(
        np.sqrt(np.mean((degrade_coefficients(enhanced, psf) - low) ** 2))
    )
    return enhanced.astype(np.float32), {
        "enabled": True,
        "method": "local_mtf_gsa",
        "coefficient_detail_strength": strength,
        "local_radius_lowres": radius,
        "local_ridge": ridge,
        "local_correlation_floor": correlation_floor,
        "mtf_fine_sigma": float(config.coefficient_detail_mtf_fine_sigma),
        "support_floor": support_floor,
        "base_residual_keep": base_residual_keep,
        "base_residual_rms_by_component": [
            float(np.sqrt(np.mean(base_residual[:, :, component] ** 2)))
            for component in range(base_residual.shape[2])
        ],
        "texture_coherence_floor": texture_coherence_floor,
        "texture_coherence_mean": float(np.mean(texture_coherence)),
        "texture_coherence_p10": float(np.percentile(texture_coherence, 10.0)),
        "texture_coherence_p90": float(np.percentile(texture_coherence, 90.0)),
        "clip_sigma": clip_sigma,
        "clip_coefficients": clip_coefficients,
        "amplitude_recovery_limit": recovery_limit,
        "nullspace_iterations": int(config.coefficient_detail_nullspace_iterations),
        "back_projection_iterations": int(
            config.coefficient_detail_back_projection_iterations
        ),
        "accepted_components": accepted_components,
        "local_abs_correlation_median_by_component_feature": correlation_medians,
        "local_abs_correlation_p90_by_component_feature": correlation_p90,
        "local_multiple_correlation_median_by_component": multiple_correlation_medians,
        "local_multiple_correlation_p90_by_component": multiple_correlation_p90,
        "raw_detail_rms_by_component": raw_detail_rms,
        "target_ungated_detail_rms_by_component": target_detail_rms,
        "projected_detail_rms_before_recovery_by_component": projected_detail_rms,
        "amplitude_recovery_by_component": amplitude_recovery,
        "rgb_detail_confidence_mean": float(np.mean(confidence)),
        "rgb_detail_confidence_dark_p10": float(np.percentile(confidence, 10.0)),
        "detail_p98_by_component": detail_p98,
        "detail_lowres_rmse": float(np.sqrt(np.mean(low_detail * low_detail))),
        "coefficient_rmse_before_detail_back_projection": rmse_before,
        "coefficient_rmse_after_detail_back_projection": rmse_after,
    }


def _inject_simplex_abundance_detail(
    abundances: np.ndarray,
    low_abundances: np.ndarray,
    rgb: np.ndarray,
    psf: PsfModel,
    rgb_confidence: np.ndarray,
    config: FusionConfig,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Recover RGB-explainable detail only as simplex abundance transfer.

    Low-resolution HSI abundances are held fixed while a three-channel RGB
    endmember matrix is fitted on the observation grid.  The MTF-matched RGB
    residual is mapped through the centred RGB endmembers, which guarantees a
    zero-sum abundance perturbation.  The perturbation is noise/coherence
    gated, bounded in L1, projected away from the observed low-resolution
    component, and followed by simplex-constrained data consistency.
    """

    current_full = np.asarray(abundances, dtype=np.float32)
    low_full = np.asarray(low_abundances, dtype=np.float32)
    constraint = str(config.coefficient_constraint).strip().lower()
    simplex_count = (
        current_full.shape[2]
        if constraint in {"simplex", "nonnegative_simplex", "abundance_simplex"}
        else min(max(1, int(config.rank)), current_full.shape[2])
    )
    if low_full.shape[2] != current_full.shape[2]:
        raise ValueError("High- and low-resolution coefficient counts differ")
    current = project_simplex(current_full[:, :, :simplex_count], axis=2)
    low = project_simplex(low_full[:, :, :simplex_count], axis=2)
    residual_coefficients = current_full[:, :, simplex_count:].copy()

    def assemble(simplex_abundance: np.ndarray) -> np.ndarray:
        if residual_coefficients.shape[2] == 0:
            return np.asarray(simplex_abundance, dtype=np.float32)
        return np.concatenate(
            [np.asarray(simplex_abundance, dtype=np.float32), residual_coefficients],
            axis=2,
        ).astype(np.float32)
    strength = max(0.0, float(config.coefficient_detail_strength))
    if strength <= 0.0:
        return assemble(current), {
            "enabled": False,
            "method": "simplex_abundance_rgb_residual",
            "coefficient_detail_strength": strength,
            "accepted_components": 0,
        }

    guide = normalize_rgb(rgb).astype(np.float32)
    low_rgb = np.stack(
        [degrade_spatial_map(guide[:, :, channel], psf) for channel in range(3)],
        axis=2,
    ).astype(np.float32)
    low_confidence = degrade_spatial_map(
        np.clip(np.asarray(rgb_confidence, dtype=np.float32), 0.0, 1.0),
        psf,
    )
    abundance_flat = low.reshape(-1, low.shape[2]).astype(np.float64)
    rgb_flat = low_rgb.reshape(-1, 3).astype(np.float64)
    weight = np.clip(low_confidence.reshape(-1).astype(np.float64), 0.0, 1.0)
    valid = (
        np.isfinite(abundance_flat).all(axis=1)
        & np.isfinite(rgb_flat).all(axis=1)
        & (weight > 0.05)
    )
    if int(np.sum(valid)) < max(64, 4 * low.shape[2]):
        return assemble(current), {
            "enabled": False,
            "method": "simplex_abundance_rgb_residual",
            "reason": "insufficient_confident_low_resolution_support",
            "coefficient_detail_strength": strength,
            "accepted_components": 0,
        }

    a = abundance_flat[valid]
    r = rgb_flat[valid]
    w = weight[valid, None]
    ridge = max(1e-6, float(config.simplex_abundance_rgb_ridge))
    normal = (a.T @ (w * a)) / float(a.shape[0])
    normal += ridge * np.eye(a.shape[1], dtype=np.float64)
    rhs = (a.T @ (w * r)) / float(a.shape[0])
    rgb_endmembers = np.maximum(np.linalg.solve(normal, rhs), 0.0)
    prediction = a @ rgb_endmembers
    weighted_mean = np.sum(w * r, axis=0) / np.maximum(np.sum(w, axis=0), 1e-12)
    residual_ss = np.sum(w * (prediction - r) ** 2, axis=0)
    total_ss = np.sum(w * (r - weighted_mean[None, :]) ** 2, axis=0)
    r2_by_channel = 1.0 - residual_ss / np.maximum(total_ss, 1e-12)
    median_r2 = float(np.median(r2_by_channel))
    minimum_r2 = float(config.simplex_abundance_min_r2)
    material_reliability = float(
        np.sqrt(
            np.clip(
                (median_r2 - minimum_r2) / max(0.75 - minimum_r2, 0.10),
                0.0,
                1.0,
            )
        )
    )
    if material_reliability <= 0.0:
        return assemble(current), {
            "enabled": False,
            "method": "simplex_abundance_rgb_residual",
            "reason": "rgb_endmember_fit_failed_material_predictability_gate",
            "coefficient_detail_strength": strength,
            "rgb_fit_r2_by_channel": r2_by_channel.astype(float).tolist(),
            "rgb_fit_r2_median": median_r2,
            "accepted_components": 0,
        }

    fine_sigma = max(0.0, float(config.coefficient_detail_mtf_fine_sigma))
    rgb_detail = np.empty_like(guide, dtype=np.float32)
    for channel in range(3):
        feature = guide[:, :, channel]
        if fine_sigma > 0.0:
            feature = cv2.GaussianBlur(
                feature,
                (0, 0),
                sigmaX=fine_sigma,
                sigmaY=fine_sigma,
                borderType=cv2.BORDER_REFLECT101,
            )
        base = cv2.resize(
            degrade_spatial_map(feature, psf),
            (guide.shape[1], guide.shape[0]),
            interpolation=cv2.INTER_CUBIC,
        )
        rgb_detail[:, :, channel] = feature - base

    centred_endmembers = rgb_endmembers - np.mean(rgb_endmembers, axis=0, keepdims=True)
    rgb_normal = centred_endmembers.T @ centred_endmembers
    rgb_normal += ridge * np.eye(3, dtype=np.float64)
    active_components = max(0, int(config.simplex_abundance_active_components))
    pair_support: dict[str, int] = {}
    alignment_values: list[np.ndarray] = []
    if active_components == 2 and current.shape[2] >= 2:
        raw_delta = np.zeros_like(current, dtype=np.float32)
        active_pair = np.argpartition(current, -2, axis=2)[:, :, -2:]
        active_pair.sort(axis=2)
        alignment_floor = float(
            np.clip(config.simplex_abundance_rgb_alignment_floor, 0.0, 0.90)
        )
        detail_energy = np.sqrt(
            np.sum(rgb_detail.astype(np.float64) ** 2, axis=2)
        )
        for first in range(current.shape[2] - 1):
            for second in range(first + 1, current.shape[2]):
                mask = (
                    (active_pair[:, :, 0] == first)
                    & (active_pair[:, :, 1] == second)
                )
                count = int(np.sum(mask))
                if count == 0:
                    continue
                direction = rgb_endmembers[first] - rgb_endmembers[second]
                direction_energy = float(np.sum(direction * direction))
                pair_support[f"{first}-{second}"] = count
                if direction_energy <= 1e-10:
                    continue
                projection = np.einsum(
                    "...c,c->...", rgb_detail, direction, optimize=True
                )
                alignment = np.abs(projection) / np.maximum(
                    detail_energy * np.sqrt(direction_energy), 1e-8
                )
                pair_reliability = np.sqrt(
                    np.clip(
                        (alignment - alignment_floor)
                        / max(0.80 - alignment_floor, 0.10),
                        0.0,
                        1.0,
                    )
                )
                transfer = (
                    projection
                    / (direction_energy + ridge)
                    * pair_reliability
                ).astype(np.float32)
                raw_delta[:, :, first][mask] = transfer[mask]
                raw_delta[:, :, second][mask] = -transfer[mask]
                alignment_values.append(alignment[mask].astype(np.float32))
    else:
        decoder = np.linalg.solve(rgb_normal, centred_endmembers.T)
        raw_delta = np.einsum(
            "...c,ck->...k", rgb_detail, decoder, optimize=True
        ).astype(np.float32)
        raw_delta -= np.mean(raw_delta, axis=2, keepdims=True)
        alignment_floor = 0.0

    texture_reliability = _rgb_texture_coherence(guide, config)
    raw_gate = (
        np.clip(np.asarray(rgb_confidence, dtype=np.float32), 0.0, 1.0)
        * texture_reliability
    )
    gate = _amplitude_preserving_confidence_gate(
        raw_gate,
        config.spatial_detail_confidence_gate_low,
        config.spatial_detail_confidence_gate_high,
    )
    raw_delta *= strength * material_reliability * gate[:, :, None]
    delta = _project_coefficients_to_observation_nullspace(
        raw_delta,
        psf,
        config.coefficient_detail_nullspace_iterations,
    )
    delta -= np.mean(delta, axis=2, keepdims=True)

    l1_limit = max(0.0, float(config.simplex_abundance_detail_l1_limit))
    delta_l1_before = np.sum(np.abs(delta), axis=2)
    if l1_limit > 0.0:
        delta *= np.minimum(
            1.0,
            l1_limit / np.maximum(delta_l1_before, 1e-8),
        )[:, :, None]
    enhanced = project_simplex(current + delta, axis=2)
    rmse_before = float(
        np.sqrt(np.mean((degrade_coefficients(enhanced, psf) - low) ** 2))
    )
    for _ in range(max(0, int(config.coefficient_detail_back_projection_iterations))):
        residual = low - degrade_coefficients(enhanced, psf)
        residual -= np.mean(residual, axis=2, keepdims=True)
        enhanced = project_simplex(
            enhanced
            + float(config.back_projection_weight)
            * upsample_coefficients(residual, enhanced.shape[:2]),
            axis=2,
        )
    rmse_after_back_projection = float(
        np.sqrt(np.mean((degrade_coefficients(enhanced, psf) - low) ** 2))
    )
    delta_l1_before_final_constraint = np.sum(
        np.abs(enhanced - current), axis=2
    )
    final_l1_constraint_applied = np.zeros(
        enhanced.shape[:2], dtype=bool
    )
    if l1_limit > 0.0:
        final_l1_constraint_applied = (
            delta_l1_before_final_constraint > l1_limit
        )
        scale = np.minimum(
            1.0,
            l1_limit
            / np.maximum(delta_l1_before_final_constraint.astype(np.float64), 1e-15),
        )
        enhanced = project_simplex(
            current.astype(np.float64)
            + (enhanced.astype(np.float64) - current.astype(np.float64))
            * scale[:, :, None],
            axis=2,
        ).astype(np.float32)

        # Simplex projection is performed in float64, but the returned
        # coefficients are float32.  Re-apply the convex L1 contraction only
        # when that cast/projection introduces a last-bit overshoot.
        for _ in range(2):
            projected_delta_l1 = np.sum(np.abs(enhanced - current), axis=2)
            overshoot = projected_delta_l1 > l1_limit
            if not np.any(overshoot):
                break
            correction_scale = np.ones(enhanced.shape[:2], dtype=np.float64)
            correction_scale[overshoot] = (
                l1_limit
                * (1.0 - 8.0 * np.finfo(np.float32).eps)
                / projected_delta_l1[overshoot].astype(np.float64)
            )
            enhanced = project_simplex(
                current.astype(np.float64)
                + (enhanced.astype(np.float64) - current.astype(np.float64))
                * correction_scale[:, :, None],
                axis=2,
            ).astype(np.float32)

    rmse_after = float(
        np.sqrt(np.mean((degrade_coefficients(enhanced, psf) - low) ** 2))
    )
    final_delta_l1 = np.sum(np.abs(enhanced - current), axis=2)
    endmember_condition = float(
        np.linalg.cond(rgb_normal)
    )
    pair_alignment = (
        np.concatenate(alignment_values)
        if alignment_values
        else np.asarray([], dtype=np.float32)
    )
    return assemble(enhanced), {
        "enabled": True,
        "method": "simplex_abundance_rgb_residual",
        "coefficient_detail_strength": strength,
        "rgb_endmembers": rgb_endmembers.astype(float).tolist(),
        "rgb_fit_r2_by_channel": r2_by_channel.astype(float).tolist(),
        "rgb_fit_r2_median": median_r2,
        "material_reliability": material_reliability,
        "rgb_tangent_normal_condition": endmember_condition,
        "active_components": active_components,
        "rgb_alignment_floor": alignment_floor,
        "active_pair_support": pair_support,
        "active_pair_alignment_mean": (
            float(np.mean(pair_alignment)) if pair_alignment.size else None
        ),
        "active_pair_alignment_p10": (
            float(np.percentile(pair_alignment, 10.0))
            if pair_alignment.size
            else None
        ),
        "mtf_fine_sigma": fine_sigma,
        "confidence_gate_low": float(config.spatial_detail_confidence_gate_low),
        "confidence_gate_high": float(config.spatial_detail_confidence_gate_high),
        "detail_gate_mean": float(np.mean(gate)),
        "detail_gate_p90": float(np.percentile(gate, 90.0)),
        "l1_limit": l1_limit,
        "delta_l1_max_before_final_constraint": float(
            np.max(delta_l1_before_final_constraint)
        ),
        "final_l1_constraint_clipped_fraction": float(
            np.mean(final_l1_constraint_applied)
        ),
        "delta_l1_p50": float(np.percentile(final_delta_l1, 50.0)),
        "delta_l1_p95": float(np.percentile(final_delta_l1, 95.0)),
        "delta_l1_max": float(np.max(final_delta_l1)),
        "simplex_minimum": float(np.min(enhanced)),
        "simplex_sum_error_max": float(
            np.max(np.abs(np.sum(enhanced, axis=2) - 1.0))
        ),
        "simplex_component_count": int(simplex_count),
        "observation_residual_component_count": int(
            residual_coefficients.shape[2]
        ),
        "nullspace_iterations": int(config.coefficient_detail_nullspace_iterations),
        "back_projection_iterations": int(
            config.coefficient_detail_back_projection_iterations
        ),
        "coefficient_rmse_before_detail_back_projection": rmse_before,
        "coefficient_rmse_after_back_projection_before_final_l1_constraint": (
            rmse_after_back_projection
        ),
        "coefficient_rmse_after_detail_back_projection": rmse_after,
        "accepted_components": int(low.shape[2]),
    }


def _inject_lowrank_coefficient_bridge(
    coefficients: np.ndarray,
    low_coefficients: np.ndarray,
    rgb: np.ndarray,
    psf: PsfModel,
    rgb_confidence: np.ndarray,
    config: FusionConfig,
    *,
    clip_min: np.ndarray | None,
    clip_max: np.ndarray | None,
    local: bool = False,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Inject only cross-block-stable rank-1/2 RGB-to-coefficient detail."""

    current = np.asarray(coefficients, dtype=np.float32)
    low = np.asarray(low_coefficients, dtype=np.float32)
    strength = max(0.0, float(config.coefficient_detail_strength))
    if strength <= 0.0:
        return current.copy(), {
            "enabled": False,
            "method": "lowrank_coefficient_bridge",
            "coefficient_detail_strength": strength,
            "bridge_rank": 0,
        }

    low_features, high_features = _mtf_matched_rgb_features(rgb, psf, config)
    feature_mean = np.nanmean(low_features, axis=(0, 1)).astype(np.float32)
    feature_scale = np.maximum(
        np.nanstd(low_features, axis=(0, 1)).astype(np.float32), 1e-4
    )
    coefficient_mean = np.nanmean(low, axis=(0, 1)).astype(np.float32)
    coefficient_scale = np.maximum(
        np.nanstd(low, axis=(0, 1)).astype(np.float32), 1e-5
    )
    x = (
        (low_features - feature_mean[None, None, :])
        / feature_scale[None, None, :]
    ).reshape(-1, low_features.shape[2])
    y = (
        (low - coefficient_mean[None, None, :])
        / coefficient_scale[None, None, :]
    ).reshape(-1, low.shape[2])
    low_confidence = degrade_spatial_map(
        np.clip(np.asarray(rgb_confidence, dtype=np.float32), 0.0, 1.0), psf
    ).reshape(-1)
    valid = (
        np.isfinite(x).all(axis=1)
        & np.isfinite(y).all(axis=1)
        & np.isfinite(low_confidence)
        & (low_confidence > 0.05)
    )
    if int(np.sum(valid)) < 256:
        return current.copy(), {
            "enabled": False,
            "method": "lowrank_coefficient_bridge",
            "reason": "insufficient_confident_low_resolution_support",
            "coefficient_detail_strength": strength,
            "bridge_rank": 0,
        }

    ridge = max(1e-6, float(config.coefficient_detail_ridge))

    def fit_mapping(mask: np.ndarray) -> np.ndarray:
        xx = x[mask].astype(np.float64)
        yy = y[mask].astype(np.float64)
        ww = low_confidence[mask].astype(np.float64)[:, None]
        normal = (xx.T @ (ww * xx)) / float(xx.shape[0])
        normal += ridge * np.eye(xx.shape[1], dtype=np.float64)
        rhs = (xx.T @ (ww * yy)) / float(xx.shape[0])
        return np.linalg.solve(normal, rhs)

    block_rows = max(2, int(config.coefficient_detail_bridge_block_rows))
    block_cols = max(2, int(config.coefficient_detail_bridge_block_cols))
    yy_grid, xx_grid = np.indices(low.shape[:2])
    row_block = np.minimum(yy_grid * block_rows // low.shape[0], block_rows - 1)
    col_block = np.minimum(xx_grid * block_cols // low.shape[1], block_cols - 1)
    fold_id = (row_block * block_cols + col_block).reshape(-1)
    prediction = np.full_like(y, np.nan, dtype=np.float64)
    fold_mappings: list[np.ndarray] = []
    for fold in range(block_rows * block_cols):
        train = valid & (fold_id != fold)
        test = valid & (fold_id == fold)
        if int(np.sum(train)) < 128 or int(np.sum(test)) < 32:
            continue
        mapping = fit_mapping(train)
        prediction[test] = x[test] @ mapping
        fold_mappings.append(mapping)
    predicted = np.isfinite(prediction).all(axis=1) & valid
    if int(np.sum(predicted)) < 128:
        return current.copy(), {
            "enabled": False,
            "method": "lowrank_coefficient_bridge",
            "reason": "insufficient_blocked_cv_support",
            "coefficient_detail_strength": strength,
            "bridge_rank": 0,
        }
    residual = prediction[predicted] - y[predicted]
    target = y[predicted]
    target_centered = target - np.mean(target, axis=0, keepdims=True)
    r2_by_component = 1.0 - np.sum(residual * residual, axis=0) / np.maximum(
        np.sum(target_centered * target_centered, axis=0), 1e-12
    )
    component_weight = np.var(target, axis=0)
    weighted_r2 = float(
        np.sum(component_weight * r2_by_component)
        / max(float(np.sum(component_weight)), 1e-12)
    )
    r2_floor = float(config.coefficient_detail_bridge_cv_r2_floor)
    cv_reliability = float(
        np.sqrt(
            np.clip(
                (weighted_r2 - r2_floor) / max(0.60 - r2_floor, 0.10),
                0.0,
                1.0,
            )
        )
    )
    if cv_reliability <= 0.0:
        return current.copy(), {
            "enabled": False,
            "method": "lowrank_coefficient_bridge",
            "reason": "blocked_cv_predictability_gate_failed",
            "coefficient_detail_strength": strength,
            "blocked_cv_r2_by_component": r2_by_component.astype(float).tolist(),
            "blocked_cv_variance_weighted_r2": weighted_r2,
            "bridge_rank": 0,
        }

    full_mapping = fit_mapping(valid)
    left, singular_values, right = np.linalg.svd(full_mapping, full_matrices=False)
    bridge_rank = min(
        max(1, int(config.coefficient_detail_bridge_rank)),
        left.shape[1],
    )
    mapping_lowrank = (
        left[:, :bridge_rank]
        * singular_values[None, :bridge_rank]
    ) @ right[:bridge_rank, :]
    high_standardized = high_features / feature_scale[None, None, :]
    local_bridge = bool(local or config.coefficient_detail_bridge_local)
    local_multiple_correlation_median: list[float] = []
    local_multiple_correlation_p90: list[float] = []
    if local_bridge:
        low_standardized = x.reshape(low.shape[:2] + (x.shape[1],)).astype(np.float32)
        coefficient_standardized = y.reshape(low.shape).astype(np.float32)
        latent_low = np.einsum(
            "...k,jk->...j",
            coefficient_standardized,
            right[:bridge_rank, :],
            optimize=True,
        ).astype(np.float32)
        radius = max(1, int(config.coefficient_detail_bridge_local_radius))
        feature_means = np.stack(
            [
                _local_mean(low_standardized[:, :, channel], radius)
                for channel in range(low_standardized.shape[2])
            ],
            axis=2,
        )
        feature_covariance = np.empty(
            low.shape[:2]
            + (low_standardized.shape[2], low_standardized.shape[2]),
            dtype=np.float32,
        )
        for first in range(low_standardized.shape[2]):
            for second in range(low_standardized.shape[2]):
                feature_covariance[:, :, first, second] = (
                    _local_mean(
                        low_standardized[:, :, first]
                        * low_standardized[:, :, second],
                        radius,
                    )
                    - feature_means[:, :, first]
                    * feature_means[:, :, second]
                )
        inverse_covariance = np.linalg.inv(
            feature_covariance
            + ridge
            * np.eye(low_standardized.shape[2], dtype=np.float32)
        )
        latent_factors = np.zeros(
            high_features.shape[:2] + (bridge_rank,), dtype=np.float32
        )
        local_floor = float(
            np.clip(
                config.coefficient_detail_bridge_local_correlation_floor,
                0.0,
                0.90,
            )
        )
        for latent_index in range(bridge_rank):
            target_field = latent_low[:, :, latent_index]
            target_mean = _local_mean(target_field, radius)
            target_variance = np.maximum(
                _local_mean(target_field * target_field, radius)
                - target_mean * target_mean,
                0.0,
            )
            cross_covariance = np.empty(
                low.shape[:2] + (low_standardized.shape[2],),
                dtype=np.float32,
            )
            for channel in range(low_standardized.shape[2]):
                cross_covariance[:, :, channel] = (
                    _local_mean(
                        low_standardized[:, :, channel] * target_field,
                        radius,
                    )
                    - feature_means[:, :, channel] * target_mean
                )
            beta = np.einsum(
                "...ij,...j->...i",
                inverse_covariance,
                cross_covariance,
                optimize=True,
            ).astype(np.float32)
            explained = np.einsum(
                "...i,...i->...", beta, cross_covariance, optimize=True
            )
            multiple_correlation = np.sqrt(
                np.clip(
                    explained / np.maximum(target_variance, 1e-10),
                    0.0,
                    1.0,
                )
            ).astype(np.float32)
            reliability_low = np.sqrt(
                np.clip(
                    (multiple_correlation - local_floor)
                    / max(0.75 - local_floor, 0.10),
                    0.0,
                    1.0,
                )
            ).astype(np.float32)
            factor = np.zeros(high_features.shape[:2], dtype=np.float32)
            for channel in range(high_standardized.shape[2]):
                beta_high = cv2.resize(
                    beta[:, :, channel],
                    (high_features.shape[1], high_features.shape[0]),
                    interpolation=cv2.INTER_CUBIC,
                )
                factor += beta_high * high_standardized[:, :, channel]
            reliability_high = cv2.resize(
                reliability_low,
                (high_features.shape[1], high_features.shape[0]),
                interpolation=cv2.INTER_LINEAR,
            )
            latent_factors[:, :, latent_index] = (
                factor * np.clip(reliability_high, 0.0, 1.0)
            )
            values = multiple_correlation[np.isfinite(multiple_correlation)]
            local_multiple_correlation_median.append(
                float(np.median(values)) if values.size else 0.0
            )
            local_multiple_correlation_p90.append(
                float(np.percentile(values, 90.0)) if values.size else 0.0
            )
        raw_delta = np.einsum(
            "...j,jk->...k",
            latent_factors,
            right[:bridge_rank, :],
            optimize=True,
        ).astype(np.float32)
    else:
        raw_delta = np.einsum(
            "...f,fk->...k", high_standardized, mapping_lowrank, optimize=True
        ).astype(np.float32)
    raw_delta *= coefficient_scale[None, None, :]

    texture_reliability = _rgb_texture_coherence(normalize_rgb(rgb), config)
    raw_gate = (
        np.clip(np.asarray(rgb_confidence, dtype=np.float32), 0.0, 1.0)
        * texture_reliability
    )
    confidence_mode = str(config.spatial_detail_confidence_mode).strip().lower()
    if confidence_mode in {"amplitude_preserving_snr", "plateau_snr"}:
        gate = _amplitude_preserving_confidence_gate(
            raw_gate,
            config.spatial_detail_confidence_gate_low,
            config.spatial_detail_confidence_gate_high,
        )
    elif confidence_mode == "none":
        gate = np.ones(raw_gate.shape, dtype=np.float32)
    else:
        gate = raw_gate.astype(np.float32)
    raw_delta *= strength * cv_reliability * gate[:, :, None]
    delta = _project_coefficients_to_observation_nullspace(
        raw_delta,
        psf,
        config.coefficient_detail_nullspace_iterations,
    )
    reliable = gate > 0.35
    raw_rms = float(np.sqrt(np.mean(raw_delta[reliable] ** 2))) if np.any(reliable) else 0.0
    projected_rms = float(np.sqrt(np.mean(delta[reliable] ** 2))) if np.any(reliable) else 0.0
    recovery_limit = max(1.0, float(config.coefficient_detail_amplitude_recovery_limit))
    recovery = (
        min(recovery_limit, raw_rms / max(projected_rms, 1e-9))
        if raw_rms > 0.0
        else 1.0
    )
    delta *= recovery
    clip_sigma = max(0.0, float(config.coefficient_detail_clip_sigma))
    detail_p98: list[float] = []
    for component in range(delta.shape[2]):
        p98 = float(np.percentile(np.abs(delta[:, :, component]), 98.0))
        limit = clip_sigma * float(coefficient_scale[component])
        if limit > 0.0 and p98 > limit:
            delta[:, :, component] *= limit / p98
            p98 = limit
        detail_p98.append(p98)

    base_residual_keep = float(
        np.clip(config.coefficient_detail_base_residual_keep, 0.0, 1.0)
    )
    base_residual = current - upsample_coefficients(
        degrade_coefficients(current, psf), current.shape[:2]
    )
    enhanced = current - (1.0 - base_residual_keep) * base_residual + delta
    clip_coefficients = bool(config.coefficient_detail_clip_coefficients)
    if clip_coefficients and clip_min is not None and clip_max is not None:
        enhanced = np.clip(enhanced, clip_min[None, None, :], clip_max[None, None, :])
    rmse_before = float(
        np.sqrt(np.mean((degrade_coefficients(enhanced, psf) - low) ** 2))
    )
    for _ in range(max(0, int(config.coefficient_detail_back_projection_iterations))):
        coefficient_residual = low - degrade_coefficients(enhanced, psf)
        enhanced += float(config.back_projection_weight) * upsample_coefficients(
            coefficient_residual, enhanced.shape[:2]
        )
        if clip_coefficients and clip_min is not None and clip_max is not None:
            enhanced = np.clip(
                enhanced, clip_min[None, None, :], clip_max[None, None, :]
            )
    rmse_after = float(
        np.sqrt(np.mean((degrade_coefficients(enhanced, psf) - low) ** 2))
    )
    fold_mapping_stack = np.stack(fold_mappings, axis=0)
    return enhanced.astype(np.float32), {
        "enabled": True,
        "method": "lowrank_coefficient_bridge",
        "coefficient_detail_strength": strength,
        "bridge_rank": int(bridge_rank),
        "local_bridge": local_bridge,
        "local_radius": (
            int(config.coefficient_detail_bridge_local_radius)
            if local_bridge
            else None
        ),
        "local_multiple_correlation_median_by_factor": local_multiple_correlation_median,
        "local_multiple_correlation_p90_by_factor": local_multiple_correlation_p90,
        "ridge": ridge,
        "blocked_cv_grid": [block_rows, block_cols],
        "blocked_cv_fold_count": int(len(fold_mappings)),
        "blocked_cv_r2_by_component": r2_by_component.astype(float).tolist(),
        "blocked_cv_variance_weighted_r2": weighted_r2,
        "blocked_cv_reliability": cv_reliability,
        "mapping_singular_values": singular_values.astype(float).tolist(),
        "mapping_left_vectors": left[:, :bridge_rank].astype(float).tolist(),
        "mapping_coefficient_loadings": right[:bridge_rank, :].astype(float).tolist(),
        "mapping_lowrank": mapping_lowrank.astype(float).tolist(),
        "fold_mapping_std": np.std(fold_mapping_stack, axis=0).astype(float).tolist(),
        "feature_mean": feature_mean.astype(float).tolist(),
        "feature_scale": feature_scale.astype(float).tolist(),
        "coefficient_mean": coefficient_mean.astype(float).tolist(),
        "coefficient_scale": coefficient_scale.astype(float).tolist(),
        "confidence_mode": confidence_mode,
        "detail_gate_mean": float(np.mean(gate)),
        "detail_gate_p90": float(np.percentile(gate, 90.0)),
        "nullspace_iterations": int(config.coefficient_detail_nullspace_iterations),
        "amplitude_recovery": float(recovery),
        "clip_sigma": clip_sigma,
        "detail_p98_by_component": detail_p98,
        "base_residual_keep": base_residual_keep,
        "coefficient_rmse_before_detail_back_projection": rmse_before,
        "coefficient_rmse_after_detail_back_projection": rmse_after,
        "accepted_components": int(np.sum(r2_by_component > r2_floor)),
    }


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

    method = str(config.coefficient_detail_method).strip().lower()
    if method in {
        "simplex_abundance_rgb_residual",
        "simplex_abundance",
        "abundance_residual",
    }:
        return _inject_simplex_abundance_detail(
            coefficients,
            low_coeff,
            rgb,
            psf,
            rgb_confidence,
            config,
        )
    if method in {
        "lowrank_coefficient_bridge",
        "lowrank_bridge",
        "band_specific_lowrank",
        "local_lowrank_coefficient_bridge",
        "local_lowrank_bridge",
    }:
        return _inject_lowrank_coefficient_bridge(
            coefficients,
            low_coeff,
            rgb,
            psf,
            rgb_confidence,
            config,
            clip_min=clip_min,
            clip_max=clip_max,
            local=method in {
                "local_lowrank_coefficient_bridge",
                "local_lowrank_bridge",
            },
        )
    if method in {"local_mtf_gsa", "local_mtf", "mtf_gsa"}:
        return _inject_local_mtf_coefficient_detail(
            coefficients,
            low_coeff,
            rgb,
            psf,
            rgb_confidence,
            config,
            clip_min=clip_min,
            clip_max=clip_max,
        )
    if method not in {"global_ridge", "legacy", "global"}:
        raise ValueError(f"Unknown coefficient_detail_method {config.coefficient_detail_method!r}")

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
    low_features, high_features, _ = _rgb_detail_features(
        rgb,
        small_sigma,
        large_sigma,
        intrinsic=bool(config.intrinsic_detail_enabled),
        log_epsilon=float(config.intrinsic_log_epsilon),
    )
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
        "method": "global_ridge",
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


def _visual_detail_denoise(
    guide: np.ndarray,
    strength: float,
) -> tuple[np.ndarray, float]:
    """Apply mild edge-preserving RGB denoising before detail decomposition."""

    source = np.clip(np.asarray(guide, dtype=np.float32), 0.0, 1.0)
    amount = float(np.clip(strength, 0.0, 1.0))
    if amount <= 0.0:
        return source.copy(), 0.0
    diameter = 5 if amount < 0.70 else 7
    sigma_color = 0.012 + 0.055 * amount
    sigma_space = 0.9 + 1.8 * amount
    denoised = np.empty_like(source)
    for channel in range(source.shape[2]):
        denoised[:, :, channel] = cv2.bilateralFilter(
            source[:, :, channel],
            d=diameter,
            sigmaColor=sigma_color,
            sigmaSpace=sigma_space,
            borderType=cv2.BORDER_REFLECT101,
        )
    residual = source - denoised
    residual_rms = float(np.sqrt(np.mean(residual * residual)))
    return np.clip(denoised, 0.0, 1.0).astype(np.float32), residual_rms


def _weighted_detail_pyramid(
    image: np.ndarray,
    sigmas: tuple[float, ...],
    weights: tuple[float, ...],
) -> tuple[np.ndarray, list[float]]:
    """Return the weighted sum of an exactly reconstructible Gaussian pyramid."""

    source = np.asarray(image, dtype=np.float32)
    previous = source
    combined = np.zeros_like(source, dtype=np.float32)
    rms_values: list[float] = []
    for sigma, weight in zip(sigmas, weights):
        smooth = cv2.GaussianBlur(
            source,
            (0, 0),
            sigmaX=float(sigma),
            sigmaY=float(sigma),
            borderType=cv2.BORDER_REFLECT101,
        )
        band = (previous - smooth).astype(np.float32)
        combined += float(weight) * band
        rms_values.append(float(np.sqrt(np.mean(band * band))))
        previous = smooth
    return combined.astype(np.float32), rms_values


def _forward_gradient(field: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(field, dtype=np.float32)
    gx = np.zeros_like(values)
    gy = np.zeros_like(values)
    gx[:, :-1] = values[:, 1:] - values[:, :-1]
    gy[:-1, :] = values[1:, :] - values[:-1, :]
    return gx, gy


def _screened_poisson_detail(
    anchor: np.ndarray,
    target_gx: np.ndarray,
    target_gy: np.ndarray,
    *,
    screen: float,
    gradient_weight: float,
    reflect_padding: int,
) -> np.ndarray:
    """Reconstruct an integrable detail field from multichannel RGB gradients."""

    pad = max(0, int(reflect_padding))
    pad_width = ((pad, pad), (pad, pad))
    anchor_pad = np.pad(anchor, pad_width, mode="reflect") if pad else anchor
    gx_pad = np.pad(target_gx, pad_width, mode="reflect") if pad else target_gx
    gy_pad = np.pad(target_gy, pad_width, mode="reflect") if pad else target_gy
    screen_value = max(1e-4, float(screen))
    gradient_value = max(0.0, float(gradient_weight))

    adjoint = (
        np.roll(gx_pad, 1, axis=1)
        - gx_pad
        + np.roll(gy_pad, 1, axis=0)
        - gy_pad
    )
    rhs = screen_value * anchor_pad + gradient_value * adjoint
    height, width = anchor_pad.shape
    omega_y = 2.0 * np.pi * np.fft.fftfreq(height)
    omega_x = 2.0 * np.pi * np.fft.fftfreq(width)
    laplacian_eigenvalue = (
        4.0
        - 2.0 * np.cos(omega_y)[:, None]
        - 2.0 * np.cos(omega_x)[None, :]
    )
    denominator = screen_value + gradient_value * laplacian_eigenvalue
    solution = np.fft.ifft2(np.fft.fft2(rhs) / denominator).real.astype(np.float32)
    if pad:
        solution = solution[pad:-pad, pad:-pad]
    return solution.astype(np.float32)


def _visual_multiscale_gradient_gain(
    rgb: np.ndarray,
    psf: PsfModel,
    config: FusionConfig,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Build V6.1 full-detail log gain and dark-area additive compensation.

    The denoised RGB is decomposed into an exact multi-scale pyramid.  Signed
    gradients from the strongest RGB channel preserve isoluminant colour
    boundaries, while screened-Poisson reconstruction makes the resulting
    detail field integrable.  A final observation-nullspace projection and
    product-level back-projection keep the measured low-resolution response
    available without suppressing the visual high-frequency target.
    """

    sigmas = tuple(float(value) for value in config.visual_detail_pyramid_sigmas)
    weights = tuple(float(value) for value in config.visual_detail_pyramid_weights)
    if not sigmas or len(sigmas) != len(weights):
        raise ValueError(
            "visual_detail_pyramid_sigmas and visual_detail_pyramid_weights "
            "must be non-empty and have equal length"
        )
    if any(value <= 0.0 for value in sigmas) or any(
        second <= first for first, second in zip(sigmas, sigmas[1:])
    ):
        raise ValueError("visual_detail_pyramid_sigmas must be positive and increasing")

    guide = normalize_rgb(rgb)
    denoised, denoise_residual_rms = _visual_detail_denoise(
        guide, config.visual_detail_denoise_strength
    )
    epsilon = max(float(config.intrinsic_log_epsilon), 1e-4)
    log_channels = np.log(denoised + epsilon).astype(np.float32)
    luminance_linear = (
        0.299 * denoised[:, :, 0]
        + 0.587 * denoised[:, :, 1]
        + 0.114 * denoised[:, :, 2]
    ).astype(np.float32)
    log_luminance = np.log(luminance_linear + epsilon).astype(np.float32)

    sharpen = max(0.0, float(config.visual_detail_sharpen_strength))
    effective_weights = list(weights)
    effective_weights[0] += sharpen
    effective_weights_tuple = tuple(effective_weights)
    anchor, luminance_band_rms = _weighted_detail_pyramid(
        log_luminance, sigmas, effective_weights_tuple
    )
    channel_band_rms: list[list[float]] = []
    channel_gx: list[np.ndarray] = []
    channel_gy: list[np.ndarray] = []
    channel_gradient_magnitude: list[np.ndarray] = []
    for channel in range(3):
        channel_detail, band_rms = _weighted_detail_pyramid(
            log_channels[:, :, channel], sigmas, effective_weights_tuple
        )
        gx, gy = _forward_gradient(channel_detail)
        channel_band_rms.append(band_rms)
        channel_gx.append(gx)
        channel_gy.append(gy)
        channel_gradient_magnitude.append(gx * gx + gy * gy)

    magnitude_stack = np.stack(channel_gradient_magnitude, axis=2)
    strongest = np.argmax(magnitude_stack, axis=2)
    gx_stack = np.stack(channel_gx, axis=2)
    gy_stack = np.stack(channel_gy, axis=2)
    selected_gx = np.take_along_axis(gx_stack, strongest[:, :, None], axis=2)[:, :, 0]
    selected_gy = np.take_along_axis(gy_stack, strongest[:, :, None], axis=2)[:, :, 0]
    anchor_gx, anchor_gy = _forward_gradient(anchor)
    chroma_weight = float(np.clip(config.visual_detail_chroma_weight, 0.0, 1.0))
    target_gx = ((1.0 - chroma_weight) * anchor_gx + chroma_weight * selected_gx).astype(
        np.float32
    )
    target_gy = ((1.0 - chroma_weight) * anchor_gy + chroma_weight * selected_gy).astype(
        np.float32
    )
    reconstructed = _screened_poisson_detail(
        anchor,
        target_gx,
        target_gy,
        screen=float(config.visual_detail_poisson_screen),
        gradient_weight=float(config.visual_detail_gradient_weight),
        reflect_padding=int(config.visual_detail_reflect_padding),
    )

    dark_boost = max(0.0, float(config.visual_detail_dark_boost))
    dark_threshold = max(
        float(np.percentile(luminance_linear, float(config.dark_detail_percentile))),
        epsilon,
    )
    darkness = np.clip((dark_threshold - luminance_linear) / dark_threshold, 0.0, 1.0)
    reconstructed *= 1.0 + dark_boost * cv2.GaussianBlur(
        darkness.astype(np.float32),
        (0, 0),
        sigmaX=1.0,
        sigmaY=1.0,
        borderType=cv2.BORDER_REFLECT101,
    )

    iterations = max(0, int(config.spatial_detail_nullspace_iterations))
    projected = _project_map_to_observation_nullspace(reconstructed, psf, iterations)
    raw_rms = float(np.sqrt(np.mean(reconstructed * reconstructed)))
    projected_rms = float(np.sqrt(np.mean(projected * projected)))
    recovery_limit = max(1.0, float(config.spatial_detail_amplitude_recovery_limit))
    recovery = min(recovery_limit, raw_rms / max(projected_rms, 1e-9))
    detail = (projected * recovery).astype(np.float32)
    log_clip = max(0.02, float(config.spatial_detail_log_detail_clip))
    detail = np.clip(detail, -log_clip, log_clip).astype(np.float32)

    strength = max(0.0, float(config.spatial_detail_strength))
    low_limit, high_limit = map(float, config.spatial_detail_gain_limits)
    gain = np.clip(np.exp(strength * detail), low_limit, high_limit).astype(np.float32)
    absolute_scale = max(
        float(np.percentile(np.abs(detail[np.isfinite(detail)]), 95.0)), 1e-6
    )
    additive_detail = (
        np.tanh(detail / absolute_scale).astype(np.float32)
        if float(config.spatial_detail_additive_strength) > 0.0
        else np.zeros_like(detail, dtype=np.float32)
    )
    low_gain = degrade_spatial_map(gain, psf)
    low_additive = degrade_spatial_map(additive_detail, psf)
    return gain, additive_detail, {
        "method": "visual_multiscale_gradient",
        "detail_policy": "all_denoised_rgb_spatial_detail",
        "denoise_method": "mild_bilateral_per_channel",
        "denoise_strength": float(config.visual_detail_denoise_strength),
        "denoise_residual_rms": denoise_residual_rms,
        "pyramid_sigmas": list(sigmas),
        "pyramid_weights": list(effective_weights_tuple),
        "luminance_band_rms": luminance_band_rms,
        "channel_band_rms": channel_band_rms,
        "chroma_weight": chroma_weight,
        "strongest_channel_fraction": [
            float(np.mean(strongest == channel)) for channel in range(3)
        ],
        "poisson_screen": float(config.visual_detail_poisson_screen),
        "poisson_gradient_weight": float(config.visual_detail_gradient_weight),
        "dark_boost": dark_boost,
        "dark_threshold": dark_threshold,
        "sharpen_strength": sharpen,
        "spatial_detail_strength": strength,
        "spatial_detail_nullspace_iterations": iterations,
        "amplitude_recovery": float(recovery),
        "raw_detail_rms": raw_rms,
        "projected_detail_rms_before_recovery": projected_rms,
        "log_detail_clip": log_clip,
        "detail_gain_min": float(np.min(gain)),
        "detail_gain_max": float(np.max(gain)),
        "detail_gain_mean": float(np.mean(gain)),
        "detail_gain_p05": float(np.percentile(gain, 5.0)),
        "detail_gain_p95": float(np.percentile(gain, 95.0)),
        "lowres_gain_rmse_from_one": float(
            np.sqrt(np.mean((low_gain - 1.0) ** 2))
        ),
        "lowres_gain_max_abs_from_one": float(np.max(np.abs(low_gain - 1.0))),
        "spatial_detail_additive_strength": float(
            config.spatial_detail_additive_strength
        ),
        "additive_detail_lowres_rmse": float(
            np.sqrt(np.mean(low_additive * low_additive))
        ),
        "additive_detail_lowres_max_abs": float(np.max(np.abs(low_additive))),
    }


def _coherent_mtf_log_hpm_gain(
    rgb: np.ndarray,
    psf: PsfModel,
    rgb_confidence: np.ndarray,
    config: FusionConfig,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """Build a noise-gated MTF-matched log-HPM gain.

    The transferred signal stays in native log-contrast units.  In reliable
    regions, a strength of one therefore targets the same local contrast as
    RGB instead of the legacy p95-normalised/tanh-compressed 20% modulation.
    """

    _, high_features = _mtf_matched_rgb_features(rgb, psf, config)
    raw_detail = high_features[:, :, 0].astype(np.float32)
    confidence_mode = str(config.spatial_detail_confidence_mode).strip().lower()
    if confidence_mode == "none":
        confidence = np.ones(raw_detail.shape, dtype=np.float32)
        texture_coherence = np.ones(raw_detail.shape, dtype=np.float32)
    elif confidence_mode in {
        "exposure",
        "local_snr",
        "amplitude_preserving_snr",
        "plateau_snr",
    }:
        texture_coherence = _rgb_texture_coherence(normalize_rgb(rgb), config)
        coherence_floor = float(
            np.clip(config.coefficient_detail_texture_coherence_floor, 0.0, 1.0)
        )
        coherence_gate = coherence_floor + (
            1.0 - coherence_floor
        ) * texture_coherence
        raw_confidence = (
            np.clip(np.asarray(rgb_confidence, dtype=np.float32), 0.0, 1.0)
            * coherence_gate
        ).astype(np.float32)
        if confidence_mode in {"amplitude_preserving_snr", "plateau_snr"}:
            confidence = _amplitude_preserving_confidence_gate(
                raw_confidence,
                config.spatial_detail_confidence_gate_low,
                config.spatial_detail_confidence_gate_high,
            )
        else:
            confidence = raw_confidence
    else:
        raise ValueError(
            f"Unknown spatial_detail_confidence_mode "
            f"{config.spatial_detail_confidence_mode!r}"
        )

    gated_detail = (raw_detail * confidence).astype(np.float32)
    iterations = max(0, int(config.spatial_detail_nullspace_iterations))
    projected = _project_map_to_observation_nullspace(gated_detail, psf, iterations)
    calibration_mask = confidence > 0.35
    raw_values = raw_detail[calibration_mask]
    projected_values = projected[calibration_mask]
    raw_rms = float(np.sqrt(np.mean(raw_values * raw_values))) if raw_values.size else 0.0
    projected_rms = (
        float(np.sqrt(np.mean(projected_values * projected_values)))
        if projected_values.size
        else 0.0
    )
    recovery_limit = max(
        1.0, float(config.spatial_detail_amplitude_recovery_limit)
    )
    recovery = (
        min(recovery_limit, raw_rms / max(projected_rms, 1e-9))
        if raw_rms > 0.0
        else 1.0
    )
    detail = (projected * recovery).astype(np.float32)
    log_clip = max(0.02, float(config.spatial_detail_log_detail_clip))
    detail = np.clip(detail, -log_clip, log_clip).astype(np.float32)

    strength = max(0.0, float(config.spatial_detail_strength))
    low_limit, high_limit = map(float, config.spatial_detail_gain_limits)
    gain = np.clip(
        np.exp(strength * detail),
        low_limit,
        high_limit,
    ).astype(np.float32)
    additive_detail = (
        detail.copy()
        if float(config.spatial_detail_additive_strength) > 0.0
        else np.zeros_like(detail, dtype=np.float32)
    )
    low_gain = degrade_spatial_map(gain, psf)
    low_additive_detail = degrade_spatial_map(additive_detail, psf)
    return gain, additive_detail, {
        "method": "coherent_mtf_log_hpm",
        "spatial_detail_strength": strength,
        "mtf_fine_sigma": float(config.coefficient_detail_mtf_fine_sigma),
        "spatial_detail_confidence_mode": confidence_mode,
        "spatial_detail_nullspace_iterations": iterations,
        "spatial_detail_back_projection_iterations": 0,
        "log_detail_clip": log_clip,
        "amplitude_recovery_limit": recovery_limit,
        "amplitude_recovery": float(recovery),
        "raw_reliable_log_detail_rms": raw_rms,
        "projected_reliable_log_detail_rms_before_recovery": projected_rms,
        "texture_coherence_mean": float(np.mean(texture_coherence)),
        "detail_confidence_mean": float(np.mean(confidence)),
        "detail_confidence_p10": float(np.percentile(confidence, 10.0)),
        "confidence_gate_low": float(config.spatial_detail_confidence_gate_low),
        "confidence_gate_high": float(config.spatial_detail_confidence_gate_high),
        "detail_gain_min": float(np.min(gain)),
        "detail_gain_max": float(np.max(gain)),
        "detail_gain_mean": float(np.mean(gain)),
        "detail_gain_p05": float(np.percentile(gain, 5.0)),
        "detail_gain_p95": float(np.percentile(gain, 95.0)),
        "lowres_gain_rmse_from_one": float(
            np.sqrt(np.mean((low_gain - 1.0) ** 2))
        ),
        "lowres_gain_max_abs_from_one": float(np.max(np.abs(low_gain - 1.0))),
        "spatial_detail_additive_strength": float(
            config.spatial_detail_additive_strength
        ),
        "additive_detail_lowres_rmse": float(
            np.sqrt(np.mean(low_additive_detail * low_additive_detail))
        ),
    }


def _spectral_shape_preserving_detail_gain(
    rgb: np.ndarray,
    psf: PsfModel,
    rgb_confidence: np.ndarray,
    config: FusionConfig,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """Create a common per-pixel gain for every band from RGB high-frequency detail."""

    method = str(config.spatial_detail_method).strip().lower()
    if method in {
        "visual_multiscale_gradient",
        "v61_visual_full_detail",
        "full_rgb_detail",
    }:
        return _visual_multiscale_gradient_gain(rgb, psf, config)
    if method in {
        "coherent_mtf_log_hpm",
        "mtf_log_hpm",
        "coherent_log_hpm",
    }:
        return _coherent_mtf_log_hpm_gain(
            rgb,
            psf,
            rgb_confidence,
            config,
        )
    if method not in {"legacy", "shared_gain", "legacy_shared_gain"}:
        raise ValueError(f"Unknown spatial_detail_method {config.spatial_detail_method!r}")

    small_sigma = max(0.5, float(config.spatial_detail_small_sigma))
    large_sigma = max(small_sigma + 0.5, float(config.spatial_detail_large_sigma))
    _, rgb_detail, edge = _rgb_detail_features(
        rgb,
        small_sigma,
        large_sigma,
        intrinsic=bool(config.intrinsic_detail_enabled),
        log_epsilon=float(config.intrinsic_log_epsilon),
    )
    detail = rgb_detail[:, :, 0]
    valid = np.isfinite(detail)
    detail_scale = float(np.percentile(np.abs(detail[valid]), 95.0)) if valid.any() else 1.0
    detail = np.tanh(detail / max(detail_scale, 1e-6)).astype(np.float32)
    texture_floor = float(np.clip(config.spatial_detail_texture_floor, 0.0, 1.0))
    confidence_mode = str(config.spatial_detail_confidence_mode).strip().lower()
    if confidence_mode == "none":
        confidence_gate = np.ones_like(rgb_confidence, dtype=np.float32)
    elif confidence_mode in {"exposure", "local_snr"}:
        confidence_gate = np.clip(rgb_confidence, 0.0, 1.0)
    else:
        raise ValueError(f"Unknown spatial_detail_confidence_mode {config.spatial_detail_confidence_mode!r}")
    detail *= (texture_floor + (1.0 - texture_floor) * edge) * confidence_gate

    iterations = max(0, int(config.spatial_detail_nullspace_iterations))
    detail = _project_map_to_observation_nullspace(detail, psf, iterations)
    projected_scale = float(np.percentile(np.abs(detail[np.isfinite(detail)]), 98.0)) if np.isfinite(detail).any() else 1.0
    detail = np.clip(detail / max(projected_scale, 1e-6), -1.0, 1.0).astype(np.float32)
    additive_detail = (
        detail.copy()
        if float(config.spatial_detail_additive_strength) > 0.0
        else np.zeros_like(detail, dtype=np.float32)
    )
    low_limit, high_limit = map(float, config.spatial_detail_gain_limits)
    if bool(config.spatial_detail_log_gain):
        gain = np.clip(
            np.exp(float(config.spatial_detail_strength) * detail),
            low_limit,
            high_limit,
        ).astype(np.float32)
    else:
        gain = np.clip(1.0 + float(config.spatial_detail_strength) * detail, low_limit, high_limit).astype(np.float32)
    back_projection_iterations = int(config.spatial_detail_back_projection_iterations)
    if back_projection_iterations <= 0:
        back_projection_iterations = iterations
    for _ in range(back_projection_iterations):
        low_gain = degrade_spatial_map(gain, psf)
        correction = cv2.resize(low_gain - 1.0, (gain.shape[1], gain.shape[0]), interpolation=cv2.INTER_CUBIC)
        gain = np.clip(gain - correction, low_limit, high_limit).astype(np.float32)
    low_gain = degrade_spatial_map(gain, psf)
    return gain, additive_detail, {
        "spatial_detail_strength": float(config.spatial_detail_strength),
        "spatial_detail_small_sigma": small_sigma,
        "spatial_detail_large_sigma": large_sigma,
        "spatial_detail_texture_floor": texture_floor,
        "intrinsic_detail_enabled": bool(config.intrinsic_detail_enabled),
        "spatial_detail_log_gain": bool(config.spatial_detail_log_gain),
        "spatial_detail_confidence_mode": confidence_mode,
        "spatial_detail_back_projection_iterations": back_projection_iterations,
        "dark_detail_boost": float(config.dark_detail_boost),
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


def build_band_adaptive_mtf_detail(
    low_cube: np.ndarray,
    rgb: np.ndarray,
    psf: PsfModel,
    config: FusionConfig,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Build a GSA-style additive detail field with band-adaptive gains.

    The RGB structural residual is matched to the estimated sensor MTF.  Each
    spectral band receives a signed regression gain learned on the observation
    grid, then smoothed along wavelength and bounded by the band's observed
    variance.  This restores absolute contrast where a multiplicative gain is
    ineffective (especially near the dark floor) without copying one fixed
    amplitude into every band.
    """

    cube = np.asarray(low_cube, dtype=np.float32)
    strength = max(0.0, float(config.spatial_detail_additive_strength))
    if strength <= 0.0:
        return (
            np.zeros(psf.high_shape, dtype=np.float32),
            np.zeros(cube.shape[2], dtype=np.float32),
            {
                "enabled": False,
                "mode": "band_adaptive_mtf_gsa",
                "spatial_detail_additive_strength": strength,
                "accepted_bands": 0,
            },
        )

    guide = normalize_rgb(rgb)
    luminance = (
        0.299 * guide[:, :, 0]
        + 0.587 * guide[:, :, 1]
        + 0.114 * guide[:, :, 2]
    ).astype(np.float32)
    if bool(config.intrinsic_detail_enabled):
        luminance = np.log(
            luminance + max(float(config.intrinsic_log_epsilon), 1e-4)
        ).astype(np.float32)
    fine_sigma = max(0.0, float(config.coefficient_detail_mtf_fine_sigma))
    upsample_mode = str(config.coefficient_detail_mtf_upsample).strip().lower()
    if upsample_mode in {"linear", "bilinear"}:
        upsample_interpolation = cv2.INTER_LINEAR
    elif upsample_mode in {"cubic", "bicubic"}:
        upsample_interpolation = cv2.INTER_CUBIC
    else:
        raise ValueError(
            f"Unknown coefficient_detail_mtf_upsample "
            f"{config.coefficient_detail_mtf_upsample!r}"
        )
    if fine_sigma > 0.0:
        luminance = cv2.GaussianBlur(
            luminance,
            (0, 0),
            sigmaX=fine_sigma,
            sigmaY=fine_sigma,
            borderType=cv2.BORDER_REFLECT101,
        )
    low_guide = degrade_spatial_map(luminance, psf)
    guide_base = cv2.resize(
        low_guide,
        (luminance.shape[1], luminance.shape[0]),
        interpolation=upsample_interpolation,
    )
    detail = (luminance - guide_base).astype(np.float32)
    confidence_mode = str(config.spatial_detail_confidence_mode).strip().lower()
    if confidence_mode in {
        "exposure",
        "local_snr",
        "amplitude_preserving_snr",
        "plateau_snr",
    }:
        detail_confidence = _rgb_weights(
            rgb,
            config.rgb_edge_sigma,
            config,
        )[2]
        texture_coherence = _rgb_texture_coherence(guide, config)
        coherence_floor = float(
            np.clip(config.coefficient_detail_texture_coherence_floor, 0.0, 1.0)
        )
        detail_confidence *= coherence_floor + (
            1.0 - coherence_floor
        ) * texture_coherence
        if confidence_mode in {"amplitude_preserving_snr", "plateau_snr"}:
            detail_confidence = _amplitude_preserving_confidence_gate(
                detail_confidence,
                config.spatial_detail_confidence_gate_low,
                config.spatial_detail_confidence_gate_high,
            )
        detail *= np.clip(detail_confidence, 0.0, 1.0)
    elif confidence_mode == "none":
        detail_confidence = np.ones(detail.shape, dtype=np.float32)
        texture_coherence = np.ones(detail.shape, dtype=np.float32)
    else:
        raise ValueError(
            f"Unknown spatial_detail_confidence_mode "
            f"{config.spatial_detail_confidence_mode!r}"
        )
    detail = _project_map_to_observation_nullspace(
        detail,
        psf,
        min(1, max(0, int(config.spatial_detail_nullspace_iterations))),
    )

    p = low_guide.reshape(-1)
    values = cube.reshape(-1, cube.shape[2])
    valid_p = np.isfinite(p)
    p_mean = float(np.mean(p[valid_p])) if valid_p.any() else 0.0
    p_center = p - p_mean
    p_std = max(float(np.std(p[valid_p])) if valid_p.any() else 0.0, 1e-6)
    band_mean = np.nanmean(values, axis=0).astype(np.float32)
    band_std = np.maximum(np.nanstd(values, axis=0).astype(np.float32), 1e-6)
    beta = np.zeros(cube.shape[2], dtype=np.float32)
    correlations = np.zeros(cube.shape[2], dtype=np.float32)
    for band in range(cube.shape[2]):
        y = values[:, band]
        valid = valid_p & np.isfinite(y)
        if int(np.sum(valid)) < 64:
            continue
        yc = y[valid] - float(np.mean(y[valid]))
        pc = p_center[valid]
        covariance = float(np.mean(pc * yc))
        y_std = max(float(np.std(yc)), 1e-6)
        correlations[band] = covariance / (p_std * y_std)
        beta[band] = covariance / (p_std * p_std + 1e-6)

    correlation_floor = float(
        np.clip(config.spatial_detail_additive_correlation_floor, 0.0, 0.90)
    )
    reliability = np.sqrt(
        np.clip(
            (np.abs(correlations) - correlation_floor)
            / max(0.65 - correlation_floor, 0.10),
            0.0,
            1.0,
        )
    ).astype(np.float32)
    gain_clip = max(0.1, float(config.spatial_detail_additive_gain_clip))
    beta_limit = gain_clip * band_std / p_std
    scale = strength * np.clip(beta, -beta_limit, beta_limit) * reliability

    spectral_sigma = max(0.0, float(config.spatial_detail_additive_spectral_sigma))
    if spectral_sigma > 0.0 and scale.size > 2:
        scale = cv2.GaussianBlur(
            scale.reshape(1, -1),
            (0, 0),
            sigmaX=spectral_sigma,
            sigmaY=0.0,
            borderType=cv2.BORDER_REFLECT101,
        ).reshape(-1)

    detail_p95 = max(
        float(np.percentile(np.abs(detail[np.isfinite(detail)]), 95.0))
        if np.isfinite(detail).any()
        else 0.0,
        1e-6,
    )
    max_contribution = np.maximum(
        max(0.0, float(config.spatial_detail_additive_std_fraction)) * band_std,
        max(0.0, float(config.spatial_detail_additive_mean_fraction))
        * np.abs(band_mean),
    )
    scale_limit = max_contribution / detail_p95
    scale = np.clip(scale, -scale_limit, scale_limit).astype(np.float32)
    low_detail = degrade_spatial_map(detail, psf)
    return detail.astype(np.float32), scale, {
        "enabled": True,
        "mode": "band_adaptive_mtf_gsa",
        "spatial_detail_additive_strength": strength,
        "mtf_fine_sigma": fine_sigma,
        "correlation_floor": correlation_floor,
        "gain_clip": gain_clip,
        "spectral_sigma_bands": spectral_sigma,
        "confidence_mode": confidence_mode,
        "detail_confidence_mean": float(np.mean(detail_confidence)),
        "detail_confidence_p10": float(np.percentile(detail_confidence, 10.0)),
        "texture_coherence_mean": float(np.mean(texture_coherence)),
        "accepted_bands": int(np.sum(reliability > 0.0)),
        "band_correlation_min": float(np.min(correlations)),
        "band_correlation_median": float(np.median(correlations)),
        "band_correlation_max": float(np.max(correlations)),
        "spectral_scale_min": float(np.min(scale)),
        "spectral_scale_median": float(np.median(scale)),
        "spectral_scale_max": float(np.max(scale)),
        "detail_p95": detail_p95,
        "detail_lowres_rmse": float(np.sqrt(np.mean(low_detail * low_detail))),
        "detail_lowres_max_abs": float(np.max(np.abs(low_detail))),
    }


def back_project_modulated_product(
    coefficients: np.ndarray,
    low_coefficients: np.ndarray,
    target_low_cube: np.ndarray,
    basis: np.ndarray,
    mean_spectrum: np.ndarray,
    detail_gain: np.ndarray,
    additive_detail: np.ndarray,
    additive_spectral_scale: np.ndarray,
    psf: PsfModel,
    config: FusionConfig,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Back-project the *final modulated cube* to the HSI observation grid.

    This is deliberately different from the legacy ``D(gain) ~= 1`` proxy.
    The residual is computed after coefficient reconstruction, multiplicative
    log-HPM modulation, and additive dark-detail compensation, then projected
    back into the fixed spectral subspace.
    """

    constraint = str(config.coefficient_constraint).strip().lower()
    pure_simplex_constraint = constraint in {
        "simplex",
        "nonnegative_simplex",
        "abundance_simplex",
    }
    hybrid_simplex_constraint = constraint in {
        "hybrid_simplex",
        "simplex_plus_residual",
        "abundance_simplex_plus_residual",
    }
    simplex_constraint = pure_simplex_constraint or hybrid_simplex_constraint
    current = np.asarray(coefficients, dtype=np.float32).copy()
    low_coeff = np.asarray(low_coefficients, dtype=np.float32)
    simplex_count = (
        current.shape[2]
        if pure_simplex_constraint
        else min(max(1, int(config.rank)), current.shape[2])
    )

    def apply_constraint(candidate: np.ndarray) -> np.ndarray:
        values = np.asarray(candidate, dtype=np.float32).copy()
        if simplex_constraint:
            values[:, :, :simplex_count] = project_simplex(
                values[:, :, :simplex_count], axis=2
            )
        return values

    if simplex_constraint:
        current = apply_constraint(current)
        low_coeff = apply_constraint(low_coeff)
    target = np.asarray(target_low_cube, dtype=np.float32)
    spectral_basis = np.asarray(basis, dtype=np.float32)
    mean = np.asarray(mean_spectrum, dtype=np.float32)
    gain = np.asarray(detail_gain, dtype=np.float32)
    additive = np.asarray(additive_detail, dtype=np.float32)
    additive_scale = np.asarray(additive_spectral_scale, dtype=np.float32)
    iterations = max(
        0, int(config.spatial_detail_product_back_projection_iterations)
    )
    weight = float(
        np.clip(config.spatial_detail_product_back_projection_weight, 0.0, 1.0)
    )
    clip_sigma = max(
        0.0, float(config.spatial_detail_product_back_projection_clip_sigma)
    )
    low_gain = degrade_spatial_map(gain, psf)
    low_additive = degrade_spatial_map(additive, psf)
    basis_pseudoinverse = (
        np.linalg.pinv(spectral_basis.astype(np.float64), rcond=1e-6).astype(
            np.float32
        )
        if simplex_constraint
        else None
    )

    def predict_low_cube(candidate: np.ndarray) -> np.ndarray:
        weighted_coeff = degrade_coefficients(candidate * gain[:, :, None], psf)
        predicted = (
            np.einsum(
                "...k,kb->...b", weighted_coeff, spectral_basis, optimize=True
            )
            + low_gain[:, :, None] * mean[None, None, :]
            + low_additive[:, :, None] * additive_scale[None, None, :]
        )
        return predicted.astype(np.float32)

    before = predict_low_cube(current)
    rmse_before = float(np.sqrt(np.mean((before - target) ** 2)))
    low_std = np.maximum(np.std(low_coeff, axis=(0, 1)).astype(np.float32), 1e-5)
    history: list[dict[str, float]] = []
    for iteration in range(iterations):
        predicted = predict_low_cube(current)
        spectral_residual = target - predicted
        if simplex_constraint:
            coefficient_residual = np.einsum(
                "...b,bk->...k",
                spectral_residual,
                basis_pseudoinverse,
                optimize=True,
            ).astype(np.float32)
            coefficient_residual[:, :, :simplex_count] -= np.mean(
                coefficient_residual[:, :, :simplex_count],
                axis=2,
                keepdims=True,
            )
        else:
            coefficient_residual = np.einsum(
                "...b,kb->...k", spectral_residual, spectral_basis, optimize=True
            ).astype(np.float32)
        if clip_sigma > 0.0:
            limit = clip_sigma * low_std
            coefficient_residual = np.clip(
                coefficient_residual,
                -limit[None, None, :],
                limit[None, None, :],
            )
        correction = upsample_coefficients(
            coefficient_residual, current.shape[:2]
        ) / np.maximum(gain[:, :, None], 0.20)
        if simplex_constraint:
            current_rmse = float(np.sqrt(np.mean(spectral_residual * spectral_residual)))
            accepted = False
            accepted_weight = weight
            for _ in range(7):
                proposal = apply_constraint(
                    current + accepted_weight * correction
                )
                proposal_residual = target - predict_low_cube(proposal)
                proposal_rmse = float(
                    np.sqrt(np.mean(proposal_residual * proposal_residual))
                )
                if proposal_rmse <= current_rmse + 1e-12:
                    current = proposal
                    accepted = True
                    break
                accepted_weight *= 0.5
            if not accepted:
                break
        else:
            current += weight * correction
            accepted_weight = weight
        history.append(
            {
                "iteration": float(iteration + 1),
                "final_product_lowres_rmse": float(
                    np.sqrt(np.mean(spectral_residual * spectral_residual))
                ),
                "coefficient_residual_rmse": float(
                    np.sqrt(np.mean(coefficient_residual * coefficient_residual))
                ),
                "accepted_weight": float(accepted_weight),
            }
        )
    after = predict_low_cube(current)
    rmse_after = float(np.sqrt(np.mean((after - target) ** 2)))
    return current.astype(np.float32), {
        "enabled": bool(iterations > 0),
        "method": "final_modulated_cube_observation_cycle",
        "iterations": iterations,
        "weight": weight,
        "clip_sigma": clip_sigma,
        "lowres_rmse_before": rmse_before,
        "lowres_rmse_after": rmse_after,
        "history": history,
        "proxy_D_gain_used": False,
        "coefficient_constraint": constraint,
        "simplex_component_count": int(simplex_count) if simplex_constraint else 0,
        "spectral_residual_solver": (
            "basis_pseudoinverse_then_simplex_tangent"
            if simplex_constraint
            else "orthonormal_basis_transpose"
        ),
    }


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
        confidence = _rgb_weights(rgb, config.rgb_edge_sigma, config)[2]
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
    wx, wy, rgb_confidence = _rgb_weights(rgb, config.rgb_edge_sigma, config)
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
        "intrinsic_detail_enabled": bool(config.intrinsic_detail_enabled),
        "intrinsic_log_epsilon": float(config.intrinsic_log_epsilon),
        "dark_detail_boost": float(config.dark_detail_boost),
        "rgb_confidence_mean": float(np.mean(rgb_confidence)),
        "rgb_confidence_p05": float(np.percentile(rgb_confidence, 5)),
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
