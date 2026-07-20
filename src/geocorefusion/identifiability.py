"""Candidate-independent RGB-to-spectral detail identifiability diagnostics.

The functions in this module deliberately operate only on registered RGB and
native low-resolution NIR/SWIR observations.  A fused high-resolution
candidate is neither accepted nor inspected.  This keeps the estimated
band-specific coherent amplitude frozen before candidate evaluation and
prevents a visually sharp product from defining its own target.

The observable relationship is evaluated after the RGB has passed through
the target band's PSF/downsampling model and after both modalities have been
filtered by the same low-grid log-domain Difference-of-Gaussians band-pass.
Spatial blocked cross-validation scores only an inward-eroded test core and
also excludes an outward guard from training.  Ridge and train-fold PCA-ridge
baselines, toroidal spatial-shift nulls, and block-bootstrap confidence
intervals are reported separately.

This is an identifiability screen, not proof of unknown HR NIR/SWIR truth.
Even an ``identifiable`` result supports only the shared structure observable
inside the selected low-resolution passband.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Literal, Sequence

import cv2
import numpy as np

from .dataset import normalize_rgb
from .degradation import PsfModel, degrade_spatial_map


IdentifiabilityStatus = Literal[
    "identifiable",
    "weakly_identifiable",
    "unidentifiable",
]


@dataclass(frozen=True, slots=True)
class IdentifiabilityConfig:
    """Frozen settings for candidate-independent band identifiability.

    Sigma values are expressed on the common native spectral grid.  The
    default exclusion gap covers three times the broad DoG sigma.  Scored test
    blocks are eroded by this gap and training support is excluded beyond the
    original test boundary by the same gap, preventing the precomputed DoG
    support from straddling train and scored-test pixels.

    The numeric epsilon defaults are engineering fallbacks for synthetic and
    smoke tests.  A real experiment must replace/freeze them from sensor- and
    band-specific dark/flat measurements before any candidate is evaluated;
    ``0.012`` is not asserted to be a calibrated noise floor.
    """

    bandpass_sigma_low_px: float = 0.6
    bandpass_sigma_high_px: float = 2.0
    epsilon_rgb: float = 0.012
    epsilon_band: float = 0.012
    block_rows: int = 4
    block_cols: int = 4
    exclusion_gap_px: int | None = None
    ridge_alpha: float = 0.03
    lowrank_feature_ranks: tuple[int, ...] = (1, 2)
    primary_model: str = "rgb_ridge"
    minimum_psf_coverage: float = 0.95
    minimum_valid_pixels: int = 256
    minimum_train_pixels: int = 192
    minimum_test_pixels: int = 24
    null_repeats: int = 31
    bootstrap_repeats: int = 200
    confidence_level: float = 0.95
    random_seed: int = 20260720

    identifiable_cv_r2_min: float = 0.15
    weak_cv_r2_min: float = 0.02
    identifiable_abs_cv_rho_min: float = 0.45
    weak_abs_cv_rho_min: float = 0.20
    identifiable_null_p_max: float = 0.05
    weak_null_p_max: float = 0.20
    identifiable_amplitude_fraction_min: float = 0.20
    weak_amplitude_fraction_min: float = 0.10
    identifiable_sign_stability_min: float = 0.75
    weak_sign_stability_min: float = 0.60
    require_null_for_classification: bool = True
    require_amplitude_ci_for_identifiable: bool = True

    def __post_init__(self) -> None:
        if self.bandpass_sigma_low_px < 0.0:
            raise ValueError("bandpass_sigma_low_px must be non-negative")
        if self.bandpass_sigma_high_px <= self.bandpass_sigma_low_px:
            raise ValueError(
                "bandpass_sigma_high_px must exceed bandpass_sigma_low_px"
            )
        if self.epsilon_rgb <= 0.0 or self.epsilon_band <= 0.0:
            raise ValueError("log-domain epsilons must be positive")
        if self.block_rows < 2 or self.block_cols < 2:
            raise ValueError("blocked CV requires at least a 2x2 block grid")
        if self.exclusion_gap_px is not None and self.exclusion_gap_px < 0:
            raise ValueError("exclusion_gap_px cannot be negative")
        if self.ridge_alpha < 0.0:
            raise ValueError("ridge_alpha cannot be negative")
        if any(rank < 1 for rank in self.lowrank_feature_ranks):
            raise ValueError("lowrank_feature_ranks must contain positive ranks")
        if not 0.0 < self.minimum_psf_coverage <= 1.0:
            raise ValueError("minimum_psf_coverage must lie in (0, 1]")
        for name in (
            "minimum_valid_pixels",
            "minimum_train_pixels",
            "minimum_test_pixels",
        ):
            if int(getattr(self, name)) < 1:
                raise ValueError(f"{name} must be positive")
        if self.null_repeats < 0 or self.bootstrap_repeats < 0:
            raise ValueError("null/bootstrap repeat counts cannot be negative")
        if not 0.0 < self.confidence_level < 1.0:
            raise ValueError("confidence_level must lie strictly between 0 and 1")
        if self.identifiable_cv_r2_min < self.weak_cv_r2_min:
            raise ValueError("identifiable_cv_r2_min cannot be below the weak bound")
        if self.identifiable_abs_cv_rho_min < self.weak_abs_cv_rho_min:
            raise ValueError(
                "identifiable_abs_cv_rho_min cannot be below the weak bound"
            )
        if self.identifiable_null_p_max > self.weak_null_p_max:
            raise ValueError("identifiable_null_p_max cannot exceed the weak bound")
        if (
            self.identifiable_amplitude_fraction_min
            < self.weak_amplitude_fraction_min
        ):
            raise ValueError(
                "identifiable_amplitude_fraction_min cannot be below the weak bound"
            )
        if self.identifiable_sign_stability_min < self.weak_sign_stability_min:
            raise ValueError(
                "identifiable_sign_stability_min cannot be below the weak bound"
            )

    @property
    def resolved_exclusion_gap_px(self) -> int:
        if self.exclusion_gap_px is not None:
            return int(self.exclusion_gap_px)
        return int(np.ceil(3.0 * self.bandpass_sigma_high_px))


@dataclass(slots=True)
class _LinearFit:
    coefficients: np.ndarray
    intercept: float
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    pca_components: np.ndarray

    def predict(self, features: np.ndarray) -> np.ndarray:
        return np.asarray(features, dtype=np.float64) @ self.coefficients + self.intercept


@dataclass(slots=True)
class _BlockedCvResult:
    prediction: np.ndarray
    baseline_prediction: np.ndarray
    fold_id: np.ndarray
    metrics: dict[str, Any]


@dataclass(slots=True)
class _CrossFittedAmplitude:
    prediction: np.ndarray
    baseline_prediction: np.ndarray
    fold_id: np.ndarray
    training_fold_slopes: np.ndarray
    training_fold_weights: np.ndarray
    metrics: dict[str, Any]


def _finite_correlation(a: np.ndarray, b: np.ndarray) -> float:
    aa = np.asarray(a, dtype=np.float64).reshape(-1)
    bb = np.asarray(b, dtype=np.float64).reshape(-1)
    valid = np.isfinite(aa) & np.isfinite(bb)
    if int(np.sum(valid)) < 3:
        return float("nan")
    x = aa[valid] - float(np.mean(aa[valid]))
    y = bb[valid] - float(np.mean(bb[valid]))
    denominator = float(np.sqrt(np.sum(x * x) * np.sum(y * y)))
    if denominator <= 1e-12:
        return float("nan")
    return float(np.clip(np.sum(x * y) / denominator, -1.0, 1.0))


def _slope(reference: np.ndarray, target: np.ndarray) -> float:
    x = np.asarray(reference, dtype=np.float64).reshape(-1)
    y = np.asarray(target, dtype=np.float64).reshape(-1)
    valid = np.isfinite(x) & np.isfinite(y)
    if int(np.sum(valid)) < 3:
        return float("nan")
    x = x[valid] - float(np.mean(x[valid]))
    y = y[valid] - float(np.mean(y[valid]))
    variance = float(np.sum(x * x))
    return float(np.sum(x * y) / variance) if variance > 1e-12 else float("nan")


def _weighted_gaussian(
    image: np.ndarray,
    valid: np.ndarray,
    sigma_px: float,
) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(image, dtype=np.float32)
    support = np.asarray(valid, dtype=bool) & np.isfinite(values)
    if sigma_px <= 1e-8:
        return np.where(support, values, 0.0).astype(np.float32), support
    numerator = cv2.GaussianBlur(
        np.where(support, values, 0.0).astype(np.float32),
        (0, 0),
        float(sigma_px),
        borderType=cv2.BORDER_REFLECT101,
    )
    denominator = cv2.GaussianBlur(
        support.astype(np.float32),
        (0, 0),
        float(sigma_px),
        borderType=cv2.BORDER_REFLECT101,
    )
    blurred = numerator / np.maximum(denominator, 1e-6)
    return blurred.astype(np.float32), denominator >= 0.995


def _log_dog_bandpass(
    image: np.ndarray,
    *,
    epsilon: float,
    sigma_low_px: float,
    sigma_high_px: float,
) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(image, dtype=np.float32)
    valid = np.isfinite(values) & (values >= 0.0)
    log_values = np.log(np.where(valid, values, 0.0) + float(epsilon)).astype(
        np.float32
    )
    narrow, valid_narrow = _weighted_gaussian(log_values, valid, sigma_low_px)
    broad, valid_broad = _weighted_gaussian(log_values, valid, sigma_high_px)
    output_valid = valid_narrow & valid_broad
    detail = narrow - broad
    detail[~output_valid] = np.nan
    return detail.astype(np.float32), output_valid


def _degrade_rgb(
    rgb: np.ndarray,
    psf: PsfModel,
    minimum_coverage: float,
) -> tuple[np.ndarray, np.ndarray]:
    rgb01 = normalize_rgb(np.asarray(rgb))
    if tuple(rgb01.shape[:2]) != tuple(psf.high_shape):
        raise ValueError(
            f"RGB shape {rgb01.shape[:2]} does not match PSF high_shape {psf.high_shape}"
        )
    output = np.empty(psf.low_shape + (3,), dtype=np.float32)
    valid_output = np.ones(psf.low_shape, dtype=bool)
    for channel in range(3):
        values = rgb01[:, :, channel]
        valid = np.isfinite(values)
        numerator = degrade_spatial_map(np.where(valid, values, 0.0), psf)
        coverage = degrade_spatial_map(valid.astype(np.float32), psf)
        output[:, :, channel] = numerator / np.maximum(coverage, 1e-6)
        valid_output &= coverage >= float(minimum_coverage)
    return output, valid_output


def _common_bandpass_fields(
    rgb: np.ndarray,
    band_low: np.ndarray,
    psf: PsfModel,
    config: IdentifiabilityConfig,
    epsilon_band: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    band = np.asarray(band_low, dtype=np.float32)
    if band.shape != tuple(psf.low_shape):
        raise ValueError(
            f"band_low shape {band.shape} does not match PSF low_shape {psf.low_shape}"
        )
    rgb_low, rgb_psf_valid = _degrade_rgb(
        rgb,
        psf,
        config.minimum_psf_coverage,
    )
    features: list[np.ndarray] = []
    feature_valid = rgb_psf_valid.copy()
    for channel in range(3):
        detail, valid = _log_dog_bandpass(
            rgb_low[:, :, channel],
            epsilon=config.epsilon_rgb,
            sigma_low_px=config.bandpass_sigma_low_px,
            sigma_high_px=config.bandpass_sigma_high_px,
        )
        features.append(detail)
        feature_valid &= valid
    luminance = (
        0.299 * rgb_low[:, :, 0]
        + 0.587 * rgb_low[:, :, 1]
        + 0.114 * rgb_low[:, :, 2]
    )
    guide, guide_valid = _log_dog_bandpass(
        luminance,
        epsilon=config.epsilon_rgb,
        sigma_low_px=config.bandpass_sigma_low_px,
        sigma_high_px=config.bandpass_sigma_high_px,
    )
    target, target_valid = _log_dog_bandpass(
        band,
        epsilon=epsilon_band,
        sigma_low_px=config.bandpass_sigma_low_px,
        sigma_high_px=config.bandpass_sigma_high_px,
    )
    valid = feature_valid & guide_valid & target_valid
    gap = config.resolved_exclusion_gap_px
    border = max(gap, int(np.ceil(3.0 * config.bandpass_sigma_high_px)))
    if border > 0:
        valid[:border, :] = False
        valid[-border:, :] = False
        valid[:, :border] = False
        valid[:, -border:] = False
    metadata = {
        "domain": "native_radiometry_log_DoG",
        "rgb_scaled_without_percentile_stretch": True,
        "rgb_psf_degraded_before_log": True,
        "same_low_grid_bandpass_for_rgb_and_spectral_band": True,
        "sigma_low_px": float(config.bandpass_sigma_low_px),
        "sigma_high_px": float(config.bandpass_sigma_high_px),
        "epsilon_rgb": float(config.epsilon_rgb),
        "epsilon_band": float(epsilon_band),
        "epsilon_requirement": (
            "Engineering defaults are not calibrated noise floors; real runs must "
            "freeze sensor- and band-specific values from dark/flat measurements "
            "before candidate evaluation."
        ),
        "excluded_border_px": int(border),
        "psf": psf.to_dict(),
    }
    return (
        np.stack(features, axis=2).astype(np.float32),
        guide.astype(np.float32),
        target.astype(np.float32),
        {"valid": valid, "metadata": metadata},
    )


def _block_map(shape: tuple[int, int], rows: int, cols: int) -> np.ndarray:
    yy, xx = np.indices(shape)
    row_id = np.minimum(yy * rows // shape[0], rows - 1)
    col_id = np.minimum(xx * cols // shape[1], cols - 1)
    return (row_id * cols + col_id).astype(np.int32)


def _expanded_block_mask(
    fold_map: np.ndarray,
    fold: int,
    gap: int,
) -> np.ndarray:
    test = fold_map == int(fold)
    yy, xx = np.where(test)
    expanded = np.zeros(test.shape, dtype=bool)
    if yy.size == 0:
        return expanded
    y0 = max(0, int(np.min(yy)) - gap)
    y1 = min(test.shape[0], int(np.max(yy)) + gap + 1)
    x0 = max(0, int(np.min(xx)) - gap)
    x1 = min(test.shape[1], int(np.max(xx)) + gap + 1)
    expanded[y0:y1, x0:x1] = True
    return expanded


def _eroded_block_mask(
    fold_map: np.ndarray,
    fold: int,
    gap: int,
) -> np.ndarray:
    """Return the scored block core after removing the filtered boundary."""

    block = fold_map == int(fold)
    yy, xx = np.where(block)
    eroded = np.zeros(block.shape, dtype=bool)
    if yy.size == 0:
        return eroded
    y0 = int(np.min(yy)) + gap
    y1 = int(np.max(yy)) - gap + 1
    x0 = int(np.min(xx)) + gap
    x1 = int(np.max(xx)) - gap + 1
    if y0 >= y1 or x0 >= x1:
        return eroded
    eroded[y0:y1, x0:x1] = True
    return eroded


def _fit_linear(
    features: np.ndarray,
    target: np.ndarray,
    *,
    ridge_alpha: float,
    pca_rank: int | None,
) -> _LinearFit:
    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(target, dtype=np.float64).reshape(-1)
    feature_mean = np.mean(x, axis=0)
    feature_scale = np.maximum(np.std(x, axis=0), 1e-8)
    standardized = (x - feature_mean[None, :]) / feature_scale[None, :]
    target_mean = float(np.mean(y))
    centered_target = y - target_mean
    feature_count = standardized.shape[1]
    rank = feature_count if pca_rank is None else min(max(1, int(pca_rank)), feature_count)
    if rank < feature_count:
        _, _, right = np.linalg.svd(standardized, full_matrices=False)
        components = right[:rank, :]
    else:
        components = np.eye(feature_count, dtype=np.float64)
    latent = standardized @ components.T
    normal = latent.T @ latent / float(latent.shape[0])
    normal += max(float(ridge_alpha), 1e-12) * np.eye(rank, dtype=np.float64)
    rhs = latent.T @ centered_target / float(latent.shape[0])
    latent_coefficients = np.linalg.solve(normal, rhs)
    standardized_coefficients = components.T @ latent_coefficients
    coefficients = standardized_coefficients / feature_scale
    intercept = float(target_mean - feature_mean @ coefficients)
    return _LinearFit(
        coefficients=coefficients,
        intercept=intercept,
        feature_mean=feature_mean,
        feature_scale=feature_scale,
        pca_components=components,
    )


def _prediction_metrics(
    observed: np.ndarray,
    predicted: np.ndarray,
    baseline: np.ndarray,
) -> dict[str, float | int]:
    y = np.asarray(observed, dtype=np.float64).reshape(-1)
    p = np.asarray(predicted, dtype=np.float64).reshape(-1)
    b = np.asarray(baseline, dtype=np.float64).reshape(-1)
    valid = np.isfinite(y) & np.isfinite(p) & np.isfinite(b)
    count = int(np.sum(valid))
    if count < 3:
        return {
            "pixel_count": count,
            "cv_predictive_r2": float("nan"),
            "cv_pearson_rho": float("nan"),
            "rmse": float("nan"),
            "normalized_rmse": float("nan"),
        }
    y = y[valid]
    p = p[valid]
    b = b[valid]
    model_ss = float(np.sum((p - y) ** 2))
    baseline_ss = float(np.sum((b - y) ** 2))
    target_std = float(np.std(y))
    return {
        "pixel_count": count,
        "cv_predictive_r2": (
            float(1.0 - model_ss / baseline_ss)
            if baseline_ss > 1e-12
            else float("nan")
        ),
        "cv_pearson_rho": _finite_correlation(y, p),
        "rmse": float(np.sqrt(model_ss / count)),
        "normalized_rmse": (
            float(np.sqrt(model_ss / count) / target_std)
            if target_std > 1e-12
            else float("nan")
        ),
    }


def _blocked_cv(
    features: np.ndarray,
    target: np.ndarray,
    valid: np.ndarray,
    config: IdentifiabilityConfig,
    *,
    pca_rank: int | None,
    include_fold_details: bool = True,
) -> _BlockedCvResult:
    shape = target.shape
    fold_map = _block_map(shape, config.block_rows, config.block_cols)
    prediction = np.full(shape, np.nan, dtype=np.float64)
    baseline = np.full(shape, np.nan, dtype=np.float64)
    prediction_fold = np.full(shape, -1, dtype=np.int32)
    fold_details: list[dict[str, Any]] = []
    completed_folds = 0
    gap = config.resolved_exclusion_gap_px
    for fold in range(config.block_rows * config.block_cols):
        test_core = _eroded_block_mask(fold_map, fold, gap)
        test = np.asarray(valid, dtype=bool) & test_core
        excluded = _expanded_block_mask(fold_map, fold, gap)
        train = np.asarray(valid, dtype=bool) & ~excluded
        train_count = int(np.sum(train))
        test_count = int(np.sum(test))
        if (
            train_count < config.minimum_train_pixels
            or test_count < config.minimum_test_pixels
        ):
            continue
        fit = _fit_linear(
            features[train],
            target[train],
            ridge_alpha=config.ridge_alpha,
            pca_rank=pca_rank,
        )
        prediction[test] = fit.predict(features[test])
        baseline[test] = float(np.mean(target[train]))
        prediction_fold[test] = fold
        completed_folds += 1
        if include_fold_details:
            fold_metric = _prediction_metrics(
                target[test],
                prediction[test],
                baseline[test],
            )
            fold_details.append(
                {
                    "fold": int(fold),
                    "train_pixel_count": train_count,
                    "test_pixel_count": test_count,
                    "test_boundary_inset_px": int(gap),
                    "training_target_mean_baseline": float(np.mean(target[train])),
                    **fold_metric,
                }
            )
    predicted = np.isfinite(prediction) & np.asarray(valid, dtype=bool)
    aggregate = _prediction_metrics(
        target[predicted],
        prediction[predicted],
        baseline[predicted],
    )
    full_fit: dict[str, Any] = {}
    if int(np.sum(valid)) >= config.minimum_train_pixels:
        fit = _fit_linear(
            features[valid],
            target[valid],
            ridge_alpha=config.ridge_alpha,
            pca_rank=pca_rank,
        )
        full_fit = {
            "coefficients_log_band_per_rgb_feature": fit.coefficients.astype(float).tolist(),
            "intercept": float(fit.intercept),
            "feature_mean": fit.feature_mean.astype(float).tolist(),
            "feature_scale": fit.feature_scale.astype(float).tolist(),
            "pca_components": fit.pca_components.astype(float).tolist(),
        }
    metrics: dict[str, Any] = {
        **aggregate,
        "fold_count": int(completed_folds),
        "requested_fold_count": int(config.block_rows * config.block_cols),
        "feature_rank": int(
            features.shape[2]
            if pca_rank is None
            else min(int(pca_rank), features.shape[2])
        ),
        "ridge_alpha": float(config.ridge_alpha),
        "predictive_r2_baseline": "per_fold_training_target_mean",
        "test_boundary_inset_px": int(gap),
        "full_observation_fit": full_fit,
    }
    if include_fold_details:
        metrics["folds"] = fold_details
    return _BlockedCvResult(
        prediction=prediction,
        baseline_prediction=baseline,
        fold_id=prediction_fold,
        metrics=metrics,
    )


def _cross_fitted_luminance_amplitude(
    guide: np.ndarray,
    target: np.ndarray,
    valid: np.ndarray,
    config: IdentifiabilityConfig,
) -> _CrossFittedAmplitude:
    """Fit the signed guide slope on training blocks only for every test block."""

    shape = target.shape
    fold_map = _block_map(shape, config.block_rows, config.block_cols)
    prediction = np.full(shape, np.nan, dtype=np.float64)
    baseline = np.full(shape, np.nan, dtype=np.float64)
    prediction_fold = np.full(shape, -1, dtype=np.int32)
    slopes: list[float] = []
    weights: list[float] = []
    fold_details: list[dict[str, Any]] = []
    gap = config.resolved_exclusion_gap_px
    guide_feature = np.asarray(guide, dtype=np.float64)[:, :, None]
    for fold in range(config.block_rows * config.block_cols):
        test_core = _eroded_block_mask(fold_map, fold, gap)
        test = np.asarray(valid, dtype=bool) & test_core
        excluded = _expanded_block_mask(fold_map, fold, gap)
        train = np.asarray(valid, dtype=bool) & ~excluded
        train_count = int(np.sum(train))
        test_count = int(np.sum(test))
        if (
            train_count < config.minimum_train_pixels
            or test_count < config.minimum_test_pixels
        ):
            continue
        fit = _fit_linear(
            guide_feature[train],
            target[train],
            ridge_alpha=config.ridge_alpha,
            pca_rank=None,
        )
        slope = float(fit.coefficients[0])
        prediction[test] = fit.predict(guide_feature[test])
        baseline[test] = float(np.mean(target[train]))
        prediction_fold[test] = fold
        slopes.append(slope)
        weights.append(float(test_count))
        fold_details.append(
            {
                "fold": int(fold),
                "train_pixel_count": train_count,
                "test_pixel_count": test_count,
                "test_boundary_inset_px": int(gap),
                "training_slope_log_band_per_log_rgb": slope,
                "training_intercept": float(fit.intercept),
                "training_target_mean_baseline": float(np.mean(target[train])),
            }
        )
    predicted = np.isfinite(prediction) & np.asarray(valid, dtype=bool)
    metrics: dict[str, Any] = {
        **_prediction_metrics(
            target[predicted],
            prediction[predicted],
            baseline[predicted],
        ),
        "fold_count": int(len(fold_details)),
        "requested_fold_count": int(config.block_rows * config.block_cols),
        "ridge_alpha": float(config.ridge_alpha),
        "predictive_r2_baseline": "per_fold_training_target_mean",
        "test_boundary_inset_px": int(gap),
        "folds": fold_details,
    }
    return _CrossFittedAmplitude(
        prediction=prediction,
        baseline_prediction=baseline,
        fold_id=prediction_fold,
        training_fold_slopes=np.asarray(slopes, dtype=np.float64),
        training_fold_weights=np.asarray(weights, dtype=np.float64),
        metrics=metrics,
    )


def _percentile_interval(
    values: Sequence[float],
    confidence_level: float,
) -> list[float] | None:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size < 4:
        return None
    tail = 50.0 * (1.0 - float(confidence_level))
    low, high = np.percentile(finite, [tail, 100.0 - tail])
    return [float(low), float(high)]


def _cluster_bootstrap(
    target: np.ndarray,
    prediction: np.ndarray,
    baseline: np.ndarray,
    guide: np.ndarray,
    fold_id: np.ndarray,
    config: IdentifiabilityConfig,
) -> dict[str, Any]:
    valid = (
        np.isfinite(target)
        & np.isfinite(prediction)
        & np.isfinite(baseline)
        & np.isfinite(guide)
        & (fold_id >= 0)
    )
    blocks = np.unique(fold_id[valid])
    if config.bootstrap_repeats <= 0 or blocks.size < 4:
        return {
            "method": "spatial_block_bootstrap",
            "repeat_count": 0,
            "confidence_level": float(config.confidence_level),
            "cv_predictive_r2_ci": None,
            "cv_pearson_rho_ci": None,
            "expected_coherent_amplitude_ci": None,
        }
    indices_by_block = {
        int(block): np.flatnonzero(valid.reshape(-1) & (fold_id.reshape(-1) == block))
        for block in blocks
    }
    flat_target = target.reshape(-1)
    flat_prediction = prediction.reshape(-1)
    flat_baseline = baseline.reshape(-1)
    flat_guide = guide.reshape(-1)
    rng = np.random.default_rng(config.random_seed + 7919)
    r2_values: list[float] = []
    rho_values: list[float] = []
    amplitude_values: list[float] = []
    for _ in range(config.bootstrap_repeats):
        sampled_blocks = rng.choice(blocks, size=blocks.size, replace=True)
        sampled_indices = np.concatenate(
            [indices_by_block[int(block)] for block in sampled_blocks]
        )
        metrics = _prediction_metrics(
            flat_target[sampled_indices],
            flat_prediction[sampled_indices],
            flat_baseline[sampled_indices],
        )
        r2_values.append(float(metrics["cv_predictive_r2"]))
        rho_values.append(float(metrics["cv_pearson_rho"]))
        amplitude_values.append(
            _slope(flat_guide[sampled_indices], flat_target[sampled_indices])
        )
    return {
        "method": "spatial_block_bootstrap",
        "resampling_unit": "held_out_spatial_block",
        "repeat_count": int(config.bootstrap_repeats),
        "confidence_level": float(config.confidence_level),
        "cv_predictive_r2_ci": _percentile_interval(
            r2_values, config.confidence_level
        ),
        "cv_pearson_rho_ci": _percentile_interval(
            rho_values, config.confidence_level
        ),
        "expected_coherent_amplitude_ci": _percentile_interval(
            amplitude_values, config.confidence_level
        ),
    }


def _spatial_shift_null(
    features: np.ndarray,
    target: np.ndarray,
    feature_valid: np.ndarray,
    target_valid: np.ndarray,
    observed_cv_r2: float,
    config: IdentifiabilityConfig,
    *,
    pca_rank: int | None,
) -> dict[str, Any]:
    if config.null_repeats <= 0:
        return {
            "method": "toroidal_spatial_shift",
            "repeat_count": 0,
            "empirical_p_value": None,
            "cv_r2_distribution": [],
            "cv_r2_95th_percentile": None,
            "shifts_yx_px": [],
        }
    height, width = target.shape
    rng = np.random.default_rng(config.random_seed + 1543)
    block_height = max(1, height // config.block_rows)
    block_width = max(1, width // config.block_cols)
    minimum_shift = max(
        1,
        config.resolved_exclusion_gap_px + 1,
        min(block_height, block_width) // 2,
    )
    shifts: list[tuple[int, int]] = []
    used: set[tuple[int, int]] = set()
    attempts = 0
    maximum_attempts = max(1000, config.null_repeats * 100)
    while len(shifts) < config.null_repeats and attempts < maximum_attempts:
        attempts += 1
        dy = int(rng.integers(0, height))
        dx = int(rng.integers(0, width))
        circular_y = min(dy, height - dy)
        circular_x = min(dx, width - dx)
        if max(circular_y, circular_x) < minimum_shift or (dy, dx) in used:
            continue
        used.add((dy, dx))
        shifts.append((dy, dx))
    null_values: list[float] = []
    accepted_shifts: list[list[int]] = []
    for dy, dx in shifts:
        shifted_features = np.roll(features, shift=(dy, dx), axis=(0, 1))
        shifted_valid = np.roll(feature_valid, shift=(dy, dx), axis=(0, 1))
        valid = shifted_valid & target_valid
        result = _blocked_cv(
            shifted_features,
            target,
            valid,
            config,
            pca_rank=pca_rank,
            include_fold_details=False,
        )
        value = float(result.metrics["cv_predictive_r2"])
        if np.isfinite(value):
            null_values.append(value)
            accepted_shifts.append([int(dy), int(dx)])
    finite_null = np.asarray(null_values, dtype=np.float64)
    if finite_null.size and np.isfinite(observed_cv_r2):
        empirical_p = float(
            (1 + np.sum(finite_null >= float(observed_cv_r2)))
            / (finite_null.size + 1)
        )
        percentile_95 = float(np.percentile(finite_null, 95.0))
    else:
        empirical_p = None
        percentile_95 = None
    return {
        "method": "toroidal_spatial_shift",
        "preserves_rgb_spatial_autocorrelation": True,
        "minimum_circular_shift_px": int(minimum_shift),
        "requested_repeat_count": int(config.null_repeats),
        "repeat_count": int(finite_null.size),
        "empirical_p_value": empirical_p,
        "cv_r2_distribution": finite_null.astype(float).tolist(),
        "cv_r2_95th_percentile": percentile_95,
        "shifts_yx_px": accepted_shifts,
    }


def _amplitude_summary(
    guide: np.ndarray,
    target: np.ndarray,
    cv_result: _BlockedCvResult,
    cross_fitted: _CrossFittedAmplitude,
    bootstrap: dict[str, Any],
) -> dict[str, Any]:
    valid = (
        np.isfinite(guide)
        & np.isfinite(target)
        & np.isfinite(cv_result.prediction)
        & (cv_result.fold_id >= 0)
    )
    direct_oof_slope = _slope(guide[valid], target[valid])
    slope_values = cross_fitted.training_fold_slopes
    slope_weights = cross_fitted.training_fold_weights
    if slope_values.size and float(np.sum(slope_weights)) > 0.0:
        estimate = float(np.sum(slope_values * slope_weights) / np.sum(slope_weights))
    else:
        estimate = float("nan")
    guide_std = float(np.std(guide[valid])) if int(np.sum(valid)) else float("nan")
    target_std = float(np.std(target[valid])) if int(np.sum(valid)) else float("nan")
    fraction = (
        float(abs(estimate) * guide_std / target_std)
        if np.isfinite(estimate) and target_std > 1e-12
        else float("nan")
    )
    block_slopes: list[float] = []
    for fold in np.unique(cv_result.fold_id[valid]):
        mask = valid & (cv_result.fold_id == fold)
        if int(np.sum(mask)) >= 8:
            block_slopes.append(_slope(guide[mask], target[mask]))
    finite_slopes = np.asarray(block_slopes, dtype=np.float64)
    finite_slopes = finite_slopes[np.isfinite(finite_slopes)]
    training_slopes = cross_fitted.training_fold_slopes
    finite_training_slopes = training_slopes[np.isfinite(training_slopes)]
    if (
        finite_training_slopes.size
        and np.isfinite(estimate)
        and abs(estimate) > 1e-12
    ):
        training_sign_stability = float(
            np.mean(np.sign(finite_training_slopes) == np.sign(estimate))
        )
    else:
        training_sign_stability = float("nan")
    if finite_slopes.size and np.isfinite(estimate) and abs(estimate) > 1e-12:
        held_out_sign_stability = float(
            np.mean(np.sign(finite_slopes) == np.sign(estimate))
        )
    else:
        held_out_sign_stability = float("nan")
    return {
        "guide": "Rec.601_luminance_native_log_DoG",
        "estimate_log_band_per_log_rgb": float(estimate),
        "direct_oof_slope_log_band_per_log_rgb": float(direct_oof_slope),
        "confidence_interval": bootstrap["expected_coherent_amplitude_ci"],
        "normalized_fraction_of_target_detail_rms": fraction,
        "training_fold_sign_stability": training_sign_stability,
        "held_out_block_sign_stability": held_out_sign_stability,
        "training_fold_slopes": finite_training_slopes.astype(float).tolist(),
        "held_out_block_slopes": finite_slopes.astype(float).tolist(),
        "cross_fitted_luminance_relation": cross_fitted.metrics,
        "frozen_value_log_band_per_log_rgb": None,
        "note": (
            "Each test block uses a slope fitted only outside its guarded test "
            "neighborhood; the frozen value is the test-support-weighted mean of "
            "those training-fold slopes. It is fixed before any fused candidate "
            "is inspected and replaces unit RGB beta as the band-specific reference."
        ),
    }


def _ci_excludes_zero(interval: list[float] | None) -> bool:
    return bool(
        interval is not None
        and len(interval) == 2
        and (float(interval[0]) > 0.0 or float(interval[1]) < 0.0)
    )


def _classify(
    primary_metrics: dict[str, Any],
    amplitude: dict[str, Any],
    null: dict[str, Any],
    config: IdentifiabilityConfig,
) -> tuple[IdentifiabilityStatus, dict[str, Any]]:
    r2 = float(primary_metrics.get("cv_predictive_r2", np.nan))
    rho = abs(float(primary_metrics.get("cv_pearson_rho", np.nan)))
    fraction = float(amplitude["normalized_fraction_of_target_detail_rms"])
    sign_stability = float(amplitude["held_out_block_sign_stability"])
    p_value_raw = null.get("empirical_p_value")
    p_value = float(p_value_raw) if p_value_raw is not None else float("nan")
    amplitude_ci_excludes_zero = _ci_excludes_zero(amplitude["confidence_interval"])
    null_strong = (
        np.isfinite(p_value) and p_value <= config.identifiable_null_p_max
    )
    null_weak = np.isfinite(p_value) and p_value <= config.weak_null_p_max
    if not config.require_null_for_classification:
        null_strong = null_strong or not np.isfinite(p_value)
        null_weak = null_weak or not np.isfinite(p_value)
    ci_strong = amplitude_ci_excludes_zero
    if not config.require_amplitude_ci_for_identifiable:
        ci_strong = True
    strong_checks = {
        "cv_r2": bool(np.isfinite(r2) and r2 >= config.identifiable_cv_r2_min),
        "abs_cv_rho": bool(
            np.isfinite(rho) and rho >= config.identifiable_abs_cv_rho_min
        ),
        "spatial_null": bool(null_strong),
        "coherent_amplitude_fraction": bool(
            np.isfinite(fraction)
            and fraction >= config.identifiable_amplitude_fraction_min
        ),
        "sign_stability": bool(
            np.isfinite(sign_stability)
            and sign_stability >= config.identifiable_sign_stability_min
        ),
        "amplitude_ci_excludes_zero": bool(ci_strong),
    }
    weak_checks = {
        "cv_r2": bool(np.isfinite(r2) and r2 >= config.weak_cv_r2_min),
        "abs_cv_rho": bool(np.isfinite(rho) and rho >= config.weak_abs_cv_rho_min),
        "spatial_null": bool(null_weak),
        "coherent_amplitude_fraction": bool(
            np.isfinite(fraction) and fraction >= config.weak_amplitude_fraction_min
        ),
        "sign_stability": bool(
            np.isfinite(sign_stability)
            and sign_stability >= config.weak_sign_stability_min
        ),
    }
    if all(strong_checks.values()):
        status: IdentifiabilityStatus = "identifiable"
    elif all(weak_checks.values()):
        status = "weakly_identifiable"
    else:
        status = "unidentifiable"
    return status, {
        "status": status,
        "identifiable_checks": strong_checks,
        "weakly_identifiable_checks": weak_checks,
        "thresholds": {
            "identifiable_cv_r2_min": float(config.identifiable_cv_r2_min),
            "weak_cv_r2_min": float(config.weak_cv_r2_min),
            "identifiable_abs_cv_rho_min": float(
                config.identifiable_abs_cv_rho_min
            ),
            "weak_abs_cv_rho_min": float(config.weak_abs_cv_rho_min),
            "identifiable_null_p_max": float(config.identifiable_null_p_max),
            "weak_null_p_max": float(config.weak_null_p_max),
            "identifiable_amplitude_fraction_min": float(
                config.identifiable_amplitude_fraction_min
            ),
            "weak_amplitude_fraction_min": float(
                config.weak_amplitude_fraction_min
            ),
            "identifiable_sign_stability_min": float(
                config.identifiable_sign_stability_min
            ),
            "weak_sign_stability_min": float(config.weak_sign_stability_min),
        },
    }


def _insufficient_report(
    *,
    band_index: int | None,
    wavelength_nm: float | None,
    support_count: int,
    config: IdentifiabilityConfig,
    bandpass: dict[str, Any],
) -> dict[str, Any]:
    return {
        "method": "candidate_independent_bandpass_blocked_cv_v1",
        "band_index": band_index,
        "wavelength_nm": wavelength_nm,
        "status": "unidentifiable",
        "reason": "insufficient_common_bandpass_support",
        "support": {
            "valid_pixel_count": int(support_count),
            "minimum_required": int(config.minimum_valid_pixels),
        },
        "bandpass": bandpass,
        "models": {},
        "expected_coherent_amplitude": {
            "estimate_log_band_per_log_rgb": None,
            "frozen_value_log_band_per_log_rgb": None,
        },
        "recommended_rgb_detail_action": "disabled_observation_supported_baseline_only",
        "candidate_independent": True,
        "claim_scope": "insufficient_support_no_high_resolution_truth_claim",
    }


def fit_blocked_bandpass_relation(
    rgb: np.ndarray,
    band_low: np.ndarray,
    psf: PsfModel,
    *,
    valid_mask: np.ndarray | None = None,
    band_index: int | None = None,
    wavelength_nm: float | None = None,
    epsilon_band: float | None = None,
    config: IdentifiabilityConfig | None = None,
) -> dict[str, Any]:
    """Assess one observed band without reading a fused candidate.

    The returned ``expected_coherent_amplitude`` is a signed log-band/log-RGB
    slope on the frozen common passband.  It is not forced to one.  An
    unidentifiable band receives no frozen evaluation target and RGB detail
    injection should remain disabled for a scientific product.
    """

    settings = config or IdentifiabilityConfig()
    resolved_epsilon_band = (
        float(settings.epsilon_band)
        if epsilon_band is None
        else float(epsilon_band)
    )
    if resolved_epsilon_band <= 0.0:
        raise ValueError("epsilon_band must be positive")
    features, guide, target, common = _common_bandpass_fields(
        rgb,
        band_low,
        psf,
        settings,
        resolved_epsilon_band,
    )
    valid = np.asarray(common["valid"], dtype=bool)
    if valid_mask is not None:
        supplied = np.asarray(valid_mask, dtype=bool)
        if supplied.shape != target.shape:
            raise ValueError(
                f"valid_mask shape {supplied.shape} does not match band shape {target.shape}"
            )
        valid &= supplied
    support_count = int(np.sum(valid))
    if support_count < settings.minimum_valid_pixels:
        return _insufficient_report(
            band_index=band_index,
            wavelength_nm=wavelength_nm,
            support_count=support_count,
            config=settings,
            bandpass=common["metadata"],
        )

    models: dict[str, dict[str, Any]] = {}
    cv_results: dict[str, _BlockedCvResult] = {}
    model_pca_ranks: dict[str, int | None] = {}
    ridge_result = _blocked_cv(
        features,
        target,
        valid,
        settings,
        pca_rank=None,
    )
    models["rgb_ridge"] = ridge_result.metrics
    cv_results["rgb_ridge"] = ridge_result
    model_pca_ranks["rgb_ridge"] = None
    for rank in sorted(set(settings.lowrank_feature_ranks)):
        if rank >= features.shape[2]:
            continue
        name = f"rgb_pca_ridge_rank_{rank}"
        result = _blocked_cv(
            features,
            target,
            valid,
            settings,
            pca_rank=rank,
        )
        models[name] = result.metrics
        cv_results[name] = result
        model_pca_ranks[name] = rank
    if settings.primary_model not in models:
        raise ValueError(
            f"primary_model {settings.primary_model!r} is unavailable; "
            f"choose one of {sorted(models)}"
        )
    primary = cv_results[settings.primary_model]
    primary_pca_rank = model_pca_ranks[settings.primary_model]
    predicted_count = int(primary.metrics["pixel_count"])
    if predicted_count < settings.minimum_valid_pixels:
        report = _insufficient_report(
            band_index=band_index,
            wavelength_nm=wavelength_nm,
            support_count=predicted_count,
            config=settings,
            bandpass=common["metadata"],
        )
        report["reason"] = "insufficient_blocked_cv_support"
        report["models"] = models
        return report

    cross_fitted_amplitude = _cross_fitted_luminance_amplitude(
        guide,
        target,
        valid,
        settings,
    )
    bootstrap = _cluster_bootstrap(
        target,
        primary.prediction,
        primary.baseline_prediction,
        guide,
        primary.fold_id,
        settings,
    )
    null = _spatial_shift_null(
        features,
        target,
        valid,
        valid,
        float(primary.metrics["cv_predictive_r2"]),
        settings,
        pca_rank=primary_pca_rank,
    )
    amplitude = _amplitude_summary(
        guide,
        target,
        primary,
        cross_fitted_amplitude,
        bootstrap,
    )
    status, classification = _classify(
        primary.metrics,
        amplitude,
        null,
        settings,
    )
    estimate = float(amplitude["estimate_log_band_per_log_rgb"])
    if status != "unidentifiable" and np.isfinite(estimate):
        amplitude["frozen_value_log_band_per_log_rgb"] = estimate
    if status == "identifiable":
        action = "eligible_for_gated_band_specific_shared_detail"
        amplitude["use_policy"] = (
            "candidate_evaluation_reference_not_independent_HR_truth"
        )
    elif status == "weakly_identifiable":
        action = "limited_ablation_only_not_a_hard_injection_target"
        amplitude["use_policy"] = "diagnostic_only_due_to_unstable_relation"
    else:
        action = "disabled_observation_supported_baseline_only"
        amplitude["use_policy"] = "no_frozen_candidate_target"
    return {
        "method": "candidate_independent_bandpass_blocked_cv_v1",
        "band_index": band_index,
        "wavelength_nm": wavelength_nm,
        "status": status,
        "support": {
            "valid_pixel_count": support_count,
            "blocked_cv_pixel_count": predicted_count,
            "block_grid": [int(settings.block_rows), int(settings.block_cols)],
            "exclusion_gap_px": int(settings.resolved_exclusion_gap_px),
            "adjacent_train_test_pixels_excluded": True,
            "test_block_eroded_before_scoring": True,
            "test_boundary_inset_px": int(settings.resolved_exclusion_gap_px),
            "training_guard_outside_test_block_px": int(
                settings.resolved_exclusion_gap_px
            ),
        },
        "bandpass": common["metadata"],
        "primary_model": settings.primary_model,
        "models": models,
        "confidence_intervals": bootstrap,
        "spatial_shuffle_null": null,
        "expected_coherent_amplitude": amplitude,
        "classification": classification,
        "recommended_rgb_detail_action": action,
        "candidate_independent": True,
        "unit_rgb_beta_is_not_a_target": True,
        "claim_scope": (
            "shared_structure_identifiability_inside_native_low_resolution_passband_only; "
            "not_independent_high_resolution_NIR_SWIR_truth"
        ),
    }


def assess_cube_identifiability(
    rgb: np.ndarray,
    observed_cube: np.ndarray,
    wavelengths_nm: Sequence[float],
    psf_models: PsfModel | Sequence[PsfModel],
    *,
    valid_mask: np.ndarray | None = None,
    band_epsilons: float | Sequence[float] | None = None,
    config: IdentifiabilityConfig | None = None,
) -> dict[str, Any]:
    """Run the frozen identifiability screen for every observed spectral band."""

    cube = np.asarray(observed_cube, dtype=np.float32)
    if cube.ndim != 3:
        raise ValueError("observed_cube must have shape (height, width, bands)")
    wavelengths = np.asarray(wavelengths_nm, dtype=np.float64).reshape(-1)
    if wavelengths.size != cube.shape[2]:
        raise ValueError("wavelength count must equal observed_cube band count")
    if isinstance(psf_models, PsfModel):
        resolved_psfs = [psf_models] * cube.shape[2]
    else:
        resolved_psfs = list(psf_models)
        if len(resolved_psfs) != cube.shape[2]:
            raise ValueError("psf_models count must equal observed_cube band count")
    if band_epsilons is None:
        resolved_epsilons: list[float | None] = [None] * cube.shape[2]
    elif np.isscalar(band_epsilons):
        resolved_epsilons = [float(band_epsilons)] * cube.shape[2]
    else:
        resolved_epsilons = [float(value) for value in band_epsilons]
        if len(resolved_epsilons) != cube.shape[2]:
            raise ValueError("band_epsilons count must equal observed_cube band count")
    if valid_mask is None:
        masks: list[np.ndarray | None] = [None] * cube.shape[2]
    else:
        mask = np.asarray(valid_mask, dtype=bool)
        if mask.shape == cube.shape[:2]:
            masks = [mask] * cube.shape[2]
        elif mask.shape == cube.shape:
            masks = [mask[:, :, index] for index in range(cube.shape[2])]
        else:
            raise ValueError(
                "valid_mask must have shape (height, width) or match observed_cube"
            )
    base_config = config or IdentifiabilityConfig()
    bands: list[dict[str, Any]] = []
    for index in range(cube.shape[2]):
        band_config = replace(
            base_config,
            random_seed=int(base_config.random_seed + 10007 * index),
        )
        bands.append(
            fit_blocked_bandpass_relation(
                rgb,
                cube[:, :, index],
                resolved_psfs[index],
                valid_mask=masks[index],
                band_index=index,
                wavelength_nm=float(wavelengths[index]),
                epsilon_band=resolved_epsilons[index],
                config=band_config,
            )
        )
    counts = {
        status: int(sum(band["status"] == status for band in bands))
        for status in (
            "identifiable",
            "weakly_identifiable",
            "unidentifiable",
        )
    }
    return {
        "method": "candidate_independent_bandpass_blocked_cv_v1",
        "candidate_independent": True,
        "band_count": int(cube.shape[2]),
        "status_counts": counts,
        "bands": bands,
        "unit_rgb_beta_is_not_a_target": True,
        "claim_scope": (
            "shared_structure_identifiability_inside_native_low_resolution_passband_only; "
            "not_independent_high_resolution_NIR_SWIR_truth"
        ),
    }


__all__ = [
    "IdentifiabilityConfig",
    "IdentifiabilityStatus",
    "assess_cube_identifiability",
    "fit_blocked_bandpass_relation",
]
