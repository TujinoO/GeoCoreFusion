"""Auditable proxy uncertainty for an already accepted registration warp.

This module deliberately does not estimate or modify a registration solution.
It converts accepted bidirectional tie-point diagnostics into a local 2x2
second-moment proxy around the final displacement field.  The proxy must be
calibrated against independent landmarks before it is interpreted as a
probabilistic registration covariance.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

import cv2
import numpy as np
from scipy.spatial import cKDTree


@dataclass(frozen=True, slots=True)
class RegistrationUncertaintyConfig:
    """Configuration for the local registration-uncertainty proxy.

    All displacement-related parameters are expressed on the analysis grid.
    ``covariance_floor_px2`` is an eigenvalue floor, not a standard deviation.
    The remaining penalty scales are heuristic and therefore require external
    calibration before probabilistic coverage claims are made.
    """

    covariance_floor_px2: float = 0.04
    spatial_bandwidth_px: float = 36.0
    neighbours: int = 12
    score_penalty_px: float = 0.35
    margin_reference: float = 0.02
    margin_penalty_px: float = 0.25
    backward_error_weight: float = 0.50
    distance_inflation_per_bandwidth_px: float = 0.25
    max_distance_inflation_px: float = 2.0
    query_chunk_size: int = 65536

    def validate(self) -> None:
        if self.covariance_floor_px2 < 0.0:
            raise ValueError("covariance_floor_px2 must be non-negative")
        if self.spatial_bandwidth_px <= 0.0:
            raise ValueError("spatial_bandwidth_px must be positive")
        if self.neighbours < 1:
            raise ValueError("neighbours must be at least one")
        if self.score_penalty_px < 0.0:
            raise ValueError("score_penalty_px must be non-negative")
        if self.margin_reference <= 0.0:
            raise ValueError("margin_reference must be positive")
        if self.margin_penalty_px < 0.0:
            raise ValueError("margin_penalty_px must be non-negative")
        if self.backward_error_weight < 0.0:
            raise ValueError("backward_error_weight must be non-negative")
        if self.distance_inflation_per_bandwidth_px < 0.0:
            raise ValueError("distance_inflation_per_bandwidth_px must be non-negative")
        if self.max_distance_inflation_px < 0.0:
            raise ValueError("max_distance_inflation_px must be non-negative")
        if self.query_chunk_size < 1:
            raise ValueError("query_chunk_size must be at least one")


@dataclass(slots=True)
class RegistrationUncertaintyEstimate:
    """Local uncertainty-proxy fields in ``(shift_y, shift_x)`` order."""

    covariance_yx_px2: np.ndarray
    nearest_tiepoint_distance_px: np.ndarray
    effective_tiepoint_count: np.ndarray
    metadata: dict[str, Any]


def _tiepoint_value(point: Any, name: str) -> float:
    if isinstance(point, Mapping):
        if name not in point:
            raise ValueError(f"tie point is missing {name!r}")
        value = point[name]
    elif hasattr(point, name):
        value = getattr(point, name)
    else:
        raise ValueError(f"tie point is missing {name!r}")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"tie point {name!r} must be finite")
    return result


def _as_tiepoint_arrays(
    tie_points: Sequence[Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if not tie_points:
        raise ValueError("at least one accepted bidirectional tie point is required")
    coordinates = []
    displacements = []
    scores = []
    margins = []
    backward_errors = []
    for point in tie_points:
        coordinates.append([_tiepoint_value(point, "ref_y"), _tiepoint_value(point, "ref_x")])
        displacements.append([_tiepoint_value(point, "shift_y"), _tiepoint_value(point, "shift_x")])
        scores.append(_tiepoint_value(point, "score"))
        margins.append(_tiepoint_value(point, "margin"))
        backward_error = _tiepoint_value(point, "backward_error")
        if backward_error < 0.0:
            raise ValueError("tie point backward_error must be non-negative")
        backward_errors.append(backward_error)
    return (
        np.asarray(coordinates, dtype=np.float64),
        np.asarray(displacements, dtype=np.float64),
        np.asarray(scores, dtype=np.float64),
        np.asarray(margins, dtype=np.float64),
        np.asarray(backward_errors, dtype=np.float64),
    )


def _sample_final_shift(field: np.ndarray, coordinates_yx: np.ndarray) -> np.ndarray:
    values = np.asarray(field, dtype=np.float32)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("final displacement fields must be finite two-dimensional arrays")
    height, width = values.shape
    y = coordinates_yx[:, 0]
    x = coordinates_yx[:, 1]
    if np.any(y < 0.0) or np.any(y > height - 1.0) or np.any(x < 0.0) or np.any(x > width - 1.0):
        raise ValueError("tie-point reference coordinates fall outside the final displacement field")
    sampled = cv2.remap(
        values,
        x.astype(np.float32).reshape(1, -1),
        y.astype(np.float32).reshape(1, -1),
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return sampled.reshape(-1).astype(np.float64)


def _quality_variance_px2(
    scores: np.ndarray,
    margins: np.ndarray,
    backward_errors: np.ndarray,
    config: RegistrationUncertaintyConfig,
) -> np.ndarray:
    score_deficit = 1.0 - np.clip(scores, 0.0, 1.0)
    nonnegative_margin = np.maximum(margins, 0.0)
    margin_deficit = 1.0 / np.sqrt(1.0 + nonnegative_margin / config.margin_reference)
    return (
        np.square(config.score_penalty_px * score_deficit)
        + np.square(config.margin_penalty_px * margin_deficit)
        + np.square(config.backward_error_weight * backward_errors)
    )


def _proxy_metadata(
    config: RegistrationUncertaintyConfig,
    residual_yx: np.ndarray,
    quality_variance_px2: np.ndarray,
    nearest_distance_px: np.ndarray,
    effective_count: np.ndarray,
) -> dict[str, Any]:
    residual_norm = np.linalg.norm(residual_yx, axis=1)
    return {
        "status": "uncertainty_proxy",
        "uncertainty_proxy": True,
        "calibration_required": True,
        "claim_boundary": (
            "Heuristic local second moments from same-data tie points are not a calibrated "
            "registration-error covariance. Validate coverage against independent landmarks "
            "or known-truth targets before probabilistic interpretation."
        ),
        "method": "quality_weighted_local_residual_second_moment_with_distance_inflation",
        "component_order": ["shift_y_px", "shift_x_px"],
        "covariance_units": "analysis_pixel_squared",
        "mean_registration_solution_changed": False,
        "tie_point_count": int(residual_yx.shape[0]),
        "configuration": asdict(config),
        "tiepoint_residual_norm_px": {
            "median": float(np.median(residual_norm)),
            "p95": float(np.percentile(residual_norm, 95.0)),
            "max": float(np.max(residual_norm)),
        },
        "tiepoint_quality_sigma_px": {
            "median": float(np.median(np.sqrt(quality_variance_px2))),
            "p95": float(np.percentile(np.sqrt(quality_variance_px2), 95.0)),
        },
        "spatial_support": {
            "nearest_distance_median_px": float(np.median(nearest_distance_px)),
            "nearest_distance_p95_px": float(np.percentile(nearest_distance_px, 95.0)),
            "effective_tiepoint_count_median": float(np.median(effective_count)),
            "effective_tiepoint_count_p05": float(np.percentile(effective_count, 5.0)),
        },
    }


def estimate_local_registration_covariance(
    tie_points: Sequence[Any],
    final_shift_y: np.ndarray,
    final_shift_x: np.ndarray,
    *,
    query_y: np.ndarray | None = None,
    query_x: np.ndarray | None = None,
    config: RegistrationUncertaintyConfig | None = None,
) -> RegistrationUncertaintyEstimate:
    """Estimate a local 2x2 uncertainty proxy around a fixed final warp.

    Tie-point residuals are evaluated against ``final_shift_y/x`` at each
    accepted reference coordinate.  At every query location, quality-weighted
    spatial neighbours contribute the residual second moment about zero.  The
    zero-centred moment intentionally retains local bias relative to the final
    warp instead of silently changing that warp.  Isotropic terms derived from
    score, peak margin, backward error, distance to support, and the configured
    covariance floor are then added.

    The result is ordered as ``(shift_y, shift_x)`` and expressed in analysis
    pixel squared.  It is an auditable proxy, not a calibrated covariance.
    """

    cfg = config or RegistrationUncertaintyConfig()
    cfg.validate()
    shift_y = np.asarray(final_shift_y, dtype=np.float32)
    shift_x = np.asarray(final_shift_x, dtype=np.float32)
    if shift_y.shape != shift_x.shape or shift_y.ndim != 2:
        raise ValueError("final_shift_y and final_shift_x must have the same two-dimensional shape")

    coordinates, displacements, scores, margins, backward_errors = _as_tiepoint_arrays(tie_points)
    sampled_final = np.stack(
        [_sample_final_shift(shift_y, coordinates), _sample_final_shift(shift_x, coordinates)],
        axis=1,
    )
    residual_yx = displacements - sampled_final
    quality_variance = _quality_variance_px2(scores, margins, backward_errors, cfg)
    precision = 1.0 / np.maximum(cfg.covariance_floor_px2 + quality_variance, 1e-12)

    if query_y is None and query_x is None:
        query_y_array, query_x_array = np.indices(shift_y.shape, dtype=np.float32)
    elif query_y is None or query_x is None:
        raise ValueError("query_y and query_x must either both be supplied or both be omitted")
    else:
        query_y_array, query_x_array = np.broadcast_arrays(
            np.asarray(query_y, dtype=np.float32),
            np.asarray(query_x, dtype=np.float32),
        )
        if not np.isfinite(query_y_array).all() or not np.isfinite(query_x_array).all():
            raise ValueError("query coordinates must be finite")

    output_shape = query_y_array.shape
    queries = np.stack([query_y_array.reshape(-1), query_x_array.reshape(-1)], axis=1).astype(np.float64)
    covariance = np.empty((queries.shape[0], 2, 2), dtype=np.float64)
    nearest_distance = np.empty(queries.shape[0], dtype=np.float64)
    effective_count = np.empty(queries.shape[0], dtype=np.float64)
    tree = cKDTree(coordinates)
    neighbour_count = min(int(cfg.neighbours), coordinates.shape[0])
    bandwidth2 = float(cfg.spatial_bandwidth_px) ** 2
    identity = np.eye(2, dtype=np.float64)

    for start in range(0, queries.shape[0], int(cfg.query_chunk_size)):
        stop = min(start + int(cfg.query_chunk_size), queries.shape[0])
        distances, indices = tree.query(queries[start:stop], k=neighbour_count)
        if neighbour_count == 1:
            distances = distances[:, None]
            indices = indices[:, None]
        minimum_distance = distances[:, 0]
        # Subtracting the closest squared distance keeps the nearest spatial
        # weight at one and avoids underflow during distant extrapolation.
        relative_distance2 = np.maximum(np.square(distances) - np.square(minimum_distance[:, None]), 0.0)
        spatial_weight = np.exp(-0.5 * relative_distance2 / bandwidth2)
        raw_weight = spatial_weight * precision[indices]
        weight = raw_weight / np.maximum(np.sum(raw_weight, axis=1, keepdims=True), 1e-15)
        local_residual = residual_yx[indices]
        residual_second_moment = np.einsum(
            "nk,nki,nkj->nij",
            weight,
            local_residual,
            local_residual,
            optimize=True,
        )
        local_quality_variance = np.sum(weight * quality_variance[indices], axis=1)
        distance_sigma = np.minimum(
            cfg.max_distance_inflation_px,
            cfg.distance_inflation_per_bandwidth_px * minimum_distance / cfg.spatial_bandwidth_px,
        )
        isotropic_variance = (
            cfg.covariance_floor_px2
            + local_quality_variance
            + np.square(distance_sigma)
        )
        covariance[start:stop] = residual_second_moment + isotropic_variance[:, None, None] * identity
        nearest_distance[start:stop] = minimum_distance
        effective_count[start:stop] = 1.0 / np.maximum(np.sum(np.square(weight), axis=1), 1e-15)

    covariance = 0.5 * (covariance + np.swapaxes(covariance, -1, -2))
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    eigenvalues = np.maximum(eigenvalues, cfg.covariance_floor_px2)
    covariance = np.einsum(
        "...ik,...k,...jk->...ij",
        eigenvectors,
        eigenvalues,
        eigenvectors,
        optimize=True,
    )
    covariance = covariance.reshape(output_shape + (2, 2)).astype(np.float32)
    nearest_distance = nearest_distance.reshape(output_shape).astype(np.float32)
    effective_count = effective_count.reshape(output_shape).astype(np.float32)
    return RegistrationUncertaintyEstimate(
        covariance_yx_px2=covariance,
        nearest_tiepoint_distance_px=nearest_distance,
        effective_tiepoint_count=effective_count,
        metadata=_proxy_metadata(
            cfg,
            residual_yx,
            quality_variance,
            nearest_distance,
            effective_count,
        ),
    )


def registration_sigma_points_5(
    mean_shift_yx: np.ndarray,
    covariance_yx_px2: np.ndarray,
    *,
    center_weight: float = 1.0 / 3.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return five positive-weight sigma points matching mean and covariance.

    Point order is ``center, +axis0, -axis0, +axis1, -axis1`` where the axes
    are the covariance eigenvectors.  The returned point array has shape
    ``broadcast_shape + (5, 2)`` and uses ``(shift_y, shift_x)`` order.  The
    one-dimensional weight array has length five and exactly reconstructs the
    supplied first two moments up to floating-point precision.
    """

    if not 0.0 <= float(center_weight) < 1.0:
        raise ValueError("center_weight must lie in [0, 1)")
    mean = np.asarray(mean_shift_yx, dtype=np.float64)
    covariance = np.asarray(covariance_yx_px2, dtype=np.float64)
    if mean.ndim < 1 or mean.shape[-1] != 2:
        raise ValueError("mean_shift_yx must end with a two-component axis")
    if covariance.ndim < 2 or covariance.shape[-2:] != (2, 2):
        raise ValueError("covariance_yx_px2 must end with a 2x2 matrix")
    batch_shape = np.broadcast_shapes(mean.shape[:-1], covariance.shape[:-2])
    mean = np.broadcast_to(mean, batch_shape + (2,))
    covariance = np.broadcast_to(covariance, batch_shape + (2, 2))
    if not np.isfinite(mean).all() or not np.isfinite(covariance).all():
        raise ValueError("mean and covariance must be finite")
    symmetric = 0.5 * (covariance + np.swapaxes(covariance, -1, -2))
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    tolerance = 1e-9 * np.maximum(1.0, np.max(np.abs(eigenvalues), axis=-1, keepdims=True))
    if np.any(eigenvalues < -tolerance):
        raise ValueError("covariance_yx_px2 must be positive semidefinite")
    root = eigenvectors * np.sqrt(np.maximum(eigenvalues, 0.0))[..., None, :]
    scale = np.sqrt(2.0 / (1.0 - float(center_weight)))
    points = np.empty(batch_shape + (5, 2), dtype=np.float64)
    points[..., 0, :] = mean
    points[..., 1, :] = mean + scale * root[..., :, 0]
    points[..., 2, :] = mean - scale * root[..., :, 0]
    points[..., 3, :] = mean + scale * root[..., :, 1]
    points[..., 4, :] = mean - scale * root[..., :, 1]
    off_center_weight = (1.0 - float(center_weight)) / 4.0
    weights = np.asarray(
        [center_weight, off_center_weight, off_center_weight, off_center_weight, off_center_weight],
        dtype=np.float64,
    )
    return points.astype(np.float32), weights
