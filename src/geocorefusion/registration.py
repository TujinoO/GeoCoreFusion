"""Cross-modal affine registration with scan-direction drift correction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
from scipy.signal import find_peaks
from scipy.spatial import cKDTree

from .config import RegistrationConfig
from .dataset import DatasetTriplet, hsi_structure, normalize_image, rgb_structure, select_band_indices


@dataclass(slots=True)
class RegistrationModel:
    sensor: str
    rgb_shape: tuple[int, int]
    sensor_shape: tuple[int, int]
    rgb_to_sensor_matrix: np.ndarray
    drift_rgb_y: np.ndarray
    drift_sensor_dx: np.ndarray
    drift_sensor_dy: np.ndarray
    ecc_score: float
    edge_correlation: float
    method: str = "cross_modal_ecc_affine_plus_scanline_drift"

    def map_rgb_to_sensor(self, y: np.ndarray | float, x: np.ndarray | float) -> tuple[np.ndarray, np.ndarray]:
        yy = np.asarray(y, dtype=np.float32)
        xx = np.asarray(x, dtype=np.float32)
        matrix = self.rgb_to_sensor_matrix
        sx = matrix[0, 0] * xx + matrix[0, 1] * yy + matrix[0, 2]
        sy = matrix[1, 0] * xx + matrix[1, 1] * yy + matrix[1, 2]
        if self.drift_rgb_y.size:
            dx = np.interp(yy.reshape(-1), self.drift_rgb_y, self.drift_sensor_dx).reshape(yy.shape)
            dy = np.interp(yy.reshape(-1), self.drift_rgb_y, self.drift_sensor_dy).reshape(yy.shape)
            sx = sx + dx
            sy = sy + dy
        return sy.astype(np.float32), sx.astype(np.float32)

    def valid_fraction(self, y: np.ndarray, x: np.ndarray, margin: float = 1.0) -> float:
        sy, sx = self.map_rgb_to_sensor(y, x)
        valid = (sy >= margin) & (sy < self.sensor_shape[0] - margin) & (sx >= margin) & (sx < self.sensor_shape[1] - margin)
        return float(valid.mean()) if valid.size else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "sensor": self.sensor,
            "method": self.method,
            "rgb_shape": list(self.rgb_shape),
            "sensor_shape": list(self.sensor_shape),
            "rgb_to_sensor_matrix": self.rgb_to_sensor_matrix.tolist(),
            "drift_rgb_y": self.drift_rgb_y.tolist(),
            "drift_sensor_dx": self.drift_sensor_dx.tolist(),
            "drift_sensor_dy": self.drift_sensor_dy.tolist(),
            "ecc_score": self.ecc_score,
            "edge_correlation": self.edge_correlation,
        }


@dataclass(slots=True)
class RegistrationBundle:
    nir: RegistrationModel
    swir: RegistrationModel
    preview_rgb: np.ndarray
    preview_nir_aligned: np.ndarray
    preview_swir_aligned: np.ndarray

    def to_dict(self) -> dict[str, Any]:
        return {"reference": "RGB", "nir": self.nir.to_dict(), "swir": self.swir.to_dict()}


@dataclass(slots=True)
class RoiRegistrationModel:
    sensor: str
    analysis_shape: tuple[int, int]
    affine_matrix: np.ndarray
    dense_shift_x: np.ndarray
    dense_shift_y: np.ndarray
    column_control_x: np.ndarray
    column_shift_x: np.ndarray
    row_control_y: np.ndarray
    row_shift_x: np.ndarray
    row_shift_y: np.ndarray
    score_before: float
    score_after: float
    ecc_score: float
    accepted_affine: bool
    method: str = "roi_affine_piecewise_geometry_plus_guarded_tiepoint_idw"
    details: dict[str, Any] | None = None

    def map_analysis_indices(self, y: np.ndarray, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        yy = np.asarray(y, dtype=np.float32)
        xx = np.asarray(x, dtype=np.float32)
        if self.dense_shift_x.size:
            sample_y = yy.copy()
            sample_x = xx.copy()
            xx = xx + _sample_dense_map(self.dense_shift_x, sample_y, sample_x)
            yy = yy + _sample_dense_map(self.dense_shift_y, sample_y, sample_x)
        if self.column_control_x.size:
            dx_column = np.interp(xx.reshape(-1), self.column_control_x, self.column_shift_x).reshape(xx.shape)
            xx = xx + dx_column.astype(np.float32)
        if self.row_control_y.size:
            dx = np.interp(yy.reshape(-1), self.row_control_y, self.row_shift_x).reshape(yy.shape)
            dy = np.interp(yy.reshape(-1), self.row_control_y, self.row_shift_y).reshape(yy.shape)
            xx = xx + dx.astype(np.float32)
            yy = yy + dy.astype(np.float32)
        matrix = self.affine_matrix
        mapped_x = matrix[0, 0] * xx + matrix[0, 1] * yy + matrix[0, 2]
        mapped_y = matrix[1, 0] * xx + matrix[1, 1] * yy + matrix[1, 2]
        return mapped_y.astype(np.float32), mapped_x.astype(np.float32)

    def to_dict(self) -> dict[str, Any]:
        return {
            "sensor": self.sensor,
            "method": self.method,
            "analysis_shape": list(self.analysis_shape),
            "affine_matrix": self.affine_matrix.tolist(),
            "dense_field_shape": list(self.dense_shift_x.shape) if self.dense_shift_x.size else [0, 0],
            "dense_shift_x_p95_abs": float(np.percentile(np.abs(self.dense_shift_x), 95)) if self.dense_shift_x.size else 0.0,
            "dense_shift_y_p95_abs": float(np.percentile(np.abs(self.dense_shift_y), 95)) if self.dense_shift_y.size else 0.0,
            "column_control_x": self.column_control_x.tolist(),
            "column_shift_x": self.column_shift_x.tolist(),
            "row_control_y": self.row_control_y.tolist(),
            "row_shift_x": self.row_shift_x.tolist(),
            "row_shift_y": self.row_shift_y.tolist(),
            "score_before": self.score_before,
            "score_after": self.score_after,
            "score_gain": self.score_after - self.score_before,
            "ecc_score": self.ecc_score,
            "accepted_affine": self.accepted_affine,
            "details": self.details or {},
        }


@dataclass(slots=True)
class RoiTiePoint:
    ref_y: float
    ref_x: float
    moving_y: float
    moving_x: float
    shift_y: float
    shift_x: float
    score: float
    margin: float
    backward_error: float

    def to_dict(self) -> dict[str, float]:
        return {
            "ref_y": self.ref_y,
            "ref_x": self.ref_x,
            "moving_y": self.moving_y,
            "moving_x": self.moving_x,
            "shift_y": self.shift_y,
            "shift_x": self.shift_x,
            "score": self.score,
            "margin": self.margin,
            "backward_error": self.backward_error,
        }


@dataclass(slots=True)
class RoiRegistrationBundle:
    roi: dict[str, int]
    analysis_shape: tuple[int, int]
    nir: RoiRegistrationModel
    swir: RoiRegistrationModel
    status: str
    pair_score_before: float
    pair_score_after: float
    pair_refinement_accepted: bool
    reference_structure: np.ndarray
    nir_initial: np.ndarray
    swir_initial: np.ndarray
    nir_aligned: np.ndarray
    swir_aligned: np.ndarray
    nir_overlap_aligned: np.ndarray
    swir_overlap_aligned: np.ndarray

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": "full_scan_coarse_then_roi_joint_cross_modal_refinement",
            "roi": self.roi,
            "analysis_shape": list(self.analysis_shape),
            "status": self.status,
            "nir": self.nir.to_dict(),
            "swir": self.swir.to_dict(),
            "nir_swir_overlap": {
                "score_before": self.pair_score_before,
                "score_after": self.pair_score_after,
                "score_gain": self.pair_score_after - self.pair_score_before,
                "pair_refinement_accepted": self.pair_refinement_accepted,
            },
        }


def _feature_base_and_support(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Normalize finite pixels while keeping an explicit invalid-data support mask."""

    values = np.asarray(image, dtype=np.float32)
    valid = np.isfinite(values)
    if not valid.any():
        return np.zeros(values.shape, dtype=np.uint8), valid
    normalized = normalize_image(values)
    fill = float(np.median(normalized[valid]))
    filled = np.where(valid, normalized, fill)
    support = valid.copy()
    if not valid.all():
        # Feature operators have a small spatial footprint. Excluding a two-pixel
        # halo prevents the finite fill from becoming an artificial boundary edge.
        invalid_halo = cv2.dilate((~valid).astype(np.uint8), np.ones((5, 5), dtype=np.uint8))
        support &= invalid_halo == 0
    base = (np.clip(filled, 0.0, 1.0) * 255.0).round().astype(np.uint8)
    return base, support


def _feature(image: np.ndarray) -> np.ndarray:
    base, support = _feature_base_and_support(image)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(base).astype(np.float32) / 255.0
    gx = cv2.Sobel(clahe, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(clahe, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = cv2.magnitude(gx, gy)
    magnitude[~support] = np.nan
    grad = normalize_image(magnitude)
    combined = 0.35 * clahe + 0.65 * np.nan_to_num(grad, nan=0.0)
    combined[~support] = np.nan
    feature = normalize_image(combined)
    feature[~support] = np.nan
    return feature


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    aa = np.asarray(a, dtype=np.float32).reshape(-1)
    bb = np.asarray(b, dtype=np.float32).reshape(-1)
    valid = np.isfinite(aa) & np.isfinite(bb)
    if valid.sum() < 32:
        return float("nan")
    aa = aa[valid] - aa[valid].mean()
    bb = bb[valid] - bb[valid].mean()
    denom = float(np.sqrt(np.sum(aa * aa) * np.sum(bb * bb)))
    return float(np.sum(aa * bb) / denom) if denom > 1e-9 else float("nan")


def _finite_cv_pair(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return finite OpenCV inputs plus their shared, scoreable support."""

    aa = np.asarray(a, dtype=np.float32)
    bb = np.asarray(b, dtype=np.float32)
    valid = np.isfinite(aa) & np.isfinite(bb)
    return (
        np.where(np.isfinite(aa), aa, 0.0).astype(np.float32),
        np.where(np.isfinite(bb), bb, 0.0).astype(np.float32),
        valid,
    )


def _modality_feature(image: np.ndarray) -> np.ndarray:
    """Build an edge-proximity representation that is stable across RGB/NIR/SWIR."""

    base, support = _feature_base_and_support(image)
    tile = (max(2, min(8, base.shape[1] // 20)), max(2, min(8, base.shape[0] // 20)))
    contrast = cv2.createCLAHE(clipLimit=2.0, tileGridSize=tile).apply(base).astype(np.float32) / 255.0
    gx = cv2.Scharr(contrast, cv2.CV_32F, 1, 0)
    gy = cv2.Scharr(contrast, cv2.CV_32F, 0, 1)
    magnitude = cv2.magnitude(gx, gy)
    magnitude[~support] = np.nan
    gradient = normalize_image(magnitude)
    finite_gradient = gradient[support & np.isfinite(gradient)]
    threshold = float(np.percentile(finite_gradient, 72.0)) if finite_gradient.size else 1.0
    edges = ((gradient >= max(threshold, 0.05)) & support).astype(np.uint8)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((3, 3), dtype=np.uint8))
    distance = cv2.distanceTransform(1 - edges, cv2.DIST_L2, 3)
    proximity = np.exp(-distance / 2.5).astype(np.float32)
    laplacian_raw = np.abs(cv2.Laplacian(contrast, cv2.CV_32F, ksize=3))
    laplacian_raw[~support] = np.nan
    laplacian = normalize_image(laplacian_raw)
    combined = (
        0.62 * proximity
        + 0.28 * cv2.GaussianBlur(np.nan_to_num(gradient, nan=0.0), (0, 0), 0.8)
        + 0.10 * cv2.GaussianBlur(np.nan_to_num(laplacian, nan=0.0), (0, 0), 0.8)
    )
    combined[~support] = np.nan
    feature = normalize_image(combined)
    feature[~support] = np.nan
    return feature


def _warp_validity(
    valid: np.ndarray,
    matrix: np.ndarray,
    shape: tuple[int, int],
) -> np.ndarray:
    return cv2.warpAffine(
        np.asarray(valid, dtype=np.float32),
        np.asarray(matrix[:2], dtype=np.float32),
        (shape[1], shape[0]),
        flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0.0,
    )


def _warp_affine(image: np.ndarray, matrix: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    source = np.asarray(image, dtype=np.float32)
    valid = np.isfinite(source)
    warped = cv2.warpAffine(
        np.where(valid, source, 0.0).astype(np.float32),
        np.asarray(matrix[:2], dtype=np.float32),
        (shape[1], shape[0]),
        flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0.0,
    )
    warped_validity = _warp_validity(valid, matrix, shape)
    warped[warped_validity < 1.0 - 1e-6] = np.nan
    return warped


def _remap_preserve_invalid(image: np.ndarray, map_x: np.ndarray, map_y: np.ndarray) -> np.ndarray:
    """Remap without synthesizing image content outside finite source support."""

    source = np.asarray(image, dtype=np.float32)
    valid = np.isfinite(source)
    remapped = cv2.remap(
        np.where(valid, source, 0.0).astype(np.float32),
        np.asarray(map_x, dtype=np.float32),
        np.asarray(map_y, dtype=np.float32),
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0.0,
    )
    remapped_validity = cv2.remap(
        valid.astype(np.float32),
        np.asarray(map_x, dtype=np.float32),
        np.asarray(map_y, dtype=np.float32),
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0.0,
    )
    remapped[remapped_validity < 1.0 - 1e-6] = np.nan
    return remapped


def _sample_dense_map(field: np.ndarray, y: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Bilinearly sample an analysis-grid displacement field."""

    values = np.asarray(field, dtype=np.float32)
    yy = np.asarray(y, dtype=np.float32)
    xx = np.asarray(x, dtype=np.float32)
    original_shape = yy.shape
    map_shape = original_shape if len(original_shape) >= 2 else (1, int(yy.size))
    map_y = yy.reshape(map_shape)
    map_x = xx.reshape(map_shape)
    sampled = cv2.remap(
        values,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT101,
    )
    return sampled.reshape(original_shape).astype(np.float32)


def _physical_roi_affine(matrix: np.ndarray, shape: tuple[int, int], config: RegistrationConfig) -> bool:
    linear = np.asarray(matrix[:2, :2], dtype=np.float64)
    scale_x = float(np.linalg.norm(linear[:, 0]))
    scale_y = float(np.linalg.norm(linear[:, 1]))
    low, high = map(float, config.roi_scale_limits)
    if not (low <= scale_x <= high and low <= scale_y <= high):
        return False
    if float(np.linalg.det(linear)) <= 0.0:
        return False
    normalized_dot = abs(float(np.dot(linear[:, 0], linear[:, 1]))) / max(scale_x * scale_y, 1e-8)
    if normalized_dot > float(config.roi_shear_limit):
        return False
    tx, ty = float(matrix[0, 2]), float(matrix[1, 2])
    limit = float(config.roi_translation_fraction)
    return abs(tx) <= shape[1] * limit and abs(ty) <= shape[0] * limit


def _estimate_roi_affine(
    reference: np.ndarray,
    moving: np.ndarray,
    config: RegistrationConfig,
    *,
    min_gain: float | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    shape = reference.shape
    ref_feature = _modality_feature(reference)
    moving_feature = _modality_feature(moving)
    before = _corr(ref_feature, moving_feature)
    ref_cv, moving_cv, cv_support = _finite_cv_pair(ref_feature, moving_feature)
    if int(cv_support.sum()) >= 32:
        shift, phase_response = cv2.phaseCorrelate(ref_cv, moving_cv, cv_support.astype(np.float32))
    else:
        shift, phase_response = (0.0, 0.0), 0.0
    starts = [
        np.asarray([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32),
        np.asarray([[1.0, 0.0, shift[0]], [0.0, 1.0, shift[1]]], dtype=np.float32),
    ]
    criteria = (
        cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
        int(config.roi_ecc_iterations),
        float(config.roi_ecc_epsilon),
    )
    candidates: list[tuple[float, float, np.ndarray, np.ndarray, str]] = []
    motions = (
        (cv2.MOTION_TRANSLATION, "translation", 0.0),
        (cv2.MOTION_EUCLIDEAN, "euclidean", 0.003),
        (cv2.MOTION_AFFINE, "affine", 0.006),
    )
    for motion, name, penalty in motions:
        for start in starts:
            warp = start.copy()
            try:
                ecc, warp = cv2.findTransformECC(
                    ref_cv,
                    moving_cv,
                    warp,
                    motion,
                    criteria,
                    (cv_support.astype(np.uint8) * 255),
                    int(config.gaussian_filter_size),
                )
            except cv2.error:
                continue
            matrix = np.vstack([warp.astype(np.float64), [0.0, 0.0, 1.0]])
            if not _physical_roi_affine(matrix, shape, config):
                continue
            aligned = _warp_affine(moving_feature, matrix, shape)
            correlation = _corr(ref_feature, aligned)
            if not np.isfinite(correlation):
                continue
            candidates.append((float(correlation - penalty), float(ecc), matrix, aligned, name))

    identity = np.eye(3, dtype=np.float64)
    if not candidates:
        return identity, moving_feature, {
            "accepted": False,
            "motion": "identity_fallback",
            "score_before": before,
            "score_after": before,
            "ecc_score": float(phase_response),
            "phase_shift": [float(shift[0]), float(shift[1])],
            "phase_response": float(phase_response),
        }

    _, ecc, candidate, aligned, motion_name = max(candidates, key=lambda item: item[0])
    after = _corr(ref_feature, aligned)
    required_gain = float(config.roi_min_score_gain if min_gain is None else min_gain)
    accepted = bool(np.isfinite(after) and after >= before + required_gain)
    if not accepted:
        candidate, aligned, after, motion_name = identity, moving_feature, before, "identity_guarded_fallback"
    return candidate, aligned, {
        "accepted": accepted,
        "motion": motion_name,
        "score_before": before,
        "score_after": after,
        "ecc_score": ecc,
        "phase_shift": [float(shift[0]), float(shift[1])],
        "phase_response": float(phase_response),
    }


def _row_warp_image(
    image: np.ndarray,
    matrix: np.ndarray,
    control_y: np.ndarray,
    shift_x: np.ndarray,
    shift_y: np.ndarray,
    dense_shift_x: np.ndarray | None = None,
    dense_shift_y: np.ndarray | None = None,
    column_control_x: np.ndarray | None = None,
    column_shift_x: np.ndarray | None = None,
) -> np.ndarray:
    shape = image.shape
    yy, xx = np.indices(shape, dtype=np.float32)
    if dense_shift_x is not None and dense_shift_y is not None and dense_shift_x.size:
        xx = xx + np.asarray(dense_shift_x, dtype=np.float32)
        yy = yy + np.asarray(dense_shift_y, dtype=np.float32)
    if column_control_x is not None and column_shift_x is not None and column_control_x.size:
        dx_column = np.interp(xx[0], column_control_x, column_shift_x).astype(np.float32)
        xx = xx + dx_column[None, :]
    if control_y.size:
        dx = np.interp(yy[:, 0], control_y, shift_x).astype(np.float32)
        dy = np.interp(yy[:, 0], control_y, shift_y).astype(np.float32)
        xx = xx + dx[:, None]
        yy = yy + dy[:, None]
    mapped_x = matrix[0, 0] * xx + matrix[0, 1] * yy + matrix[0, 2]
    mapped_y = matrix[1, 0] * xx + matrix[1, 1] * yy + matrix[1, 2]
    return _remap_preserve_invalid(image, mapped_x, mapped_y)


def _column_line_profile(image: np.ndarray) -> np.ndarray:
    base = normalize_image(np.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0))
    response = np.abs(cv2.Scharr(base, cv2.CV_32F, 1, 0))
    margin = max(2, response.shape[0] // 12)
    profile = np.quantile(response[margin:-margin], 0.82, axis=0).astype(np.float32)
    profile = cv2.GaussianBlur(profile[:, None], (1, 9), 0).reshape(-1)
    return normalize_image(profile)


def _estimate_column_geometry_refinement(
    reference: np.ndarray,
    moving: np.ndarray,
    matrix: np.ndarray,
    config: RegistrationConfig,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    empty = np.zeros(0, dtype=np.float32)
    if not config.enable_roi_column_geometry_refinement:
        return empty, empty, {"accepted": False, "reason": "disabled"}

    aligned = _warp_affine(moving, matrix, reference.shape)
    ref_profile = _column_line_profile(reference)
    moving_profile = _column_line_profile(aligned)
    distance = max(3, int(config.roi_column_peak_distance))
    prominence = float(config.roi_column_peak_min_prominence)
    ref_peaks, ref_props = find_peaks(ref_profile, distance=distance, prominence=prominence)
    moving_peaks, moving_props = find_peaks(moving_profile, distance=distance, prominence=prominence)
    if ref_peaks.size < 3 or moving_peaks.size < 3:
        return empty, empty, {
            "accepted": False,
            "reason": "insufficient_vertical_line_peaks",
            "reference_peak_count": int(ref_peaks.size),
            "moving_peak_count": int(moving_peaks.size),
        }

    ref_prominence = np.asarray(ref_props["prominences"], dtype=np.float32)
    moving_prominence = np.asarray(moving_props["prominences"], dtype=np.float32)
    relative_floor = float(config.roi_column_reference_relative_prominence) * float(np.max(ref_prominence))
    eligible_ref = np.flatnonzero(ref_prominence >= max(prominence, relative_floor))
    ref_order = eligible_ref[np.argsort(ref_prominence[eligible_ref])[::-1]][: min(7, eligible_ref.size)]
    moving_order = np.argsort(moving_prominence)[::-1][: min(12, moving_peaks.size)]
    candidate_moving = sorted(int(index) for index in moving_order.tolist())
    used: set[int] = set()
    matches: list[tuple[int, int, float, float]] = []
    max_shift = int(config.roi_column_max_shift)
    for ref_index in ref_order:
        ref_x = int(ref_peaks[ref_index])
        candidates: list[tuple[float, int]] = []
        for moving_index in candidate_moving:
            if moving_index in used:
                continue
            moving_x = int(moving_peaks[moving_index])
            delta = abs(moving_x - ref_x)
            if delta > max_shift:
                continue
            score = float(moving_prominence[moving_index]) - 0.018 * float(delta)
            candidates.append((score, moving_index))
        if not candidates:
            continue
        _, moving_index = max(candidates, key=lambda item: item[0])
        used.add(moving_index)
        matches.append((
            ref_x,
            int(moving_peaks[moving_index]),
            float(ref_prominence[ref_index]),
            float(moving_prominence[moving_index]),
        ))

    matches.sort(key=lambda item: item[0])
    monotonic: list[tuple[int, int, float, float]] = []
    for match in matches:
        if monotonic and match[1] <= monotonic[-1][1]:
            if match[2] + match[3] > monotonic[-1][2] + monotonic[-1][3]:
                monotonic[-1] = match
            continue
        monotonic.append(match)
    matches = monotonic
    if len(matches) < 3:
        return empty, empty, {
            "accepted": False,
            "reason": "insufficient_monotonic_peak_matches",
            "matches": matches,
        }

    control_x = np.asarray([item[0] for item in matches], dtype=np.float32)
    shift_x = np.asarray([item[1] - item[0] for item in matches], dtype=np.float32)
    width = reference.shape[1]
    if control_x[0] > 0:
        control_x = np.insert(control_x, 0, 0.0)
        shift_x = np.insert(shift_x, 0, shift_x[0])
    if control_x[-1] < width - 1:
        control_x = np.append(control_x, float(width - 1))
        shift_x = np.append(shift_x, shift_x[-1])
    mapped = control_x + shift_x
    if np.any(np.diff(mapped) <= 0.25):
        return empty, empty, {"accepted": False, "reason": "non_monotonic_column_map", "matches": matches}

    refined = _row_warp_image(
        moving,
        matrix,
        empty,
        empty,
        empty,
        column_control_x=control_x,
        column_shift_x=shift_x,
    )
    profile_before = _corr(ref_profile, moving_profile)
    profile_after = _corr(ref_profile, _column_line_profile(refined))
    feature_before = _corr(_modality_feature(reference), _modality_feature(aligned))
    feature_after = _corr(_modality_feature(reference), _modality_feature(refined))
    accepted = bool(
        profile_after >= profile_before + float(config.roi_column_min_profile_gain)
        and feature_after >= feature_before - float(config.roi_column_max_feature_score_loss)
    )
    details = {
        "accepted": accepted,
        "profile_score_before": profile_before,
        "profile_score_after": profile_after,
        "feature_score_before": feature_before,
        "feature_score_after": feature_after,
        "matches": [
            {
                "reference_x": item[0],
                "moving_x": item[1],
                "shift_x": item[1] - item[0],
                "reference_prominence": item[2],
                "moving_prominence": item[3],
            }
            for item in matches
        ],
    }
    if not accepted:
        return empty, empty, details
    return control_x.astype(np.float32), shift_x.astype(np.float32), details


def _estimate_row_refinement(
    reference: np.ndarray,
    moving: np.ndarray,
    matrix: np.ndarray,
    config: RegistrationConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    if not config.enable_roi_row_refinement or config.roi_row_control_points < 3:
        empty = np.zeros(0, dtype=np.float32)
        return empty, empty, empty, {"accepted": False, "reason": "disabled"}

    ref_feature = _modality_feature(reference)
    aligned_feature = _warp_affine(_modality_feature(moving), matrix, reference.shape)
    height, width = reference.shape
    count = int(config.roi_row_control_points)
    half = max(28, int(np.ceil(height / max(4, count - 1))))
    centers = np.linspace(min(half, height // 3), max(height - half - 1, height * 2 // 3), count)
    radius = int(config.roi_row_search_radius)
    records: list[dict[str, Any]] = []
    accepted_points: list[tuple[float, float, float, float]] = []

    for center in centers:
        y0 = max(0, int(round(center)) - half)
        y1 = min(height, int(round(center)) + half + 1)
        margin = min(max(4, radius + 2), max(4, width // 8))
        ref_patch = ref_feature[y0:y1, margin : width - margin]
        yy, xx = np.indices((y1 - y0, width), dtype=np.float32)
        yy += y0
        scored: list[tuple[float, int, int]] = []
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                sampled = _remap_preserve_invalid(
                    aligned_feature,
                    xx + float(dx),
                    yy + float(dy),
                )[:, margin : width - margin]
                scored.append((_corr(ref_patch, sampled), dx, dy))
        scored.sort(key=lambda item: item[0], reverse=True)
        best = scored[0]
        base = next(item for item in scored if item[1] == 0 and item[2] == 0)
        peak_margin = best[0] - scored[1][0] if len(scored) > 1 else 0.0
        gain = best[0] - base[0]
        accepted = bool(
            (best[1] != 0 or best[2] != 0)
            and gain >= float(config.roi_row_min_score_gain)
            and peak_margin >= 0.00015
        )
        records.append({
            "rgb_y": float(center),
            "shift_x_analysis_px": int(best[1]),
            "shift_y_analysis_px": int(best[2]),
            "score": float(best[0]),
            "gain": float(gain),
            "peak_margin": float(peak_margin),
            "accepted": accepted,
        })
        if accepted:
            accepted_points.append((float(center), float(best[1]), float(best[2]), float(best[0])))

    if len(accepted_points) < 3:
        empty = np.zeros(0, dtype=np.float32)
        return empty, empty, empty, {"accepted": False, "reason": "insufficient_confident_controls", "controls": records}

    points = np.asarray(accepted_points, dtype=np.float32)
    for column in (1, 2):
        median = float(np.median(points[:, column]))
        mad = float(np.median(np.abs(points[:, column] - median))) + 1e-6
        points[np.abs(points[:, column] - median) > max(3.5 * mad, 3.0), column] = median
    control_y = points[:, 0]
    shift_x = points[:, 1]
    shift_y = points[:, 2]
    refined_feature = _row_warp_image(_modality_feature(moving), matrix, control_y, shift_x, shift_y)
    before = _corr(ref_feature, aligned_feature)
    after = _corr(ref_feature, refined_feature)
    if not np.isfinite(after) or after < before + 0.003:
        empty = np.zeros(0, dtype=np.float32)
        return empty, empty, empty, {
            "accepted": False,
            "reason": "global_score_guard",
            "score_before": before,
            "score_after": after,
            "controls": records,
        }
    return control_y, shift_x, shift_y, {
        "accepted": True,
        "score_before": before,
        "score_after": after,
        "controls": records,
    }


def _tiepoint_subpixel_peak(response: np.ndarray, py: int, px: int) -> tuple[float, float]:
    dy = dx = 0.0
    if 0 < px < response.shape[1] - 1:
        left, center, right = map(float, response[py, px - 1 : px + 2])
        denominator = left - 2.0 * center + right
        if np.isfinite([left, center, right]).all() and abs(denominator) > 1e-6:
            dx = float(np.clip(0.5 * (left - right) / denominator, -0.75, 0.75))
    if 0 < py < response.shape[0] - 1:
        top, center, bottom = map(float, response[py - 1 : py + 2, px])
        denominator = top - 2.0 * center + bottom
        if np.isfinite([top, center, bottom]).all() and abs(denominator) > 1e-6:
            dy = float(np.clip(0.5 * (top - bottom) / denominator, -0.75, 0.75))
    return dy, dx


def _match_tiepoint_one(
    reference: np.ndarray,
    moving: np.ndarray,
    y: int,
    x: int,
    template_radius: int,
    search_radius: int,
) -> tuple[float, float, float, float] | None:
    radius = int(template_radius)
    search_radius = int(search_radius)
    if (
        y - radius < 0
        or y + radius >= reference.shape[0]
        or x - radius < 0
        or x + radius >= reference.shape[1]
    ):
        return None
    template = np.asarray(
        reference[y - radius : y + radius + 1, x - radius : x + radius + 1],
        dtype=np.float32,
    )
    if not np.isfinite(template).all():
        return None
    if float(np.std(template)) < 0.035:
        return None
    y0 = max(0, y - radius - search_radius)
    y1 = min(moving.shape[0], y + radius + search_radius + 1)
    x0 = max(0, x - radius - search_radius)
    x1 = min(moving.shape[1], x + radius + search_radius + 1)
    search = np.asarray(moving[y0:y1, x0:x1], dtype=np.float32)
    if search.shape[0] < template.shape[0] or search.shape[1] < template.shape[1]:
        return None
    search_valid = np.isfinite(search)
    response = cv2.matchTemplate(
        np.where(search_valid, search, 0.0).astype(np.float32),
        template,
        cv2.TM_CCOEFF_NORMED,
    )
    support = cv2.matchTemplate(
        search_valid.astype(np.float32),
        np.ones(template.shape, dtype=np.float32),
        cv2.TM_CCORR,
    )
    response[support < float(template.size) - 0.5] = -np.inf
    if not np.isfinite(response).any():
        return None
    py, px = np.unravel_index(int(np.argmax(response)), response.shape)
    score = float(response[py, px])
    # A peak on the response boundary says only that the optimum lies outside
    # the tested search window; it cannot support a subpixel displacement.
    if py == 0 or px == 0 or py == response.shape[0] - 1 or px == response.shape[1] - 1:
        return None
    suppressed = response.copy()
    suppressed[max(0, py - 1) : py + 2, max(0, px - 1) : px + 2] = -1.0
    finite_suppressed = suppressed[np.isfinite(suppressed)]
    second = float(np.max(finite_suppressed)) if finite_suppressed.size > 9 else -1.0
    sub_y, sub_x = _tiepoint_subpixel_peak(response, py, px)
    moving_y = y0 + py + sub_y + radius
    moving_x = x0 + px + sub_x + radius
    return float(moving_y), float(moving_x), float(score), float(score - second)


def _estimate_tiepoint_field(
    reference: np.ndarray,
    moving: np.ndarray,
    config: RegistrationConfig,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    empty = np.zeros(reference.shape, dtype=np.float32)
    if not config.enable_roi_tiepoint_refinement:
        return empty, empty, {"accepted": False, "reason": "disabled"}
    ref_feature = _modality_feature(reference)
    moving_feature = _modality_feature(moving)
    radius = max(3, int(config.roi_tiepoint_template_radius))
    search_radius = max(1, int(config.roi_tiepoint_search_radius))
    ys = np.linspace(
        radius + 2,
        reference.shape[0] - radius - 3,
        max(3, int(config.roi_tiepoint_grid_rows)),
    ).round().astype(int)
    xs = np.linspace(
        radius + 2,
        reference.shape[1] - radius - 3,
        max(3, int(config.roi_tiepoint_grid_cols)),
    ).round().astype(int)
    candidates: list[RoiTiePoint] = []
    rejected: list[dict[str, Any]] = []
    for y in ys:
        for x in xs:
            forward = _match_tiepoint_one(ref_feature, moving_feature, int(y), int(x), radius, search_radius)
            if forward is None:
                rejected.append({"ref_y": int(y), "ref_x": int(x), "reason": "low_texture_or_border"})
                continue
            moving_y, moving_x, score, margin = forward
            backward = _match_tiepoint_one(
                moving_feature,
                ref_feature,
                int(round(moving_y)),
                int(round(moving_x)),
                radius,
                search_radius,
            )
            if backward is None:
                rejected.append({"ref_y": int(y), "ref_x": int(x), "score": score, "reason": "backward_failed"})
                continue
            back_y, back_x, _, _ = backward
            backward_error = float(np.hypot(back_y - y, back_x - x))
            point = RoiTiePoint(
                ref_y=float(y),
                ref_x=float(x),
                moving_y=moving_y,
                moving_x=moving_x,
                shift_y=moving_y - float(y),
                shift_x=moving_x - float(x),
                score=score,
                margin=margin,
                backward_error=backward_error,
            )
            if (
                score < float(config.roi_tiepoint_min_score)
                or margin < float(config.roi_tiepoint_min_margin)
                or backward_error > float(config.roi_tiepoint_max_backward_error)
            ):
                rejected.append({**point.to_dict(), "reason": "confidence_guard"})
            else:
                candidates.append(point)
    minimum = max(4, int(config.roi_tiepoint_min_points))
    if len(candidates) < minimum:
        return empty, empty, {
            "accepted": False,
            "reason": "insufficient_confident_tie_points",
            "tie_point_count": len(candidates),
            "rejected_tie_point_count": len(rejected),
            "tie_points": [point.to_dict() for point in candidates],
        }
    shifts = np.asarray([[point.shift_y, point.shift_x] for point in candidates], dtype=np.float32)
    median = np.median(shifts, axis=0)
    mad = np.median(np.abs(shifts - median), axis=0) + 0.35
    points: list[RoiTiePoint] = []
    for point, shift in zip(candidates, shifts, strict=True):
        robust_z = float(np.max(np.abs(shift - median) / (1.4826 * mad)))
        if robust_z > 3.5 or float(np.hypot(*shift)) > float(config.roi_tiepoint_max_shift):
            rejected.append({**point.to_dict(), "reason": "robust_displacement_guard"})
        else:
            points.append(point)
    if len(points) < minimum:
        return empty, empty, {
            "accepted": False,
            "reason": "insufficient_robust_tie_points",
            "tie_point_count": len(points),
            "rejected_tie_point_count": len(rejected),
            "tie_points": [point.to_dict() for point in points],
        }
    coordinates = np.asarray([[point.ref_y, point.ref_x] for point in points], dtype=np.float32)
    displacements = np.asarray([[point.shift_y, point.shift_x] for point in points], dtype=np.float32)
    confidence = np.asarray([max(0.05, point.score - 0.20) ** 2 for point in points], dtype=np.float32)
    yy, xx = np.indices(reference.shape, dtype=np.float32)
    query = np.stack([yy.reshape(-1), xx.reshape(-1)], axis=1)
    tree = cKDTree(coordinates)
    distances, indices = tree.query(query, k=min(int(config.roi_tiepoint_idw_neighbours), len(points)))
    if distances.ndim == 1:
        distances = distances[:, None]
        indices = indices[:, None]
    weights = confidence[indices] / (distances * distances + float(config.roi_tiepoint_idw_smoothing))
    weights /= np.maximum(np.sum(weights, axis=1, keepdims=True), 1e-8)
    dense = np.sum(weights[:, :, None] * displacements[indices], axis=1).reshape(reference.shape + (2,))
    sigma = max(0.0, float(config.roi_tiepoint_field_sigma))
    if sigma > 0:
        dense[:, :, 0] = cv2.GaussianBlur(dense[:, :, 0], (0, 0), sigma)
        dense[:, :, 1] = cv2.GaussianBlur(dense[:, :, 1], (0, 0), sigma)
    magnitude = np.sqrt(dense[:, :, 0] ** 2 + dense[:, :, 1] ** 2)
    return dense[:, :, 0].astype(np.float32), dense[:, :, 1].astype(np.float32), {
        "accepted": True,
        "method": "bidirectional_grid_tie_points_idw",
        "tie_point_count": len(points),
        "rejected_tie_point_count": len(rejected),
        "tie_points": [point.to_dict() for point in points],
        "rejected_tie_points": rejected[:64],
        "field_magnitude_median": float(np.median(magnitude)),
        "field_magnitude_p95": float(np.percentile(magnitude, 95)),
        "field_magnitude_max": float(np.max(magnitude)),
    }


def _warp_aligned_residual(
    image: np.ndarray,
    shift_y: np.ndarray,
    shift_x: np.ndarray,
    factor: float,
) -> np.ndarray:
    yy, xx = np.indices(image.shape, dtype=np.float32)
    return _remap_preserve_invalid(
        image,
        xx + float(factor) * np.asarray(shift_x, dtype=np.float32),
        yy + float(factor) * np.asarray(shift_y, dtype=np.float32),
    )


def _dense_field_jacobian_p01(shift_y: np.ndarray, shift_x: np.ndarray, factor: float) -> float:
    dy_y, dy_x = np.gradient(float(factor) * np.asarray(shift_y, dtype=np.float32))
    dx_y, dx_x = np.gradient(float(factor) * np.asarray(shift_x, dtype=np.float32))
    determinant = (1.0 + dx_x) * (1.0 + dy_y) - dx_y * dy_x
    return float(np.percentile(determinant, 1.0))


def _roi_registration_scores(
    reference: np.ndarray,
    nir: np.ndarray,
    swir: np.ndarray,
    nir_overlap: np.ndarray,
    swir_overlap: np.ndarray,
) -> dict[str, float]:
    return {
        "nir_rgb_score": _corr(_modality_feature(reference), _modality_feature(nir)),
        "swir_rgb_score": _corr(_modality_feature(reference), _modality_feature(swir)),
        "nir_swir_score": _corr(_modality_feature(swir_overlap), _modality_feature(nir_overlap)),
    }


def _select_tiepoint_factor(
    reference: np.ndarray,
    nir: np.ndarray,
    swir: np.ndarray,
    nir_overlap: np.ndarray,
    swir_overlap: np.ndarray,
    shift_y: np.ndarray,
    shift_x: np.ndarray,
    target: str,
    config: RegistrationConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, dict[str, Any]]:
    baseline = _roi_registration_scores(reference, nir, swir, nir_overlap, swir_overlap)
    trials: list[dict[str, float | bool]] = []
    best: tuple[float, float, np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None = None
    for factor in config.roi_tiepoint_factors:
        factor = float(factor)
        if target == "swir":
            nir_trial, nir_overlap_trial = nir, nir_overlap
            swir_trial = _warp_aligned_residual(swir, shift_y, shift_x, factor)
            swir_overlap_trial = _warp_aligned_residual(swir_overlap, shift_y, shift_x, factor)
        elif target == "nir":
            nir_trial = _warp_aligned_residual(nir, shift_y, shift_x, factor)
            nir_overlap_trial = _warp_aligned_residual(nir_overlap, shift_y, shift_x, factor)
            swir_trial, swir_overlap_trial = swir, swir_overlap
        elif target == "shared":
            nir_trial = _warp_aligned_residual(nir, shift_y, shift_x, factor)
            swir_trial = _warp_aligned_residual(swir, shift_y, shift_x, factor)
            nir_overlap_trial = _warp_aligned_residual(nir_overlap, shift_y, shift_x, factor)
            swir_overlap_trial = _warp_aligned_residual(swir_overlap, shift_y, shift_x, factor)
        else:
            raise ValueError(target)
        scores = _roi_registration_scores(reference, nir_trial, swir_trial, nir_overlap_trial, swir_overlap_trial)
        jacobian = _dense_field_jacobian_p01(shift_y, shift_x, factor)
        pair_loss = float(config.roi_tiepoint_pair_score_loss)
        if target == "swir":
            feasible = bool(scores["nir_swir_score"] >= baseline["nir_swir_score"] - pair_loss)
            objective = (
                scores["swir_rgb_score"] - baseline["swir_rgb_score"]
                + 0.40 * (scores["nir_swir_score"] - baseline["nir_swir_score"])
            )
        elif target == "nir":
            feasible = bool(scores["nir_swir_score"] >= baseline["nir_swir_score"] - pair_loss)
            objective = (
                scores["nir_rgb_score"] - baseline["nir_rgb_score"]
                + 0.40 * (scores["nir_swir_score"] - baseline["nir_swir_score"])
            )
        else:
            feasible = bool(scores["nir_swir_score"] >= baseline["nir_swir_score"] - 0.60 * pair_loss)
            objective = (
                scores["nir_rgb_score"] - baseline["nir_rgb_score"]
                + scores["swir_rgb_score"] - baseline["swir_rgb_score"]
                + 0.25 * (scores["nir_swir_score"] - baseline["nir_swir_score"])
            )
        feasible = feasible and jacobian > float(config.roi_tiepoint_jacobian_floor)
        trial = {"factor": factor, "feasible": feasible, "objective": objective, "jacobian_p01": jacobian, **scores}
        trials.append(trial)
        if feasible and (best is None or objective > best[0]):
            best = (objective, factor, nir_trial, swir_trial, nir_overlap_trial, swir_overlap_trial)
    if best is None:
        factor = 0.0
        selected = nir, swir, nir_overlap, swir_overlap
    else:
        _, factor, *selected = best
    return selected[0], selected[1], selected[2], selected[3], float(factor), {
        "target": target,
        "baseline": baseline,
        "trials": trials,
        "selected_factor": float(factor),
        "selected": next((trial for trial in trials if trial["factor"] == float(factor)), None),
    }


def _compose_dense_field(
    base_y: np.ndarray,
    base_x: np.ndarray,
    added_y: np.ndarray,
    added_x: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compose output-space warps: first base, then added."""

    yy, xx = np.indices(base_y.shape, dtype=np.float32)
    sampled_base_y = cv2.remap(base_y, xx + added_x, yy + added_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT101)
    sampled_base_x = cv2.remap(base_x, xx + added_x, yy + added_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT101)
    return (added_y + sampled_base_y).astype(np.float32), (added_x + sampled_base_x).astype(np.float32)


def _mean_selected_bands(
    cube: np.ndarray,
    meta: Any,
    model: RegistrationModel,
    grid_y: np.ndarray,
    grid_x: np.ndarray,
    wavelengths: list[float],
) -> np.ndarray:
    indices = select_band_indices(meta, wavelengths)
    sampled = sample_cube_on_rgb_grid(cube, model, grid_y, grid_x, bands=indices)
    finite = np.isfinite(sampled)
    count = finite.sum(axis=2)
    total = np.where(finite, sampled, 0.0).sum(axis=2)
    mean = np.full_like(total, np.nan, dtype=np.float32)
    np.divide(total, count, out=mean, where=count > 0)
    normalized = normalize_image(mean)
    normalized[count == 0] = np.nan
    return normalized


def estimate_roi_registration(
    dataset: DatasetTriplet,
    coarse: RegistrationBundle,
    roi: dict[str, int],
    analysis_shape: tuple[int, int],
    config: RegistrationConfig,
) -> RoiRegistrationBundle:
    """Refine the actual ROI observation grid and jointly lock NIR to SWIR overlap bands."""

    grid_y, grid_x = analysis_rgb_grid(roi, *analysis_shape)
    rgb_crop = np.asarray(
        dataset.rgb.cube[
            roi["y"] : roi["y"] + roi["height"],
            roi["x"] : roi["x"] + roi["width"],
            :3,
        ]
    )
    reference = rgb_structure(rgb_crop, target_shape=analysis_shape)
    nir_initial = _mean_selected_bands(
        dataset.nir.cube, dataset.nir.meta, coarse.nir, grid_y, grid_x,
        [750, 850, 950, 1050, 1250, 1400],
    )
    swir_initial = _mean_selected_bands(
        dataset.swir.cube, dataset.swir.meta, coarse.swir, grid_y, grid_x,
        [1050, 1250, 1650, 2200, 2350],
    )
    overlap_wavelengths = [1050, 1150, 1250, 1350, 1450]
    nir_overlap_initial = _mean_selected_bands(
        dataset.nir.cube, dataset.nir.meta, coarse.nir, grid_y, grid_x, overlap_wavelengths,
    )
    swir_overlap_initial = _mean_selected_bands(
        dataset.swir.cube, dataset.swir.meta, coarse.swir, grid_y, grid_x, overlap_wavelengths,
    )

    nir_matrix, _, nir_details = _estimate_roi_affine(reference, nir_initial, config)
    swir_matrix, _, swir_details = _estimate_roi_affine(reference, swir_initial, config)
    nir_overlap_direct = _warp_affine(nir_overlap_initial, nir_matrix, analysis_shape)
    swir_overlap_direct = _warp_affine(swir_overlap_initial, swir_matrix, analysis_shape)
    pair_before = _corr(_modality_feature(swir_overlap_direct), _modality_feature(nir_overlap_direct))
    pair_matrix, _, pair_details = _estimate_roi_affine(
        swir_overlap_direct,
        nir_overlap_direct,
        config,
        min_gain=float(config.roi_pair_min_score_gain),
    )
    pair_accepted = bool(pair_details["accepted"])
    if pair_accepted:
        candidate_nir_matrix = nir_matrix @ pair_matrix
        candidate_nir_rgb = _warp_affine(nir_initial, candidate_nir_matrix, analysis_shape)
        candidate_rgb_score = _corr(_modality_feature(reference), _modality_feature(candidate_nir_rgb))
        direct_rgb = _warp_affine(nir_initial, nir_matrix, analysis_shape)
        direct_rgb_score = _corr(_modality_feature(reference), _modality_feature(direct_rgb))
        candidate_overlap = _warp_affine(nir_overlap_initial, candidate_nir_matrix, analysis_shape)
        candidate_pair_score = _corr(_modality_feature(swir_overlap_direct), _modality_feature(candidate_overlap))
        if (
            candidate_rgb_score < direct_rgb_score - float(config.roi_pair_max_rgb_score_loss)
            or candidate_pair_score < pair_before + float(config.roi_pair_min_score_gain)
        ):
            pair_accepted = False
        else:
            nir_matrix = candidate_nir_matrix

    nir_overlap_column_base = _warp_affine(nir_overlap_initial, nir_matrix, analysis_shape)
    swir_overlap_column_base = _warp_affine(swir_overlap_initial, swir_matrix, analysis_shape)
    pair_column_baseline = _corr(
        _modality_feature(swir_overlap_column_base),
        _modality_feature(nir_overlap_column_base),
    )
    nir_column_x, nir_column_shift_x, nir_column_details = _estimate_column_geometry_refinement(
        reference, nir_initial, nir_matrix, config,
    )
    swir_column_x, swir_column_shift_x, swir_column_details = _estimate_column_geometry_refinement(
        reference, swir_initial, swir_matrix, config,
    )
    if nir_column_x.size and swir_column_x.size:
        shared_control_x = np.unique(np.concatenate([nir_column_x, swir_column_x])).astype(np.float32)
        full_shared_shift_x = 0.5 * (
            np.interp(shared_control_x, nir_column_x, nir_column_shift_x)
            + np.interp(shared_control_x, swir_column_x, swir_column_shift_x)
        )
        empty = np.zeros(0, dtype=np.float32)
        reference_feature = _modality_feature(reference)
        nir_base = _warp_affine(nir_initial, nir_matrix, analysis_shape)
        swir_base = _warp_affine(swir_initial, swir_matrix, analysis_shape)
        nir_overlap_base = _warp_affine(nir_overlap_initial, nir_matrix, analysis_shape)
        swir_overlap_base = _warp_affine(swir_overlap_initial, swir_matrix, analysis_shape)
        nir_feature_base = _corr(reference_feature, _modality_feature(nir_base))
        swir_feature_base = _corr(reference_feature, _modality_feature(swir_base))
        pair_feature_base = _corr(_modality_feature(swir_overlap_base), _modality_feature(nir_overlap_base))
        nir_profile_base = _corr(_column_line_profile(reference), _column_line_profile(nir_base))
        swir_profile_base = _corr(_column_line_profile(reference), _column_line_profile(swir_base))
        consensus_trials: list[dict[str, float | bool]] = []
        best_consensus: tuple[float, float, np.ndarray] | None = None
        for factor in (1.0, 0.75, 0.5, 0.25, 0.0):
            trial_shift = (full_shared_shift_x * factor).astype(np.float32)
            trial_controls = shared_control_x if factor > 0 else empty
            trial_values = trial_shift if factor > 0 else empty
            nir_trial = _row_warp_image(
                nir_initial, nir_matrix, empty, empty, empty,
                column_control_x=trial_controls, column_shift_x=trial_values,
            )
            swir_trial = _row_warp_image(
                swir_initial, swir_matrix, empty, empty, empty,
                column_control_x=trial_controls, column_shift_x=trial_values,
            )
            nir_overlap_trial = _row_warp_image(
                nir_overlap_initial, nir_matrix, empty, empty, empty,
                column_control_x=trial_controls, column_shift_x=trial_values,
            )
            swir_overlap_trial = _row_warp_image(
                swir_overlap_initial, swir_matrix, empty, empty, empty,
                column_control_x=trial_controls, column_shift_x=trial_values,
            )
            nir_feature = _corr(reference_feature, _modality_feature(nir_trial))
            swir_feature = _corr(reference_feature, _modality_feature(swir_trial))
            pair_feature = _corr(_modality_feature(swir_overlap_trial), _modality_feature(nir_overlap_trial))
            nir_profile = _corr(_column_line_profile(reference), _column_line_profile(nir_trial))
            swir_profile = _corr(_column_line_profile(reference), _column_line_profile(swir_trial))
            feasible = bool(
                pair_feature >= pair_feature_base - float(config.roi_column_max_pair_score_loss)
                and nir_feature >= nir_feature_base - float(config.roi_column_max_feature_score_loss)
                and swir_feature >= swir_feature_base - float(config.roi_column_max_feature_score_loss)
            )
            geometry_gain = (nir_profile - nir_profile_base) + (swir_profile - swir_profile_base)
            objective = geometry_gain + 0.04 * (nir_feature + swir_feature + pair_feature)
            consensus_trials.append({
                "factor": factor,
                "feasible": feasible,
                "objective": objective,
                "nir_rgb_score": nir_feature,
                "swir_rgb_score": swir_feature,
                "nir_swir_score": pair_feature,
                "nir_profile_score": nir_profile,
                "swir_profile_score": swir_profile,
            })
            if feasible and (best_consensus is None or objective > best_consensus[0]):
                best_consensus = (objective, factor, trial_shift)
        selected_factor = best_consensus[1] if best_consensus is not None else 0.0
        selected_shift = best_consensus[2] if best_consensus is not None else np.zeros_like(full_shared_shift_x)
        if selected_factor > 0:
            nir_column_x = swir_column_x = shared_control_x
            nir_column_shift_x = swir_column_shift_x = selected_shift.astype(np.float32)
        else:
            nir_column_x = nir_column_shift_x = empty
            swir_column_x = swir_column_shift_x = empty
        nir_column_details = dict(nir_column_details)
        swir_column_details = dict(swir_column_details)
        nir_column_details.update({
            "accepted": selected_factor > 0,
            "joint_consensus_factor": selected_factor,
            "joint_consensus_shift": selected_shift.tolist(),
            "joint_consensus_trials": consensus_trials,
        })
        swir_column_details.update({
            "accepted": selected_factor > 0,
            "joint_consensus_factor": selected_factor,
            "joint_consensus_shift": selected_shift.tolist(),
            "joint_consensus_trials": consensus_trials,
        })
    nir_control_y, nir_shift_x, nir_shift_y, nir_row_details = _estimate_row_refinement(
        reference, nir_initial, nir_matrix, config,
    )
    swir_control_y, swir_shift_x, swir_shift_y, swir_row_details = _estimate_row_refinement(
        reference, swir_initial, swir_matrix, config,
    )
    nir_aligned = _row_warp_image(
        nir_initial, nir_matrix, nir_control_y, nir_shift_x, nir_shift_y,
        column_control_x=nir_column_x, column_shift_x=nir_column_shift_x,
    )
    swir_aligned = _row_warp_image(
        swir_initial, swir_matrix, swir_control_y, swir_shift_x, swir_shift_y,
        column_control_x=swir_column_x, column_shift_x=swir_column_shift_x,
    )
    nir_overlap_aligned = _row_warp_image(
        nir_overlap_initial, nir_matrix, nir_control_y, nir_shift_x, nir_shift_y,
        column_control_x=nir_column_x, column_shift_x=nir_column_shift_x,
    )
    swir_overlap_aligned = _row_warp_image(
        swir_overlap_initial, swir_matrix, swir_control_y, swir_shift_x, swir_shift_y,
        column_control_x=swir_column_x, column_shift_x=swir_column_shift_x,
    )
    nir_before = _corr(_modality_feature(reference), _modality_feature(nir_initial))
    swir_before = _corr(_modality_feature(reference), _modality_feature(swir_initial))
    nir_after = _corr(_modality_feature(reference), _modality_feature(nir_aligned))
    swir_after = _corr(_modality_feature(reference), _modality_feature(swir_aligned))
    pair_after = _corr(_modality_feature(swir_overlap_aligned), _modality_feature(nir_overlap_aligned))
    if pair_after < pair_column_baseline - float(config.roi_column_max_pair_score_loss) and (
        nir_column_x.size or swir_column_x.size
    ):
        empty = np.zeros(0, dtype=np.float32)
        nir_column_details = dict(nir_column_details)
        swir_column_details = dict(swir_column_details)
        nir_column_details.update({
            "accepted": False,
            "reason": "nir_swir_pair_consistency_guard",
            "candidate_pair_score": pair_after,
        })
        swir_column_details.update({
            "accepted": False,
            "reason": "nir_swir_pair_consistency_guard",
            "candidate_pair_score": pair_after,
        })
        nir_column_x = nir_column_shift_x = empty
        swir_column_x = swir_column_shift_x = empty
        nir_aligned = _row_warp_image(nir_initial, nir_matrix, nir_control_y, nir_shift_x, nir_shift_y)
        swir_aligned = _row_warp_image(swir_initial, swir_matrix, swir_control_y, swir_shift_x, swir_shift_y)
        nir_overlap_aligned = _row_warp_image(
            nir_overlap_initial, nir_matrix, nir_control_y, nir_shift_x, nir_shift_y,
        )
        swir_overlap_aligned = _row_warp_image(
            swir_overlap_initial, swir_matrix, swir_control_y, swir_shift_x, swir_shift_y,
        )
        nir_after = _corr(_modality_feature(reference), _modality_feature(nir_aligned))
        swir_after = _corr(_modality_feature(reference), _modality_feature(swir_aligned))
        pair_after = _corr(_modality_feature(swir_overlap_aligned), _modality_feature(nir_overlap_aligned))

    dense_nir_y = np.zeros(analysis_shape, dtype=np.float32)
    dense_nir_x = np.zeros(analysis_shape, dtype=np.float32)
    dense_swir_y = np.zeros(analysis_shape, dtype=np.float32)
    dense_swir_x = np.zeros(analysis_shape, dtype=np.float32)
    swir_tie_y, swir_tie_x, swir_tiepoint_details = _estimate_tiepoint_field(
        reference, swir_aligned, config,
    )
    if swir_tiepoint_details.get("accepted"):
        (
            nir_aligned,
            swir_aligned,
            nir_overlap_aligned,
            swir_overlap_aligned,
            swir_tie_factor,
            swir_tiepoint_selection,
        ) = _select_tiepoint_factor(
            reference,
            nir_aligned,
            swir_aligned,
            nir_overlap_aligned,
            swir_overlap_aligned,
            swir_tie_y,
            swir_tie_x,
            "swir",
            config,
        )
        dense_swir_y = (swir_tie_factor * swir_tie_y).astype(np.float32)
        dense_swir_x = (swir_tie_factor * swir_tie_x).astype(np.float32)
        swir_tiepoint_details = {**swir_tiepoint_details, "selection": swir_tiepoint_selection}

    nir_tie_y, nir_tie_x, nir_tiepoint_details = _estimate_tiepoint_field(
        reference, nir_aligned, config,
    )
    if nir_tiepoint_details.get("accepted"):
        (
            nir_aligned,
            swir_aligned,
            nir_overlap_aligned,
            swir_overlap_aligned,
            nir_tie_factor,
            nir_tiepoint_selection,
        ) = _select_tiepoint_factor(
            reference,
            nir_aligned,
            swir_aligned,
            nir_overlap_aligned,
            swir_overlap_aligned,
            nir_tie_y,
            nir_tie_x,
            "nir",
            config,
        )
        dense_nir_y = (nir_tie_factor * nir_tie_y).astype(np.float32)
        dense_nir_x = (nir_tie_factor * nir_tie_x).astype(np.float32)
        nir_tiepoint_details = {**nir_tiepoint_details, "selection": nir_tiepoint_selection}

    pair_tie_y, pair_tie_x, pair_tiepoint_details = _estimate_tiepoint_field(
        swir_overlap_aligned, nir_overlap_aligned, config,
    )
    if pair_tiepoint_details.get("accepted"):
        (
            nir_aligned,
            swir_aligned,
            nir_overlap_aligned,
            swir_overlap_aligned,
            pair_tie_factor,
            pair_tiepoint_selection,
        ) = _select_tiepoint_factor(
            reference,
            nir_aligned,
            swir_aligned,
            nir_overlap_aligned,
            swir_overlap_aligned,
            pair_tie_y,
            pair_tie_x,
            "nir",
            config,
        )
        added_pair_y = (pair_tie_factor * pair_tie_y).astype(np.float32)
        added_pair_x = (pair_tie_factor * pair_tie_x).astype(np.float32)
        dense_nir_y, dense_nir_x = _compose_dense_field(
            dense_nir_y, dense_nir_x, added_pair_y, added_pair_x,
        )
        pair_tiepoint_details = {**pair_tiepoint_details, "selection": pair_tiepoint_selection}

    joint_structure = normalize_image(
        0.5 * normalize_image(nir_aligned) + 0.5 * normalize_image(swir_aligned)
    )
    joint_tie_y, joint_tie_x, joint_tiepoint_details = _estimate_tiepoint_field(
        reference, joint_structure, config,
    )
    if joint_tiepoint_details.get("accepted"):
        (
            nir_aligned,
            swir_aligned,
            nir_overlap_aligned,
            swir_overlap_aligned,
            joint_tie_factor,
            joint_tiepoint_selection,
        ) = _select_tiepoint_factor(
            reference,
            nir_aligned,
            swir_aligned,
            nir_overlap_aligned,
            swir_overlap_aligned,
            joint_tie_y,
            joint_tie_x,
            "shared",
            config,
        )
        added_y = (joint_tie_factor * joint_tie_y).astype(np.float32)
        added_x = (joint_tie_factor * joint_tie_x).astype(np.float32)
        dense_nir_y, dense_nir_x = _compose_dense_field(dense_nir_y, dense_nir_x, added_y, added_x)
        dense_swir_y, dense_swir_x = _compose_dense_field(dense_swir_y, dense_swir_x, added_y, added_x)
        joint_tiepoint_details = {**joint_tiepoint_details, "selection": joint_tiepoint_selection}

    nir_after = _corr(_modality_feature(reference), _modality_feature(nir_aligned))
    swir_after = _corr(_modality_feature(reference), _modality_feature(swir_aligned))
    pair_after = _corr(_modality_feature(swir_overlap_aligned), _modality_feature(nir_overlap_aligned))

    nir_model = RoiRegistrationModel(
        sensor="NIR",
        analysis_shape=analysis_shape,
        affine_matrix=nir_matrix,
        dense_shift_x=dense_nir_x,
        dense_shift_y=dense_nir_y,
        column_control_x=nir_column_x,
        column_shift_x=nir_column_shift_x,
        row_control_y=nir_control_y,
        row_shift_x=nir_shift_x,
        row_shift_y=nir_shift_y,
        score_before=nir_before,
        score_after=nir_after,
        ecc_score=float(nir_details["ecc_score"]),
        accepted_affine=bool(nir_details["accepted"]),
        details={
            "affine": nir_details,
            "column_geometry_refinement": nir_column_details,
            "row_refinement": nir_row_details,
            "tiepoint_refinement": nir_tiepoint_details,
            "nir_to_swir_tiepoint_refinement": pair_tiepoint_details,
            "joint_hsi_to_rgb_tiepoint_refinement": joint_tiepoint_details,
        },
    )
    swir_model = RoiRegistrationModel(
        sensor="SWIR",
        analysis_shape=analysis_shape,
        affine_matrix=swir_matrix,
        dense_shift_x=dense_swir_x,
        dense_shift_y=dense_swir_y,
        column_control_x=swir_column_x,
        column_shift_x=swir_column_shift_x,
        row_control_y=swir_control_y,
        row_shift_x=swir_shift_x,
        row_shift_y=swir_shift_y,
        score_before=swir_before,
        score_after=swir_after,
        ecc_score=float(swir_details["ecc_score"]),
        accepted_affine=bool(swir_details["accepted"]),
        details={
            "affine": swir_details,
            "column_geometry_refinement": swir_column_details,
            "row_refinement": swir_row_details,
            "tiepoint_refinement": swir_tiepoint_details,
            "nir_to_swir_tiepoint_refinement": pair_tiepoint_details,
            "joint_hsi_to_rgb_tiepoint_refinement": joint_tiepoint_details,
        },
    )
    weakest = min(nir_after, swir_after, pair_after)
    if weakest < float(config.roi_fail_score):
        status = "failed"
    elif min(nir_after, swir_after) < float(config.roi_warning_score) or pair_after < float(config.roi_pair_warning_score):
        status = "warning"
    else:
        status = "passed"
    return RoiRegistrationBundle(
        roi=dict(roi),
        analysis_shape=analysis_shape,
        nir=nir_model,
        swir=swir_model,
        status=status,
        pair_score_before=pair_before,
        pair_score_after=pair_after,
        pair_refinement_accepted=pair_accepted,
        reference_structure=reference,
        nir_initial=nir_initial,
        swir_initial=swir_initial,
        nir_aligned=nir_aligned,
        swir_aligned=swir_aligned,
        nir_overlap_aligned=nir_overlap_aligned,
        swir_overlap_aligned=swir_overlap_aligned,
    )


def refined_analysis_rgb_grid(
    roi: dict[str, int],
    model: RoiRegistrationModel,
) -> tuple[np.ndarray, np.ndarray]:
    """Return RGB coordinates whose coarse sensor mapping includes the ROI residual warp."""

    height, width = model.analysis_shape
    yy, xx = np.indices((height, width), dtype=np.float32)
    mapped_y, mapped_x = model.map_analysis_indices(yy, xx)
    scale_y = (roi["height"] - 1) / float(max(height - 1, 1))
    scale_x = (roi["width"] - 1) / float(max(width - 1, 1))
    rgb_y = roi["y"] + mapped_y * scale_y
    rgb_x = roi["x"] + mapped_x * scale_x
    return rgb_y.astype(np.float32), rgb_x.astype(np.float32)


def _endpoint_coordinate_scale(source_size: int, target_size: int) -> float:
    """Scale pixel-center coordinates while mapping both raster endpoints exactly."""

    if source_size <= 1 or target_size <= 1:
        return 0.0
    return (float(target_size) - 1.0) / (float(source_size) - 1.0)


def _estimate_one(
    rgb_preview: np.ndarray,
    sensor_structure: np.ndarray,
    *,
    rgb_shape: tuple[int, int],
    sensor_shape: tuple[int, int],
    sensor_name: str,
    config: RegistrationConfig,
) -> tuple[RegistrationModel, np.ndarray]:
    height, width = rgb_preview.shape
    moving = cv2.resize(sensor_structure, (width, height), interpolation=cv2.INTER_AREA)
    template_feature = _feature(rgb_preview)
    moving_feature = _feature(moving)

    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, int(config.ecc_iterations), float(config.ecc_epsilon))
    template_cv, moving_cv, cv_support = _finite_cv_pair(template_feature, moving_feature)
    if int(cv_support.sum()) >= 32:
        shift, phase_response = cv2.phaseCorrelate(template_cv, moving_cv, cv_support.astype(np.float32))
    else:
        shift, phase_response = (0.0, 0.0), 0.0
    initial_translation = np.array([[1.0, 0.0, shift[0]], [0.0, 1.0, shift[1]]], dtype=np.float32)
    requested = config.motion.lower()
    candidate_types = [cv2.MOTION_TRANSLATION, cv2.MOTION_EUCLIDEAN, cv2.MOTION_AFFINE] if requested in {"auto", "auto_physical"} else [
        cv2.MOTION_AFFINE if requested == "affine" else cv2.MOTION_EUCLIDEAN if requested == "euclidean" else cv2.MOTION_TRANSLATION
    ]
    candidates: list[tuple[float, float, np.ndarray, np.ndarray]] = []
    for motion in candidate_types:
        candidate = initial_translation.copy()
        try:
            ecc, candidate = cv2.findTransformECC(
                template_cv,
                moving_cv,
                candidate,
                motion,
                criteria,
                (cv_support.astype(np.uint8) * 255),
                int(config.gaussian_filter_size),
            )
        except cv2.error:
            continue
        linear_candidate = candidate[:, :2]
        physical = (
            0.92 <= linear_candidate[0, 0] <= 1.08
            and 0.92 <= linear_candidate[1, 1] <= 1.08
            and abs(float(linear_candidate[0, 1])) <= 0.04
            and abs(float(linear_candidate[1, 0])) <= 0.04
        )
        if requested == "auto_physical" and not physical:
            continue
        aligned_candidate = _warp_affine(moving_feature, candidate, (height, width))
        correlation = _corr(template_feature, aligned_candidate)
        if not np.isfinite(correlation):
            continue
        complexity_penalty = 0.006 * (motion == cv2.MOTION_EUCLIDEAN) + 0.012 * (motion == cv2.MOTION_AFFINE)
        candidates.append((float(correlation - complexity_penalty), float(ecc), candidate.copy(), aligned_candidate))
    if candidates:
        _, ecc_score, warp, aligned = max(candidates, key=lambda item: item[0])
    else:
        warp = initial_translation
        ecc_score = float(phase_response)
        aligned = _warp_affine(moving_feature, warp, (height, width))

    rgb_to_preview = np.array(
        [
            [_endpoint_coordinate_scale(rgb_shape[1], width), 0.0, 0.0],
            [0.0, _endpoint_coordinate_scale(rgb_shape[0], height), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    warp3 = np.vstack([warp.astype(np.float64), [0.0, 0.0, 1.0]])
    preview_to_sensor = np.array(
        [
            [_endpoint_coordinate_scale(width, sensor_shape[1]), 0.0, 0.0],
            [0.0, _endpoint_coordinate_scale(height, sensor_shape[0]), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    raw_matrix = preview_to_sensor @ warp3 @ rgb_to_preview

    drift_y: list[float] = []
    drift_dx: list[float] = []
    drift_dy: list[float] = []
    if config.enable_strip_drift and config.strip_count >= 3:
        strip_h = max(48, int(np.ceil(height / config.strip_count)))
        overlap = int(round(strip_h * config.strip_overlap))
        linear = warp3[:2, :2]
        for center in np.linspace(strip_h // 2, height - strip_h // 2 - 1, config.strip_count):
            y0 = max(0, int(round(center - strip_h / 2 - overlap)))
            y1 = min(height, int(round(center + strip_h / 2 + overlap)))
            if y1 - y0 < 32:
                continue
            ref_strip = template_feature[y0:y1]
            mov_strip = aligned[y0:y1]
            window = cv2.createHanningWindow((width, y1 - y0), cv2.CV_32F)
            ref_cv, mov_cv, strip_support = _finite_cv_pair(ref_strip, mov_strip)
            window *= strip_support.astype(np.float32)
            if int(strip_support.sum()) >= 32:
                shift, response = cv2.phaseCorrelate(ref_cv, mov_cv, window)
            else:
                shift, response = (0.0, 0.0), 0.0
            dx_preview = float(np.clip(shift[0], -config.strip_max_shift_preview_px, config.strip_max_shift_preview_px))
            dy_preview = float(np.clip(shift[1], -config.strip_max_shift_preview_px, config.strip_max_shift_preview_px))
            if not np.isfinite(response) or response < 0.02:
                dx_preview = dy_preview = 0.0
            correction_input_preview = linear @ np.asarray([dx_preview, dy_preview], dtype=np.float64)
            drift_y.append(float(center * _endpoint_coordinate_scale(height, rgb_shape[0])))
            drift_dx.append(float(correction_input_preview[0] * _endpoint_coordinate_scale(width, sensor_shape[1])))
            drift_dy.append(float(correction_input_preview[1] * _endpoint_coordinate_scale(height, sensor_shape[0])))

    if drift_y:
        dx_array = np.asarray(drift_dx, dtype=np.float64)
        dy_array = np.asarray(drift_dy, dtype=np.float64)
        for values in (dx_array, dy_array):
            median = float(np.median(values))
            mad = float(np.median(np.abs(values - median))) + 1e-6
            values[np.abs(values - median) > max(3.5 * mad, 3.0)] = median
        median_dx = float(np.median(dx_array))
        median_dy = float(np.median(dy_array))
        raw_matrix[0, 2] += median_dx
        raw_matrix[1, 2] += median_dy
        drift_dx = (dx_array - median_dx).tolist()
        drift_dy = (dy_array - median_dy).tolist()

    model = RegistrationModel(
        sensor=sensor_name,
        rgb_shape=rgb_shape,
        sensor_shape=sensor_shape,
        rgb_to_sensor_matrix=raw_matrix,
        drift_rgb_y=np.asarray(drift_y, dtype=np.float32),
        drift_sensor_dx=np.asarray(drift_dx, dtype=np.float32),
        drift_sensor_dy=np.asarray(drift_dy, dtype=np.float32),
        ecc_score=float(ecc_score),
        edge_correlation=_corr(template_feature, aligned),
    )
    return model, aligned


def estimate_registration(dataset: DatasetTriplet, config: RegistrationConfig) -> RegistrationBundle:
    cv2.setNumThreads(1)
    cv2.setRNGSeed(20260717)
    cv2.setUseOptimized(False)
    if hasattr(cv2, "ocl"):
        cv2.ocl.setUseOpenCL(False)
    preview_height = min(
        int(config.preview_max_height),
        max(128, int(round(dataset.rgb.meta.lines * config.preview_width / dataset.rgb.meta.samples))),
    )
    preview_shape = (preview_height, int(config.preview_width))
    rgb_preview = rgb_structure(dataset.rgb.cube, target_shape=preview_shape)
    nir_structure = hsi_structure(dataset.nir.cube, dataset.nir.meta, "NIR", target_shape=preview_shape)
    swir_structure = hsi_structure(dataset.swir.cube, dataset.swir.meta, "SWIR", target_shape=preview_shape)
    nir_model, nir_aligned = _estimate_one(
        rgb_preview, nir_structure,
        rgb_shape=dataset.rgb.meta.shape[:2], sensor_shape=dataset.nir.meta.shape[:2],
        sensor_name="NIR", config=config,
    )
    swir_model, swir_aligned = _estimate_one(
        rgb_preview, swir_structure,
        rgb_shape=dataset.rgb.meta.shape[:2], sensor_shape=dataset.swir.meta.shape[:2],
        sensor_name="SWIR", config=config,
    )
    return RegistrationBundle(nir=nir_model, swir=swir_model, preview_rgb=rgb_preview, preview_nir_aligned=nir_aligned, preview_swir_aligned=swir_aligned)


def sample_cube_on_rgb_grid(
    cube: np.ndarray,
    model: RegistrationModel,
    rgb_y: np.ndarray,
    rgb_x: np.ndarray,
    *,
    bands: slice | list[int] | np.ndarray | None = None,
) -> np.ndarray:
    """Sample a sensor cube on an arbitrary RGB-coordinate grid."""

    sy, sx = model.map_rgb_to_sensor(rgb_y, rgb_x)
    band_indices = np.arange(cube.shape[2]) if bands is None else np.arange(cube.shape[2])[bands] if isinstance(bands, slice) else np.asarray(bands, dtype=int)
    margin = 2
    y0 = max(0, int(np.floor(np.nanmin(sy))) - margin)
    y1 = min(cube.shape[0], int(np.ceil(np.nanmax(sy))) + margin + 1)
    x0 = max(0, int(np.floor(np.nanmin(sx))) - margin)
    x1 = min(cube.shape[1], int(np.ceil(np.nanmax(sx))) + margin + 1)
    if y1 <= y0 or x1 <= x0:
        raise ValueError("Requested RGB grid does not overlap sensor coverage")
    map_y = (sy - y0).astype(np.float32)
    map_x = (sx - x0).astype(np.float32)
    out = np.empty(rgb_y.shape + (band_indices.size,), dtype=np.float32)
    for out_idx, band in enumerate(band_indices):
        source = np.asarray(cube[y0:y1, x0:x1, int(band)], dtype=np.float32)
        out[:, :, out_idx] = cv2.remap(
            source,
            map_x,
            map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=float("nan"),
        )
    return out


def analysis_rgb_grid(roi: dict[str, int], height: int, width: int) -> tuple[np.ndarray, np.ndarray]:
    ys = np.linspace(roi["y"], roi["y"] + roi["height"] - 1, height, dtype=np.float32)
    xs = np.linspace(roi["x"], roi["x"] + roi["width"] - 1, width, dtype=np.float32)
    return np.meshgrid(ys, xs, indexing="ij")
