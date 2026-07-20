import cv2
import numpy as np
import pytest

from geocorefusion.config import RegistrationConfig
from geocorefusion.registration import (
    _corr,
    _estimate_one,
    _estimate_roi_affine,
    _estimate_tiepoint_field,
    _match_tiepoint_one,
    _modality_feature,
    _warp_affine,
    _warp_aligned_residual,
)


def _texture(shape: tuple[int, int]) -> np.ndarray:
    y, x = np.indices(shape, dtype=np.float32)
    image = 0.3 * np.sin(x / 5.0) + 0.2 * np.cos(y / 11.0)
    image += ((x - 42) ** 2 + (y - 118) ** 2 < 18**2) * 1.2
    image += ((x > 75) & (x < 88) & (y > 20) & (y < 205)) * 0.8
    rng = np.random.default_rng(20260720)
    image += cv2.GaussianBlur(rng.normal(0.0, 0.22, shape).astype(np.float32), (0, 0), 1.0)
    return cv2.GaussianBlur(image.astype(np.float32), (0, 0), 0.7)


def _affine_tre_stats(
    estimated: np.ndarray,
    truth: np.ndarray,
    shape: tuple[int, int],
    margin: int = 20,
) -> tuple[float, float]:
    ys, xs = np.meshgrid(
        np.linspace(margin, shape[0] - margin - 1, 11),
        np.linspace(margin, shape[1] - margin - 1, 9),
        indexing="ij",
    )
    points = np.stack([xs.reshape(-1), ys.reshape(-1), np.ones(xs.size)], axis=0)
    errors = np.linalg.norm(estimated[:2] @ points - truth[:2] @ points, axis=0)
    return float(np.median(errors)), float(np.percentile(errors, 95.0))


def test_ecc_coordinate_direction() -> None:
    shape = (240, 128)
    reference = _texture(shape)
    true_rgb_to_sensor = np.array([[1.0, 0.006, 4.0], [-0.004, 1.0, -3.0]], dtype=np.float32)
    sensor = cv2.warpAffine(
        reference,
        true_rgb_to_sensor,
        (shape[1], shape[0]),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=float("nan"),
    )
    cfg = RegistrationConfig(preview_width=shape[1], preview_max_height=shape[0], ecc_iterations=500, enable_strip_drift=False)
    model, aligned = _estimate_one(reference, sensor, rgb_shape=shape, sensor_shape=shape, sensor_name="TEST", config=cfg)
    assert aligned.shape == shape
    assert model.edge_correlation > 0.80
    np.testing.assert_allclose(model.rgb_to_sensor_matrix[:2], true_rgb_to_sensor, atol=0.8)
    tre_median, tre_p95 = _affine_tre_stats(model.rgb_to_sensor_matrix, true_rgb_to_sensor, shape, margin=16)
    assert tre_median < 0.5, {"synthetic_tre_median_px": tre_median, "synthetic_tre_p95_px": tre_p95}
    assert tre_p95 < 0.9, {"synthetic_tre_median_px": tre_median, "synthetic_tre_p95_px": tre_p95}


def test_preview_raw_conversion_maps_pixel_center_endpoints() -> None:
    preview_shape = (121, 201)
    reference = _texture(preview_shape)
    rgb_shape = (481, 1001)
    sensor_shape = (241, 501)
    cfg = RegistrationConfig(
        preview_width=preview_shape[1],
        preview_max_height=preview_shape[0],
        motion="translation",
        enable_strip_drift=False,
    )
    model, _ = _estimate_one(
        reference,
        reference,
        rgb_shape=rgb_shape,
        sensor_shape=sensor_shape,
        sensor_name="TEST",
        config=cfg,
    )
    sy, sx = model.map_rgb_to_sensor(
        np.asarray([0.0, rgb_shape[0] - 1.0], dtype=np.float32),
        np.asarray([0.0, rgb_shape[1] - 1.0], dtype=np.float32),
    )
    np.testing.assert_allclose(sy, [0.0, sensor_shape[0] - 1.0], atol=0.08)
    np.testing.assert_allclose(sx, [0.0, sensor_shape[1] - 1.0], atol=0.08)


def test_roi_cross_modal_affine_recovers_scale_and_translation() -> None:
    shape = (280, 176)
    reference = _texture(shape)
    true_reference_to_moving = np.array(
        [[1.055, 0.008, -7.0], [-0.012, 0.985, 4.5]],
        dtype=np.float32,
    )
    moving = cv2.warpAffine(
        reference,
        true_reference_to_moving,
        (shape[1], shape[0]),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=float("nan"),
    )
    valid = np.isfinite(moving)
    moving[valid] = np.sqrt(
        np.clip((moving[valid] - np.nanmin(moving)) / (np.nanmax(moving) - np.nanmin(moving)), 0, 1)
    )
    cfg = RegistrationConfig(roi_ecc_iterations=900, enable_roi_row_refinement=False)
    matrix, _, details = _estimate_roi_affine(reference, moving, cfg)
    assert details["accepted"]
    assert details["score_after"] > details["score_before"] + 0.08
    np.testing.assert_allclose(matrix[:2], true_reference_to_moving, atol=1.1)
    tre_median, tre_p95 = _affine_tre_stats(matrix, true_reference_to_moving, shape)
    assert tre_median < 0.5, {"synthetic_tre_median_px": tre_median, "synthetic_tre_p95_px": tre_p95}
    assert tre_p95 < 0.9, {"synthetic_tre_median_px": tre_median, "synthetic_tre_p95_px": tre_p95}


def test_affine_and_residual_warps_preserve_invalid_support() -> None:
    shape = (48, 64)
    yy, xx = np.indices(shape, dtype=np.float32)
    image = xx + 0.1 * yy
    shift_x = np.full(shape, 4.0, dtype=np.float32)
    shift_y = np.zeros(shape, dtype=np.float32)
    residual = _warp_aligned_residual(image, shift_y, shift_x, 1.0)
    affine = _warp_affine(
        image,
        np.asarray([[1.0, 0.0, 4.0], [0.0, 1.0, 0.0]], dtype=np.float32),
        shape,
    )
    assert np.isfinite(residual[:, :-4]).all()
    assert np.isnan(residual[:, -4:]).all()
    assert np.isfinite(affine[:, :-4]).all()
    assert np.isnan(affine[:, -4:]).all()
    assert np.isnan(_modality_feature(residual)[:, -4:]).all()
    masked_copy = image.copy()
    masked_copy[:, -4:] = np.nan
    assert _corr(image, masked_copy) == pytest.approx(1.0, abs=1e-6)


def test_tiepoint_response_boundary_peak_is_rejected() -> None:
    rng = np.random.default_rng(7)
    reference = cv2.GaussianBlur(rng.normal(size=(96, 96)).astype(np.float32), (0, 0), 0.8)
    interior_shift = cv2.warpAffine(
        reference,
        np.asarray([[1.0, 0.0, 2.0], [0.0, 1.0, 0.0]], dtype=np.float32),
        (96, 96),
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=float("nan"),
    )
    boundary_shift = cv2.warpAffine(
        reference,
        np.asarray([[1.0, 0.0, 3.0], [0.0, 1.0, 0.0]], dtype=np.float32),
        (96, 96),
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=float("nan"),
    )
    assert _match_tiepoint_one(reference, interior_shift, 48, 48, 6, 3) is not None
    assert _match_tiepoint_one(reference, boundary_shift, 48, 48, 6, 3) is None


def test_guarded_tiepoints_improve_smooth_local_warp() -> None:
    shape = (280, 176)
    reference = _texture(shape)
    yy, xx = np.indices(shape, dtype=np.float32)
    displacement_y = lambda y, x: 1.2 * np.sin(y / 48.0) + 0.45 * np.sin(x / 21.0)
    displacement_x = lambda y, x: 1.8 * np.sin(x / 38.0) - 0.55 * np.cos(y / 33.0)
    true_dy = displacement_y(yy, xx)
    true_dx = displacement_x(yy, xx)
    inverse_y = yy.copy()
    inverse_x = xx.copy()
    for _ in range(12):
        next_y = yy - displacement_y(inverse_y, inverse_x)
        next_x = xx - displacement_x(inverse_y, inverse_x)
        inverse_y, inverse_x = next_y, next_x
    moving = cv2.remap(
        reference,
        inverse_x,
        inverse_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=float("nan"),
    )
    valid = np.isfinite(moving)
    moving[valid] = np.sqrt(
        np.clip((moving[valid] - np.nanmin(moving)) / (np.nanmax(moving) - np.nanmin(moving)), 0, 1)
    )
    cfg = RegistrationConfig(
        roi_tiepoint_grid_rows=13,
        roi_tiepoint_grid_cols=8,
        roi_tiepoint_search_radius=5,
    )
    shift_y, shift_x, details = _estimate_tiepoint_field(reference, moving, cfg)
    assert details["accepted"]
    assert details["tie_point_count"] >= cfg.roi_tiepoint_min_points
    before = _corr(_modality_feature(reference), _modality_feature(moving))
    aligned = _warp_aligned_residual(moving, shift_y, shift_x, 1.0)
    after = _corr(_modality_feature(reference), _modality_feature(aligned))
    assert after > before + 0.04
    assert details["field_magnitude_p95"] < cfg.roi_tiepoint_max_shift
    interior = np.s_[20:-20, 20:-20]
    epe = np.hypot(shift_y[interior] - true_dy[interior], shift_x[interior] - true_dx[interior])
    epe_median = float(np.median(epe))
    epe_p95 = float(np.percentile(epe, 95.0))
    assert epe_median < 0.5, {"synthetic_epe_median_px": epe_median, "synthetic_epe_p95_px": epe_p95}
    assert epe_p95 < 0.9, {"synthetic_epe_median_px": epe_median, "synthetic_epe_p95_px": epe_p95}
