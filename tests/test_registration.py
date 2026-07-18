import cv2
import numpy as np

from geocorefusion.config import RegistrationConfig
from geocorefusion.registration import (
    _corr,
    _estimate_one,
    _estimate_roi_affine,
    _estimate_tiepoint_field,
    _modality_feature,
    _warp_aligned_residual,
)


def _texture(shape: tuple[int, int]) -> np.ndarray:
    y, x = np.indices(shape, dtype=np.float32)
    image = 0.3 * np.sin(x / 5.0) + 0.2 * np.cos(y / 11.0)
    image += ((x - 42) ** 2 + (y - 118) ** 2 < 18**2) * 1.2
    image += ((x > 75) & (x < 88) & (y > 20) & (y < 205)) * 0.8
    return cv2.GaussianBlur(image.astype(np.float32), (0, 0), 0.7)


def test_ecc_coordinate_direction() -> None:
    shape = (240, 128)
    reference = _texture(shape)
    true_rgb_to_sensor = np.array([[1.0, 0.006, 4.0], [-0.004, 1.0, -3.0]], dtype=np.float32)
    sensor = cv2.warpAffine(reference, true_rgb_to_sensor, (shape[1], shape[0]), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT101)
    cfg = RegistrationConfig(preview_width=shape[1], preview_max_height=shape[0], ecc_iterations=500, enable_strip_drift=False)
    model, aligned = _estimate_one(reference, sensor, rgb_shape=shape, sensor_shape=shape, sensor_name="TEST", config=cfg)
    assert aligned.shape == shape
    assert model.edge_correlation > 0.80
    np.testing.assert_allclose(model.rgb_to_sensor_matrix[:2], true_rgb_to_sensor, atol=0.8)


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
        borderMode=cv2.BORDER_REFLECT101,
    )
    moving = np.sqrt(np.clip((moving - moving.min()) / (moving.max() - moving.min()), 0, 1)).astype(np.float32)
    cfg = RegistrationConfig(roi_ecc_iterations=900, enable_roi_row_refinement=False)
    matrix, _, details = _estimate_roi_affine(reference, moving, cfg)
    assert details["accepted"]
    assert details["score_after"] > details["score_before"] + 0.08
    np.testing.assert_allclose(matrix[:2], true_reference_to_moving, atol=1.1)


def test_guarded_tiepoints_improve_smooth_local_warp() -> None:
    shape = (280, 176)
    reference = _texture(shape)
    yy, xx = np.indices(shape, dtype=np.float32)
    true_dy = 1.2 * np.sin(yy / 48.0) + 0.45 * np.sin(xx / 21.0)
    true_dx = 1.8 * np.sin(xx / 38.0) - 0.55 * np.cos(yy / 33.0)
    moving = cv2.remap(
        reference,
        xx - true_dx,
        yy - true_dy,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT101,
    )
    moving = np.sqrt(np.clip((moving - moving.min()) / (moving.max() - moving.min()), 0, 1)).astype(np.float32)
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
