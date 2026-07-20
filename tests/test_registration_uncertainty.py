import numpy as np
import pytest

from geocorefusion.registration_uncertainty import (
    RegistrationUncertaintyConfig,
    estimate_local_registration_covariance,
    registration_sigma_points_5,
)


def _point(
    y: float,
    x: float,
    shift_y: float,
    shift_x: float,
    *,
    score: float = 0.92,
    margin: float = 0.08,
    backward_error: float = 0.05,
) -> dict[str, float]:
    return {
        "ref_y": y,
        "ref_x": x,
        "shift_y": shift_y,
        "shift_x": shift_x,
        "score": score,
        "margin": margin,
        "backward_error": backward_error,
    }


def test_covariance_uses_residuals_relative_to_fixed_final_warp() -> None:
    shape = (16, 18)
    final_y = np.full(shape, 1.0, dtype=np.float32)
    final_x = np.full(shape, -2.0, dtype=np.float32)
    matched = [
        _point(4.0, 4.0, 1.0, -2.0),
        _point(4.0, 13.0, 1.0, -2.0),
        _point(11.0, 4.0, 1.0, -2.0),
        _point(11.0, 13.0, 1.0, -2.0),
    ]
    mismatched = [dict(point, shift_x=point["shift_x"] + 0.8) for point in matched]
    cfg = RegistrationUncertaintyConfig(
        covariance_floor_px2=0.01,
        distance_inflation_per_bandwidth_px=0.0,
    )
    query_y = np.asarray([7.5], dtype=np.float32)
    query_x = np.asarray([8.5], dtype=np.float32)
    low = estimate_local_registration_covariance(
        matched,
        final_y,
        final_x,
        query_y=query_y,
        query_x=query_x,
        config=cfg,
    )
    high = estimate_local_registration_covariance(
        mismatched,
        final_y,
        final_x,
        query_y=query_y,
        query_x=query_x,
        config=cfg,
    )
    assert high.covariance_yx_px2[0, 1, 1] > low.covariance_yx_px2[0, 1, 1] + 0.60
    assert high.covariance_yx_px2[0, 0, 0] == pytest.approx(
        low.covariance_yx_px2[0, 0, 0], rel=1e-5
    )
    assert high.metadata["mean_registration_solution_changed"] is False


def test_covariance_is_psd_obeys_floor_and_is_audited_as_proxy() -> None:
    shape = (12, 14)
    zeros = np.zeros(shape, dtype=np.float32)
    points = [
        _point(2.0, 2.0, -0.5, 1.0),
        _point(2.0, 11.0, 0.5, -1.0),
        _point(9.0, 2.0, 0.5, 1.0),
        _point(9.0, 11.0, -0.5, -1.0),
    ]
    floor = 0.16
    estimate = estimate_local_registration_covariance(
        points,
        zeros,
        zeros,
        config=RegistrationUncertaintyConfig(covariance_floor_px2=floor, query_chunk_size=17),
    )
    assert estimate.covariance_yx_px2.shape == shape + (2, 2)
    np.testing.assert_allclose(
        estimate.covariance_yx_px2,
        np.swapaxes(estimate.covariance_yx_px2, -1, -2),
        atol=1e-6,
    )
    eigenvalues = np.linalg.eigvalsh(estimate.covariance_yx_px2)
    assert float(np.min(eigenvalues)) >= floor - 1e-6
    assert estimate.metadata["status"] == "uncertainty_proxy"
    assert estimate.metadata["uncertainty_proxy"] is True
    assert estimate.metadata["calibration_required"] is True
    assert estimate.metadata["component_order"] == ["shift_y_px", "shift_x_px"]


def test_poor_matching_quality_and_extrapolation_increase_proxy_variance() -> None:
    shape = (8, 64)
    zeros = np.zeros(shape, dtype=np.float32)
    points = [
        _point(4.0, 4.0, 0.0, 0.0, score=0.98, margin=0.20, backward_error=0.01),
        _point(4.0, 28.0, 0.0, 0.0, score=0.35, margin=0.013, backward_error=1.20),
    ]
    estimate = estimate_local_registration_covariance(
        points,
        zeros,
        zeros,
        query_y=np.asarray([4.0, 4.0, 4.0], dtype=np.float32),
        query_x=np.asarray([4.0, 28.0, 60.0], dtype=np.float32),
        config=RegistrationUncertaintyConfig(
            covariance_floor_px2=0.01,
            spatial_bandwidth_px=3.0,
            neighbours=1,
            distance_inflation_per_bandwidth_px=0.20,
            max_distance_inflation_px=4.0,
        ),
    )
    trace = np.trace(estimate.covariance_yx_px2, axis1=-2, axis2=-1)
    assert trace[1] > trace[0]
    assert trace[2] > trace[1]
    np.testing.assert_allclose(estimate.nearest_tiepoint_distance_px, [0.0, 0.0, 32.0], atol=1e-6)


def test_five_sigma_points_reconstruct_batched_first_two_moments() -> None:
    mean = np.asarray([[1.2, -0.7], [-2.0, 0.5]], dtype=np.float64)
    covariance = np.asarray(
        [
            [[0.50, 0.18], [0.18, 0.25]],
            [[0.09, -0.02], [-0.02, 0.16]],
        ],
        dtype=np.float64,
    )
    points, weights = registration_sigma_points_5(mean, covariance, center_weight=0.30)
    reconstructed_mean = np.einsum("p,bpi->bi", weights, points)
    centered = points - reconstructed_mean[:, None, :]
    reconstructed_covariance = np.einsum("p,bpi,bpj->bij", weights, centered, centered)
    np.testing.assert_allclose(reconstructed_mean, mean, atol=2e-7)
    np.testing.assert_allclose(reconstructed_covariance, covariance, atol=2e-7)
    assert points.shape == (2, 5, 2)
    assert np.all(weights > 0.0)
    assert float(np.sum(weights)) == pytest.approx(1.0)


def test_uncertainty_inputs_are_guarded() -> None:
    zeros = np.zeros((4, 4), dtype=np.float32)
    with pytest.raises(ValueError, match="at least one accepted"):
        estimate_local_registration_covariance([], zeros, zeros)
    with pytest.raises(ValueError, match="positive semidefinite"):
        registration_sigma_points_5(
            np.zeros(2, dtype=np.float32),
            np.asarray([[1.0, 2.0], [2.0, 1.0]], dtype=np.float32),
        )
