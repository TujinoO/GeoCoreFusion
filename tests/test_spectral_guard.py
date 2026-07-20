from __future__ import annotations

import numpy as np
import pytest

from geocorefusion.spectral_guard import project_spectral_cone


def _axis_angle_deg(candidate: np.ndarray, reference: np.ndarray) -> np.ndarray:
    candidate64 = np.asarray(candidate, dtype=np.float64)
    reference64 = np.asarray(reference, dtype=np.float64)
    denominator = np.sum(reference64 * reference64, axis=-1)
    axial_scale = np.sum(candidate64 * reference64, axis=-1) / denominator
    parallel = axial_scale[..., None] * reference64
    orthogonal = candidate64 - parallel
    return np.degrees(
        np.arctan2(
            np.linalg.norm(orthogonal, axis=-1),
            axial_scale * np.sqrt(denominator),
        )
    )


def _parallel_delta_scale(candidate: np.ndarray, reference: np.ndarray) -> np.ndarray:
    candidate64 = np.asarray(candidate, dtype=np.float64)
    reference64 = np.asarray(reference, dtype=np.float64)
    denominator = np.sum(reference64 * reference64, axis=-1)
    return np.sum((candidate64 - reference64) * reference64, axis=-1) / denominator


def test_zero_reference_fails_closed_without_inventing_axis() -> None:
    reference = np.zeros((2, 3), dtype=np.float32)
    candidate = np.asarray([[0.2, 0.4, 0.8], [1.0, 0.0, 0.5]], dtype=np.float32)

    projected, diagnostics = project_spectral_cone(
        candidate,
        reference,
        0.5,
        return_diagnostics=True,
    )

    np.testing.assert_array_equal(projected, reference)
    assert diagnostics.spectrum_count == 2
    assert diagnostics.valid_reference_count == 0
    assert diagnostics.zero_reference_count == 2
    assert diagnostics.cone_clipped_count == 0
    assert diagnostics.cone_clip_fraction == 0.0


def test_zero_candidate_preserves_its_parallel_delta() -> None:
    reference = np.asarray([0.2, 0.4, 0.8], dtype=np.float64)
    candidate = np.zeros_like(reference)

    projected = project_spectral_cone(candidate, reference, 0.5)

    np.testing.assert_array_equal(projected, candidate)
    np.testing.assert_allclose(
        _parallel_delta_scale(projected, reference),
        _parallel_delta_scale(candidate, reference),
        atol=0.0,
        rtol=0.0,
    )


def test_spectrum_already_inside_cone_is_bitwise_unchanged() -> None:
    reference = np.asarray([1.0, 1.0, 1.0], dtype=np.float32)
    orthogonal = np.asarray([0.002, -0.001, -0.001], dtype=np.float32)
    candidate = 1.2 * reference + orthogonal
    assert float(_axis_angle_deg(candidate, reference)) < 0.5

    projected, diagnostics = project_spectral_cone(
        candidate,
        reference,
        0.5,
        return_diagnostics=True,
    )

    np.testing.assert_array_equal(projected, candidate)
    assert diagnostics.cone_clipped_count == 0
    assert diagnostics.cone_clip_fraction == 0.0


def test_projection_truncates_only_orthogonal_component() -> None:
    reference = np.asarray([1.0, 0.0, 0.0], dtype=np.float64)
    candidate = np.asarray([2.0, 1.0, 0.0], dtype=np.float64)
    before_parallel = _parallel_delta_scale(candidate, reference)

    projected, diagnostics = project_spectral_cone(
        candidate,
        reference,
        10.0,
        return_diagnostics=True,
    )

    np.testing.assert_allclose(
        _parallel_delta_scale(projected, reference),
        before_parallel,
        atol=1e-12,
        rtol=1e-12,
    )
    assert float(_axis_angle_deg(projected, reference)) <= 10.0 + 1e-10
    assert projected[1] < candidate[1]
    assert diagnostics.cone_clipped_count == 1
    assert diagnostics.cone_clip_fraction == 1.0


def test_float32_batched_image_respects_tolerance_and_parallel_component() -> None:
    rng = np.random.default_rng(7)
    reference = rng.uniform(0.05, 0.75, size=(2, 5, 4, 11)).astype(np.float32)
    candidate = (reference + rng.normal(0.0, 0.10, reference.shape)).astype(
        np.float32
    )
    # Keep the physically relevant positive projection while retaining large
    # orthogonal perturbations that require clipping.
    candidate = np.maximum(candidate, 0.001).astype(np.float32)
    before_parallel = _parallel_delta_scale(candidate, reference)

    projected = project_spectral_cone(candidate, reference, 0.5)

    assert projected.shape == candidate.shape
    assert projected.dtype == np.float32
    assert float(np.max(_axis_angle_deg(projected, reference))) <= 0.50001
    np.testing.assert_allclose(
        _parallel_delta_scale(projected, reference),
        before_parallel,
        atol=2e-7,
        rtol=2e-6,
    )


def test_zero_degree_cone_removes_orthogonal_component() -> None:
    reference = np.asarray([[1.0, 1.0, 1.0]], dtype=np.float64)
    candidate = np.asarray([[2.0, 1.0, 1.0]], dtype=np.float64)

    projected = project_spectral_cone(candidate, reference, 0.0)

    assert float(_axis_angle_deg(projected, reference).item()) <= 1e-12
    np.testing.assert_allclose(
        _parallel_delta_scale(projected, reference),
        _parallel_delta_scale(candidate, reference),
        atol=1e-12,
        rtol=1e-12,
    )


@pytest.mark.parametrize("angle", [-0.1, 90.0, float("nan"), float("inf")])
def test_invalid_angle_raises(angle: float) -> None:
    spectrum = np.ones(3, dtype=np.float32)
    with pytest.raises(ValueError, match="max_angle_deg"):
        project_spectral_cone(spectrum, spectrum, angle)


def test_shape_and_input_errors_are_explicit() -> None:
    with pytest.raises(ValueError, match="identical shapes"):
        project_spectral_cone(np.ones((2, 3)), np.ones((3,)), 0.5)
    with pytest.raises(ValueError, match="at least one spectrum"):
        project_spectral_cone(np.empty((0, 3)), np.empty((0, 3)), 0.5)
    with pytest.raises(ValueError, match="finite values"):
        project_spectral_cone(
            np.asarray([1.0, np.nan]), np.asarray([1.0, 1.0]), 0.5
        )
    with pytest.raises(TypeError, match="real-valued"):
        project_spectral_cone(
            np.asarray([1.0 + 1.0j, 0.0]),
            np.asarray([1.0 + 0.0j, 0.0]),
            0.5,
        )
    with pytest.raises(ValueError, match="negative projection"):
        project_spectral_cone(
            np.asarray([-1.0, 0.0]), np.asarray([1.0, 0.0]), 0.5
        )
