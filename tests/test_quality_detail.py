import cv2
import numpy as np

from geocorefusion.degradation import PsfModel, degrade_coefficients, degrade_spatial_map
from geocorefusion.lowrank import SubspaceModel
from geocorefusion.output import reconstruct_modulated
from geocorefusion.quality import (
    _degrade_final_modulated_cube,
    _fit_low_resolution_log_relation,
    _halo_overshoot_proxy,
    _log_high_frequency,
    _masked_detail_statistics,
    _selected_band_detail_metrics,
    build_quality_report,
)
from geocorefusion.spectral import SpectralHarmonizationResult


def _subspace() -> SubspaceModel:
    return SubspaceModel(
        mean_spectrum=np.array([0.08, 0.14, 0.20], dtype=np.float32),
        basis=np.array(
            [
                [0.75, 0.45, 0.25],
                [-0.30, 0.20, 0.55],
            ],
            dtype=np.float32,
        ),
        explained_variance_ratio=np.array([0.72, 0.22], dtype=np.float32),
        # Deliberately narrow fitted quantiles: the V7 physical path must not
        # use these as output truth limits.
        clip_min=np.array([0.05, 0.10, 0.16], dtype=np.float32),
        clip_max=np.array([0.18, 0.24, 0.30], dtype=np.float32),
    )


def _synthetic_fields(shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    y, x = np.indices(shape, dtype=np.float32)
    texture = np.sin(x / 2.1) + 0.65 * np.cos(y / 3.3)
    coefficients = np.stack(
        [0.12 + 0.13 * (x > shape[1] // 2) + 0.035 * texture, -0.04 + 0.025 * texture],
        axis=2,
    ).astype(np.float32)
    gain = (1.0 + 0.22 * np.sin(x / 1.7)).astype(np.float32)
    additive = (0.08 * np.cos(y / 2.4) - 0.055 * np.sin(x / 1.9)).astype(np.float32)
    scales = np.array([0.55, -0.35, 0.75], dtype=np.float32)
    return coefficients, gain, additive, scales


def test_final_observation_degrades_the_actual_single_clip_product() -> None:
    high_shape = (36, 30)
    psf = PsfModel(1.7, 2.2, 0.8, (9, 7), high_shape)
    coefficients, gain, additive, scales = _synthetic_fields(high_shape)
    subspace = _subspace()

    expected_hr = reconstruct_modulated(
        coefficients,
        subspace,
        detail_gain=gain,
        additive_detail=additive,
        additive_spectral_scale=scales,
        physical_clip_limits=(0.0, 0.32),
    )
    expected_low = np.stack(
        [degrade_spatial_map(expected_hr[:, :, band], psf) for band in range(3)],
        axis=2,
    )
    actual_low, retained = _degrade_final_modulated_cube(
        coefficients,
        subspace,
        psf,
        gain,
        additive,
        scales,
        physical_clip_limits=(0.0, 0.32),
        chunk_bands=2,
        retained_band_indices=(0, 2),
    )

    np.testing.assert_allclose(actual_low, expected_low, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(retained[0], expected_hr[:, :, 0], rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(retained[2], expected_hr[:, :, 2], rtol=1e-6, atol=1e-6)

    # The former coefficient-domain D(gain) proxy does not commute with the
    # final nonlinear clip and therefore must differ for this construction.
    low_gain = degrade_spatial_map(gain, psf)
    low_coeff = degrade_coefficients(coefficients * gain[:, :, None], psf)
    proxy = (
        np.einsum("...k,kb->...b", low_coeff, subspace.basis, optimize=True)
        + low_gain[:, :, None] * subspace.mean_spectrum[None, None, :]
        + degrade_spatial_map(additive, psf)[:, :, None] * scales[None, None, :]
    )
    proxy = np.clip(proxy, 0.0, 0.32)
    assert float(np.max(np.abs(actual_low - proxy))) > 1e-4


def test_detail_statistics_separate_correlation_amplitude_and_residual() -> None:
    rng = np.random.default_rng(17)
    reference = rng.normal(size=(96, 80)).astype(np.float32)
    mask = np.ones(reference.shape, dtype=bool)

    half = _masked_detail_statistics(reference, 0.5 * reference, mask)
    assert half["rho"] > 0.9999
    assert abs(float(half["beta"]) - 0.5) < 1e-5
    assert abs(float(half["energy_ratio_A"]) - 0.5) < 1e-5
    assert float(half["orthogonal_residual_ratio_R_perp"]) < 1e-5

    noise = rng.normal(size=reference.shape).astype(np.float32)
    noisy = 0.8 * reference + 0.6 * noise
    separated = _masked_detail_statistics(reference, noisy, mask)
    assert 0.76 < float(separated["beta"]) < 0.84
    assert 0.95 < float(separated["energy_ratio_A"]) < 1.05
    assert 0.55 < float(separated["orthogonal_residual_ratio_R_perp"]) < 0.65
    assert float(separated["aligned_variance_fraction_rho2"]) < 0.75

    flat_reference = np.zeros(reference.shape, dtype=np.float32)
    flat_artifact = _masked_detail_statistics(flat_reference, 0.1 * noise, mask)
    assert np.isfinite(float(flat_artifact["energy_ratio_A"]))
    assert float(flat_artifact["energy_ratio_A"]) > 100.0


def test_native_log_detail_does_not_erase_tenfold_amplitude_loss() -> None:
    y, x = np.indices((128, 112), dtype=np.float32)
    texture = np.sin(x / 2.7) * np.cos(y / 3.9)
    reference = (0.50 + 0.10 * texture).astype(np.float32)
    candidate = (0.50 + 0.01 * texture).astype(np.float32)
    mask = np.ones(reference.shape, dtype=bool)

    ref_detail = _log_high_frequency(reference, sigma_px=2.4)
    candidate_detail = _log_high_frequency(candidate, sigma_px=2.4)
    metrics = _masked_detail_statistics(ref_detail, candidate_detail, mask)

    assert float(metrics["rho"]) > 0.99
    assert 0.07 < float(metrics["beta"]) < 0.14
    assert 0.07 < float(metrics["energy_ratio_A"]) < 0.14

    # Demonstrate why candidate-specific percentile stretching is forbidden.
    stretched_ref = _log_high_frequency(reference, sigma_px=2.4, normalize=True)
    stretched_candidate = _log_high_frequency(candidate, sigma_px=2.4, normalize=True)
    stretched = _masked_detail_statistics(stretched_ref, stretched_candidate, mask)
    assert 0.95 < float(stretched["beta"]) < 1.05


def test_lr_log_relation_is_candidate_independent_and_flags_weak_support() -> None:
    y, x = np.indices((40, 32), dtype=np.float32)
    rgb_low = np.clip(0.08 + 0.70 * (x / 31.0) + 0.04 * np.sin(y / 3.0), 0.01, 1.0)
    alpha_true = -0.65
    intercept = -1.2
    band_low = np.exp(intercept + alpha_true * np.log(rgb_low + 0.012)) - 0.012
    band_low = np.clip(band_low, 0.0, None).astype(np.float32)
    relation = _fit_low_resolution_log_relation(rgb_low, band_low)

    assert relation["identifiable"] is True
    assert abs(float(relation["alpha_log_band_per_log_rgb"]) - alpha_true) < 0.02
    assert float(relation["rho_low_resolution"]) < -0.99

    weak = _fit_low_resolution_log_relation(
        rgb_low,
        np.full_like(rgb_low, 0.2),
    )
    assert weak["identifiable"] is False


def test_halo_proxy_detects_edge_overshoot() -> None:
    reference = np.zeros((128, 128), dtype=np.float32)
    reference[:, 64:] = 1.0
    clean = cv2.GaussianBlur(reference, (0, 0), 0.45)
    ringing = clean.copy()
    ringing[:, 62:64] -= 0.18
    ringing[:, 64:66] += 0.20
    mask = np.ones(reference.shape, dtype=bool)

    clean_metrics = _halo_overshoot_proxy(reference, clean, mask)
    ringing_metrics = _halo_overshoot_proxy(reference, ringing, mask)
    assert clean_metrics["status"] == "diagnostic_only"
    assert ringing_metrics["status"] == "diagnostic_only"
    assert float(ringing_metrics["overshoot_plus_undershoot_p95_edge_step"]) > 0.15
    assert float(ringing_metrics["overshoot_plus_undershoot_p95_edge_step"]) > (
        float(clean_metrics["overshoot_plus_undershoot_p95_edge_step"]) + 0.10
    )


def test_selected_band_metrics_keep_legacy_fields_and_add_v7_closure() -> None:
    shape = (128, 96)
    y, x = np.indices(shape, dtype=np.float32)
    texture = 0.10 * np.sin(x / 2.5) + 0.08 * np.cos(y / 3.7)
    luminance = np.clip(0.08 + 0.50 * (x > 44) + texture, 0.005, 0.95)
    rgb = np.stack([luminance, 0.92 * luminance, 0.84 * luminance], axis=2).astype(np.float32)
    coefficients, gain, additive, scales = _synthetic_fields(shape)
    subspace = _subspace()
    wavelengths = np.array([900.0, 1650.0, 2200.0], dtype=np.float32)
    final_cube = reconstruct_modulated(
        coefficients,
        subspace,
        detail_gain=gain,
        additive_detail=additive,
        additive_spectral_scale=scales,
    )
    metrics = _selected_band_detail_metrics(
        coefficients,
        subspace,
        rgb,
        wavelengths,
        gain,
        additive,
        scales,
        final_hr_bands={index: final_cube[:, :, index] for index in range(3)},
    )

    assert metrics["status"] == "diagnostic_only"
    assert set(metrics["log_high_frequency_correlation"]["900.0nm"]) == {
        "all_valid",
        "darkest_20pct",
    }
    band = metrics["bands"]["2200.0nm"]
    selected = band["multiscale_log_high_frequency"]["sigma_2.4px"]["reliable_rgb_detail"]
    assert {"rho", "beta", "energy_ratio_A", "orthogonal_residual_ratio_R_perp"} <= set(selected)
    assert "edge_f1_1px" in band["gradient_and_edge"]["reliable_rgb_detail"]
    assert band["conservative_screening"]["claim_status"] == (
        "same_data_rgb_guide_diagnostic_not_independent_swir_truth"
    )


def test_quality_report_preserves_old_keys_and_reports_exact_forward_method() -> None:
    high_shape = (48, 36)
    low_shape = (12, 9)
    psf = PsfModel(1.5, 2.0, 0.8, low_shape, high_shape)
    coefficients, gain, additive, scales = _synthetic_fields(high_shape)
    subspace = _subspace()
    wavelengths = np.array([900.0, 1650.0, 2200.0], dtype=np.float32)
    low_cube, _ = _degrade_final_modulated_cube(
        coefficients,
        subspace,
        psf,
        gain,
        additive,
        scales,
        chunk_bands=2,
    )
    y, x = np.indices(high_shape, dtype=np.float32)
    texture = 0.08 * np.sin(x / 2.0) + 0.06 * np.cos(y / 2.7)
    rgb_luma = np.clip(0.08 + 0.55 * (x > 17) + texture, 0.004, 0.95)
    rgb = np.stack([rgb_luma, 0.9 * rgb_luma, 0.82 * rgb_luma], axis=2).astype(np.float32)
    spectral = SpectralHarmonizationResult(
        cube=low_cube.copy(),
        wavelengths_nm=wavelengths,
        calibrated_swir=low_cube.copy(),
        swir_gain=np.ones(3, dtype=np.float32),
        swir_offset=np.zeros(3, dtype=np.float32),
        nir_reliability=np.ones(3, dtype=np.float32),
        swir_reliability=np.ones(3, dtype=np.float32),
        uncertainty_by_band=np.zeros(3, dtype=np.float32),
        band_metadata=[],
        model={"overlap_rmse_before": 0.01, "overlap_rmse_after": 0.005},
    )
    report = build_quality_report(
        coefficients,
        degrade_coefficients(coefficients, psf),
        subspace,
        psf,
        rgb,
        spectral,
        low_cube,
        low_cube,
        wavelengths,
        wavelengths,
        np.zeros(high_shape, dtype=np.float32),
        gain,
        additive,
        scales,
        observation_chunk_bands=2,
    )

    assert report["continuous_cube_observation"]["rmse"] < 1e-7
    assert report["continuous_cube_observation"]["proxy_D_gain_used"] is False
    assert report["final_hr_product_observation"]["method"].startswith("final_hr_unbounded")
    assert report["summary"]["status_scope"] == "low_resolution_observation_consistency_only"
    assert report["summary"]["independent_hr_hsi_truth_status"] == "not_evaluated"
    assert "rgb_material_boundary_correlation" in report["spatial"]
    assert "log_high_frequency_correlation" in report["spatial"]["band_detail_by_brightness"]
    assert "bands" in report["spatial"]["band_detail_by_brightness"]
