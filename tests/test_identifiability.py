import numpy as np

from geocorefusion.degradation import PsfModel
from geocorefusion.identifiability import (
    IdentifiabilityConfig,
    _blocked_cv,
    _block_map,
    _eroded_block_mask,
    _expanded_block_mask,
    _fit_linear,
    _prediction_metrics,
    assess_cube_identifiability,
    fit_blocked_bandpass_relation,
)


def _config(**overrides: object) -> IdentifiabilityConfig:
    values: dict[str, object] = {
        "bandpass_sigma_low_px": 0.45,
        "bandpass_sigma_high_px": 1.6,
        "block_rows": 4,
        "block_cols": 4,
        "exclusion_gap_px": 3,
        "minimum_valid_pixels": 320,
        "minimum_train_pixels": 300,
        "minimum_test_pixels": 36,
        "null_repeats": 19,
        "bootstrap_repeats": 120,
        "random_seed": 73,
    }
    values.update(overrides)
    return IdentifiabilityConfig(**values)


def _shared_log_fields(
    shape: tuple[int, int],
    *,
    alpha: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    raw = rng.normal(size=shape)
    # A deterministic FFT low-pass gives a spatially correlated, non-periodic
    # test texture without importing the implementation's OpenCV filters.
    frequencies_y = np.fft.fftfreq(shape[0])[:, None]
    frequencies_x = np.fft.fftfreq(shape[1])[None, :]
    transfer = np.exp(-35.0 * (frequencies_x**2 + frequencies_y**2))
    texture = np.fft.ifft2(np.fft.fft2(raw) * transfer).real
    texture = (texture - np.mean(texture)) / np.std(texture)
    log_rgb = -1.35 + 0.28 * texture
    rgb_channel = np.exp(log_rgb)
    rgb = np.stack([rgb_channel, rgb_channel, rgb_channel], axis=2).astype(np.float32)
    independent = rng.normal(scale=0.012, size=shape)
    band = np.exp(-1.8 + alpha * 0.28 * texture + independent).astype(np.float32)
    return rgb, band


def test_blocked_bandpass_relation_freezes_signed_band_amplitude() -> None:
    shape = (88, 76)
    alpha = -0.43
    rgb, band = _shared_log_fields(shape, alpha=alpha, seed=11)
    psf = PsfModel(0.0, 0.0, 1.0, shape, shape, method="synthetic_identity")

    report = fit_blocked_bandpass_relation(
        rgb,
        band,
        psf,
        band_index=7,
        wavelength_nm=2201.0,
        config=_config(),
    )

    assert report["candidate_independent"] is True
    assert report["unit_rgb_beta_is_not_a_target"] is True
    assert report["status"] == "identifiable"
    assert report["support"]["adjacent_train_test_pixels_excluded"] is True
    assert report["support"]["test_block_eroded_before_scoring"] is True
    assert report["bandpass"]["same_low_grid_bandpass_for_rgb_and_spectral_band"] is True
    assert "dark/flat" in report["bandpass"]["epsilon_requirement"]
    assert report["primary_model"] == "rgb_ridge"
    assert {"rgb_ridge", "rgb_pca_ridge_rank_1", "rgb_pca_ridge_rank_2"} <= set(
        report["models"]
    )
    primary = report["models"]["rgb_ridge"]
    assert primary["cv_predictive_r2"] > 0.85
    assert primary["cv_pearson_rho"] > 0.94
    assert report["spatial_shuffle_null"]["empirical_p_value"] <= 0.05
    assert report["spatial_shuffle_null"]["preserves_rgb_spatial_autocorrelation"] is True
    assert report["spatial_shuffle_null"]["minimum_circular_shift_px"] >= 9

    amplitude = report["expected_coherent_amplitude"]
    assert abs(amplitude["estimate_log_band_per_log_rgb"] - alpha) < 0.035
    assert abs(amplitude["frozen_value_log_band_per_log_rgb"] - alpha) < 0.035
    assert amplitude["confidence_interval"][1] < 0.0
    assert abs(amplitude["frozen_value_log_band_per_log_rgb"] - 1.0) > 1.0
    assert len(amplitude["training_fold_slopes"]) >= 12
    assert amplitude["training_fold_sign_stability"] > 0.9
    assert amplitude["cross_fitted_luminance_relation"]["cv_predictive_r2"] > 0.85


def test_independent_band_is_unidentifiable_and_has_no_frozen_target() -> None:
    shape = (88, 76)
    rgb, _ = _shared_log_fields(shape, alpha=0.4, seed=31)
    _, unrelated = _shared_log_fields(shape, alpha=0.55, seed=911)
    psf = PsfModel(0.0, 0.0, 1.0, shape, shape, method="synthetic_identity")

    report = fit_blocked_bandpass_relation(
        rgb,
        unrelated,
        psf,
        config=_config(random_seed=912),
    )

    assert report["status"] == "unidentifiable"
    assert (
        report["expected_coherent_amplitude"][
            "frozen_value_log_band_per_log_rgb"
        ]
        is None
    )
    assert report["recommended_rgb_detail_action"].startswith("disabled")
    assert not all(
        report["classification"]["weakly_identifiable_checks"].values()
    )


def test_cube_wrapper_reports_band_specific_status_and_weak_policy() -> None:
    shape = (84, 72)
    rgb, shared = _shared_log_fields(shape, alpha=0.31, seed=5)
    _, unrelated = _shared_log_fields(shape, alpha=-0.5, seed=105)
    cube = np.stack([shared, unrelated], axis=2)
    psf = PsfModel(0.0, 0.0, 1.0, shape, shape, method="synthetic_identity")
    config = _config(
        primary_model="rgb_pca_ridge_rank_1",
        identifiable_cv_r2_min=0.9999,
        weak_cv_r2_min=0.02,
        identifiable_abs_cv_rho_min=0.9999,
        weak_abs_cv_rho_min=0.2,
    )

    report = assess_cube_identifiability(
        rgb,
        cube,
        [901.0, 2201.0],
        psf,
        config=config,
    )

    assert report["band_count"] == 2
    assert report["bands"][0]["primary_model"] == "rgb_pca_ridge_rank_1"
    assert report["bands"][0]["status"] == "weakly_identifiable"
    assert report["bands"][0]["wavelength_nm"] == 901.0
    assert report["bands"][0]["expected_coherent_amplitude"][
        "frozen_value_log_band_per_log_rgb"
    ] is not None
    assert report["bands"][1]["status"] == "unidentifiable"
    assert report["status_counts"] == {
        "identifiable": 0,
        "weakly_identifiable": 1,
        "unidentifiable": 1,
    }


def test_insufficient_support_fails_closed() -> None:
    shape = (36, 30)
    rgb, band = _shared_log_fields(shape, alpha=0.4, seed=7)
    psf = PsfModel(0.0, 0.0, 1.0, shape, shape, method="synthetic_identity")
    mask = np.zeros(shape, dtype=bool)
    mask[10:15, 10:15] = True

    report = fit_blocked_bandpass_relation(
        rgb,
        band,
        psf,
        valid_mask=mask,
        config=_config(minimum_valid_pixels=64, null_repeats=0, bootstrap_repeats=0),
    )

    assert report["status"] == "unidentifiable"
    assert report["reason"] == "insufficient_common_bandpass_support"
    assert report["models"] == {}


def test_configuration_rejects_overlapping_passband_scales() -> None:
    try:
        IdentifiabilityConfig(
            bandpass_sigma_low_px=2.0,
            bandpass_sigma_high_px=1.0,
        )
    except ValueError as error:
        assert "must exceed" in str(error)
    else:
        raise AssertionError("overlapping/reversed band-pass scales must be rejected")


def test_guard_expands_held_out_block_before_training() -> None:
    fold_map = _block_map((40, 36), 4, 3)
    test = fold_map == 4
    excluded = _expanded_block_mask(fold_map, 4, gap=3)
    test_y, test_x = np.where(test)
    excluded_y, excluded_x = np.where(excluded)

    assert excluded_y.min() == test_y.min() - 3
    assert excluded_y.max() == test_y.max() + 3
    assert excluded_x.min() == test_x.min() - 3
    assert excluded_x.max() == test_x.max() + 3
    assert np.all(excluded[test])

    scored_core = _eroded_block_mask(fold_map, 4, gap=3)
    core_y, core_x = np.where(scored_core)
    assert core_y.min() == test_y.min() + 3
    assert core_y.max() == test_y.max() - 3
    assert core_x.min() == test_x.min() + 3
    assert core_x.max() == test_x.max() - 3


def test_eroded_test_core_rejects_boundary_only_filter_leakage() -> None:
    shape = (96, 96)
    gap = 4
    rng = np.random.default_rng(2201)
    feature = rng.normal(size=shape)
    features = feature[:, :, None].astype(np.float32)
    target = (0.02 * rng.normal(size=shape)).astype(np.float32)
    fold_map = _block_map(shape, 4, 4)

    # Adversarially place a perfect feature/target relation only inside the
    # boundary width that a pre-split DoG can contaminate. Block interiors are
    # independent. A leaky test mask scores the contaminated rim as success.
    contaminated_boundary = np.zeros(shape, dtype=bool)
    for fold in range(16):
        block = fold_map == fold
        core = _eroded_block_mask(fold_map, fold, gap)
        contaminated_boundary |= block & ~core
    target[contaminated_boundary] = feature[contaminated_boundary]
    valid = np.ones(shape, dtype=bool)
    config = _config(
        block_rows=4,
        block_cols=4,
        exclusion_gap_px=gap,
        ridge_alpha=0.001,
        minimum_valid_pixels=128,
        minimum_train_pixels=512,
        minimum_test_pixels=64,
        null_repeats=0,
        bootstrap_repeats=0,
    )

    guarded = _blocked_cv(
        features,
        target,
        valid,
        config,
        pca_rank=None,
    )

    # Reconstruct the audited, unsafe behavior: training has an outer guard,
    # but the entire un-eroded test block is scored.
    leaky_prediction = np.full(shape, np.nan, dtype=np.float64)
    leaky_baseline = np.full(shape, np.nan, dtype=np.float64)
    for fold in range(16):
        test = fold_map == fold
        train = ~_expanded_block_mask(fold_map, fold, gap)
        fit = _fit_linear(
            features[train],
            target[train],
            ridge_alpha=config.ridge_alpha,
            pca_rank=None,
        )
        leaky_prediction[test] = fit.predict(features[test])
        leaky_baseline[test] = float(np.mean(target[train]))
    leaky = _prediction_metrics(target, leaky_prediction, leaky_baseline)

    assert float(leaky["cv_predictive_r2"]) > 0.35
    assert float(guarded.metrics["cv_predictive_r2"]) < 0.05
    assert guarded.metrics["predictive_r2_baseline"] == (
        "per_fold_training_target_mean"
    )
    for fold_detail in guarded.metrics["folds"]:
        fold = int(fold_detail["fold"])
        train = ~_expanded_block_mask(fold_map, fold, gap)
        expected_baseline = float(np.mean(target[train]))
        assert abs(
            float(fold_detail["training_target_mean_baseline"])
            - expected_baseline
        ) < 1e-12
