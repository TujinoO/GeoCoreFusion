import numpy as np
import cv2

from geocorefusion.config import FusionConfig
from geocorefusion.degradation import PsfModel, degrade_coefficients, degrade_spatial_map
from geocorefusion.fusion import (
    _amplitude_preserving_confidence_gate,
    _dark_texture_confidence,
    _inject_simplex_abundance_detail,
    back_project_modulated_product,
    refine_coefficients,
)


def test_amplitude_preserving_confidence_gate_has_off_transition_and_plateau() -> None:
    reliability = np.array([-0.1, 0.08, 0.13, 0.18, 0.23, 0.28, 0.7], dtype=np.float32)
    gate = _amplitude_preserving_confidence_gate(reliability, 0.08, 0.28)
    assert np.all(np.diff(gate) >= 0.0)
    assert gate[0] == 0.0
    assert gate[1] == 0.0
    assert 0.0 < gate[3] < 1.0
    assert gate[-2] == 1.0
    assert gate[-1] == 1.0


def test_variational_refiner_preserves_observation_consistency() -> None:
    high_shape = (128, 96)
    low_shape = (24, 16)
    y, x = np.indices(high_shape)
    rgb = np.zeros(high_shape + (3,), dtype=np.float32)
    rgb[:, :, 0] = (x > 46).astype(np.float32) * 0.75 + 0.1
    rgb[:, :, 1] = (y > 65).astype(np.float32) * 0.55 + 0.2
    rgb[:, :, 2] = 0.2 + 0.1 * np.sin(x / 5.0)
    low_y, low_x = np.indices(low_shape)
    low = np.stack([
        (low_x > 7).astype(np.float32),
        (low_y > 11).astype(np.float32),
        0.2 * np.sin(low_x / 2.0) + 0.1 * np.cos(low_y / 3.0),
    ], axis=2)
    psf = PsfModel(2.0, 3.0, 0.8, low_shape, high_shape)
    cfg = FusionConfig(rank=3, variational_iterations=10, back_projection_interval=1, diffusion_strength=0.18)
    result = refine_coefficients(low, rgb, psf, cfg)
    assert result.coefficients.shape == high_shape + (3,)
    assert result.uncertainty_map.shape == high_shape
    assert result.detail_gain_map.shape == high_shape
    assert result.additive_detail_map.shape == high_shape
    assert np.percentile(np.abs(result.detail_gain_map - 1.0), 95) > 0.005
    assert np.sqrt(np.mean((degrade_spatial_map(result.detail_gain_map, psf) - 1.0) ** 2)) < 0.02
    assert np.sqrt(np.mean(degrade_spatial_map(result.additive_detail_map, psf) ** 2)) < 0.02
    assert np.sqrt(np.mean((degrade_coefficients(result.coefficients, psf) - low) ** 2)) < 0.08


def test_coefficient_detail_restores_rgb_aligned_high_frequency() -> None:
    high_shape = (128, 96)
    low_shape = (24, 16)
    y, x = np.indices(high_shape)
    texture = 0.08 * np.sin(x / 2.3) + 0.06 * np.cos(y / 3.1)
    rgb = np.stack(
        [
            0.15 + 0.65 * (x > 46) + texture,
            0.20 + 0.50 * (y > 65) + 0.7 * texture,
            0.25 + 0.20 * (x > 46) - 0.5 * texture,
        ],
        axis=2,
    ).astype(np.float32)
    low_y, low_x = np.indices(low_shape)
    low = np.stack(
        [
            (low_x > 7).astype(np.float32),
            (low_y > 11).astype(np.float32),
            0.25 * (low_x > 7).astype(np.float32) - 0.15 * (low_y > 11).astype(np.float32),
        ],
        axis=2,
    )
    psf = PsfModel(2.0, 3.0, 0.8, low_shape, high_shape)
    base_cfg = FusionConfig(
        rank=3,
        refiner="bicubic",
        coefficient_detail_strength=0.0,
        spatial_detail_strength=0.0,
    )
    detail_cfg = FusionConfig(
        rank=3,
        refiner="bicubic",
        coefficient_detail_strength=0.24,
        coefficient_detail_min_correlation=0.02,
        coefficient_detail_clip_sigma=0.40,
        spatial_detail_strength=0.0,
    )
    base = refine_coefficients(low, rgb, psf, base_cfg)
    enhanced = refine_coefficients(low, rgb, psf, detail_cfg)

    def high_frequency_energy(coeff: np.ndarray) -> float:
        values = []
        for component in range(coeff.shape[2]):
            smooth = cv2.GaussianBlur(coeff[:, :, component], (0, 0), 2.0)
            values.append(float(np.mean(np.abs(coeff[:, :, component] - smooth))))
        return float(np.mean(values))

    assert enhanced.details["coefficient_detail"]["accepted_components"] >= 2
    assert high_frequency_energy(enhanced.coefficients) > 1.08 * high_frequency_energy(base.coefficients)
    assert np.sqrt(np.mean((degrade_coefficients(enhanced.coefficients, psf) - low) ** 2)) < 0.08


def test_intrinsic_log_detail_recovers_dark_texture_without_breaking_observation() -> None:
    high_shape = (160, 128)
    low_shape = (30, 20)
    y, x = np.indices(high_shape, dtype=np.float32)
    texture = 0.55 * np.sin(x / 2.8) + 0.45 * np.cos(y / 3.7)
    dark = x < 32
    illumination = np.where(dark, 0.012, 0.48).astype(np.float32)
    rgb_detail = np.where(dark, 0.0035, 0.045) * texture
    rgb = np.stack(
        [
            illumination + rgb_detail,
            0.92 * illumination + 0.78 * rgb_detail,
            0.84 * illumination - 0.55 * rgb_detail,
        ],
        axis=2,
    ).astype(np.float32)
    rgb = np.clip(rgb, 0.002, 1.0)

    coarse = 0.35 + 0.18 * (x > 72) + 0.10 * np.sin(y / 28.0)
    true_coeff = np.stack(
        [
            coarse + 0.055 * texture,
            0.55 * coarse - 0.025 * texture,
            0.20 * np.cos(x / 32.0) + 0.018 * texture,
        ],
        axis=2,
    ).astype(np.float32)
    psf = PsfModel(2.2, 3.0, 0.8, low_shape, high_shape)
    low = degrade_coefficients(true_coeff, psf)

    common = dict(
        rank=3,
        refiner="bicubic",
        coefficient_detail_strength=0.28,
        coefficient_detail_min_correlation=0.01,
        coefficient_detail_clip_sigma=0.55,
        coefficient_detail_support_floor=0.45,
        spatial_detail_strength=0.28,
        spatial_detail_additive_strength=0.0,
        spatial_detail_nullspace_iterations=3,
    )
    legacy = refine_coefficients(low, rgb, psf, FusionConfig(**common))
    intrinsic = refine_coefficients(
        low,
        rgb,
        psf,
        FusionConfig(
            **common,
            intrinsic_detail_enabled=True,
            intrinsic_log_epsilon=1.0 / 255.0,
            dark_detail_boost=1.0,
            dark_detail_percentile=30.0,
            dark_texture_noise_floor=0.008,
            spatial_detail_log_gain=True,
            spatial_detail_confidence_mode="none",
            spatial_detail_back_projection_iterations=4,
        ),
    )

    def dark_log_detail_correlation(result) -> float:
        predicted = np.maximum(result.coefficients[:, :, 0] * result.detail_gain_map, 1e-4)
        truth = np.maximum(true_coeff[:, :, 0], 1e-4)
        predicted_detail = np.log(predicted) - cv2.GaussianBlur(np.log(predicted), (0, 0), 2.4)
        truth_detail = np.log(truth) - cv2.GaussianBlur(np.log(truth), (0, 0), 2.4)
        mask = dark & (y > 8) & (y < high_shape[0] - 9) & (x > 8)
        return float(np.corrcoef(predicted_detail[mask], truth_detail[mask])[0, 1])

    legacy_corr = dark_log_detail_correlation(legacy)
    intrinsic_corr = dark_log_detail_correlation(intrinsic)
    assert intrinsic_corr > legacy_corr + 0.02
    assert intrinsic_corr > 0.45
    assert np.sqrt(np.mean((degrade_coefficients(intrinsic.coefficients, psf) - low) ** 2)) < 0.08
    assert np.sqrt(np.mean((degrade_spatial_map(intrinsic.detail_gain_map, psf) - 1.0) ** 2)) < 0.025


def _high_pass(image: np.ndarray, sigma: float = 2.4) -> np.ndarray:
    arr = np.asarray(image, dtype=np.float32)
    return arr - cv2.GaussianBlur(arr, (0, 0), sigma, borderType=cv2.BORDER_REFLECT101)


def _masked_slope_and_correlation(
    reference: np.ndarray,
    estimate: np.ndarray,
    mask: np.ndarray,
) -> tuple[float, float]:
    x = np.asarray(reference, dtype=np.float32)[mask]
    y = np.asarray(estimate, dtype=np.float32)[mask]
    x = x - float(np.mean(x))
    y = y - float(np.mean(y))
    variance = float(np.sum(x * x))
    slope = float(np.sum(x * y) / variance) if variance > 1e-9 else 0.0
    denominator = float(np.sqrt(variance * np.sum(y * y)))
    correlation = float(np.sum(x * y) / denominator) if denominator > 1e-9 else 0.0
    return slope, correlation


def test_local_mtf_gsa_handles_opposite_material_detail_signs() -> None:
    high_shape = (192, 160)
    low_shape = (48, 40)
    y, x = np.indices(high_shape, dtype=np.float32)
    coarse = 0.12 * np.sin(2.0 * np.pi * x / 52.0) + 0.08 * np.cos(2.0 * np.pi * y / 58.0)
    fine = np.sin(2.0 * np.pi * x / 7.0) + 0.65 * np.cos(2.0 * np.pi * y / 9.0)
    sign = np.where(x < high_shape[1] / 2.0, 1.0, -1.0).astype(np.float32)
    rgb_structure = coarse + 0.035 * fine
    rgb = np.stack(
        [
            0.52 + rgb_structure,
            0.48 + 0.82 * rgb_structure,
            0.44 - 0.45 * rgb_structure,
        ],
        axis=2,
    ).astype(np.float32)
    rgb = np.clip(rgb, 0.05, 0.95)
    true_coeff = np.stack(
        [
            0.42 + sign * (0.65 * coarse + 0.65 * 0.035 * fine),
            0.31 - sign * (0.35 * coarse + 0.35 * 0.035 * fine),
            0.20 + 0.14 * np.sin(y / 27.0),
        ],
        axis=2,
    ).astype(np.float32)
    psf = PsfModel(1.6, 1.9, 0.9, low_shape, high_shape)
    low = degrade_coefficients(true_coeff, psf)

    common = dict(
        rank=3,
        refiner="bicubic",
        coefficient_detail_strength=0.95,
        coefficient_detail_clip_sigma=1.20,
        coefficient_detail_support_floor=0.45,
        coefficient_detail_nullspace_iterations=1,
        coefficient_detail_back_projection_iterations=1,
        coefficient_detail_clip_coefficients=False,
        spatial_detail_strength=0.0,
        dark_detail_boost=0.0,
    )
    global_result = refine_coefficients(
        low,
        rgb,
        psf,
        FusionConfig(**common, coefficient_detail_method="global_ridge"),
    )
    local_result = refine_coefficients(
        low,
        rgb,
        psf,
        FusionConfig(
            **common,
            coefficient_detail_method="local_mtf_gsa",
            coefficient_detail_local_radius=4,
            coefficient_detail_local_ridge=0.025,
            coefficient_detail_local_correlation_floor=0.05,
            coefficient_detail_amplitude_recovery_limit=1.6,
        ),
    )

    truth_detail = _high_pass(true_coeff[:, :, 0])
    global_detail = _high_pass(global_result.coefficients[:, :, 0])
    local_detail = _high_pass(local_result.coefficients[:, :, 0])
    border = 14
    stable = (
        (y > border)
        & (y < high_shape[0] - border)
        & (x > border)
        & (x < high_shape[1] - border)
        & (np.abs(x - high_shape[1] / 2.0) > 14)
    )
    local_slope, local_corr = _masked_slope_and_correlation(
        truth_detail, local_detail, stable
    )
    _, global_corr = _masked_slope_and_correlation(
        truth_detail, global_detail, stable
    )
    assert local_result.details["coefficient_detail"]["accepted_components"] >= 2
    assert local_corr > global_corr + 0.18
    assert local_corr > 0.70
    assert 0.65 < local_slope < 1.35
    assert np.sqrt(
        np.mean((degrade_coefficients(local_result.coefficients, psf) - low) ** 2)
    ) < 0.035


def test_local_mtf_gsa_preserves_isoluminant_colour_texture() -> None:
    high_shape = (160, 144)
    low_shape = (40, 36)
    y, x = np.indices(high_shape, dtype=np.float32)
    chroma = (
        0.13 * np.sin(2.0 * np.pi * x / 49.0)
        + 0.08 * np.cos(2.0 * np.pi * y / 57.0)
        + 0.040 * np.sin(2.0 * np.pi * x / 6.5)
        + 0.028 * np.cos(2.0 * np.pi * y / 7.5)
    )
    rgb = np.stack(
        [
            0.50 + chroma,
            0.50 - (0.299 / 0.587) * chroma,
            np.full_like(chroma, 0.50),
        ],
        axis=2,
    ).astype(np.float32)
    true_coeff = np.stack(
        [
            0.40 + 0.55 * chroma,
            0.28 - 0.24 * chroma,
            0.18 + 0.08 * np.sin(x / 31.0),
        ],
        axis=2,
    ).astype(np.float32)
    psf = PsfModel(1.5, 1.8, 0.9, low_shape, high_shape)
    low = degrade_coefficients(true_coeff, psf)
    base = refine_coefficients(
        low,
        rgb,
        psf,
        FusionConfig(
            rank=3,
            refiner="bicubic",
            coefficient_detail_strength=0.0,
            spatial_detail_strength=0.0,
        ),
    )
    enhanced = refine_coefficients(
        low,
        rgb,
        psf,
        FusionConfig(
            rank=3,
            refiner="bicubic",
            coefficient_detail_method="local_mtf_gsa",
            coefficient_detail_strength=0.90,
            coefficient_detail_local_radius=4,
            coefficient_detail_local_ridge=0.02,
            coefficient_detail_local_correlation_floor=0.04,
            coefficient_detail_clip_sigma=1.2,
            coefficient_detail_support_floor=0.5,
            coefficient_detail_nullspace_iterations=1,
            coefficient_detail_back_projection_iterations=1,
            coefficient_detail_clip_coefficients=False,
            spatial_detail_strength=0.0,
        ),
    )
    truth_detail = _high_pass(true_coeff[:, :, 0], sigma=1.6)
    base_detail = _high_pass(base.coefficients[:, :, 0], sigma=1.6)
    enhanced_detail = _high_pass(enhanced.coefficients[:, :, 0], sigma=1.6)
    mask = (x > 12) & (x < high_shape[1] - 13) & (y > 12) & (y < high_shape[0] - 13)
    base_slope, base_corr = _masked_slope_and_correlation(truth_detail, base_detail, mask)
    slope, enhanced_corr = _masked_slope_and_correlation(
        truth_detail, enhanced_detail, mask
    )
    assert enhanced_corr > base_corr + 0.12
    assert enhanced_corr > 0.82
    assert slope > base_slope + 0.25
    assert 0.65 < slope < 1.35


def test_lowrank_coefficient_bridge_recovers_fixed_cross_block_direction() -> None:
    high_shape = (176, 144)
    low_shape = (44, 36)
    y, x = np.indices(high_shape, dtype=np.float32)
    structure = (
        0.11 * np.sin(2.0 * np.pi * x / 53.0)
        + 0.08 * np.cos(2.0 * np.pi * y / 61.0)
        + 0.042 * np.sin(2.0 * np.pi * x / 7.0)
        + 0.026 * np.cos(2.0 * np.pi * y / 8.5)
    )
    rgb = np.stack(
        [
            0.50 + structure,
            0.47 + 0.78 * structure,
            0.43 - 0.38 * structure,
        ],
        axis=2,
    ).astype(np.float32)
    true_coeff = np.stack(
        [
            0.42 + 0.64 * structure,
            0.30 - 0.31 * structure,
            0.20 + 0.11 * structure,
            0.14 + 0.05 * np.sin(y / 29.0),
        ],
        axis=2,
    ).astype(np.float32)
    psf = PsfModel(1.4, 1.8, 0.9, low_shape, high_shape)
    low = degrade_coefficients(true_coeff, psf)
    base = refine_coefficients(
        low,
        rgb,
        psf,
        FusionConfig(
            rank=4,
            refiner="bicubic",
            coefficient_detail_strength=0.0,
            spatial_detail_strength=0.0,
        ),
    )
    enhanced = refine_coefficients(
        low,
        rgb,
        psf,
        FusionConfig(
            rank=4,
            refiner="bicubic",
            coefficient_detail_method="lowrank_coefficient_bridge",
            coefficient_detail_strength=0.90,
            coefficient_detail_bridge_rank=1,
            coefficient_detail_bridge_cv_r2_floor=-0.1,
            coefficient_detail_ridge=0.02,
            coefficient_detail_clip_sigma=1.5,
            coefficient_detail_nullspace_iterations=1,
            coefficient_detail_back_projection_iterations=1,
            coefficient_detail_amplitude_recovery_limit=1.8,
            coefficient_detail_clip_coefficients=False,
            coefficient_detail_base_residual_keep=0.0,
            spatial_detail_confidence_mode="none",
            spatial_detail_strength=0.0,
        ),
    )
    local_enhanced = refine_coefficients(
        low,
        rgb,
        psf,
        FusionConfig(
            rank=4,
            refiner="bicubic",
            coefficient_detail_method="local_lowrank_coefficient_bridge",
            coefficient_detail_strength=0.90,
            coefficient_detail_bridge_rank=1,
            coefficient_detail_bridge_cv_r2_floor=-0.1,
            coefficient_detail_bridge_local_radius=4,
            coefficient_detail_bridge_local_correlation_floor=0.02,
            coefficient_detail_ridge=0.02,
            coefficient_detail_clip_sigma=1.5,
            coefficient_detail_nullspace_iterations=1,
            coefficient_detail_back_projection_iterations=1,
            coefficient_detail_amplitude_recovery_limit=1.8,
            coefficient_detail_clip_coefficients=False,
            coefficient_detail_base_residual_keep=0.0,
            spatial_detail_confidence_mode="none",
            spatial_detail_strength=0.0,
        ),
    )
    mask = (x > 12) & (x < high_shape[1] - 13) & (y > 12) & (y < high_shape[0] - 13)
    truth_detail = _high_pass(true_coeff[:, :, 0], 2.2)
    base_detail = _high_pass(base.coefficients[:, :, 0], 2.2)
    enhanced_detail = _high_pass(enhanced.coefficients[:, :, 0], 2.2)
    _, base_corr = _masked_slope_and_correlation(truth_detail, base_detail, mask)
    slope, enhanced_corr = _masked_slope_and_correlation(
        truth_detail, enhanced_detail, mask
    )
    bridge = enhanced.details["coefficient_detail"]
    assert bridge["bridge_rank"] == 1
    assert bridge["blocked_cv_variance_weighted_r2"] > 0.35
    assert enhanced_corr > base_corr + 0.18
    assert enhanced_corr > 0.80
    assert 0.55 < slope < 1.35
    local_detail = _high_pass(local_enhanced.coefficients[:, :, 0], 2.2)
    _, local_corr = _masked_slope_and_correlation(
        truth_detail, local_detail, mask
    )
    assert local_enhanced.details["coefficient_detail"]["local_bridge"] is True
    assert local_corr > 0.78


def test_dark_texture_confidence_rejects_independent_rgb_noise() -> None:
    height, width = 128, 160
    y, x = np.indices((height, width), dtype=np.float32)
    rng = np.random.default_rng(7)
    guide = np.full((height, width, 3), 0.012, dtype=np.float32)
    independent = rng.normal(0.0, 0.0018, size=(height, width // 2, 3)).astype(np.float32)
    guide[:, : width // 2, :] += independent
    common = 0.0024 * (
        np.sin(2.0 * np.pi * x[:, width // 2 :] / 8.0)
        + 0.55 * np.cos(2.0 * np.pi * y[:, width // 2 :] / 11.0)
    )
    guide[:, width // 2 :, 0] += common
    guide[:, width // 2 :, 1] += 0.82 * common
    guide[:, width // 2 :, 2] -= 0.58 * common
    guide = np.clip(guide, 0.002, 0.08)
    confidence = _dark_texture_confidence(
        guide,
        FusionConfig(
            intrinsic_detail_enabled=True,
            intrinsic_log_epsilon=1.0 / 255.0,
            dark_detail_boost=1.0,
            dark_detail_percentile=45.0,
            dark_texture_noise_floor=0.008,
            dark_texture_correlation_floor=0.18,
            dark_texture_window_radius=3,
        ),
    )
    noise_confidence = float(np.mean(confidence[10:-10, 10 : width // 2 - 10]))
    structure_confidence = float(np.mean(confidence[10:-10, width // 2 + 10 : -10]))
    assert noise_confidence < 0.08
    assert structure_confidence > 0.18
    assert structure_confidence > 4.0 * max(noise_confidence, 1e-4)


def test_simplex_abundance_rgb_residual_recovers_shared_material_texture() -> None:
    high_shape = (160, 128)
    low_shape = (40, 32)
    y, x = np.indices(high_shape, dtype=np.float32)
    texture = np.sin(2.0 * np.pi * x / 9.0) * np.cos(2.0 * np.pi * y / 13.0)
    coarse = 0.06 * np.sin(x / 27.0) + 0.04 * np.cos(y / 31.0)
    abundance0 = 0.46 + coarse + 0.075 * texture
    abundance1 = 0.31 - 0.035 * texture - 0.45 * coarse
    abundance2 = 1.0 - abundance0 - abundance1
    truth = np.stack([abundance0, abundance1, abundance2], axis=2).astype(np.float32)
    assert float(np.min(truth)) > 0.0
    rgb_endmembers = np.array(
        [[0.78, 0.31, 0.18], [0.19, 0.72, 0.27], [0.16, 0.24, 0.79]],
        dtype=np.float32,
    )
    rgb = np.einsum("...k,kc->...c", truth, rgb_endmembers, optimize=True)
    psf = PsfModel(1.1, 1.4, 0.9, low_shape, high_shape)
    low = degrade_coefficients(truth, psf)
    base = refine_coefficients(
        low,
        rgb,
        psf,
        FusionConfig(
            rank=3,
            refiner="bicubic",
            coefficient_detail_method="simplex_abundance_rgb_residual",
            coefficient_detail_strength=0.0,
            coefficient_constraint="simplex",
            spatial_detail_method="coherent_mtf_log_hpm",
            spatial_detail_strength=0.0,
        ),
    )
    enhanced = refine_coefficients(
        low,
        rgb,
        psf,
        FusionConfig(
            rank=3,
            refiner="bicubic",
            coefficient_detail_method="simplex_abundance_rgb_residual",
            coefficient_detail_strength=0.85,
            coefficient_detail_nullspace_iterations=1,
            coefficient_detail_back_projection_iterations=1,
            coefficient_detail_mtf_fine_sigma=0.35,
            coefficient_constraint="simplex",
            simplex_abundance_rgb_ridge=0.002,
            simplex_abundance_min_r2=-0.2,
            simplex_abundance_detail_l1_limit=0.30,
            spatial_detail_confidence_gate_low=0.0,
            spatial_detail_confidence_gate_high=0.08,
            dark_texture_noise_floor=0.001,
            dark_texture_correlation_floor=0.02,
            spatial_detail_method="coherent_mtf_log_hpm",
            spatial_detail_strength=0.0,
        ),
    )
    mask = (x > 8) & (x < high_shape[1] - 9) & (y > 8) & (y < high_shape[0] - 9)
    truth_detail = _high_pass(truth[:, :, 0], 2.0)
    base_detail = _high_pass(base.coefficients[:, :, 0], 2.0)
    enhanced_detail = _high_pass(enhanced.coefficients[:, :, 0], 2.0)
    base_corr = float(np.corrcoef(truth_detail[mask], base_detail[mask])[0, 1])
    enhanced_corr = float(
        np.corrcoef(truth_detail[mask], enhanced_detail[mask])[0, 1]
    )
    assert enhanced_corr > base_corr + 0.12
    assert enhanced_corr > 0.75
    assert float(np.min(enhanced.coefficients)) >= 0.0
    assert float(np.max(np.abs(np.sum(enhanced.coefficients, axis=2) - 1.0))) < 2e-6
    assert enhanced.details["coefficient_detail"]["rgb_fit_r2_median"] > 0.60


def test_simplex_abundance_final_l1_limit_survives_back_projection() -> None:
    high_shape = (32, 32)
    low_shape = (8, 8)
    y, x = np.indices(high_shape, dtype=np.float32)
    truth = np.stack(
        [
            0.42 + 0.16 * np.sin(x / 7.0),
            0.31 + 0.12 * np.cos(y / 6.0),
            np.zeros(high_shape, dtype=np.float32),
        ],
        axis=2,
    )
    truth[:, :, 2] = 1.0 - truth[:, :, 0] - truth[:, :, 1]
    rgb_endmembers = np.array(
        [[0.82, 0.26, 0.18], [0.21, 0.76, 0.29], [0.14, 0.22, 0.83]],
        dtype=np.float32,
    )
    rgb = np.einsum("...k,kc->...c", truth, rgb_endmembers, optimize=True)
    psf = PsfModel(1.0, 1.2, 0.9, low_shape, high_shape)
    low = degrade_coefficients(truth, psf)
    current = np.empty_like(truth)
    current[:, :, 0] = 0.90
    current[:, :, 1:] = 0.05
    l1_limit = 0.04

    enhanced, details = _inject_simplex_abundance_detail(
        current,
        low,
        rgb,
        psf,
        np.ones(high_shape, dtype=np.float32),
        FusionConfig(
            rank=3,
            coefficient_constraint="simplex",
            coefficient_detail_strength=0.8,
            coefficient_detail_nullspace_iterations=0,
            coefficient_detail_back_projection_iterations=2,
            coefficient_detail_mtf_fine_sigma=0.0,
            simplex_abundance_rgb_ridge=1e-6,
            simplex_abundance_min_r2=-1.0,
            simplex_abundance_detail_l1_limit=l1_limit,
            simplex_abundance_active_components=0,
            spatial_detail_confidence_gate_low=0.0,
            spatial_detail_confidence_gate_high=0.01,
            dark_texture_noise_floor=1e-4,
            dark_texture_correlation_floor=0.0,
        ),
    )

    final_delta_l1 = np.sum(np.abs(enhanced - current), axis=2)
    actual_rmse = float(
        np.sqrt(np.mean((degrade_coefficients(enhanced, psf) - low) ** 2))
    )
    assert details["delta_l1_max_before_final_constraint"] > l1_limit + 0.10
    assert details["final_l1_constraint_clipped_fraction"] > 0.95
    assert float(np.max(final_delta_l1)) <= l1_limit + 2e-6
    assert np.isclose(details["delta_l1_max"], np.max(final_delta_l1), atol=1e-8)
    assert np.isclose(
        details["coefficient_rmse_after_detail_back_projection"],
        actual_rmse,
        atol=1e-8,
    )
    assert float(np.min(enhanced)) >= 0.0
    assert float(np.max(np.abs(np.sum(enhanced, axis=2) - 1.0))) < 2e-6


def test_final_product_cycle_preserves_simplex_abundances() -> None:
    high_shape = (80, 64)
    low_shape = (20, 16)
    y, x = np.indices(high_shape, dtype=np.float32)
    truth = np.stack(
        [
            0.48 + 0.08 * np.sin(x / 13.0),
            0.29 + 0.06 * np.cos(y / 17.0),
            np.zeros(high_shape, dtype=np.float32),
        ],
        axis=2,
    )
    truth[:, :, 2] = 1.0 - truth[:, :, 0] - truth[:, :, 1]
    psf = PsfModel(1.0, 1.2, 0.9, low_shape, high_shape)
    low = degrade_coefficients(truth, psf)
    initial = np.stack(
        [
            cv2.resize(low[:, :, component], high_shape[::-1], interpolation=cv2.INTER_CUBIC)
            for component in range(3)
        ],
        axis=2,
    )
    basis = np.array(
        [[0.62, 0.28, 0.16, 0.11], [0.21, 0.57, 0.33, 0.18], [0.12, 0.19, 0.51, 0.69]],
        dtype=np.float32,
    )
    target = np.einsum("...k,kb->...b", low, basis, optimize=True)
    gain = np.exp(0.05 * np.sin(2.0 * np.pi * x / 8.0)).astype(np.float32)
    refined, details = back_project_modulated_product(
        initial,
        low,
        target,
        basis,
        np.zeros(4, dtype=np.float32),
        gain,
        np.zeros(high_shape, dtype=np.float32),
        np.zeros(4, dtype=np.float32),
        psf,
        FusionConfig(
            coefficient_constraint="simplex",
            spatial_detail_product_back_projection_iterations=4,
            spatial_detail_product_back_projection_weight=0.8,
            spatial_detail_product_back_projection_clip_sigma=0.8,
        ),
    )
    assert details["spectral_residual_solver"] == "basis_pseudoinverse_then_simplex_tangent"
    assert details["lowres_rmse_after"] < details["lowres_rmse_before"]
    assert float(np.min(refined)) >= 0.0
    assert float(np.max(np.abs(np.sum(refined, axis=2) - 1.0))) < 2e-6


def test_final_product_back_projection_uses_modulated_cube_residual() -> None:
    high_shape = (96, 72)
    low_shape = (24, 18)
    y, x = np.indices(high_shape, dtype=np.float32)
    true_coeff = np.stack(
        [
            0.42 + 0.12 * np.sin(x / 19.0) + 0.08 * np.cos(y / 23.0),
            0.28 + 0.09 * np.cos(x / 17.0) - 0.05 * np.sin(y / 21.0),
        ],
        axis=2,
    ).astype(np.float32)
    psf = PsfModel(1.4, 1.8, 0.9, low_shape, high_shape)
    low_coeff = degrade_coefficients(true_coeff, psf)
    initial = np.empty_like(true_coeff)
    for component in range(2):
        initial[:, :, component] = cv2.resize(
            low_coeff[:, :, component],
            (high_shape[1], high_shape[0]),
            interpolation=cv2.INTER_CUBIC,
        )
    basis = np.array(
        [[0.8, 0.6, 0.0], [-0.6, 0.8, 0.0]], dtype=np.float32
    )
    mean = np.array([0.22, 0.26, 0.18], dtype=np.float32)
    target_low_cube = (
        np.einsum("...k,kb->...b", low_coeff, basis, optimize=True)
        + mean[None, None, :]
    ).astype(np.float32)
    log_detail = 0.16 * np.sin(2.0 * np.pi * x / 7.0)
    gain = np.exp(log_detail).astype(np.float32)
    additive = np.zeros(high_shape, dtype=np.float32)
    scale = np.zeros(3, dtype=np.float32)
    refined, details = back_project_modulated_product(
        initial,
        low_coeff,
        target_low_cube,
        basis,
        mean,
        gain,
        additive,
        scale,
        psf,
        FusionConfig(
            spatial_detail_product_back_projection_iterations=4,
            spatial_detail_product_back_projection_weight=0.8,
            spatial_detail_product_back_projection_clip_sigma=0.8,
        ),
    )
    assert details["proxy_D_gain_used"] is False
    assert details["lowres_rmse_after"] < 0.45 * details["lowres_rmse_before"]
    final_band = (
        np.einsum("...k,k->...", refined, basis[:, 0], optimize=True)
        + mean[0]
    ) * gain
    final_detail = _high_pass(np.log(np.maximum(final_band, 1e-4)), sigma=2.0)
    reference_detail = _high_pass(np.log(gain), sigma=2.0)
    mask = (x > 8) & (x < high_shape[1] - 9) & (y > 8) & (y < high_shape[0] - 9)
    _, correlation = _masked_slope_and_correlation(
        reference_detail, final_detail, mask
    )
    assert correlation > 0.80


def test_v61_visual_full_detail_transfers_isoluminant_edges_with_lowres_control() -> None:
    high_shape = (144, 112)
    low_shape = (36, 28)
    y, x = np.indices(high_shape, dtype=np.float32)
    texture = 0.065 * np.sin(2.0 * np.pi * x / 8.5) * np.cos(
        2.0 * np.pi * y / 11.5
    )
    base = 0.36 + 0.05 * np.sin(y / 31.0)
    rng = np.random.default_rng(61)
    noise = rng.normal(0.0, 0.008, size=high_shape).astype(np.float32)
    rgb = np.stack(
        [
            base + texture + noise,
            base - (0.299 / 0.587) * texture + noise,
            base + 0.15 * texture + noise,
        ],
        axis=2,
    ).astype(np.float32)
    rgb = np.clip(rgb, 0.01, 0.99)
    low_y, low_x = np.indices(low_shape, dtype=np.float32)
    low = np.stack(
        [
            0.42 + 0.06 * np.sin(low_x / 7.0),
            0.27 + 0.04 * np.cos(low_y / 8.0),
        ],
        axis=2,
    ).astype(np.float32)
    psf = PsfModel(1.2, 1.5, 0.9, low_shape, high_shape)
    result = refine_coefficients(
        low,
        rgb,
        psf,
        FusionConfig(
            rank=2,
            refiner="bicubic",
            coefficient_detail_strength=0.0,
            spatial_detail_method="visual_multiscale_gradient",
            spatial_detail_strength=1.0,
            spatial_detail_gain_limits=(0.55, 1.80),
            spatial_detail_log_detail_clip=0.55,
            spatial_detail_nullspace_iterations=2,
            spatial_detail_amplitude_recovery_limit=2.0,
            visual_detail_denoise_strength=0.55,
            visual_detail_pyramid_sigmas=(0.65, 1.35, 2.8, 5.6),
            visual_detail_pyramid_weights=(1.15, 1.0, 0.82, 0.58),
            visual_detail_chroma_weight=0.70,
            visual_detail_poisson_screen=0.22,
            visual_detail_gradient_weight=1.0,
            visual_detail_sharpen_strength=0.25,
        ),
    )

    log_gain = np.log(np.maximum(result.detail_gain_map, 1e-6))
    mask = (x > 8) & (x < high_shape[1] - 9) & (y > 8) & (y < high_shape[0] - 9)
    correlation = float(np.corrcoef(texture[mask], log_gain[mask])[0, 1])
    lowres_rmse = float(
        np.sqrt(
            np.mean((degrade_spatial_map(result.detail_gain_map, psf) - 1.0) ** 2)
        )
    )
    assert correlation > 0.55
    assert np.percentile(np.abs(log_gain[mask]), 95.0) > 0.01
    assert lowres_rmse < 0.035
    assert result.details["spatial_detail"]["method"] == "visual_multiscale_gradient"
    assert result.details["spatial_detail"]["denoise_residual_rms"] > 0.0
