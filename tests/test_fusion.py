import numpy as np
import cv2

from geocorefusion.config import FusionConfig
from geocorefusion.degradation import PsfModel, degrade_coefficients, degrade_spatial_map
from geocorefusion.fusion import refine_coefficients


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
