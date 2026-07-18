import numpy as np

from geocorefusion.config import SpectralConfig
from geocorefusion.spectral import harmonize_sensors


def _field(wavelengths: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    y, x = np.indices(shape, dtype=np.float32)
    abundance = 0.25 + 0.55 * (x / max(shape[1] - 1, 1)) + 0.15 * np.sin(y / 7.0)
    base = 0.45 + 0.00008 * (wavelengths - 1200.0)
    absorption = 0.16 * np.exp(-0.5 * ((wavelengths - 1400.0) / 35.0) ** 2) + 0.22 * np.exp(-0.5 * ((wavelengths - 2200.0) / 25.0) ** 2)
    return abundance[:, :, None] * (base - absorption)[None, None, :] + 0.08


def test_wavelength_dependent_harmonization_reduces_overlap_error() -> None:
    nir_w = np.arange(700.0, 1500.1, 10.0)
    swir_w = np.arange(1000.0, 2500.1, 20.0)
    nir = _field(nir_w, (24, 20)).astype(np.float32)
    truth_swir = _field(swir_w, (24, 20)).astype(np.float32)
    gain = 0.8 + 0.00018 * (swir_w - 1000.0)
    offset = -0.025 + 0.00002 * (swir_w - 1000.0)
    swir_observed = (truth_swir - offset[None, None, :]) / gain[None, None, :]
    result = harmonize_sensors(nir, swir_observed.astype(np.float32), nir_w, swir_w, SpectralConfig(output_start_nm=700, output_end_nm=2500, output_step_nm=5))
    assert result.model["overlap_rmse_after"] < 0.45 * result.model["overlap_rmse_before"]
    assert result.wavelengths_nm[0] == 700
    assert result.wavelengths_nm[-1] == 2500
    assert result.cube.shape == (24, 20, 361)
    assert np.isfinite(result.cube).all()

