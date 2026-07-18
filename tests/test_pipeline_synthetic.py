from pathlib import Path

import cv2
import numpy as np

from geocorefusion.config import PipelineConfig
from geocorefusion.envi import create_bip_writer
from geocorefusion.pipeline import run_pipeline
from geocorefusion.validation import validate_run


def _write(path: Path, cube: np.ndarray, wavelengths: np.ndarray | None = None) -> None:
    writer, _, _ = create_bip_writer(path, cube.shape, dtype=str(cube.dtype), wavelengths=wavelengths)
    writer[:] = cube
    writer.flush()
    del writer


def test_full_pipeline_on_synthetic_triplet(tmp_path: Path) -> None:
    data = tmp_path / "data"
    data.mkdir()
    high_shape = (240, 128)
    low_shape = (48, 32)
    y, x = np.indices(high_shape, dtype=np.float32)
    material = 0.15 + 0.65 * (x > 62) + 0.15 * np.sin(y / 13.0)
    rgb = np.stack([
        np.clip(material, 0, 1),
        np.clip(0.25 + 0.65 * (y > 118), 0, 1),
        np.clip(0.2 + 0.15 * np.sin(x / 6.0), 0, 1),
    ], axis=2)
    rgb_u8 = (rgb * 255).round().astype(np.uint8)
    low_material = cv2.resize(material, (low_shape[1], low_shape[0]), interpolation=cv2.INTER_AREA)
    nir_w = np.linspace(700, 1500, 25)
    swir_w = np.linspace(1000, 2500, 26)
    def cube(wavelengths: np.ndarray) -> np.ndarray:
        spectral = 0.45 + 0.00005 * (wavelengths - 1200) - 0.12 * np.exp(-0.5 * ((wavelengths - 2200) / 50) ** 2)
        return (0.08 + low_material[:, :, None] * spectral[None, None, :]).astype(np.float32)
    _write(data / "RGB-synthetic.hdr", rgb_u8)
    _write(data / "NIR-synthetic.hdr", cube(nir_w), nir_w)
    _write(data / "SWIR-synthetic.hdr", cube(swir_w), swir_w)

    config = PipelineConfig(data_dir=data, output_dir=tmp_path / "run")
    config.roi.mode = "manual"
    config.roi.x = 0
    config.roi.y = 0
    config.roi.width = high_shape[1]
    config.roi.height = high_shape[0]
    config.registration.preview_width = 96
    config.registration.preview_max_height = 180
    config.registration.ecc_iterations = 80
    config.registration.enable_strip_drift = False
    config.registration.enable_roi_refinement = False
    config.spectral.output_start_nm = 700
    config.spectral.output_end_nm = 2500
    config.spectral.output_step_nm = 50
    config.fusion.rank = 4
    config.fusion.variational_iterations = 3
    config.fusion.psf_backoff_factors = (1.0, 0.5)
    config.output.write_previews = False
    result = run_pipeline(config)
    assert result.quality_report["summary"]["status"] in {"passed", "warning"}
    assert validate_run(result.output_dir)["checks"]["passed"]
