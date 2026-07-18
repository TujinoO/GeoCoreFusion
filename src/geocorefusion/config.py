"""Configuration schema for the fusion pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(slots=True)
class RoiConfig:
    mode: str = "auto"
    x: int | None = None
    y: int | None = None
    width: int = 1024
    height: int = 1536
    auto_candidates: int = 80


@dataclass(slots=True)
class RegistrationConfig:
    preview_width: int = 320
    preview_max_height: int = 4096
    motion: str = "auto_physical"
    ecc_iterations: int = 250
    ecc_epsilon: float = 1e-6
    gaussian_filter_size: int = 5
    strip_count: int = 12
    strip_overlap: float = 0.25
    strip_max_shift_preview_px: float = 10.0
    enable_strip_drift: bool = False
    enable_roi_refinement: bool = True
    roi_ecc_iterations: int = 700
    roi_ecc_epsilon: float = 1e-7
    roi_scale_limits: tuple[float, float] = (0.84, 1.16)
    roi_shear_limit: float = 0.14
    roi_translation_fraction: float = 0.22
    roi_min_score_gain: float = 0.012
    roi_pair_min_score_gain: float = 0.003
    roi_pair_max_rgb_score_loss: float = 0.035
    enable_roi_column_geometry_refinement: bool = True
    roi_column_peak_min_prominence: float = 0.06
    roi_column_reference_relative_prominence: float = 0.25
    roi_column_peak_distance: int = 10
    roi_column_max_shift: int = 12
    roi_column_min_profile_gain: float = 0.03
    roi_column_max_feature_score_loss: float = 0.065
    roi_column_max_pair_score_loss: float = 0.03
    enable_roi_tiepoint_refinement: bool = True
    roi_tiepoint_grid_rows: int = 13
    roi_tiepoint_grid_cols: int = 8
    roi_tiepoint_template_radius: int = 7
    roi_tiepoint_search_radius: int = 8
    roi_tiepoint_min_points: int = 8
    roi_tiepoint_min_score: float = 0.27
    roi_tiepoint_min_margin: float = 0.012
    roi_tiepoint_max_backward_error: float = 1.75
    roi_tiepoint_max_shift: float = 9.2
    roi_tiepoint_idw_neighbours: int = 10
    roi_tiepoint_idw_smoothing: float = 18.0
    roi_tiepoint_field_sigma: float = 2.2
    roi_tiepoint_pair_score_loss: float = 0.020
    roi_tiepoint_jacobian_floor: float = 0.45
    roi_tiepoint_factors: tuple[float, ...] = (1.0, 0.75, 0.5, 0.25, 0.0)
    enable_roi_row_refinement: bool = False
    roi_row_control_points: int = 9
    roi_row_search_radius: int = 5
    roi_row_min_score_gain: float = 0.007
    roi_fail_score: float = 0.25
    roi_warning_score: float = 0.42
    roi_pair_warning_score: float = 0.50


@dataclass(slots=True)
class SpectralConfig:
    output_start_nm: float = 691.0
    output_end_nm: float = 2518.0
    output_step_nm: float = 5.0
    calibration_method: str = "robust_quantile_smooth"
    gain_limits: tuple[float, float] = (0.25, 4.0)
    offset_limits: tuple[float, float] = (-2.0, 2.0)
    smoothing_window: int = 11
    smoothing_polyorder: int = 2
    overlap_taper_fraction: float = 0.20
    bad_band_noise_quantile: float = 0.97


@dataclass(slots=True)
class DegradationConfig:
    estimate_psf: bool = True
    psf_sigma_x_candidates: tuple[float, ...] = (0.0, 0.5, 0.9, 1.3, 1.8)
    psf_sigma_y_candidates: tuple[float, ...] = (0.0, 0.5, 1.0, 1.6, 2.2)
    default_sigma_x: float = 0.8
    default_sigma_y: float = 1.2
    minimum_identifiable_score: float = 0.12


@dataclass(slots=True)
class FusionConfig:
    rank: int = 12
    max_basis_pixels: int = 50000
    refiner: str = "variational"
    variational_iterations: int = 18
    diffusion_strength: float = 0.22
    anchor_weight: float = 0.20
    rgb_edge_sigma: float = 0.10
    guided_radius: int = 10
    guided_epsilon: float = 0.0025
    boundary_injection_strength: float = 0.85
    boundary_support_radius: int = 9
    coefficient_detail_strength: float = 0.0
    coefficient_detail_ridge: float = 0.08
    coefficient_detail_min_correlation: float = 0.08
    coefficient_detail_clip_sigma: float = 0.35
    coefficient_detail_support_floor: float = 0.30
    coefficient_detail_nullspace_iterations: int = 3
    coefficient_detail_back_projection_iterations: int = 3
    spatial_detail_strength: float = 0.34
    spatial_detail_small_sigma: float = 1.2
    spatial_detail_large_sigma: float = 5.5
    spatial_detail_texture_floor: float = 0.55
    spatial_detail_gain_limits: tuple[float, float] = (0.72, 1.28)
    spatial_detail_nullspace_iterations: int = 2
    spatial_detail_additive_strength: float = 0.0
    spatial_detail_additive_std_fraction: float = 0.35
    spatial_detail_additive_mean_fraction: float = 0.04
    back_projection_weight: float = 0.80
    back_projection_interval: int = 2
    back_projection_clip_sigma: float = 0.35
    coefficient_clip_margin: float = 0.12
    safety_observation_rmse: float = 0.08
    psf_backoff_factors: tuple[float, ...] = (1.0, 0.75, 0.5, 0.25)
    reconstruct_tile: int = 256
    output_dtype: str = "float32"
    clip_quantiles: tuple[float, float] = (0.002, 0.998)
    random_seed: int = 42
    neural_steps: int = 300
    neural_learning_rate: float = 2e-3
    neural_hidden_channels: int = 48


@dataclass(slots=True)
class OutputConfig:
    overwrite_files: bool = False
    write_envi: bool = True
    write_coefficients: bool = True
    write_previews: bool = True
    write_uncertainty: bool = True


@dataclass(slots=True)
class ProjectInfo:
    project_id: str = "GeoCoreFusion"
    borehole_id: str | None = None
    core_interval: str | None = None
    notes: str | None = None


@dataclass(slots=True)
class PipelineConfig:
    data_dir: Path
    output_dir: Path
    roi: RoiConfig = field(default_factory=RoiConfig)
    registration: RegistrationConfig = field(default_factory=RegistrationConfig)
    spectral: SpectralConfig = field(default_factory=SpectralConfig)
    degradation: DegradationConfig = field(default_factory=DegradationConfig)
    fusion: FusionConfig = field(default_factory=FusionConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    project: ProjectInfo = field(default_factory=ProjectInfo)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["data_dir"] = str(self.data_dir)
        payload["output_dir"] = str(self.output_dir)
        return payload


def _update_dataclass(instance: Any, values: dict[str, Any]) -> Any:
    for key, value in values.items():
        if not hasattr(instance, key):
            raise ValueError(f"Unknown configuration key {type(instance).__name__}.{key}")
        current = getattr(instance, key)
        if hasattr(current, "__dataclass_fields__") and isinstance(value, dict):
            _update_dataclass(current, value)
        else:
            if isinstance(current, tuple) and isinstance(value, list):
                value = tuple(value)
            setattr(instance, key, value)
    return instance


def load_config(path: str | Path) -> PipelineConfig:
    source = Path(path)
    payload = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
    if "data_dir" not in payload or "output_dir" not in payload:
        raise ValueError("Configuration requires data_dir and output_dir")
    config = PipelineConfig(data_dir=Path(payload.pop("data_dir")), output_dir=Path(payload.pop("output_dir")))
    _update_dataclass(config, payload)
    config.data_dir = Path(config.data_dir)
    config.output_dir = Path(config.output_dir)
    return config
