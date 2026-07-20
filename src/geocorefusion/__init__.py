"""GeoCoreFusion research prototype."""

from ._version import __version__
from .config import PipelineConfig, load_config
from .identifiability import (
    IdentifiabilityConfig,
    assess_cube_identifiability,
    fit_blocked_bandpass_relation,
)
from .pipeline import run_pipeline
from .registration_uncertainty import (
    RegistrationUncertaintyConfig,
    RegistrationUncertaintyEstimate,
    estimate_local_registration_covariance,
    registration_sigma_points_5,
)
from .spectral_guard import SpectralConeDiagnostics, project_spectral_cone

__all__ = [
    "IdentifiabilityConfig",
    "PipelineConfig",
    "RegistrationUncertaintyConfig",
    "RegistrationUncertaintyEstimate",
    "SpectralConeDiagnostics",
    "assess_cube_identifiability",
    "estimate_local_registration_covariance",
    "fit_blocked_bandpass_relation",
    "load_config",
    "project_spectral_cone",
    "registration_sigma_points_5",
    "run_pipeline",
    "__version__",
]
