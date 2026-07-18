"""GeoCoreFusion research prototype."""

from .config import PipelineConfig, load_config
from .pipeline import run_pipeline

__all__ = ["PipelineConfig", "load_config", "run_pipeline"]
__version__ = "0.5.0"
