"""CMIP6-like cube diagnosis with tree models and SHAP."""

from rf_shap.config import PipelineConfig, load_config
from rf_shap.pipeline import DiagnosisPipeline, run_pipeline

__all__ = [
    "PipelineConfig",
    "load_config",
    "DiagnosisPipeline",
    "run_pipeline",
]
__version__ = "0.1.0"
