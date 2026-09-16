"""End-to-end demo on a synthetic CMIP6-like cube."""

from __future__ import annotations

from pathlib import Path

from rf_shap.config import PipelineConfig
from rf_shap.pipeline import DiagnosisPipeline
from rf_shap.synthetic import make_synthetic_cubes


def main() -> None:
    x, y = make_synthetic_cubes(n_time=18, n_lat=6, n_lon=8, seed=7)
    out = Path("outputs/demo")
    cfg = PipelineConfig.from_dict(
        {
            "seed": 7,
            "output_dir": str(out),
            "x_transform": {"mode": "lag", "lags": [1]},
            "preprocess": {
                "x": {"deseasonalize": False, "detrend": False},
                "y": {"deseasonalize": False, "detrend": False},
            },
            "feature_selection": {
                "methods": ["collinearity", "pairwise_importance"],
                "collinearity": {"corr_threshold": 0.98, "vif_threshold": 20.0, "linreg_r2_threshold": 0.99},
                "pairwise_importance": {"corr_threshold": 0.95},
            },
            "split": {"method": "time", "test_size": 0.25},
            "sampling": {"method": "fscs", "n_samples": 400},
            "model": {
                "name": "extra_trees",
                "params": {"n_estimators": 80, "max_depth": 8, "n_jobs": -1, "random_state": 7},
            },
            "predict": {"restore_cube": True},
            "explain": {
                "rf_importance": True,
                "permutation_importance": True,
                "permutation_n_repeats": 2,
                "shap": {
                    "enabled": True,
                    "method": "auto",
                    "max_samples": 250,
                    "background_size": 40,
                },
                "ale": True,
                "ice_pdp": True,
                "ice_n_samples": 20,
                "pdp_grid": 10,
            },
        }
    )
    result = DiagnosisPipeline(cfg).run(x=x, y=y)
    result.save(out)
    print("metrics:", result.metrics)
    print("features:", result.artifacts.get("features"))
    print("shap method:", result.shap_cube.attrs.get("shap_method") if result.shap_cube is not None else None)
    print("saved to", out.resolve())


if __name__ == "__main__":
    main()
