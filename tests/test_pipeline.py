from rf_shap.config import PipelineConfig
from rf_shap.pipeline import DiagnosisPipeline
from rf_shap.synthetic import make_synthetic_cubes


def test_pipeline_smoke():
    x, y = make_synthetic_cubes(n_time=12, n_lat=4, n_lon=5, seed=3)
    cfg = PipelineConfig.from_dict(
        {
            "seed": 3,
            "x_transform": {"mode": "raw"},
            "feature_selection": {"methods": ["collinearity"]},
            "split": {"method": "random", "test_size": 0.3},
            "sampling": {"method": "random", "n_samples": 80},
            "model": {"name": "extra_trees", "params": {"n_estimators": 30, "max_depth": 6, "n_jobs": 1}},
            "explain": {
                "rf_importance": True,
                "permutation_importance": False,
                "shap": {"enabled": False},
                "ale": False,
                "ice_pdp": False,
            },
        }
    )
    result = DiagnosisPipeline(cfg).run(x=x, y=y)
    assert result.prediction_cube is not None
    assert result.prediction_cube.dims == ("time", "lat", "lon")
    assert result.metrics["n_test"] > 0
    assert result.metrics["rmse"] >= 0
