from rf_shap.config import PipelineConfig
from rf_shap.pipeline import DiagnosisPipeline
from rf_shap.synthetic import make_synthetic_cubes


def test_shap_restores_cube():
    x, y = make_synthetic_cubes(n_time=8, n_lat=3, n_lon=4, seed=4)
    cfg = PipelineConfig.from_dict(
        {
            "seed": 4,
            "x_transform": {"mode": "raw"},
            "feature_selection": {"methods": []},
            "split": {"method": "time", "test_size": 0.3},
            "sampling": {"method": "random", "n_samples": 50},
            "model": {"name": "extra_trees", "params": {"n_estimators": 20, "max_depth": 5, "n_jobs": 1}},
            "explain": {
                "rf_importance": False,
                "permutation_importance": False,
                "shap": {"enabled": True, "method": "treeshap", "max_samples": 30, "background_size": 10},
                "ale": False,
                "ice_pdp": False,
            },
        }
    )
    result = DiagnosisPipeline(cfg).run(x=x, y=y)
    assert result.shap_cube is not None
    assert set(result.shap_cube.data_vars) == set(result.artifacts["features"])
    assert result.shap_cube["tas"].dims == ("time", "lat", "lon")
    assert int(result.shap_cube["tas"].notnull().sum()) <= 30
