import importlib.util

import numpy as np
import pytest

from rf_shap.config import PipelineConfig
from rf_shap.explain.shap_explain import fast_tree_shap
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


def test_fast_tree_shap_for_random_forest():
    if importlib.util.find_spec("fasttreeshap") is None:
        pytest.skip("fasttreeshap is not installed")
    from sklearn.ensemble import RandomForestRegressor

    rng = np.random.default_rng(0)
    x = rng.normal(size=(80, 4))
    y = 0.5 * x[:, 0] + x[:, 1] + rng.normal(scale=0.05, size=80)
    model = RandomForestRegressor(n_estimators=15, max_depth=4, random_state=0, n_jobs=1)
    model.fit(x[:60], y[:60])
    x_explain = x[60:]
    values, expected = fast_tree_shap(model, x_explain, algorithm="v2", n_jobs=1)
    assert values.shape == x_explain.shape
    base = float(np.asarray(expected).reshape(-1)[0])
    np.testing.assert_allclose(values.sum(axis=1) + base, model.predict(x_explain), rtol=1e-4, atol=1e-3)


def test_linear_tree_shap_matches_treeshap_on_a_forest():
    if importlib.util.find_spec("linear_tree_shap") is None:
        pytest.skip("linear_tree_shap is not installed")
    import shap
    from sklearn.ensemble import RandomForestRegressor

    from rf_shap.explain.shap_explain import linear_tree_shap

    rng = np.random.default_rng(1)
    x = rng.normal(size=(40, 3))
    y = x[:, 0] - 0.4 * x[:, 2]
    model = RandomForestRegressor(n_estimators=4, max_depth=4, random_state=1, n_jobs=1)
    model.fit(x[:30], y[:30])
    x_explain = x[30:]
    values, expected = linear_tree_shap(model, x_explain, version="v2")
    reference = shap.TreeExplainer(model).shap_values(x_explain, check_additivity=False)
    assert values.shape == x_explain.shape
    np.testing.assert_allclose(values, np.asarray(reference), rtol=1e-3, atol=1e-2)
    np.testing.assert_allclose(values.sum(axis=1) + expected, model.predict(x_explain), rtol=1e-3, atol=1e-2)
