import pytest
from sklearn.ensemble import RandomForestRegressor

from rf_shap.explain.shap_explain import _is_tree_model
from rf_shap.models.fast_forest import FastForestRegressor, fast_forest
from rf_shap.models.registry import available_models, build_model


def test_existing_rf_is_unchanged():
    model = build_model("rf", {"n_estimators": 3, "max_depth": 2, "n_jobs": 1}, seed=0)
    assert isinstance(model, RandomForestRegressor)
    assert model.n_estimators == 3
    assert "rf" in available_models()
    assert "extra_trees" in available_models()


def test_fastforest_is_not_a_treeshap_model():
    model = FastForestRegressor(n_estimators=2, max_depth=2)
    assert model.supports_tree_shap is False
    assert _is_tree_model(model) is False
    assert _is_tree_model(RandomForestRegressor(n_estimators=2)) is True


def test_fast_forest_requires_the_repository_or_builds():
    try:
        import data_structures.wrappers.histogram_random_forest_regressor  # noqa: F401
    except ImportError:
        with pytest.raises(ImportError):
            fast_forest(n_estimators=2)
        assert "fastforest" not in available_models()
        return
    model = build_model("fastforest", {"n_estimators": 2, "max_depth": 2, "n_jobs": 1}, seed=0)
    assert isinstance(model, FastForestRegressor)
    assert "fastforest" in available_models()
