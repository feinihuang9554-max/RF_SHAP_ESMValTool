import numpy as np
import pytest

from rf_shap.cube.tabular import cube_to_table, table_to_cube
from rf_shap.features.collinearity import drop_by_correlation
from rf_shap.models.sampling import sample_train
from rf_shap.models.split import split_table
from rf_shap.predict.metrics import regression_metrics
from rf_shap.synthetic import make_synthetic_cubes


def test_cube_roundtrip():
    x, y = make_synthetic_cubes(n_time=6, n_lat=4, n_lon=5, seed=1)
    table = cube_to_table(x, y)
    restored = table_to_cube(table.y, table.index, table.extras["template"], name="obs")
    np.testing.assert_allclose(restored.values, y["obs"].values, equal_nan=True)
    assert table.X.shape[1] == 3
    assert table.n_samples == 6 * 4 * 5


def test_metrics_and_split():
    x, y = make_synthetic_cubes(n_time=10, n_lat=3, n_lon=3, seed=2)
    table = cube_to_table(x, y)
    split = split_table(table, {"method": "time", "test_size": 0.3})
    assert split.train.n_samples > 0 and split.test.n_samples > 0
    pred = np.full_like(split.test.y, split.train.y.mean())
    metrics = regression_metrics(split.test.y, pred)
    assert "r2" in metrics and "ubrmse" in metrics


def test_sampling_and_collinearity():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(200, 4))
    x[:, 3] = x[:, 0] * 0.99 + 0.01 * rng.normal(size=200)
    y = x[:, 0] + 0.2 * x[:, 1]
    kept = drop_by_correlation(x, ["a", "b", "c", "d"], threshold=0.9)
    assert "d" not in kept or "a" not in kept
    idx = sample_train(x, y, method="fscs", n_samples=40, seed=0)
    assert len(idx) <= 40
    idx2 = sample_train(x, y, method="stratified", n_samples=50, seed=0)
    assert len(idx2) == 50
