"""The ESMValTool diagnostic compares a reference explanation with another fit."""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_COMPARE = Path(__file__).resolve().parents[1] / "esmvaltool" / "diag_scripts" / "rf_shap" / "compare.py"
_spec = importlib.util.spec_from_file_location("rf_shap_esmval_compare", _COMPARE)
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)
compare_pipeline_results = _module.compare_pipeline_results


class _Shap:
    def __init__(self, values, names):
        self.values = values
        self.feature_names = names
        self.sample_positions = np.arange(len(values))


class _Test:
    def __init__(self, x):
        self.X = x


class _Split:
    def __init__(self, x):
        self.test = _Test(x)


class _Result:
    def __init__(self, x, shap, ale):
        self.split = _Split(x)
        self.artifacts = {"shap_result": _Shap(shap, ["a", "b"]), "ale": ale}
        self.model = None


def test_compare_pipeline_results_writes_curve_and_slope_tables():
    x = np.linspace(0, 1, 30)
    features = np.column_stack([x, x + 0.2])
    shap_ref = np.column_stack([2 * x, -x])
    shap_other = np.column_stack([2 * x + 0.5, -x])
    ale = {
        "a": pd.DataFrame({"x": x, "ale": x}),
        "b": pd.DataFrame({"x": x, "ale": -x}),
    }
    tables = compare_pipeline_results(
        _Result(features, shap_ref, ale),
        _Result(features, shap_other, ale),
        interaction=False,
    )
    assert set(tables) == {"ale_distance", "dependence_distance", "slope_difference"}
    assert tables["ale_distance"]["distance"].max() == pytest.approx(0.0)
    slopes = tables["slope_difference"].set_index("feature")
    assert slopes.loc["a", "difference"] == pytest.approx(0.0)
    assert slopes.loc["b", "difference"] == pytest.approx(0.0)


def test_compare_pipeline_results_uses_shared_features():
    x = np.linspace(0, 1, 20)
    ref = _Result(
        np.column_stack([x, x + 0.1, x + 0.2]),
        np.column_stack([x, -x, 0.5 * x]),
        {
            "a": pd.DataFrame({"x": x, "ale": x}),
            "b": pd.DataFrame({"x": x, "ale": -x}),
            "c": pd.DataFrame({"x": x, "ale": 0.5 * x}),
        },
    )
    ref.artifacts["shap_result"].feature_names = ["a", "b", "c"]
    other = _Result(
        np.column_stack([x, x + 0.2]),
        np.column_stack([x, 0.5 * x]),
        {
            "a": pd.DataFrame({"x": x, "ale": x}),
            "c": pd.DataFrame({"x": x, "ale": 0.5 * x}),
        },
    )
    other.artifacts["shap_result"].feature_names = ["a", "c"]
    tables = compare_pipeline_results(ref, other, interaction=False)
    assert set(tables["dependence_distance"]["feature"]) == {"a", "c"}
