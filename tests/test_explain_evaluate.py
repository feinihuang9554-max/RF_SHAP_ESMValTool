import numpy as np
import pandas as pd
import pytest

from rf_shap.explain.evaluate import (
    ale_curve_distance,
    compare_explanations,
    curve_distance,
    dependence_curve_distance,
    dependence_slope_difference,
    interaction_curve_distance,
)


def test_identical_curves_have_zero_distance():
    x = np.linspace(0, 1, 12)
    y = x ** 2
    assert curve_distance(x, y, x[::-1], y[::-1]) == 0.0


def test_vertical_shift_is_the_curve_distance():
    x = np.linspace(-2, 2, 30)
    y = 0.5 * x
    distance = curve_distance(x, y, x, y + 1.5)
    assert distance == pytest.approx(1.5, abs=1e-6)


def test_dependence_slope_difference_and_distance():
    x_ref = np.column_stack([np.linspace(0, 10, 80), np.linspace(-1, 1, 80)])
    shap_ref = np.column_stack([2.0 * x_ref[:, 0], -x_ref[:, 1]])
    x_other = x_ref + 0.01
    shap_other = np.column_stack([5.0 * x_other[:, 0], -x_other[:, 1]])
    slopes = dependence_slope_difference(x_ref, shap_ref, x_other, shap_other, ["vpd", "tas"])
    vpd = slopes.set_index("feature").loc["vpd"]
    tas = slopes.set_index("feature").loc["tas"]
    assert vpd["slope_ref"] == pytest.approx(2.0, abs=0.05)
    assert vpd["slope_other"] == pytest.approx(5.0, abs=0.05)
    assert vpd["difference"] == pytest.approx(3.0, abs=0.1)
    assert tas["difference"] == pytest.approx(0.0, abs=0.05)
    distances = dependence_curve_distance(x_ref, shap_ref, x_ref, shap_ref, ["vpd", "tas"])
    assert distances["distance"].tolist() == [0.0, 0.0]


def test_ale_and_interaction_distances():
    curve = pd.DataFrame({"x": [0.0, 1.0, 2.0], "ale": [0.0, 1.0, 0.0]})
    same = ale_curve_distance({"tas": curve}, {"tas": curve})
    assert same.loc[0, "distance"] == 0.0
    shifted = curve.assign(ale=curve["ale"] + 0.4)
    moved = ale_curve_distance({"tas": curve}, {"tas": shifted})
    assert moved.loc[0, "distance"] == pytest.approx(0.4, abs=1e-6)

    x = np.linspace(0, 1, 40).reshape(-1, 1)
    x = np.column_stack([x[:, 0], 1 - x[:, 0]])
    interaction = np.zeros((len(x), 2, 2))
    interaction[:, 0, 1] = x[:, 0]
    other = interaction.copy()
    other[:, 0, 1] = x[:, 0] + 0.25
    distances = interaction_curve_distance(x, interaction, x, other, ["vpd", "tas"])
    pair = distances.set_index(["feature", "interacts_with"])
    assert pair.loc[("vpd", "tas"), "distance"] == pytest.approx(0.25, abs=1e-6)
    assert pair.loc[("tas", "vpd"), "distance"] == 0.0


def test_compare_explanations_returns_the_requested_tables():
    x = np.column_stack([np.linspace(0, 1, 30), np.linspace(0, 1, 30)])
    shap = np.column_stack([x[:, 0], 0.0 * x[:, 1]])
    compared = compare_explanations(
        x_ref=x,
        shap_ref=shap,
        x_other=x,
        shap_other=shap,
        feature_names=["vpd", "tas"],
    )
    assert set(compared) == {"dependence_distance", "slope_difference"}
    assert compared["dependence_distance"]["distance"].sum() == 0.0
