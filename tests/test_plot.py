import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from rf_shap.plot import (
    plot_comparison_bars,
    plot_shap_dependence,
    plot_shap_dependence_overlay,
    plot_shap_maps,
    write_three_column_html,
)


def test_plot_comparison_bars_uses_feature_labels():
    frame = pd.DataFrame({"feature": ["tas", "pr"], "distance": [0.2, 0.05]})
    fig = plot_comparison_bars(frame, "distance", "ALE distance")
    ax = fig.axes[0]
    labels = [t.get_text() for t in ax.get_yticklabels()]
    assert labels == ["pr", "tas"]
    fig.clf()


def test_plot_comparison_bars_joins_interaction_pairs():
    frame = pd.DataFrame(
        {
            "feature": ["tas"],
            "interacts_with": ["pr"],
            "distance": [1.0],
        }
    )
    fig = plot_comparison_bars(frame, "distance", "interaction")
    labels = [t.get_text() for t in fig.axes[0].get_yticklabels()]
    assert labels == ["tas × pr"]
    fig.clf()


def test_plot_shap_dependence_uses_feature_as_x_axis():
    x = np.linspace(0, 1, 40)
    features = np.column_stack([x, 2 * x])
    shap = np.column_stack([x, -x])
    fig = plot_shap_dependence(features, shap, ["tas", "pr"])
    assert fig.axes[0].get_xlabel() == "tas"
    assert "SHAP of tas" in fig.axes[0].get_ylabel()
    fig.clf()


def test_plot_shap_dependence_overlay_shared_features():
    x = np.linspace(0, 1, 30)
    fig = plot_shap_dependence_overlay(
        np.column_stack([x, x]),
        np.column_stack([x, -x]),
        ["tas", "pr"],
        np.column_stack([x]),
        np.column_stack([0.5 * x]),
        ["tas"],
    )
    assert fig.axes[0].get_xlabel() == "tas"
    fig.clf()


def test_write_three_column_html(tmp_path):
    png = tmp_path / "png"
    png.mkdir()
    obs = png / "obs.png"
    cmip = png / "cmip.png"
    compare = png / "compare.png"
    for path in (obs, cmip, compare):
        path.write_bytes(b"x")
    html = write_three_column_html(
        tmp_path / "gallery.html",
        title="gallery",
        obs_label="FLUXCOM / ERA5",
        cmip_label="ACCESS-ESM1-5",
        rows=[{"title": "SHAP dependence", "obs": obs, "cmip": cmip, "compare": compare}],
    )
    text = html.read_text()
    assert "Observations" in text
    assert "png/obs.png" in text
    assert "png/compare.png" in text


def test_plot_shap_maps_time_mean_shows_test_cells():
    time = pd.date_range("2000-01-01", periods=3, freq="MS")
    lat = [0.0, 1.0]
    lon = [10.0, 11.0]
    data = np.full((3, 2, 2), np.nan)
    data[2, 0, 0] = 1e-8
    ds = xr.Dataset(
        {"tas": (("time", "lat", "lon"), data)},
        coords={"time": time, "lat": lat, "lon": lon},
    )
    fig = plot_shap_maps(ds)
    assert fig.axes
    fig.clf()
