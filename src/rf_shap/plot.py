from __future__ import annotations

from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr


def _style():
    plt.rcParams.update(
        {
            "figure.dpi": 110,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "font.size": 10,
        }
    )


def plot_cube_maps(ds: xr.Dataset, time_index: int = 0, cmap: str = "RdYlBu_r", title: str | None = None):
    """One spatial map per variable at a single time."""
    _style()
    names = list(ds.data_vars)
    n = len(names)
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 3.6), squeeze=False)
    t = ds.time.values[time_index]
    for ax, name in zip(axes[0], names):
        da = ds[name].isel(time=time_index)
        im = ax.pcolormesh(ds.lon, ds.lat, da, shading="auto", cmap=cmap)
        ax.set_xlabel("lon")
        ax.set_ylabel("lat")
        ax.set_title(name)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(title or f"spatial maps @ {pd.to_datetime(t)}")
    fig.tight_layout()
    return fig


def plot_spatial_mean_timeseries(ds: xr.Dataset, title: str = "domain-mean time series"):
    _style()
    fig, ax = plt.subplots(figsize=(8, 3.5))
    for name, da in ds.data_vars.items():
        series = da.mean(("lat", "lon"))
        ax.plot(ds.time, series, label=name)
    ax.set_xlabel("time")
    ax.set_ylabel("value")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    return fig


def plot_histograms(table_like: pd.DataFrame, columns: Sequence[str], title: str = "feature distributions"):
    _style()
    cols = list(columns)
    n = len(cols)
    fig, axes = plt.subplots(1, n, figsize=(3.2 * n, 3.2), squeeze=False)
    for ax, col in zip(axes[0], cols):
        ax.hist(table_like[col].dropna(), bins=25, color="steelblue", edgecolor="white")
        ax.set_title(col)
        ax.set_xlabel(col)
        ax.set_ylabel("count")
    fig.suptitle(title)
    fig.tight_layout()
    return fig


def plot_correlation(x: np.ndarray, names: Sequence[str], title: str = "feature correlation"):
    _style()
    corr = pd.DataFrame(x, columns=list(names)).corr()
    fig, ax = plt.subplots(figsize=(5.5, 4.8))
    im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(names)))
    ax.set_yticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_yticklabels(names)
    for i in range(len(names)):
        for j in range(len(names)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046)
    ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_split_coverage(train_index: pd.MultiIndex, test_index: pd.MultiIndex, title: str = "train / test coverage"):
    _style()
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.8))
    for ax, index, label, color in (
        (axes[0], train_index, "train", "tab:blue"),
        (axes[1], test_index, "test", "tab:orange"),
    ):
        ax.scatter(index.get_level_values("lon"), index.get_level_values("lat"), s=8, alpha=0.35, c=color)
        ax.set_xlabel("lon")
        ax.set_ylabel("lat")
        ax.set_title(f"{label} n={len(index)}")
        ax.set_aspect("equal", adjustable="box")
    fig.suptitle(title)
    fig.tight_layout()
    return fig


def plot_split_time(train_index: pd.MultiIndex, test_index: pd.MultiIndex):
    _style()
    fig, ax = plt.subplots(figsize=(8, 2.8))
    tr = pd.to_datetime(np.unique(train_index.get_level_values("time")))
    te = pd.to_datetime(np.unique(test_index.get_level_values("time")))
    ax.scatter(tr, np.zeros(len(tr)), marker="|", s=400, label="train")
    ax.scatter(te, np.ones(len(te)), marker="|", s=400, label="test")
    ax.set_yticks([0, 1], ["train", "test"])
    ax.set_xlabel("time")
    ax.set_title("time split")
    ax.legend(loc="upper left")
    fig.tight_layout()
    return fig


def plot_sampling_map(all_index: pd.MultiIndex, sampled_index: pd.MultiIndex, title: str = "training sample subset"):
    _style()
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.scatter(all_index.get_level_values("lon"), all_index.get_level_values("lat"), s=6, alpha=0.15, c="gray", label="all train")
    ax.scatter(sampled_index.get_level_values("lon"), sampled_index.get_level_values("lat"), s=12, alpha=0.6, c="crimson", label="sampled")
    ax.set_xlabel("lon")
    ax.set_ylabel("lat")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    return fig


def plot_pred_scatter(y_true, y_pred, metrics: dict | None = None):
    _style()
    yt = np.asarray(y_true).ravel()
    yp = np.asarray(y_pred).ravel()
    fig, ax = plt.subplots(figsize=(4.8, 4.8))
    ax.scatter(yt, yp, s=10, alpha=0.4)
    lims = [np.nanmin([yt, yp]), np.nanmax([yt, yp])]
    ax.plot(lims, lims, "k--", lw=1)
    ax.set_xlabel("observed Y")
    ax.set_ylabel("predicted Y")
    title = "predicted vs observed"
    if metrics:
        title += f"\nR$^2$={metrics['r2']:.3f}  RMSE={metrics['rmse']:.3f}  ubRMSE={metrics['ubrmse']:.3f}"
    ax.set_title(title)
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    return fig


def plot_error_map(obs: xr.DataArray, pred: xr.DataArray, time_index: int | None = None):
    _style()
    if time_index is None:
        obs_m = obs.mean("time")
        pred_m = pred.mean("time")
        subtitle = "time mean"
    else:
        obs_m = obs.isel(time=time_index)
        pred_m = pred.isel(time=time_index)
        subtitle = str(pd.to_datetime(obs.time.values[time_index]))
    err = pred_m - obs_m
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    for ax, da, title, cmap in (
        (axes[0], obs_m, "observed", "viridis"),
        (axes[1], pred_m, "predicted", "viridis"),
        (axes[2], err, "pred − obs", "RdBu_r"),
    ):
        vmax = np.nanpercentile(np.abs(da.values), 98) if title.startswith("pred") and "obs" in title else None
        kw = dict(shading="auto", cmap=cmap)
        if vmax is not None:
            kw.update(vmin=-vmax, vmax=vmax)
        im = ax.pcolormesh(da.lon, da.lat, da, **kw)
        ax.set_title(title)
        ax.set_xlabel("lon")
        ax.set_ylabel("lat")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(f"prediction maps ({subtitle})")
    fig.tight_layout()
    return fig


def plot_importance_bars(frame: pd.DataFrame, value_col: str = "importance", title: str = "feature importance"):
    _style()
    df = frame.sort_values(value_col)
    fig, ax = plt.subplots(figsize=(6, max(2.5, 0.35 * len(df) + 1)))
    ax.barh(df["feature"], df[value_col], color="steelblue", xerr=df["std"] if "std" in df.columns else None)
    ax.set_xlabel(value_col)
    ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_shap_summary(shap_values: np.ndarray, x: np.ndarray, names: Sequence[str], title: str = "SHAP summary"):
    _style()
    names = list(names)
    order = np.argsort(np.abs(shap_values).mean(axis=0))
    fig, ax = plt.subplots(figsize=(6.5, max(2.8, 0.4 * len(names) + 1)))
    rng = np.random.default_rng(0)
    for row, j in enumerate(order):
        vals = shap_values[:, j]
        feat = x[:, j]
        colors = (feat - np.nanmin(feat)) / (np.nanmax(feat) - np.nanmin(feat) + 1e-12)
        jitter = rng.normal(0, 0.08, size=len(vals))
        sc = ax.scatter(vals, np.full(len(vals), row) + jitter, c=colors, cmap="coolwarm", s=8, alpha=0.7, vmin=0, vmax=1)
    ax.set_yticks(range(len(order)), [names[j] for j in order])
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel("SHAP value")
    ax.set_title(title)
    fig.colorbar(sc, ax=ax, fraction=0.03, pad=0.02, label="feature value (low→high)")
    fig.tight_layout()
    return fig


def plot_shap_maps(shap_cube: xr.Dataset, time_index: int = 0, title: str = "SHAP restored to cube"):
    return plot_cube_maps(shap_cube, time_index=min(time_index, shap_cube.sizes["time"] - 1), cmap="RdBu_r", title=title)


def plot_ale(ale: dict, title: str = "Accumulated Local Effects"):
    _style()
    names = list(ale)
    n = len(names)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 3.3), squeeze=False)
    for ax, name in zip(axes[0], names):
        df = ale[name]
        ax.plot(df["x"], df["ale"], color="tab:green")
        ax.axhline(0, color="k", lw=0.6)
        ax.set_xlabel(name)
        ax.set_ylabel("ALE")
        ax.set_title(name)
    fig.suptitle(title)
    fig.tight_layout()
    return fig


def plot_ice_pdp(ice_pdp: dict, title: str = "ICE (thin) + PDP (thick)"):
    _style()
    names = list(ice_pdp)
    n = len(names)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 3.4), squeeze=False)
    for ax, name in zip(axes[0], names):
        pack = ice_pdp[name]
        grid, ice, pdp = pack["grid"], pack["ice"], pack["pdp"]
        for row in ice:
            ax.plot(grid, row, color="0.75", lw=0.7, alpha=0.7)
        ax.plot(grid, pdp, color="tab:red", lw=2.2, label="PDP")
        ax.set_xlabel(name)
        ax.set_ylabel("prediction")
        ax.set_title(name)
        ax.legend()
    fig.suptitle(title)
    fig.tight_layout()
    return fig
