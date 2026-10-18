from pathlib import Path

import numpy as np
import xarray as xr

from rf_shap.esmval_bridge import load_xy_from_files, open_field, run_pipeline_on_cubes
from rf_shap.synthetic import make_synthetic_cubes


def test_open_field_renames_latitude(tmp_path: Path):
    time = np.array(["2000-01-01", "2000-02-01"], dtype="datetime64[ns]")
    lat = np.linspace(0, 10, 3)
    lon = np.linspace(0, 20, 4)
    da = xr.DataArray(
        np.ones((2, 3, 4)),
        dims=("time", "latitude", "longitude"),
        coords={"time": time, "latitude": lat, "longitude": lon},
        name="tas",
    )
    path = tmp_path / "tas.nc"
    da.to_dataset().to_netcdf(path)
    out = open_field(path, "tas")
    assert tuple(out.dims) == ("time", "lat", "lon")
    assert out.shape == (2, 3, 4)


def test_load_xy_and_pipeline(tmp_path: Path):
    x, y = make_synthetic_cubes(n_time=12, n_lat=4, n_lon=5, seed=1)
    x_files = {}
    for name, da in x.data_vars.items():
        p = tmp_path / f"{name}.nc"
        da.to_dataset().to_netcdf(p)
        x_files[name] = p
    y_path = tmp_path / "y.nc"
    y.to_netcdf(y_path)
    xx, yy = load_xy_from_files(x_files, y_path, y_name="obs")
    result = run_pipeline_on_cubes(
        xx,
        yy,
        override={
            "sampling": {"n_samples": 80},
            "model": {"name": "extra_trees", "params": {"n_estimators": 8, "max_depth": 4, "n_jobs": 1}},
            "explain": {
                "shap": {"enabled": False},
                "ale": False,
                "ice_pdp": False,
                "permutation_importance": False,
            },
        },
    )
    assert "r2" in result.metrics
    assert result.prediction_cube is not None
