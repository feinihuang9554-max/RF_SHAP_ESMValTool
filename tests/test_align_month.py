import numpy as np
import pandas as pd
import xarray as xr

from rf_shap.cube.align import align_month_cubes, stamp_year_month


def _cube(times, name="gpp"):
    lat = np.array([10.0, 9.0])
    lon = np.array([0.0, 1.0])
    values = np.ones((len(times), 2, 2))
    return xr.Dataset(
        {name: (("time", "lat", "lon"), values)},
        coords={"time": np.asarray(times, dtype="datetime64[ns]"), "lat": lat, "lon": lon},
    )


def test_stamp_year_month_collapses_mid_month_dates():
    cube = _cube(["2001-01-16", "2001-02-16"])
    stamped = stamp_year_month(cube)
    assert pd.Timestamp(stamped.time.values[0]) == pd.Timestamp("2001-01-01")
    assert pd.Timestamp(stamped.time.values[1]) == pd.Timestamp("2001-02-01")


def test_align_month_cubes_inner_join():
    gpp = _cube(["2001-01-16", "2001-02-16", "2001-03-16"], "gpp")
    sm = _cube(["2001-02-01", "2001-03-01", "2001-04-01"], "sm")
    pr = _cube(["2001-01-01", "2001-02-01", "2001-03-01"], "pr")
    aligned = xr.merge(align_month_cubes(gpp, sm, pr))
    months = pd.to_datetime(aligned.time.values)
    assert list(months) == [pd.Timestamp("2001-02-01"), pd.Timestamp("2001-03-01")]
    assert set(aligned.data_vars) == {"gpp", "sm", "pr"}
