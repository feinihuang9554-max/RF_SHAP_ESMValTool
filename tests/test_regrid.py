from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from rf_shap.cube.regrid import regrid_file_to_1deg, regrid_to_1deg


def _regular_cube(lat, lon, value):
    data = np.full((2, lat.size, lon.size), value, dtype=float)
    data[..., 0] = value
    return xr.Dataset(
        {"tas": (("time", "lat", "lon"), data)},
        coords={"time": [0, 1], "lat": lat, "lon": lon},
    )


def test_constant_field_stays_constant_on_the_1deg_grid():
    lat = np.linspace(-89.0, 89.0, 90)
    lon = np.linspace(0.0, 358.0, 180)
    remapped = regrid_to_1deg(_regular_cube(lat, lon, 3.0))
    assert remapped.sizes["lat"] == 181
    assert remapped.sizes["lon"] == 360
    assert remapped.lat.values[0] == pytest.approx(90.0)
    assert remapped.lat.values[-1] == pytest.approx(-90.0)
    np.testing.assert_allclose(remapped.tas.values, 3.0)


def test_longitude_outside_0_360_still_covers_the_globe():
    lat = np.linspace(-89.0, 89.0, 45)
    lon = np.linspace(-179.0, 179.0, 90)
    remapped = regrid_to_1deg(_regular_cube(lat, lon, 1.5))
    assert np.isfinite(remapped.tas.values).all()
    np.testing.assert_allclose(remapped.tas.values, 1.5)


def test_regrid_file_refuses_to_replace_the_original(tmp_path: Path):
    source = tmp_path / "tas.nc"
    _regular_cube(np.linspace(-89.0, 89.0, 20), np.linspace(0.0, 342.0, 20), 1.0).to_netcdf(source)
    with pytest.raises(ValueError, match="original"):
        regrid_file_to_1deg(source, source)
    destination = tmp_path / "out" / "tas_1deg.nc"
    written = regrid_file_to_1deg(source, destination)
    assert written == destination.resolve()
    assert source.stat().st_size > 0
    with pytest.raises(FileExistsError):
        regrid_file_to_1deg(source, destination)
    with xr.open_dataset(destination) as saved:
        assert saved.sizes["lat"] == 181
        np.testing.assert_allclose(saved.tas.values, 1.0)
