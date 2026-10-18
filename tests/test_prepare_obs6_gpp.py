import importlib.util
from pathlib import Path

import numpy as np
import xarray as xr

_SPEC = importlib.util.spec_from_file_location(
    "prepare_obs6_gpp",
    Path(__file__).resolve().parents[1] / "esmvaltool" / "prepare_obs6_gpp.py",
)
_MOD = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(_MOD)


def test_obs6_filename():
    path = _MOD.obs6_path(
        Path("/obs6"),
        tier=3,
        dataset="ERA5",
        type_="reanaly",
        version="1",
        mip="Amon",
        short_name="tas",
    )
    assert path == Path("/obs6/Tier3/ERA5/OBS6_ERA5_reanaly_1_Amon_tas.nc")


def test_scalar_height_coord():
    height = _MOD._scalar_coord(
        2.0,
        name="height",
        standard_name="height",
        units="m",
        long_name="height",
        positive="up",
    )
    assert float(height) == 2.0
    assert height.attrs["units"] == "m"


def test_relative_humidity_saturated():
    tas = xr.DataArray([293.15])
    rh = _MOD.relative_humidity(tas, tas)
    assert float(np.asarray(rh).reshape(-1)[0]) == 100.0
