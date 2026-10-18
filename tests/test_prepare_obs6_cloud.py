import importlib.util
from pathlib import Path

import numpy as np
import xarray as xr

_SPEC = importlib.util.spec_from_file_location(
    "prepare_obs6_cloud",
    Path(__file__).resolve().parents[1] / "esmvaltool" / "prepare_obs6_cloud.py",
)
_MOD = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(_MOD)


def test_fraction_scaled_to_percent():
    da = xr.DataArray([0.0, 0.5, 1.0])
    out = _MOD.as_percent_cloud(da)
    assert float(out.max()) == 100.0
    assert float(out[1]) == 50.0


def test_already_percent_not_rescaled():
    da = xr.DataArray([0.0, 40.0, 100.0])
    out = _MOD.as_percent_cloud(da)
    assert float(out[1]) == 40.0
