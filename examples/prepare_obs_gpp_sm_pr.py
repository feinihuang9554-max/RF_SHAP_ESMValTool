"""Concatenate, remap, and time-align FluxCom GPP, ESA CCI SM, and GPCP rain.

Raw yearly files stay in ``esm_case_data/obs``. Outputs go to
``vpd_gpp_1deg/obs`` on the ERA5 1° grid with time stamped at YYYY-MM-01.

The three-way overlap is 2001-01 through 2013-12 (GPP starts 2001, GPCP ends
2013). Each product is also written for its full native period.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rf_shap.cube.align import align_month_cubes, stamp_year_month
from rf_shap.cube.regrid import regrid_to_1deg

OBS = Path("/Users/huangfeini/Documents/esm_case_data/obs")
DEST = Path("/Users/huangfeini/Documents/esm_case_data/vpd_gpp_1deg/obs")
LAT_NAMES = ("lat", "latitude")
LON_NAMES = ("lon", "longitude")
TIME_NAMES = ("time", "valid_time")


def _rename_axes(ds: xr.Dataset) -> xr.Dataset:
    mapping = {}
    for names, dest in ((TIME_NAMES, "time"), (LAT_NAMES, "lat"), (LON_NAMES, "lon")):
        for name in names:
            if name in ds.dims or name in ds.coords:
                if name != dest:
                    mapping[name] = dest
                break
    return ds.rename(mapping) if mapping else ds


def _pick_field(ds: xr.Dataset, name: str | None) -> xr.DataArray:
    if name and name in ds.data_vars:
        return ds[name]
    spatial = [v for v in ds.data_vars if {"lat", "lon"} <= set(ds[v].dims)]
    if not spatial:
        raise ValueError(f"No lat/lon field in {list(ds.data_vars)}")
    return ds[spatial[0]]


def _write(dataset: xr.Dataset, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    encoding = {
        name: {"zlib": True, "complevel": 4, "dtype": "float32"}
        for name, variable in dataset.data_vars.items()
        if np.issubdtype(variable.dtype, np.floating)
    }
    dataset.to_netcdf(temporary, encoding=encoding)
    temporary.replace(path)
    size = path.stat().st_size / 1024**2
    t0 = pd.Timestamp(dataset.time.values[0]).strftime("%Y-%m")
    t1 = pd.Timestamp(dataset.time.values[-1]).strftime("%Y-%m")
    print(f"wrote {path.name} {dict(dataset.sizes)} {t0}–{t1} {size:.1f} MB", flush=True)


def process_yearly(
    files: list[Path],
    *,
    source_name: str,
    out_name: str,
    long_name: str,
    units: str,
    source: str,
) -> xr.DataArray:
    pieces = []
    for index, path in enumerate(files, start=1):
        with xr.open_dataset(path) as raw:
            field = _pick_field(_rename_axes(raw), source_name)
            field = stamp_year_month(field.load())
            remapped = regrid_to_1deg(field)
            remapped = remapped.rename(out_name)
            remapped.attrs.update(long_name=long_name, units=units, source=source)
            pieces.append(remapped)
        if index == 1 or index == len(files) or index % 10 == 0:
            print(f"  {out_name} {index}/{len(files)} {path.name}", flush=True)
    cube = xr.concat(pieces, dim="time").sortby("time")
    if pd.Index(cube.time.values).duplicated().any():
        cube = cube.groupby("time").mean(keep_attrs=True)
    cube.attrs.update(source=source, regrid=remapped.attrs.get("regrid"), regrid_grid="era5")
    return cube


def main() -> None:
    DEST.mkdir(parents=True, exist_ok=True)
    print("GPP", flush=True)
    gpp = process_yearly(
        sorted((OBS / "GPP_files").glob("GPP.*.nc")),
        source_name="GPP",
        out_name="gpp",
        long_name="gross primary productivity",
        units="gC m-2 d-1",
        source="FLUXCOM RS GPP ensemble NT/DT (Jung / Weber)",
    )
    _write(gpp.to_dataset(), DEST / "fluxcom_gpp_monthly_1deg_2001-2020.nc")

    print("ESA CCI SM", flush=True)
    sm = process_yearly(
        sorted((OBS / "ESACCI_sm").glob("sm.ESACCI.*.nc")),
        source_name="sm",
        out_name="sm",
        long_name="surface soil moisture",
        units="m3 m-3",
        source="ESA CCI SM v04.4 COMBINED monthly mean of valid daily values",
    )
    _write(sm.to_dataset(), DEST / "esacci_sm_monthly_1deg_1978-2021.nc")

    print("GPCP", flush=True)
    pr = process_yearly(
        sorted((OBS / "GPCP_files").glob("Precip.gpcp_v22.*.nc")),
        source_name="Precip",
        out_name="pr",
        long_name="precipitation",
        units="mm/month",
        source="GPCP v2.2 satellite/gauge combined monthly precipitation",
    )
    _write(pr.to_dataset(), DEST / "gpcp_pr_monthly_1deg_1979-2013.nc")

    aligned = xr.merge(align_month_cubes(gpp.to_dataset(), sm.to_dataset(), pr.to_dataset()))
    aligned.attrs.update(
        comment="Inner join on year-month after area-weighted remap onto the ERA5 1° grid",
        time_coverage="overlap of FluxCom GPP, ESA CCI SM, and GPCP (2001-01 to 2013-12)",
    )
    _write(aligned, DEST / "aligned_gpp_sm_pr_monthly_1deg_2001-2013.nc")
    print("aligned months", int(aligned.sizes["time"]), flush=True)


if __name__ == "__main__":
    main()
