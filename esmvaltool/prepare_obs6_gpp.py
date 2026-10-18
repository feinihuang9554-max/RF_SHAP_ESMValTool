"""Write FluxCom / ERA5 / GPCP cubes as OBS6 files ESMValTool can find.

Layout (config-user.yml ``rootpath.OBS6`` + ``drs.OBS6: default``)::

    <obs6>/Tier3/ERA5/OBS6_ERA5_reanaly_1_Amon_tas_....nc
    <obs6>/Tier3/FLUXCOM/OBS6_FLUXCOM_sat_1_Lmon_gpp_....nc

Short names match the CMIP GPP recipe (``gpp, tas, hurs, rsds, mrsos, pr``)
so ``reference: {predictor_dataset: ERA5, target_dataset: FLUXCOM}`` works.
``pr`` is GPCP stored under the ERA5 dataset name so every predictor shares
one ``dataset`` facet. ``hurs`` is derived from ERA5 ``t2m`` and ``d2m``.

Example::

    python esmvaltool/prepare_obs6_gpp.py \\
        --src /path/to/vpd_gpp_1deg/obs \\
        --dst /path/to/obs6
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import xarray as xr

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

from rf_shap.cube.align import stamp_year_month
from rf_shap.esmval_bridge import rename_axes

SECONDS_PER_DAY = 86400.0


def obs6_path(root: Path, *, tier: int, dataset: str, type_: str, version: str, mip: str, short_name: str) -> Path:
    name = f"OBS6_{dataset}_{type_}_{version}_{mip}_{short_name}.nc"
    return root / f"Tier{tier}" / dataset / name


def relative_humidity(tas_k: xr.DataArray, td_k: xr.DataArray) -> xr.DataArray:
    """August–Roche–Magnus relative humidity (%) from T and dewpoint (K)."""
    tas_c = tas_k - 273.15
    td_c = td_k - 273.15
    e_air = np.exp((17.625 * tas_c) / (243.04 + tas_c))
    e_dew = np.exp((17.625 * td_c) / (243.04 + td_c))
    rh = 100.0 * (e_dew / e_air)
    return rh.clip(0.0, 100.0)


def _field(path: Path, name: str) -> xr.DataArray:
    ds = rename_axes(xr.open_dataset(path))
    da = ds[name] if name in ds.data_vars else ds[list(ds.data_vars)[0]]
    extra = [d for d in da.dims if d not in ("time", "lat", "lon")]
    if extra:
        da = da.isel({d: 0 for d in extra})
    da = da.transpose("time", "lat", "lon", missing_dims="ignore")
    if "time" in da.coords:
        dummy = da.to_dataset(name=da.name or "z")
        dummy = stamp_year_month(dummy)
        da = dummy[da.name or "z"]
    return da.load()


def _scalar_coord(value: float, *, name: str, standard_name: str, units: str, long_name: str, positive: str, axis: str = "Z") -> xr.DataArray:
    return xr.DataArray(
        np.float64(value),
        name=name,
        attrs={
            "standard_name": standard_name,
            "long_name": long_name,
            "units": units,
            "positive": positive,
            "axis": axis,
        },
    )


def _write(
    da: xr.DataArray,
    dest: Path,
    *,
    standard_name: str,
    units: str,
    long_name: str,
    source: str,
    extra_coords: dict[str, xr.DataArray] | None = None,
    extra_vars: dict[str, xr.DataArray] | None = None,
) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    out = da
    out.attrs.update(standard_name=standard_name, units=units, long_name=long_name)
    ds = out.to_dataset()
    if extra_coords:
        ds = ds.assign_coords(extra_coords)
        out.attrs["coordinates"] = " ".join(extra_coords)
        ds[out.name].attrs["coordinates"] = " ".join(extra_coords)
    if extra_vars:
        for name, var in extra_vars.items():
            ds[name] = var
    ds.attrs.update(source=source, project="OBS6")
    if dest.exists():
        dest.unlink()
    ds.to_netcdf(dest)
    return dest


def build_obs6(src: Path, dst: Path) -> list[Path]:
    tas = _field(src / "era5_monthly_1deg_1980-2014_2m_temperature.nc", "t2m")
    td = _field(src / "era5_monthly_1deg_1980-2014_2m_dewpoint_temperature.nc", "d2m")
    ssrd = _field(src / "era5_monthly_1deg_1980-2014_surface_solar_radiation_downwards.nc", "ssrd")
    swvl1 = _field(src / "era5_monthly_1deg_1980-2014_volumetric_soil_water_layer_1.nc", "swvl1")
    gpp = _field(src / "fluxcom_gpp_monthly_1deg_2001-2020.nc", "gpp")
    pr = _field(src / "gpcp_pr_monthly_1deg_1979-2013.nc", "pr")

    height2m = _scalar_coord(
        2.0,
        name="height",
        standard_name="height",
        units="m",
        long_name="height",
        positive="up",
    )
    depth = _scalar_coord(
        0.05,
        name="depth",
        standard_name="depth",
        units="m",
        long_name="depth",
        positive="down",
    )
    depth.attrs["bounds"] = "depth_bnds"
    depth_bnds = xr.DataArray(np.array([0.0, 0.1], dtype=np.float64), dims=("bnds",), name="depth_bnds")

    hurs = relative_humidity(tas, td).rename("hurs")
    rsds = (ssrd / SECONDS_PER_DAY).rename("rsds")
    # ERA5 swvl1 (m3 m-3) in the top 10 cm → CMOR mrsos (kg m-2).
    mrsos = (swvl1.where(swvl1 >= 0) * 100.0).rename("mrsos")
    tas = tas.rename("tas")
    # GPCP mm/month → kg m-2 s-1 (1 mm = 1 kg m-2).
    days = pr["time"].dt.days_in_month
    pr = (pr / (days * SECONDS_PER_DAY)).rename("pr")
    # FluxCom gC m-2 d-1 → kg m-2 s-1.
    gpp = (gpp * 1.0e-3 / SECONDS_PER_DAY).rename("gpp")

    written = [
        _write(
            tas,
            obs6_path(dst, tier=3, dataset="ERA5", type_="reanaly", version="1", mip="Amon", short_name="tas"),
            standard_name="air_temperature",
            units="K",
            long_name="Near-Surface Air Temperature",
            source="ERA5 monthly 2m temperature, 1 degree",
            extra_coords={"height": height2m},
        ),
        _write(
            hurs,
            obs6_path(dst, tier=3, dataset="ERA5", type_="reanaly", version="1", mip="Amon", short_name="hurs"),
            standard_name="relative_humidity",
            units="%",
            long_name="Near-Surface Relative Humidity",
            source="ERA5 derived from 2m temperature and dewpoint",
            extra_coords={"height": height2m},
        ),
        _write(
            rsds,
            obs6_path(dst, tier=3, dataset="ERA5", type_="reanaly", version="1", mip="Amon", short_name="rsds"),
            standard_name="surface_downwelling_shortwave_flux_in_air",
            units="W m-2",
            long_name="Surface Downwelling Shortwave Radiation",
            source="ERA5 monthly ssrd / 86400",
        ),
        _write(
            mrsos,
            obs6_path(dst, tier=3, dataset="ERA5", type_="reanaly", version="1", mip="Lmon", short_name="mrsos"),
            standard_name="volume_fraction_of_condensed_water_in_soil",
            units="kg m-2",
            long_name="Moisture in Upper Portion of Soil Column",
            source="ERA5 volumetric soil water layer 1 converted to kg m-2 over 10 cm",
            extra_coords={"depth": depth},
            extra_vars={"depth_bnds": depth_bnds},
        ),
        _write(
            pr,
            obs6_path(dst, tier=3, dataset="ERA5", type_="reanaly", version="1", mip="Amon", short_name="pr"),
            standard_name="precipitation_flux",
            units="kg m-2 s-1",
            long_name="Precipitation",
            source="GPCP v2.2 monthly precipitation, packed as ERA5 drivers for one dataset facet",
        ),
        _write(
            gpp,
            obs6_path(dst, tier=3, dataset="FLUXCOM", type_="sat", version="1", mip="Lmon", short_name="gpp"),
            standard_name="gross_primary_productivity_of_biomass_expressed_as_carbon",
            units="kg m-2 s-1",
            long_name="Carbon Mass Flux out of Atmosphere due to Gross Primary Production on Land",
            source="FLUXCOM RS GPP ensemble, converted from gC m-2 d-1",
        ),
    ]
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", type=Path, required=True)
    parser.add_argument("--dst", type=Path, required=True)
    args = parser.parse_args()
    written = build_obs6(args.src, args.dst)
    print(f"wrote {len(written)} OBS6 files under {args.dst}")
    for path in written:
        print(" ", path.relative_to(args.dst))


if __name__ == "__main__":
    main()
