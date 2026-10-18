"""Write ERA5 total cloud cover as OBS6 ``clt`` (CMOR %).

``tas``, ``hurs`` and ``rsds`` are produced by ``prepare_obs6_gpp.py`` and are
reused. ERA5 ``tcc`` is 0–1; CMIP ``clt`` is 0–100.

Example::

    python esmvaltool/prepare_obs6_cloud.py \\
      --src /path/to/esm_case_data/obs \\
      --dst /path/to/esm_case_data/obs6
"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import numpy as np

_GPP = Path(__file__).with_name("prepare_obs6_gpp.py")
_spec = importlib.util.spec_from_file_location("prepare_obs6_gpp", _GPP)
_gpp = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_gpp)


def as_percent_cloud(da):
    """Map 0–1 cloud fraction onto CMOR percent if needed."""
    finite = np.asarray(da.values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    out = da
    if finite.size and float(np.nanmax(finite)) <= 1.5:
        out = da * 100.0
    return out.clip(0.0, 100.0)


def build_obs6_cloud(src: Path, dst: Path) -> list[Path]:
    tcc = _gpp._field(src / "era5_monthly_1deg_1980-2014_total_cloud_cover.nc", "tcc")
    clt = as_percent_cloud(tcc).rename("clt")
    path = _gpp.obs6_path(
        dst, tier=3, dataset="ERA5", type_="reanaly", version="1", mip="Amon", short_name="clt"
    )
    written = _gpp._write(
        clt,
        path,
        standard_name="cloud_area_fraction",
        units="%",
        long_name="Total Cloud Cover",
        source="ERA5 monthly total cloud cover, 1 degree, scaled to percent",
    )
    return [written]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", type=Path, required=True)
    parser.add_argument("--dst", type=Path, required=True)
    args = parser.parse_args()
    written = build_obs6_cloud(args.src, args.dst)
    print(f"wrote {len(written)} OBS6 files under {args.dst}")
    for path in written:
        print(" ", path.relative_to(args.dst))


if __name__ == "__main__":
    main()
