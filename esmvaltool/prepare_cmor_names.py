"""Give flat CMIP NetCDFs the filenames ESMValTool's default DRS expects.

ESMValTool CMIP6 ``input_file`` is
``{short_name}_{mip}_{dataset}_{exp}_{ensemble}_{grid}*.nc``.
Many local copies omit ``_gn_`` (or another grid label). This script creates
symlinks that insert ``grid`` so ``esmvaltool run recipe_rf_shap.yml`` can
find the files when ``drs: CMIP6: default`` and ``rootpath`` points at the
output folder.

Example
-------
    python esmvaltool/prepare_cmor_names.py \\
        --src /path/to/ACCESS-CM2 \\
        --dst /path/to/cmip6_esmval \\
        --grid gn
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

# short_mip_dataset_exp_ensemble_rest.nc  (dataset may contain hyphens)
_NAME = re.compile(
    r"^(?P<short>[^_]+)_(?P<mip>[^_]+)_(?P<dataset>[^_]+)_"
    r"(?P<exp>[^_]+)_(?P<ensemble>[^_]+)_(?P<rest>.+)\.nc$"
)


def cmor_filename(path: Path, grid: str) -> str | None:
    match = _NAME.match(path.name)
    if not match:
        return None
    rest = match.group("rest")
    if rest == grid or rest.startswith(f"{grid}_"):
        return path.name
    return (
        f"{match.group('short')}_{match.group('mip')}_{match.group('dataset')}_"
        f"{match.group('exp')}_{match.group('ensemble')}_{grid}_{rest}.nc"
    )


def link_tree(src: Path, dst: Path, grid: str) -> list[Path]:
    dst.mkdir(parents=True, exist_ok=True)
    created: list[Path] = []
    for path in sorted(src.rglob("*.nc")):
        name = cmor_filename(path, grid)
        if name is None:
            continue
        target = dst / name
        if target.exists() or target.is_symlink():
            target.unlink()
        target.symlink_to(path.resolve())
        created.append(target)
    return created


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", type=Path, required=True)
    parser.add_argument("--dst", type=Path, required=True)
    parser.add_argument("--grid", default="gn")
    args = parser.parse_args()
    created = link_tree(args.src, args.dst, args.grid)
    print(f"linked {len(created)} files into {args.dst}")
    for path in created[:8]:
        print(" ", path.name)
    if len(created) > 8:
        print(f"  ... {len(created) - 8} more")


if __name__ == "__main__":
    main()
