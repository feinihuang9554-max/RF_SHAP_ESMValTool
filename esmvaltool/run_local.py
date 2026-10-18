"""Run the rf_shap ESMValTool diagnostic logic on local files (no CMOR DRS).

Use this when cubes sit in flat folders (e.g. ACCESS-CM2/*.nc) instead of
the ESGF directory tree that `esmvaltool run` expects.

Examples
--------
Synthetic:
    python esmvaltool/run_local.py --synthetic

Local ACCESS-CM2 tas from humidity/pressure:
    python esmvaltool/run_local.py --config esmvaltool/local_access.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rf_shap.esmval_bridge import load_xy_from_files, run_pipeline_on_cubes, save_result_files
from rf_shap.synthetic import make_synthetic_cubes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, help="YAML with x_files/y_file/rf_shap")
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--work-dir", type=Path, default=ROOT / "outputs" / "esmval_local")
    args = parser.parse_args()

    override = {}
    x_files = None
    y_file = None
    y_name = "y"
    if args.config:
        spec = yaml.safe_load(args.config.read_text()) or {}
        x_files = spec.get("x_files")
        y_file = spec.get("y_file")
        y_name = str(spec.get("target_name") or spec.get("target") or "y")
        override = spec.get("rf_shap") or {}
        if spec.get("work_dir"):
            args.work_dir = Path(spec["work_dir"])
    if args.synthetic or not x_files:
        x, y = make_synthetic_cubes(n_time=24, n_lat=8, n_lon=10, seed=7)
    else:
        x, y = load_xy_from_files(x_files, y_file, y_name=y_name)
    result = run_pipeline_on_cubes(x, y, override=override)
    save_result_files(result, args.work_dir)
    print("metrics:", result.metrics)
    print("wrote", args.work_dir)


if __name__ == "__main__":
    main()
