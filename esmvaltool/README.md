# ESMValTool wrapper for rf_shap

## What an ESMValTool user should run

**New testers:** follow [`../START.md`](../START.md) from clone to an arbitrary
CMIP-versus-reference recipe.

Requires ESMValTool **2.15** in the active conda environment (this overlay is
not yet inside the official ESMValTool conda package).

```bash
conda activate esmvaltool
git clone https://github.com/feinihuang9554-max/RF_SHAP_ESMValTool.git
cd RF_SHAP_ESMValTool
bash esmvaltool/install_links.sh
esmvaltool run recipe_rf_shap_demo.yml
```

That smoke test uses synthetic cubes. It does **not** need CMIP or observations.
If `recipe_rf_shap_demo.yml` is not on ESMValTool's recipe search path, run the
file in this repository:

```bash
esmvaltool run esmvaltool/recipes/recipe_rf_shap_demo.yml
```

Climate recipes (`recipe_rf_shap.yml`, `recipe_rf_shap_gpp.yml`) need your own
CMOR `rootpath`. Copy `config-user.example.yml` to
`~/.config/esmvaltool/config-user.yml` and point `CMIP6` / `OBS6` at your data.
Flattened CMIP filenames: `prepare_cmor_names.py`. FluxCom/ERA5 as OBS6:
`prepare_obs6_gpp.py`.

ESMValTool is normally a **recipe runner**, not a function library. Typical
use is YAML + `esmvaltool run`, which finds CMOR files, preprocesses them,
then executes a diagnostic script. This folder also mounts the `rf_shap`
API under the `esmvaltool` Python package so you can import sub-functions.

## How ESMValTool is usually used

1. Write a **recipe** (datasets, preprocessor, `scripts:` pointing at a diagnostic).
2. Point `config-user.yml` `rootpath` at CMIP6 / obs (not needed for the demo recipe).
3. Run:

```bash
conda activate esmvaltool
esmvaltool run recipe_rf_shap_demo.yml   # no CMIP files; install check
esmvaltool run recipe_rf_shap.yml        # ACCESS-CM2 tas from humidity/pressure
esmvaltool run recipe_rf_shap_gpp.yml    # ACCESS-ESM1-5 GPP from tas/hurs/rsds/mrsos/pr
```

ESMValCore writes one NetCDF per variable; `diag_scripts/rf_shap/diagnostic.py`
loads those files and calls `DiagnosisPipeline`.

## Python import API (after `install_links.sh`)

```python
import esmvaltool

esmvaltool.rf_shap                          # same package as rf_shap
esmvaltool.rf_shap.DiagnosisPipeline
esmvaltool.rf_shap.cube.align_xy
esmvaltool.rf_shap.models.build_model
esmvaltool.rf_shap.explain.compute_shap
esmvaltool.rf_shap.compare_explanations
esmvaltool.rf_shap.features.select_features
esmvaltool.rf_shap.predict.regression_metrics

# diagnostic-script path (ESMValTool layout)
from esmvaltool.diag_scripts.rf_shap import DiagnosisPipeline
from esmvaltool.diag_scripts.rf_shap.models import build_model
from esmvaltool.diag_scripts.rf_shap.cube import align_xy
```

Equivalent explicit imports:

```python
from esmvaltool.rf_shap import DiagnosisPipeline, run_pipeline
from esmvaltool.rf_shap.pipeline import DiagnosisPipeline
from esmvaltool.rf_shap.models.registry import build_model
```

This folder does **not** copy the scientific code into site-packages. It
symlinks the diagnostic, recipe, and a thin `esmvaltool.rf_shap` package
that re-exports `rf_shap`.

## What is connected

| rf_shap capability | How it is exposed |
| --- | --- |
| Load / align cubes | Preprocessor `regrid: 1x1` plus `align_xy` in the diagnostic |
| Detrend, season, lag, PCA, feature selection, split, sampling, models, SHAP, ALE, ICE | Nested YAML under `scripts.rf_shap.rf_shap` (same keys as `configs/default.yaml`) |
| Compare two explanations (ALE / dependence / interaction curve distance, dependence slope) | Optional `reference:` block; writes `ale_distance.csv`, `dependence_distance.csv`, `slope_difference.csv` |
| Prediction and SHAP cubes | `work_dir/prediction.nc`, `work_dir/shap.nc` |
| Maps and scatter | `plot_dir` via ESMValTool `save_figure` |

## Install into the `esmvaltool` conda env

```bash
# from the clone root, with the esmvaltool env active
bash esmvaltool/install_links.sh
# or, if `python` is not that env:
# ESMVALTOOL_PYTHON=/path/to/envs/esmvaltool/bin/python bash esmvaltool/install_links.sh
```

This **symlinks** the diagnostic and recipes and does `pip install -e ".[explain]"`
into that environment. It does not copy scientific code into site-packages.

After this, an ESMValTool user can:

```python
from esmvaltool.rf_shap import DiagnosisPipeline, compare_explanations
```

and `esmvaltool recipes list` includes `recipe_rf_shap.yml`, `recipe_rf_shap_demo.yml`, and `recipe_rf_shap_gpp.yml`.

## Run

**1. Smoke test (any ESMValTool install, no climate files)**

```bash
conda activate esmvaltool
esmvaltool run recipe_rf_shap_demo.yml
```

The diagnostic builds synthetic cubes inside `diagnostic.py` (`use_synthetic: true`). There is no `datasets:` block, so ESMValCore does not search `rootpath`.

**2. CMIP6 example (CMOR filenames)**

ESMValTool looks for
`{short_name}_{mip}_{dataset}_{exp}_{ensemble}_{grid}*.nc`.
If your local copy omitted `_gn_`, create links once:

```bash
python esmvaltool/prepare_cmor_names.py \
  --src /path/to/ACCESS-CM2 \
  --dst /path/to/cmip6_esmval \
  --grid gn
```

Point `rootpath.CMIP6` at that folder and `drs.CMIP6: default` (see `config-user.example.yml`). Users with an ESGF DRS tree set `drs.CMIP6: ESGF` instead.

```bash
esmvaltool run recipe_rf_shap.yml
```

Or with an explicit recipe path:

```bash
esmvaltool run esmvaltool/recipes/recipe_rf_shap.yml
```

Default CMIP recipe: ACCESS-CM2 historical monthly `huss`, `hurs`, `ps` → `tas`
for 1980–2014, 1° linear regrid, ExtraTrees, time split, SHAP restored to a cube.

**3. CMOR GPP case (`recipe_rf_shap_gpp.yml`)**

ACCESS-ESM1-5 `tas`, `hurs`, `rsds`, `mrsos`, `pr` → `gpp` is the judged mapping. The **reference** is FluxCom GPP with ERA5 drivers, written as OBS6 files:

```bash
python esmvaltool/prepare_cmor_names.py \
  --src /path/to/ACCESS-ESM1-5 \
  --dst /path/to/cmip6_esmval \
  --grid gn
python esmvaltool/prepare_obs6_gpp.py \
  --src /path/to/vpd_gpp_1deg/obs \
  --dst /path/to/obs6
esmvaltool run recipe_rf_shap_gpp.yml
```

`variables:` lists both sources (`additional_datasets` on each field). Script section:

```yaml
predictor_dataset: ACCESS-ESM1-5
target_dataset: ACCESS-ESM1-5
reference:
  predictor_dataset: ERA5
  target_dataset: FLUXCOM
```

ERA5 OBS6 `pr` is GPCP packed under the ERA5 dataset name so every predictor shares one facet. `hurs` is derived from ERA5 2 m temperature and dewpoint.

## Local files that are not CMOR/DRS

If ESMValTool cannot see a dataset, pass paths in the script section
(the preprocessor variables can be dropped only if you run the diagnostic
yourself; a normal `esmvaltool run` still needs findable datasets):

```yaml
scripts:
  rf_shap:
    script: rf_shap/diagnostic.py
    target: gpp
    x_files:
      vpd: /path/to/vpd.nc
      tas: /path/to/tas.nc
    y_file: /path/to/gpp.nc
```

Observational Y plus model X: keep both in `variables:` and set

```yaml
predictor_dataset: ACCESS-CM2
target_dataset: ERA5
target: tas
predictors: [huss, hurs, ps]
```

## Compare a process-model explanation with a reference

The script section is the explanation being judged. `reference` is the baseline
(observations, or another dataset). Both fits use the same `rf_shap` settings
unless `reference` overrides them. Curve distances and dependence-slope
differences are written next to the other diagnostic tables.

```yaml
scripts:
  rf_shap:
    script: rf_shap/diagnostic.py
    target: gpp
    predictors: [tas, hurs]
    predictor_dataset: ACCESS-ESM1-5
    target_dataset: ACCESS-ESM1-5
    reference:
      predictor_dataset: ERA5
      target_dataset: GIMMS
      interaction: false
      n_bins: 20
```

`interaction: true` also writes `interaction_distance.csv`. The reference fit
is the baseline: slope difference is the judged fit minus the reference.


## Jupyter test notebook

`examples/esmvaltool_test.ipynb` runs the same load → pipeline → plot path as the diagnostic, including an optional ACCESS-CM2 cell.

```bash
PYTHONPATH=src jupyter notebook examples/esmvaltool_test.ipynb
```

## Synthetic test of the diagnostic (no CMIP)

Preferred: `esmvaltool run recipe_rf_shap_demo.yml` after `install_links.sh`.

The same cubes can also be run outside the recipe runner:

```bash
PYTHONPATH=src python esmvaltool/run_local.py --synthetic
PYTHONPATH=src python examples/run_demo.py
```

## Output

Under `~/esmvaltool_output/<recipe>_<timestamp>/`:

- `work/rf_shap/rf_shap/metrics.csv`
- `work/rf_shap/rf_shap/prediction.nc`
- `work/rf_shap/rf_shap/shap.nc`
- `plots/rf_shap/rf_shap/*.png`
