# Start here

This repository is an **overlay** on an existing ESMValTool 2.15 install. It is
**not** yet inside the official `ESMValGroup/ESMValTool` conda package. Testers
clone this repo, link it, then run recipes.

## 0. Can this go into the ESMValTool GitHub organisation?

**Not as an official PR yet.** What you can do now:

| Path | Ready? |
| --- | --- |
| Testers clone `RF_SHAP_ESMValTool` and run recipes | Yes |
| `pip` / conda package of ESMValTool already contains `rf_shap` | No |
| Pull request into `ESMValGroup/ESMValTool` | Not yet |

An official diagnostic still needs: ESMValGroup membership and CLA, files
copied into the ESMValTool tree (not only symlinks), your name as an author
**tag** in `config-references.yml`, Sphinx recipe docs, and a bibtex entry.
Until that lands, send testers this file.

---

## 1. One-time setup (every tester)

You need conda **ESMValTool 2.15** already working (`esmvaltool --help`).

```bash
conda activate esmvaltool
git clone https://github.com/feinihuang9554-max/RF_SHAP_ESMValTool.git
cd RF_SHAP_ESMValTool
bash esmvaltool/install_links.sh
```

Always pass the **recipe file path** (ESMValTool does not search your home
folder for the short name):

```bash
esmvaltool run /full/path/to/RF_SHAP_ESMValTool/esmvaltool/recipes/recipe_rf_shap_demo.yml
```

That smoke test uses synthetic cubes. **No CMIP or observations.** If it
finishes, the overlay is installed.

Check the Python API:

```python
from esmvaltool.rf_shap import DiagnosisPipeline, compare_explanations
```

---

## 2. Point ESMValTool at your data

Copy the example config, then **edit the paths** (do not leave the
`~/esmvaltool_data` placeholders):

```bash
mkdir -p ~/.config/esmvaltool
cp esmvaltool/config-user.example.yml ~/.config/esmvaltool/config-user.yml
```

```yaml
rootpath:
  CMIP6: /path/to/your/cmip6   # ESGF tree, or a flat folder of *_gn_*.nc
  OBS6:  /path/to/your/obs6    # only if you use OBS6 datasets
drs:
  CMIP6: ESGF      # use default if files sit in one folder with CMOR names
  OBS6: default
```

CMOR file names look like
`tas_Amon_ACCESS-ESM1-5_historical_r1i1p1f1_gn_...nc`.
If your download omitted `_gn_`:

```bash
python esmvaltool/prepare_cmor_names.py \
  --src /path/to/raw_model_folder \
  --dst /path/to/cmip6_esmval \
  --grid gn
```

Then set `rootpath.CMIP6` to that `--dst` and `drs.CMIP6: default`.

Observational products must be **OBS / OBS6 CMOR names** (same `short_name`
and `mip` as in the recipe). The GPP example builder is
`python esmvaltool/prepare_obs6_gpp.py --src ... --dst ...`.

Confirm ESMValTool can see a file:

```bash
esmvaltool config show
```

---

## 3. Run a shipped climate recipe

```bash
REPO=/path/to/RF_SHAP_ESMValTool
esmvaltool run $REPO/esmvaltool/recipes/recipe_rf_shap.yml
# ACCESS-CM2: huss, hurs, ps → tas  (no observational reference)

esmvaltool run $REPO/esmvaltool/recipes/recipe_rf_shap_gpp.yml
# ACCESS-ESM1-5 GPP vs FluxCom GPP + ERA5 drivers
```

Outputs: `~/esmvaltool_output/<recipe>_<timestamp>/`

| File | What it is |
| --- | --- |
| `rf_shap_gallery.html` | Three columns: observations, CMIP, comparison |
| `index.html` | ESMValTool default figure list |
| `work/rf_shap/rf_shap/shap.nc` | SHAP restored to `(time, lat, lon)` |
| `work/rf_shap/rf_shap/obs_shap.nc` | Same for the reference fit (if `reference:` is set) |
| `work/rf_shap/rf_shap/*_distance.csv` | ALE / SHAP-dependence distances |

---

## 4. Your own case: any CMIP model vs any reference

Copy `esmvaltool/recipes/recipe_rf_shap_gpp.yml` to a new name. Change four
things only.

### 4.1 The judged model (`datasets:`)

```yaml
datasets:
  - dataset: CESM2          # any CMOR CMIP6 model you have
    project: CMIP6
    exp: historical
    ensemble: r1i1p1f1
    grid: gn
    institute: NCAR
```

Use the institute / ensemble / grid that match **your** files.

### 4.2 Variables (same list for model and reference)

Every predictor and the target must exist for **both** sides.

```yaml
variables:
  gpp:
    mip: Lmon
    preprocessor: rf_shap_cube
    timerange: "2001/2013"
    additional_datasets:
      - dataset: FLUXCOM      # or another CMIP6 model, or OBS6
        project: OBS6
        type: sat
        version: 1
        tier: 3
  tas:
    mip: Amon
    preprocessor: rf_shap_cube
    timerange: "2001/2013"
    additional_datasets:
      - dataset: ERA5
        project: OBS6
        type: reanaly
        version: 1
        tier: 3
  # ... same additional_datasets on every other predictor
```

Two CMIP models (no obs): put the second model in `additional_datasets` with
`project: CMIP6` and the same `exp` / `ensemble` / `grid` as in its files.

### 4.3 Which dataset is X/Y for each fit (`scripts:`)

```yaml
scripts:
  rf_shap:
    script: rf_shap/diagnostic.py
    target: gpp                 # Y short_name
    predictors: [tas, hurs, rsds, mrsos, pr]
    predictor_dataset: CESM2    # judged mapping: this model's X
    target_dataset: CESM2       # judged mapping: this model's Y
    caption: CESM2 GPP mapping versus FluxCom/ERA5.
    reference:                  # baseline mapping (left column on the gallery)
      predictor_dataset: ERA5
      target_dataset: FLUXCOM
      n_bins: 20
      interaction: false
    rf_shap:
      feature_selection:
        methods: []             # keep the same drivers on both fits
```

| You want | `predictor_dataset` / `target_dataset` | `reference` |
| --- | --- | --- |
| One model, no comparison | that model / that model | omit `reference` |
| Model vs observations | model / model | obs drivers / obs target |
| Model A vs model B | A / A | B / B |
| Model X vs observed Y only (no second mapping) | model / obs | omit `reference` |

Omit `reference` if you only want one ExtraTrees + SHAP cube. Keep it if you
want the three-column page (observations or model B | judged CMIP | distances).

### 4.4 Shared settings that usually stay

```yaml
rf_shap:
  seed: 42
  feature_selection:
    methods: []
  x_transform:
    mode: raw
  split:
    method: time
    test_size: 0.2
  sampling:
    method: random
    n_samples: 8000
  model:
    name: extra_trees
    params:
      n_estimators: 120
      max_depth: 12
      n_jobs: -1
  explain:
    shap:
      enabled: true
      method: auto
      max_samples: 1500
```

Run:

```bash
esmvaltool run /path/to/your_new_recipe.yml
```

Open `rf_shap_gallery.html` in the output directory.

---

## 5. What the diagnostic actually does

1. ESMValCore finds files and regrids to 1°.
2. `diagnostic.py` groups preprocessor NetCDFs into cubes X and Y for the
   judged datasets, then calls the original `DiagnosisPipeline`
   (ExtraTrees, SHAP restored to a cube, ALE, ICE/PDP).
3. If `reference:` is set, it does the same for the baseline datasets.
4. `compare_pipeline_results` writes ALE / SHAP-dependence distances and
   slope differences (judged minus reference).
5. Figures: observational (or B) column, CMIP column, overlay / distance
   column, including SHAP dependence (x = driver, y = SHAP of that driver).

---

## 6. Frequent failures

| Log message | Fix |
| --- | --- |
| `Specified recipe file does not exist` | Use the full path under `esmvaltool/recipes/` |
| `Configured 'CMIP6'/'OBS6' rootpaths ... do not exist` | Edit `~/.config/esmvaltool/config-user.yml`; do not recopy the example over a working file |
| `No files were found for Dataset` | Wrong `rootpath`/`drs`, missing `_gn_`, or OBS6 name/mip/tier do not match the file |
| `height2m does not exist` | OBS6 tas needs a 2 m height coordinate (the GPP prep script writes it) |
| Different features on the two fits | Set `rf_shap.feature_selection.methods: []` |
| Gallery missing, only `index.html` | Open `rf_shap_gallery.html` next to `index.html` |
| SHAP maps look empty | Time split leaves early times empty; the gallery uses the time mean of test cells |

---

## 7. Python without a recipe

```bash
conda activate esmvaltool
cd RF_SHAP_ESMValTool
PYTHONPATH=src python examples/run_demo.py
PYTHONPATH=src jupyter notebook examples/esmvaltool_test.ipynb
```
