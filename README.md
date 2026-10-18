# RF_SHAP_ESMValTool (`rf_shap`)

Created by Feini Huang (feini.huang@uv.es)
Python toolkit for diagnosing **input–output relationships** between **Earth-system model fields** (CMIP6-like data cubes, as used with ESMValTool-style workflows) and **observations**. Tree ensembles map collocated predictors \(X\) to a target \(Y\); SHAP and related methods explain that mapping in space and time.

**Repository:** https://github.com/feinihuang9554-max/RF_SHAP_ESMValTool

Typical question: *at each `(time, lat, lon)`, given model-like inputs, how is the observation generated, which predictors matter, and where?*

---

## Table of contents

1. [Scientific workflow](#1-scientific-workflow)
2. [Install](#2-install)
3. [Quick start](#3-quick-start)
4. [Pipeline order](#4-pipeline-order)
5. [Data: formats, cubes, alignment](#5-data-formats-cubes-alignment)
6. [Preprocessing](#6-preprocessing)
7. [X feature construction](#7-x-feature-construction)
8. [Flattening and restoring cubes](#8-flattening-and-restoring-cubes)
9. [Feature selection](#9-feature-selection)
10. [Train / test splits](#10-train--test-splits)
11. [Training-set sampling](#11-training-set-sampling)
12. [Models (including fast RF)](#12-models-including-fast-rf)
13. [Prediction and metrics](#13-prediction-and-metrics)
14. [Explanation: importance, SHAP, ALE, ICE/PDP](#14-explanation-importance-shap-ale-icepdp)
15. [Plotting](#15-plotting)
16. [Outputs](#16-outputs)
17. [Full configuration reference](#17-full-configuration-reference)
18. [Python API (every public function)](#18-python-api-every-public-function)
19. [CLI, examples, tests](#19-cli-examples-tests)
20. [Package layout](#20-package-layout)

---

## 1. Scientific workflow

Model output and observations are **data cubes**, not a spreadsheet. This package:

1. Loads \(X\) (predictors: e.g. `tas`, `pr`, `huss` from CMIP6) and \(Y\) (target: e.g. observed soil moisture).
2. Forces every variable onto dimensions `(time, lat, lon)`.
3. Aligns \(X\) and \(Y\) to the **same timestamps and grid** (one-to-one correspondence).
4. Optionally detrends, deseasonalizes, takes anomalies, or standardizes \(X\) and \(Y\) independently.
5. Builds extra \(X\) options: raw values, time lags, spatial EOF/PCA, table-level PCA.
6. Flattens valid cells to samples, selects features, splits train/test, optionally downsamples train.
7. Fits a tree model (RF, ExtraTrees, LightGBM-RF, XGBoost, …).
8. Predicts, scores (R², RMSE, **ubRMSE**, NSE, KGE, …), **writes predictions back to a cube**.
9. Explains with impurity importance, permutation importance, TreeSHAP / FastTreeSHAP / Kernel / permutation / sampling SHAP, ALE, ICE+PDP.
10. **Writes SHAP values back to a `(time, lat, lon)` cube**, one variable per feature.

---

## 2. Install

Python **≥ 3.9**.

```bash
git clone https://github.com/feinihuang9554-max/RF_SHAP_ESMValTool.git
cd RF_SHAP_ESMValTool
pip install -e ".[all]"
```

| Extra | Packages | What it unlocks |
| --- | --- | --- |
| *(core)* | numpy, pandas, xarray, netCDF4, scikit-learn, scipy, pyyaml, matplotlib | I/O, preprocess, RF / ExtraTrees / HistGBM, metrics, plots |
| `models` | xgboost, lightgbm | `xgboost`, `lightgbm`, `lightgbm_rf` |
| `explain` | shap | TreeSHAP, KernelSHAP, permutation SHAP, sampling SHAP |
| `fastshap` | fasttreeshap | `fast_tree_shap`: fast random-forest SHAP (v1 / v2) |
| `dev` | pytest | unit tests |
| `all` | all of the above | full pipeline |

### ESMValTool

**Start here:** [`START.md`](START.md) (install overlay, smoke test, point `rootpath`, copy a recipe for any CMIP model vs any reference). This overlay is not yet inside the official ESMValGroup package.

Testers who already have **ESMValTool 2.15**:

```bash
conda activate esmvaltool
git clone https://github.com/feinihuang9554-max/RF_SHAP_ESMValTool.git
cd RF_SHAP_ESMValTool
bash esmvaltool/install_links.sh
esmvaltool run esmvaltool/recipes/recipe_rf_shap_demo.yml
```

Climate recipes need a CMOR archive and `~/.config/esmvaltool/config-user.yml`. Details: `START.md` and `esmvaltool/README.md`.

```python
from esmvaltool.rf_shap import DiagnosisPipeline, compare_explanations
from esmvaltool.diag_scripts.rf_shap.models import build_model
from esmvaltool.rf_shap.cube import align_xy
```

Details: `esmvaltool/README.md`.

Entry point after install: `rf-shap -c configs/default.yaml`.

Without installing the package:

```bash
PYTHONPATH=src python -m rf_shap.cli -c configs/default.yaml
```

---

## 3. Quick start

### In-memory synthetic cubes (script)

```bash
PYTHONPATH=src python examples/run_demo.py
```

### Step-by-step plots (Jupyter)

```bash
PYTHONPATH=src jupyter notebook examples/run_demo.ipynb
```

The notebook runs the same demo as the script, but **plots after every stage**: maps, time series, histograms, correlation, train/test coverage, FSCS samples, scatter, error maps, importance bars, SHAP summary + SHAP maps, ALE, ICE+PDP.

### Config-driven (NetCDF / npy / tables)

Edit `configs/default.yaml` (`data.x.path`, `data.y.path`, variable lists) then:

```bash
PYTHONPATH=src python -m rf_shap.cli -c configs/default.yaml
```

### Python

```python
from rf_shap import DiagnosisPipeline, load_config
from rf_shap.synthetic import make_synthetic_cubes

x, y = make_synthetic_cubes()
cfg = load_config("configs/default.yaml")
result = DiagnosisPipeline(cfg).run(x=x, y=y)
result.save("outputs/run")
print(result.metrics)
```

Skip files entirely by passing cubes: `DiagnosisPipeline(cfg).run(x=x, y=y)`.

---

## 4. Pipeline order

YAML (`configs/default.yaml`) or a Python dict drives every switch.

```
load X, Y
  → rename time / lat / lon
  → regrid + align time
  → preprocess X and Y independently
  → build X (raw | lag | EOF-PCA | lag_pca)
  → flatten to SampleTable (drop NaNs)
  → optional table-level PCA
  → train / test split
  → feature selection on TRAIN only
  → subsample train
  → fit model
  → predict test → metrics → restore prediction cube
  → importance, SHAP, ALE, ICE/PDP
  → restore SHAP cube → save artifacts
```

If `split.method` is `time_window`, the pipeline fits **one model per fold** and writes `output_dir/fold_1`, `fold_2`, …

---

## 5. Data: formats, cubes, alignment

Module: `rf_shap.cube.io`, `rf_shap.cube.align`.

### 5.1 Cube geometry (required)

Every variable is a 3-D field:

| Dimension | Meaning |
| --- | --- |
| `time` | time steps (datetime or integer) |
| `lat` | latitude |
| `lon` | longitude |

Internally arrays are transposed to `(time, lat, lon)`. File coordinate names may differ; set:

- `data.time_name` (default `time`)
- `data.lat_name` (default `lat`)
- `data.lon_name` (default `lon`)

After alignment, **X and Y share the same `time`, `lat`, and `lon`**. Each finite grid cell at one time is one sample.

### 5.2 Loaders (`load_cube`)

Dispatch is by path suffix / directory.

| Input | How it is read |
| --- | --- |
| NetCDF `.nc`, `.nc4`, `.cdf` | `xarray.open_dataset`. Keep `data.x.variables` / `data.y.variables`. |
| Directory of `.npy` | `time.npy`, `lat.npy`, `lon.npy` plus one array per variable, shape `(n_time, n_lat, n_lon)`. |
| Archive `.npz` | Same keys as the directory layout. |
| Single `.npy` | Must already be `(time, lat, lon)`. Dummy lat/lon is invented if coordinates are missing — prefer a directory or NPZ for real grids. |
| Table `.csv`, `.txt`, `.parquet`, `.pq` | **Already arranged** with columns `time`, `lat`, `lon` and one column per variable. Duplicate `(time, lat, lon)` rows are **averaged**, then pivoted to a cube. |

Helpers:

- `load_netcdf`, `load_npy`, `load_table` — format-specific.
- `dataset_from_arrays(arrays, time, lat, lon)` — build a cube from numpy.
- `table_to_dataset(frame, variables, ...)` — pivot a table to a cube.

### 5.3 Alignment (`align_xy`)

| Config key | Options | Behaviour |
| --- | --- | --- |
| `data.regrid` | `nearest`, `linear`, `none` | Interpolate the **non-target** cube onto the target lat/lon. |
| `data.join_time` | `inner`, `nearest` | `inner`: timestamps present in **both**. `nearest`: reindex the other cube’s time by nearest stamp. |
| `data.target` | `y` (default), `x` | Whose grid is truth. Default **`y`**: interpolate the **model onto observations** (usual CMIP6-vs-obs diagnosis). `x`: interpolate observations onto the model grid. |

Empty time intersection raises an error. Flattening afterwards drops any cell that is non-finite in **any** X feature or in Y.

### 5.4 Regrid to 1° (`regrid_to_1deg`)

`regrid_to_1deg(data, grid="era5")` remaps a regular latitude/longitude cube onto one global 1° grid with an area-weighted mean. `grid="era5"` uses latitude 90…−90 and longitude 0…359, the same centres as the monthly ERA5 files. The input object is not changed. `regrid_file_to_1deg(source, destination)` writes that result to a new NetCDF file and refuses to replace `source`.

### 5.5 Synthetic data (`make_synthetic_cubes`)

`rf_shap.synthetic.make_synthetic_cubes(n_time=24, n_lat=8, n_lon=10, seed=0)` builds:

- X: `tas`, `pr`, `huss` with a seasonal cycle, lat/lon gradients, and noise
- Y: `obs` as a nonlinear mix of those fields

Used by demos and tests.

---

## 6. Preprocessing

Module: `rf_shap.cube.preprocess`. Config: `preprocess.x` and `preprocess.y` (independent flags, same operators).

Applied **per variable, along `time`, independently at every grid cell**.

| Flag | Function | Behaviour |
| --- | --- | --- |
| `detrend` | `detrend_along_time` | Linear regression `value ~ time_index`; keep residuals. Cells with fewer than 3 finite times stay NaN. |
| `deseasonalize` | `deseasonalize` | Subtract climatology. `climatology: month` (calendar month) or `dayofyear`. If time is not datetime, bins cycle every 12 (month) or `min(365, n_time)` steps. |
| `anomaly` | `anomaly` | Subtract the time mean at each cell. |
| `standardize` | `standardize` | \((x - \mathrm{mean}_t) / \mathrm{std}_t\). Zero std → NaN. |

If several flags are true, order is always:

**detrend → deseasonalize → anomaly → standardize.**

Purpose: stop the forest from fitting a shared trend or seasonal cycle instead of the \(X\)–\(Y\) mapping you want to diagnose.

`preprocess_dataset(ds, options)` applies the flags to every data variable. `preprocess_dataarray` does one field.

---

## 7. X feature construction

Module: `rf_shap.cube.features`. Config: `x_transform`. Cube shape is **preserved** so \(X\) and \(Y\) stay collocated.

| `mode` | What is built |
| --- | --- |
| `raw` | Original predictor fields only. |
| `lag` | Original fields **plus** time-shifted copies. `lags: [1, 2]` creates `tas_lag1`, `tas_lag2`, … The first `max(lags)` times are dropped so lags are defined. |
| `pca` | Spatial EOF **per variable**: PCA on the flattened spatial field, keep `pca_n_components` (integer rank; default **8** if a float is passed here), **reconstruct** the field. Cube unchanged; high-frequency spatial noise is reduced. |
| `lag_pca` | `lag`, then EOF reconstruction. |

Additional keys:

| Key | Meaning |
| --- | --- |
| `lags` | Positive integers, e.g. `[1, 2]`. |
| `pca_n_components` | EOF rank (cube PCA) or sklearn `n_components` (table PCA: int or variance fraction such as `0.95`). |
| `pca_whiten` | Whitening for table PCA. |
| `table_pca` | After flatten, PCA on **feature columns** (not space). Replaces columns with `pc1`, `pc2`, … |

`add_lags(ds, lags)` and `eof_compress(ds, n_components)` can be called directly. `pca_transform_table(X, n_components, whiten, random_state)` is the table-level PCA.

Use cube EOF when you want spatially smoothed fields still on lat/lon. Use `table_pca` for a compact feature basis after stacking variables.

---

## 8. Flattening and restoring cubes

Module: `rf_shap.cube.tabular`.

### `SampleTable`

| Attribute / method | Role |
| --- | --- |
| `X` | `ndarray (n_samples, n_features)` |
| `y` | `ndarray (n_samples,)` |
| `feature_names` | column names |
| `index` | `MultiIndex` of `(time, lat, lon)` |
| `y_name` | target name |
| `extras` | includes `template` (Y cube for scatter-back), selection history, PCA model, … |
| `n_samples` | row count |
| `subset(mask)` | boolean row filter |
| `take(idx)` | integer row index |
| `with_features(X, names)` | replace the feature matrix |
| `select_columns(names)` | keep named columns |
| `to_frame()` | pandas table with time/lat/lon restored as columns |

### Functions

| Function | Role |
| --- | --- |
| `cube_to_table(x, y, y_name=None)` | Stack `(time, lat, lon)` → rows. Drop rows with any NaN in X or Y. |
| `table_to_cube(values, index, template, name)` | Scatter a 1-D vector onto the template cube. Unused cells stay NaN. |
| `values_to_cubes(matrix, index, template, names)` | Same for a 2-D matrix (one cube variable per column) — used for SHAP. |

---

## 9. Feature selection

Module: `rf_shap.features`. Config: `feature_selection.methods` — a **list, run in order, on TRAIN only**. Test columns are then restricted to the survivors.

### 9.1 `collinearity` — `drop_collinear`

Three filters, in this order:

1. **`drop_by_correlation`** — pairwise \(|r|\). If two features exceed `corr_threshold` (default `0.9`), keep the **larger variance**, drop the other.
2. **`drop_by_vif` / `vif_scores`** — for each feature, regress it on the rest; \(\mathrm{VIF} = 1/(1-R^2)\). Iteratively drop the largest VIF until all VIF ≤ `vif_threshold` (default `10`).
3. **`drop_by_linreg`** — if a feature is predicted by the others with \(R^2 \ge\) `linreg_r2_threshold` (default `0.95`), drop it (almost no unique linear information).

### 9.2 `pairwise_importance` — `drop_weaker_of_pairs`

If \(|\mathrm{corr}(i,j)| >\) `corr_threshold` (default `0.8`), fit a small ExtraTrees model and **drop the less important** of the pair (“two related variables, keep the stronger one”).

### 9.3 `permutation` — `permutation_elimination`

Iterative permutation-importance pruning:

1. Fit ExtraTrees.
2. Shuffle each remaining feature `n_repeats` times (default 5).
3. Drop the lowest importance.
4. Repeat up to `max_rounds` (default 20).
5. Stop when every remaining importance ≥ `drop_below` (default `0`).

Slower than collinearity; enable when you need model-based pruning.

### 9.4 `rfe` — `recursive_feature_elimination`

- If `n_features_to_select` is `null` and `cv` is set: **RFECV** (cross-validated count, scoring `r2`).
- Else: **RFE** down to `n_features_to_select` (or half the features if unset).
- `step`: features dropped per round.

`select_features(table, options, seed)` runs the configured list. History is stored in `result.artifacts["selection"]` / `table.extras["selection_history"]`.

---

## 10. Train / test splits

Module: `rf_shap.models.split`. Config: `split`. `test_size` is the approximate test fraction where a fraction is needed.

`split_table(table, options, seed)` returns a `SplitResult(train, test, method, extras)`, or a **list** of those for sliding windows.

### 10.1 `time` — `time_split`

Chronological. Last `test_size` of unique times → test; earlier → train.

Optional `time.train_end`: explicit cutoff. Everything `≤ train_end` is train.

Use this to test **forward-in-time** generalization.

### 10.2 `space` — `space_split`

Hold out a geographic region (`space.mode`, `space.n_blocks`).

| Mode | Behaviour |
| --- | --- |
| `lon_band` | Highest longitudes (fraction `test_size`) are test. |
| `lat_band` | Highest latitudes are test. |
| `checkerboard` | Tile lat/lon into `n_blocks × n_blocks`; even tiles train, odd tiles test. |
| `block` | Same tiling; randomly assign a fraction of blocks to test. |

Use this to test **spatial** generalization.

### 10.3 `time_window` (alias `sliding`) — `time_window_splits`

Several folds:

| Key | Meaning |
| --- | --- |
| `n_splits` | Number of folds |
| `test_span` | Test window length in **time steps** |
| `expanding` | `true`: train = all times before the test window. `false`: train = a window of length `test_span` just before the test window |

### 10.4 `random` — `random_split`

Shuffle samples (cells × times). Mixes time and space; **optimistic** if neighbours are correlated.

### 10.5 `extreme` — `extreme_split`

Tails of Y using `quantile` (default `0.9`): values ≤ \((1-q)\) quantile or ≥ \(q\) quantile are “extreme”.

| `hold_extremes_as` | Meaning |
| --- | --- |
| `test` | Train on the bulk, test on extremes (can the mapping predict tails?). |
| `train` | Train on extremes, test on the bulk. |

---

## 11. Training-set sampling

Module: `rf_shap.models.sampling`. Config: `sampling`. Applied **after the split, on TRAIN only**. `n_samples: null` keeps every training row.

Every design below selects rows that already exist in the training pool. `sample_train(X, y, method, n_samples, ...)` returns integer indices. `return_weights=True` also returns inverse-probability weights for `area` and `extreme`; other designs return `None`. Those weights are not passed into `model.fit` unless you do that yourself.

| `method` | Aliases | Function | What it keeps | Main limit | Reference |
| --- | --- | --- | --- | --- | --- |
| `random` | | `random_sample` | Uniform draw | Common states dominate | |
| `area` | `area_weighted`, `pps` | `area_weighted_sample` | States in proportion to cell area and an optional per-row time weight. Equal-angle area is \(\cos(\mathrm{lat})\). | Rare environments and thresholds are under-sampled | Hansen & Hurwitz (1943); Cochran (1977) |
| `stratified` | | `stratified_sample` | Y quantile bins, drawn in proportion to bin size | Only stratifies the target | Cochran (1977) |
| `process` | `process_stratified`, `env_stratified` | `process_stratified_sample` | Season, covariate bins, or labels you supply (`strata`, `stratum_columns`). `balance: equal` or `proportional`. Datetime `time` with no columns uses DJF/MAM/JJA/SON. | Bins need a scientific reason; too many dimensions leave empty strata, which are skipped | Cochran (1977); Metzger et al. (2005); de Gruijter et al. (2006) |
| `clhs` | `lhs` | `clhs_sample` | Marginal strata of each covariate, the correlation matrix, and class proportions if `categorical` is set. Swap annealing. | Cost grows with iterations; marginal and correlation match is not the full joint distribution | Minasny & McBratney (2006), doi:10.1016/j.cageo.2005.12.009 |
| `fscs` | `cluster`, `coverage` | `fs_coverage_sample` | One existing row nearest each k-means centre in standardized X. \(k = n\_samples\). | Centre only; misses cluster edges and extremes | Brus et al. (2007); Brus (2019), doi:10.1016/j.geoderma.2018.07.036 |
| `cluster_medoid` | `medoid` | `cluster_sample(..., draw="medoid")` | The medoid of each cluster (Kaufman & Rousseeuw). Clusters larger than 800 use the nearest-to-centre row. | Same edge/extreme gap as taking only a centre | Kaufman & Rousseeuw (1990) |
| `cluster_within` | | `cluster_sample(..., draw="within")` | Real rows drawn inside each cluster. `n_clusters` sets the number of clusters. | Depends on scaling, features, and cluster count | Brus (2019) for the partition; draws are stratified across clusters |
| `maximin` | `farthest`, `farthest_point` | `maximin_sample` | Sequential farthest point, starting from the farthest pair when the pool has at most 2500 rows | Favours outliers; does not preserve frequency | Kennard & Stone (1969); Johnson, Moore & Ylvisaker (1990) |
| `kennard_stone` | `ks` | `kennard_stone` | Same farthest-point rule, but the first point is random | Same as maximin | Kennard & Stone (1969) |
| `extreme` | `threshold`, `tail` | `extreme_sample` | A fraction `extreme_fraction` from the tails (`extreme_quantile`, `extreme_side`) or from `threshold` / `threshold_band`, the rest from the complement | Changes the training distribution. Use the returned weights, or `sample_weight`, to recover the population | Coles (2001) |
| `active` | `active_learning`, `pool` | `active_learning_sample` | Batches that maximise extra-trees committee disagreement and are far from rows already chosen. Refit each round (`active_rounds`). | Repeated fits; high uncertainty is not automatically a useful new state | Lewis & Gale (1994); Seung et al. (1992); Abe & Mamitsuka (1998); Settles (2009) |
| `aclhs` | | `aclhs_sample` | cLHS on X, plus a classical variogram of y along `time` and/or lat/lon (`dependence: auto`) | Keeps dependence for this design; it does not replace an independent validation split. Search swaps real rows; it is not the R package's differential-evolution solver | Le & Vargas (2024), doi:10.1016/j.cageo.2024.105539, on top of Minasny & McBratney (2006) |
| `spatial` | | `spatial_sample` | FSCS on `(lat, lon)` | Geographic spread only | Brus (2019) |
| `systematic` | | `systematic_sample` | Evenly spaced indices, random start | Ignores covariates | |
| `none` / `all` | | | No downsampling | | |

`objective_weights` is `(O1, O2 or correlation, O3 or variogram)`. For acLHS each term is divided by its value on the initial sample, so `(1, 1, 1)` balances them. That is not the unnormalised `(10, 1000, 0.001)` example in the `aclhs` R package.

---

## 12. Models (including fast RF)

Module: `rf_shap.models.registry`. Config: `model.name`, `model.params`.

`build_model(name, params, seed)` fills defaults then constructs the estimator. `available_models()` lists backends that import successfully. Top-level `seed` is used as `random_state` unless overridden in `params`.

| `name` | Aliases | Backend | Defaults / role |
| --- | --- | --- | --- |
| `extra_trees` | `fast_rf`, `et` | `sklearn.ensemble.ExtraTreesRegressor` | **Default fast RF.** Random split thresholds; usually much faster than sklearn RF; still has `feature_importances_` and TreeSHAP. Defaults: 300 trees, `n_jobs=-1`, `max_depth=16`, `min_samples_leaf=2`. |
| `rf` | | `RandomForestRegressor` | Classical RF. Defaults: 300 trees, `n_jobs=-1`, `max_depth=16`, `min_samples_leaf=2`, `max_samples=0.7`. |
| `hist_gbm` | `hgb` | `HistGradientBoostingRegressor` | Histogram boosting; strong and fast on large cubes. `max_depth=8`, `learning_rate=0.08`, `max_iter=300`. |
| `xgboost` | | `XGBRegressor` | Requires `xgboost`. `tree_method=hist`, 400 trees, `max_depth=8`. |
| `lightgbm` | | `LGBMRegressor` | Requires `lightgbm`. 400 trees. Often the fastest TreeSHAP among boosters. |
| `lightgbm_rf` | | LightGBM `boosting_type=rf` | **RF mode**: `bagging_freq=1`, `bagging_fraction=0.8`, `feature_fraction=0.8`. Fast RF-like ensemble. |
| `fastforest` | `fast_forest`, `mabforest` | `fast_forest` → ThrunGroup `HistogramRandomForestRegressor` with solver `MAB` | **Optional fast training.** [FastForest](https://github.com/ThrunGroup/FastForest) (Tiwari et al., NeurIPS 2022) bins features and picks splits with MABSplit. Defaults: 100 trees, `max_depth=16`, `num_bins=11`. Not a scikit-learn tree, so TreeSHAP / `fast_tree_shap` stay on `rf` and the other backends; `method: auto` uses a model-agnostic SHAP explainer for this model. |

Any extra key in `model.params` is passed through to the backend (e.g. `n_estimators: 80` in the demo). `fastforest` ignores sklearn-only keys such as `n_jobs` and `max_samples`. It stays unavailable until the FastForest repository root is on `PYTHONPATH`. Do not install that repository's `requirements.txt` into the ESMValTool environment.

Python stand-ins for SoftwareX 2026 **fru** (Rust RF in R): ExtraTrees, LightGBM-RF, HistGBM. `fru` has no stable Python API here.

Explaining a fitted forest is a separate cost from fitting it. `fast_tree_shap` is the fast option for that step: random forest (or ExtraTrees / boosting trees) plus [LinkedIn FastTreeSHAP](https://github.com/linkedin/fasttreeshap) v1 or v2, instead of single-core `shap.TreeExplainer`. See §14.3.

---

## 13. Prediction and metrics

Modules: `rf_shap.predict.metrics`, `rf_shap.predict.reconstruct`.

After `model.predict` on the test table, `regression_metrics(y_true, y_pred)` uses only finite pairs:

| Name | Definition |
| --- | --- |
| `n` | Number of finite test samples |
| `r2` | Coefficient of determination |
| `rmse` | Root mean squared error |
| `mae` | Mean absolute error |
| `bias` | Mean(`pred − obs`) |
| `ubrmse` | Unbiased RMSE: \(\sqrt{\max(\mathrm{rmse}^2 - \mathrm{bias}^2, 0)}\) — scatter after removing mean bias |
| `nse` | Nash–Sutcliffe efficiency |
| `kge` | Kling–Gupta efficiency from Pearson \(r\), variability ratio \(\alpha\), bias ratio \(\beta\) |
| `pearson_r` | Pearson correlation |
| `mape` | Mean absolute percentage error (undefined where \(Y=0\)) |
| `n_train` / `n_test` / `n_features` | Sizes after sampling and selection |

`reconstruct_prediction(pred, table, name="prediction")` scatters predictions onto `table.extras["template"]`.

Config `predict.restore_cube` (default `true`) writes `prediction.nc`. Train cells and dropped NaNs remain NaN.

---

## 14. Explanation: importance, SHAP, ALE, ICE/PDP

Module package: `rf_shap.explain`. Each block can be turned off in `explain`.

### 14.1 Impurity / gain importance — `model_importance`

Uses `model.feature_importances_` (RF, ExtraTrees, XGBoost, LightGBM). Fast ranking of how often a feature reduces error. Biased toward high-cardinality or correlated features.

Config: `explain.rf_importance`. File: `rf_importance.csv`.

### 14.2 Permutation importance — `permutation_scores`

Shuffle one feature at a time on the **test** set, `explain.permutation_n_repeats` times; record the drop in score. Model-agnostic; more reliable under correlation; slower.

Config: `explain.permutation_importance`. File: `permutation_importance.csv`.

### 14.3 SHAP — `compute_shap`, `fast_tree_shap`, `shap_to_cube`

Local attributions: for each explained sample, \(\sum \mathrm{SHAP} + \mathbb{E}[f] \approx \hat{y}\).

Returns `ShapResult(values, expected_value, method, index, feature_names, sample_positions, algorithm)` with `.to_frame()`.

**Fast option for random forest + SHAP:** `fast_tree_shap(model, x, algorithm="v2", n_jobs=-1)`. This calls [LinkedIn FastTreeSHAP](https://github.com/linkedin/fasttreeshap) (Yang, NeurIPS 2021 workshop). scikit-learn random forests have no built-in TreeSHAP, and `shap.TreeExplainer` explains them on one core. FastTreeSHAP v1 and v2 are parallel. On one core the paper reports about 1.5× (v1, same memory) and 2.5× (v2, higher memory). On an 8-core machine, sklearn random forests in that paper were about 7–15× faster than single-core TreeSHAP. `compute_shap(..., method="fasttreeshap")` and the default `method: auto` use this function for tree models, then `shap_to_cube` writes the values back onto `(time, lat, lon)`.

The PyPI wheel targets NumPy 1. Under NumPy 2, build it from source in the same environment: `pip install --no-deps --no-build-isolation --no-binary :all: fasttreeshap`. On macOS the build also needs `brew install libomp`.

The default `feature_perturbation` is `tree_path_dependent`: exact for trees, and it does not take a background set, so a large training cube does not slow the explanation. Set `feature_perturbation: interventional` only when a small background (`background_size`) is required.

| `explain.shap.method` | Algorithm | When to use |
| --- | --- | --- |
| `auto` | `fast_tree_shap` if `fasttreeshap` is installed, else path-dependent TreeSHAP | Default. |
| `fasttreeshap` | `fast_tree_shap` (`algorithm`: `v0`, `v1`, `v2`, `auto`) | Fast RF + SHAP. Requires `pip install fasttreeshap`. |
| `linear_treeshap` | [Linear TreeSHAP](https://github.com/yupbank/linear_tree_shap), exact O(LD) per sklearn tree. `algorithm: v2` uses the numerically stable Chebyshev form. | Another exact option for sklearn forests. A forest value is the mean of its trees. Not used by `auto`. |
| `treeshap` | `shap.TreeExplainer` path-dependent (no background) | Exact tree SHAP without the FastTreeSHAP package. |
| `kernel` | KernelSHAP | Any model; slow. Keep `background_size` and `max_samples` small. |
| `permutation` | `shap.PermutationExplainer` | Model-agnostic permutation SHAP. |
| `sampling` | `shap.SamplingExplainer` | Model-agnostic sampling approximation. |

Speed / quality knobs:

| Key | Effect |
| --- | --- |
| `enabled` | Skip SHAP entirely if `false`. |
| `max_samples` | Explain only this many test rows (random). Other cells are NaN in the SHAP cube. `null` = all test samples. |
| `background_size` | K-means (fallback: random) background size for interventional / model-agnostic SHAP. |
| `n_jobs` | Threads for `fast_tree_shap`. `-1` uses every core. |
| `algorithm` | `v1`, `v2` (default), `auto`, or `v0`. |
| `feature_perturbation` | `tree_path_dependent` (default, fast) or `interventional`. |

Other practical speed-ups: shallower `max_depth`, ExtraTrees or LightGBM instead of a deep sklearn RF, GPU TreeSHAP if your `shap` build supports it.

**Restore to cube:** `shap_to_cube` writes one data variable per feature, dims `(time, lat, lon)`, onto the Y template. Attributes: `shap_method`, `expected_value`. Files: `shap.nc`, `shap_samples.csv`.

### 14.4 ALE — `ale_1d`

**Accumulated Local Effects**, one curve per feature:

1. Bin the feature on a quantile grid (`n_grid`).
2. In each bin, compare predictions when that feature is set to the left vs right edge.
3. Accumulate those local differences and center the curve.

More robust than PDP when features are correlated. Result: `result.artifacts["ale"][feature]` with columns `x`, `ale`.

Config: `explain.ale`.

### 14.5 ICE + PDP — `ice_pdp`

For each feature:

- `pdp_grid`: quantile grid size (default 20).
- ICE: `ice_n_samples` rows; vary only that feature along the grid.
- PDP: mean of the ICE curves.

Result: `result.artifacts["ice_pdp"][feature]` with `grid`, `pdp`, `ice`, `ice_index`.

Config: `explain.ice_pdp`, `explain.ice_n_samples`, `explain.pdp_grid`.

### 14.6 Compare two explanations — `compare_explanations`

`compare_explanations(...)` measures how far one explanation is from a reference. Pass the pieces you have. It returns a dict of tables:

| Key | What is compared |
| --- | --- |
| `ale_distance` | 1D ALE curves (`ale_1d` output). Distance is RMSE after interpolating both curves onto their overlapping x range. |
| `dependence_distance` | Dependence SHAP. Each feature is binned, the bin-mean SHAP is the curve, then the same RMSE is used. |
| `interaction_distance` | Interactive SHAP. For each ordered pair of distinct features, the curve is mean interaction SHAP against the first feature. Obtain the values with `interaction_shap(model, x)`. |
| `slope_difference` | Slope of the dependence-SHAP curve, `slope_other - slope_ref`. The slope is SHAP per unit of x, a sensitivity of that explanation. |

The same pieces are available on their own: `curve_distance`, `ale_curve_distance`, `dependence_curve_distance`, `interaction_curve_distance`, `dependence_slopes`, `dependence_slope_difference`.

---

## 15. Plotting

Module: `rf_shap.plot`. Used by `examples/run_demo.ipynb`. Each function returns a matplotlib `Figure`.

| Function | What it draws |
| --- | --- |
| `plot_cube_maps(ds, time_index, cmap, title)` | One spatial map per variable at a chosen time. |
| `plot_spatial_mean_timeseries(ds, title)` | Domain-mean time series of every variable. |
| `plot_histograms(frame, columns, title)` | Histograms of selected columns. |
| `plot_correlation(X, names, title)` | Annotated correlation heatmap. |
| `plot_split_coverage(train_index, test_index)` | Train vs test points in lon/lat. |
| `plot_split_time(train_index, test_index)` | Which timestamps are train vs test. |
| `plot_sampling_map(all_index, sampled_index)` | All train points vs the downsampled subset. |
| `plot_pred_scatter(y_true, y_pred, metrics)` | Observed vs predicted, optional R²/RMSE/ubRMSE in the title. |
| `plot_error_map(obs, pred, time_index)` | Obs, pred, and `pred−obs` maps (time mean if `time_index` is omitted). |
| `plot_importance_bars(frame, value_col, title)` | Horizontal importance bars (`std` as error bars if present). |
| `plot_shap_summary(shap_values, X, names)` | Beeswarm-style SHAP vs feature value colour. |
| `plot_shap_maps(shap_cube, time_index)` | SHAP fields restored to the cube. |
| `plot_ale(ale_dict)` | ALE curves. |
| `plot_ice_pdp(ice_pdp_dict)` | Thin ICE lines + thick PDP. |

---

## 16. Outputs

`PipelineResult.save(out_dir)` writes:

| File | Content |
| --- | --- |
| `metrics.csv` | All scores in §13 |
| `prediction.nc` | Predicted Y cube |
| `shap.nc` | SHAP cube, one variable per feature |
| `rf_importance.csv` | Tree importance |
| `permutation_importance.csv` | Permutation importance |
| `shap_samples.csv` | SHAP values plus `time`, `lat`, `lon` |

In memory: `result.metrics`, `result.model`, `result.table`, `result.split`, `result.prediction`, `result.prediction_cube`, `result.shap_cube`, `result.artifacts` (`features`, `selection`, `sampled_index`, `shap_result`, `shap_table`, `ale`, `ice_pdp`, …).

---

## 17. Full configuration reference

Canonical file: [`configs/default.yaml`](configs/default.yaml).

| Key | Type / options | Meaning |
| --- | --- | --- |
| `seed` | int | RNG for split, sampling, models, SHAP subset |
| `output_dir` | path | Where `run_pipeline` / `save` write |
| **`data.x.path` / `data.y.path`** | path | Predictor / target files |
| `data.x.variables` / `data.y.variables` | list | Fields to keep |
| `data.time_name`, `lat_name`, `lon_name` | str | Coordinate names in the files |
| `data.regrid` | `nearest` \| `linear` \| `none` | Spatial interpolation |
| `data.join_time` | `inner` \| `nearest` | Time matching |
| `data.target` | `y` \| `x` | Whose grid is truth |
| `preprocess.{x,y}.detrend` | bool | Linear detrend |
| `preprocess.{x,y}.deseasonalize` | bool | Remove climatology |
| `preprocess.{x,y}.anomaly` | bool | Remove time mean |
| `preprocess.{x,y}.standardize` | bool | Z-score in time |
| `preprocess.{x,y}.climatology` | `month` \| `dayofyear` | Season bins |
| `x_transform.mode` | `raw` \| `lag` \| `pca` \| `lag_pca` | How to build X |
| `x_transform.lags` | list[int] | Lag steps |
| `x_transform.pca_n_components` | int or float | EOF rank or table PCA variance |
| `x_transform.pca_whiten` | bool | Table PCA whitening |
| `x_transform.table_pca` | bool | PCA after flatten |
| `feature_selection.methods` | list | `collinearity`, `pairwise_importance`, `permutation`, `rfe` |
| `feature_selection.collinearity.corr_threshold` | float | Pairwise \|r\| drop |
| `feature_selection.collinearity.vif_threshold` | float | Max VIF |
| `feature_selection.collinearity.linreg_r2_threshold` | float | Linear redundancy \(R^2\) |
| `feature_selection.pairwise_importance.corr_threshold` | float | Pair drop threshold |
| `feature_selection.permutation.n_repeats` | int | Shuffle repeats |
| `feature_selection.permutation.drop_below` | float | Keep if importance ≥ this |
| `feature_selection.permutation.max_rounds` | int | Max drop rounds |
| `feature_selection.rfe.n_features_to_select` | int or null | Target count (`null` → RFECV) |
| `feature_selection.rfe.step` | int | Drops per RFE round |
| `feature_selection.rfe.cv` | int | RFECV folds |
| `split.method` | `time` \| `space` \| `time_window` \| `random` \| `extreme` | Protocol |
| `split.test_size` | float | Test fraction |
| `split.time.train_end` | timestamp or null | Explicit time cutoff |
| `split.space.mode` | `lon_band` \| `lat_band` \| `block` \| `checkerboard` | Spatial hold-out |
| `split.space.n_blocks` | int | Tiles along each axis |
| `split.time_window.n_splits` | int | Number of folds |
| `split.time_window.test_span` | int | Test length in time steps |
| `split.time_window.expanding` | bool | Expanding vs rolling train |
| `split.extreme.quantile` | float | Tail definition |
| `split.extreme.hold_extremes_as` | `test` \| `train` | Where tails go |
| `sampling.method` | see §11 | Downsample train. Default remains `random`. |
| `sampling.n_samples` | int or null | Subset size (`null` = all) |
| `sampling.n_bins` | int | Stratified Y bins, or process-stratum bins |
| `sampling.n_clusters` | int | Clusters for `cluster_within` |
| `sampling.balance` | `equal` \| `proportional` | Allocation inside process strata |
| `sampling.extreme_fraction` | float | Share of an extreme design taken from the tail or threshold pool |
| `sampling.extreme_quantile` | float | Tail probability for `extreme` |
| `sampling.n_iter` | int | Annealing steps for `clhs` and `aclhs` |
| `sampling.dependence` | `auto` \| `time` \| `space` \| `both` | Coordinate used by the acLHS variogram |
| `model.name` | see §12 | Estimator |
| `model.params` | dict | Passed to the backend |
| `predict.restore_cube` | bool | Write `prediction.nc` |
| `explain.rf_importance` | bool | Impurity importance |
| `explain.permutation_importance` | bool | Permutation importance |
| `explain.permutation_n_repeats` | int | Permutation repeats |
| `explain.shap.enabled` | bool | Compute SHAP |
| `explain.shap.method` | `auto` \| `treeshap` \| `fasttreeshap` \| `linear_treeshap` \| `kernel` \| `permutation` \| `sampling` | SHAP algorithm. `fasttreeshap` calls `fast_tree_shap`. `linear_treeshap` calls `linear_tree_shap`. |
| `explain.shap.algorithm` | `v0` \| `v1` \| `v2` \| `auto` | FastTreeSHAP variant. `v2` is the fast default for random forests. |
| `explain.shap.feature_perturbation` | `tree_path_dependent` \| `interventional` | Path-dependent is the fast exact tree option. |
| `explain.shap.max_samples` | int or null | SHAP subsample |
| `explain.shap.background_size` | int | Background size |
| `explain.shap.n_jobs` | int | FastTreeSHAP threads |
| `explain.ale` | bool | ALE curves |
| `explain.ice_pdp` | bool | ICE + PDP |
| `explain.ice_n_samples` | int | ICE curves |
| `explain.pdp_grid` | int | PDP/ALE grid size |

`PipelineConfig` (`rf_shap.config`): `from_yaml`, `from_dict`, `get(*keys, default=)`, `updated(override)`, properties `seed` and `output_dir`. `load_config(path, override)` merges a YAML file with an optional dict.

---

## 18. Python API (every public function)

### Orchestration

| Symbol | Role |
| --- | --- |
| `DiagnosisPipeline(config)` | `.load_xy()`, `.prepare_table(x, y)`, `.run(x=None, y=None)` |
| `run_pipeline(config_path, override, x, y)` | Load config, run, save under `output_dir` |
| `PipelineResult` | metrics, model, tables, cubes, `.save(dir)` |
| `rf_shap.cli.main` | argparse `-c/--config` |

### Cube I/O and features

`load_cube`, `load_netcdf`, `load_npy`, `load_table`, `dataset_from_arrays`, `table_to_dataset`, `align_xy`, `one_degree_coordinates`, `regrid_to_1deg`, `regrid_file_to_1deg`, `preprocess_dataset`, `preprocess_dataarray`, `detrend_along_time`, `deseasonalize`, `anomaly`, `standardize`, `build_x_features`, `add_lags`, `eof_compress`, `pca_transform_table`, `cube_to_table`, `table_to_cube`, `values_to_cubes`, `SampleTable`.

### Feature selection

`select_features`, `drop_collinear`, `drop_by_correlation`, `drop_by_vif`, `vif_scores`, `drop_by_linreg`, `drop_weaker_of_pairs`, `permutation_elimination`, `recursive_feature_elimination`.

### Models / split / sampling

`build_model`, `available_models`, `fast_forest`, `split_table`, `time_split`, `space_split`, `time_window_splits`, `random_split`, `extreme_split`, `SplitResult`, `sample_train`, `random_sample`, `stratified_sample`, `cluster_sample`, `fs_coverage_sample`, `clhs_sample`, `kennard_stone`, `maximin_sample`, `area_weighted_sample`, `process_stratified_sample`, `extreme_sample`, `active_learning_sample`, `aclhs_sample`, `pps_weights`, `spatial_sample`, `systematic_sample`.

### Predict / explain / plot / util

`regression_metrics`, `reconstruct_prediction`, `model_importance`, `permutation_scores`, `compute_shap`, `fast_tree_shap`, `shap_to_cube`, `ShapResult`, `ale_1d`, `ice_pdp`, `curve_distance`, `binned_dependence`, `ale_curve_distance`, `dependence_curve_distance`, `dependence_slopes`, `dependence_slope_difference`, `interaction_curve_distance`, `interaction_shap`, `compare_explanations`, all `plot_*` in §15, `make_synthetic_cubes`, `load_config`, `PipelineConfig`.

---

## 19. CLI, examples, tests

```bash
rf-shap -c configs/default.yaml
# or
PYTHONPATH=src python -m rf_shap.cli -c configs/default.yaml
```

| Path | Role |
| --- | --- |
| `examples/run_demo.py` | End-to-end synthetic run (lag X, time split, FSCS, ExtraTrees, SHAP). |
| `examples/run_demo.ipynb` | Same pipeline, **a figure after every step**. |
| `examples/data/.gitkeep` | Place your NetCDF / npy / tables here. |
| `tests/test_core.py` | Cube round-trip, metrics, split, collinearity, sampling. |
| `tests/test_pipeline.py` | Smoke test of `DiagnosisPipeline` without SHAP. |
| `tests/test_shap.py` | SHAP restore-to-cube. |
| `esmvaltool/` | ESMValTool recipe, diagnostic, install symlink, local runner. |

```bash
PYTHONPATH=src pytest -q
```

---

## 20. Package layout

```
src/rf_shap/
  __init__.py          # PipelineConfig, load_config, DiagnosisPipeline, run_pipeline
  config.py            # YAML / dict config
  cli.py               # command line
  utils.py             # small array helpers
  synthetic.py         # demo cubes
  plot.py              # matplotlib helpers
  pipeline.py          # DiagnosisPipeline, PipelineResult
  cube/                # I/O, align, preprocess, lag/PCA, flatten/restore
  features/            # collinearity, pairwise, permutation drop, RFE
  models/              # factory, sampling, split
  predict/             # metrics, cube reconstruction
  explain/             # importance, SHAP, ALE, ICE/PDP
configs/default.yaml
esmvaltool/           # recipe_rf_shap.yml + diag_scripts/rf_shap
examples/
tests/
```

Public imports: `from rf_shap import DiagnosisPipeline, load_config, PipelineConfig, run_pipeline`.
