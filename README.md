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
| `fastshap` | fasttreeshap | FastTreeSHAP v1 / v2 |
| `dev` | pytest | unit tests |
| `all` | all of the above | full pipeline |

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

### 5.4 Synthetic data (`make_synthetic_cubes`)

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

These cover i.i.d. draws plus **feature-space coverage** designs used in hydrological ML (FSCS-style clustering, conditioned Latin hypercube, Kennard–Stone), in the spirit of Journal of Hydrology / WRR sampling studies.

`sample_train(X, y, method, n_samples, n_bins, seed, lat, lon)` returns integer indices.

| `method` | Aliases | Function | Behaviour |
| --- | --- | --- | --- |
| `random` | | `random_sample` | Uniform without replacement. |
| `stratified` | | `stratified_sample` | Bin Y into `n_bins` quantile bins; draw from each bin in proportion so rare Y values are not missed. |
| `fscs` | `cluster`, `coverage` | `fs_coverage_sample` / `cluster_sample` | Standardize X, MiniBatch K-means with \(k = n\_samples\), keep the point closest to each centroid. Covers predictor space instead of oversampling common climates. |
| `clhs` | `lhs` | `clhs_sample` | Simplified **conditioned Latin hypercube**: each X column split into `n_samples` strata; swap samples so each stratum is represented (~2000 iterations). |
| `kennard_stone` | `ks` | `kennard_stone` | Start from a random point plus its farthest neighbour; repeatedly add the point with largest minimum distance to the selected set (space-filling in X). |
| `spatial` | | `spatial_sample` | Same clustering as FSCS but on `(lat, lon)` so the subset is geographically spread. |
| `systematic` | | `systematic_sample` | Evenly spaced indices with a random start. |
| `none` / `all` | | | No downsampling. |

Note: FSCS cluster count equals `n_samples`. Config key `n_clusters` is currently unused.

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

Any extra key in `model.params` is passed through to the backend (e.g. `n_estimators: 80` in the demo).

Python stand-ins for SoftwareX 2026 **fru** (Rust RF in R): ExtraTrees, LightGBM-RF, HistGBM. `fru` has no stable Python API here.

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

### 14.3 SHAP — `compute_shap`, `shap_to_cube`

Local attributions: for each explained sample, \(\sum \mathrm{SHAP} + \mathbb{E}[f] \approx \hat{y}\).

Returns `ShapResult(values, expected_value, method, index, feature_names, sample_positions)` with `.to_frame()`.

| `explain.shap.method` | Algorithm | When to use |
| --- | --- | --- |
| `auto` | FastTreeSHAP if installed, else path-dependent TreeSHAP | Default. |
| `fasttreeshap` | LinkedIn FastTreeSHAP (`algorithm`: `v0`, `v1`, `v2`, `auto`) | Fastest tree SHAP. v1 ≈ 1.5× TreeSHAP, same RAM. v2 ≈ 2.5×, precomputes tree terms, more RAM. |
| `treeshap` | `shap.TreeExplainer` path-dependent (no background) | Exact tree SHAP; usually the fastest **exact** option without FastTreeSHAP. |
| `kernel` | KernelSHAP | Any model; slow. Keep `background_size` and `max_samples` small. |
| `permutation` | `shap.PermutationExplainer` | Model-agnostic permutation SHAP. |
| `sampling` | `shap.SamplingExplainer` | Model-agnostic sampling approximation. |

Speed / quality knobs:

| Key | Effect |
| --- | --- |
| `enabled` | Skip SHAP entirely if `false`. |
| `max_samples` | Explain only this many test rows (random). Other cells are NaN in the SHAP cube. `null` = all test samples. |
| `background_size` | K-means (fallback: random) background size for interventional / model-agnostic SHAP. |
| `n_jobs` | Passed to FastTreeSHAP. |
| `algorithm` | FastTreeSHAP only. |

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
| `sampling.method` | `random` \| `stratified` \| `fscs` \| `clhs` \| `kennard_stone` \| `cluster` \| `spatial` \| `systematic` | Downsample train |
| `sampling.n_samples` | int or null | Subset size (`null` = all) |
| `sampling.n_bins` | int | Stratified Y bins |
| `model.name` | see §12 | Estimator |
| `model.params` | dict | Passed to the backend |
| `predict.restore_cube` | bool | Write `prediction.nc` |
| `explain.rf_importance` | bool | Impurity importance |
| `explain.permutation_importance` | bool | Permutation importance |
| `explain.permutation_n_repeats` | int | Permutation repeats |
| `explain.shap.enabled` | bool | Compute SHAP |
| `explain.shap.method` | `auto` \| `treeshap` \| `fasttreeshap` \| `kernel` \| `permutation` \| `sampling` | SHAP algorithm |
| `explain.shap.algorithm` | `v0` \| `v1` \| `v2` \| `auto` | FastTreeSHAP variant |
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

`load_cube`, `load_netcdf`, `load_npy`, `load_table`, `dataset_from_arrays`, `table_to_dataset`, `align_xy`, `preprocess_dataset`, `preprocess_dataarray`, `detrend_along_time`, `deseasonalize`, `anomaly`, `standardize`, `build_x_features`, `add_lags`, `eof_compress`, `pca_transform_table`, `cube_to_table`, `table_to_cube`, `values_to_cubes`, `SampleTable`.

### Feature selection

`select_features`, `drop_collinear`, `drop_by_correlation`, `drop_by_vif`, `vif_scores`, `drop_by_linreg`, `drop_weaker_of_pairs`, `permutation_elimination`, `recursive_feature_elimination`.

### Models / split / sampling

`build_model`, `available_models`, `split_table`, `time_split`, `space_split`, `time_window_splits`, `random_split`, `extreme_split`, `SplitResult`, `sample_train`, `random_sample`, `stratified_sample`, `cluster_sample`, `fs_coverage_sample`, `clhs_sample`, `kennard_stone`, `spatial_sample`, `systematic_sample`.

### Predict / explain / plot / util

`regression_metrics`, `reconstruct_prediction`, `model_importance`, `permutation_scores`, `compute_shap`, `shap_to_cube`, `ShapResult`, `ale_1d`, `ice_pdp`, all `plot_*` in §15, `make_synthetic_cubes`, `load_config`, `PipelineConfig`.

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
examples/
tests/
```

Public imports: `from rf_shap import DiagnosisPipeline, load_config, PipelineConfig, run_pipeline`.
