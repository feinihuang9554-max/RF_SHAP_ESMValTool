# rf_shap (English)

Python toolkit for diagnosing **input–output relationships** between **model fields** (CMIP6-like climate/hydrology cubes) and **observations**, using tree ensembles and SHAP-style explanations.

Chinese README: [`README.md`](README.md)

## What the package does

Climate model output and station/gridded observations are rarely used as a single table. This package:

1. Loads X (predictors, e.g. CMIP6 variables) and Y (target, e.g. observed soil moisture or temperature) as **data cubes** with dimensions `(time, lat, lon)`.
2. Aligns X and Y onto the **same timestamps and grid**.
3. Optionally detrends, deseasonalizes, and transforms X (raw / lags / PCA).
4. Flattens valid grid cells into samples, selects features, samples the training set, and fits a tree model.
5. Predicts, scores the fit, and **writes predictions back to a cube**.
6. Explains the fitted mapping (importance, permutation, SHAP, ALE, ICE/PDP) and **writes SHAP values back to a cube**.

Typical scientific question: *given model inputs at each time–space location, how does the model–observation relationship work, which predictors matter, and where?*

---

## Install

```bash
pip install -e ".[all]"
```

| Extra | Packages | Needed for |
| --- | --- | --- |
| (core) | numpy, pandas, xarray, netCDF4, scikit-learn, scipy, pyyaml, matplotlib | load, preprocess, RF/ExtraTrees/HistGBM, metrics |
| `models` | xgboost, lightgbm | boosting and LightGBM random-forest mode |
| `explain` | shap | TreeSHAP, KernelSHAP, permutation/sampling SHAP |
| `fastshap` | fasttreeshap | FastTreeSHAP v1/v2 |
| `dev` | pytest | tests |
| `all` | all of the above | full pipeline |

Python ≥ 3.9.

```bash
PYTHONPATH=src pytest -q
PYTHONPATH=src python examples/run_demo.py
PYTHONPATH=src jupyter notebook examples/run_demo.ipynb   # step-by-step plots
PYTHONPATH=src python -m rf_shap.cli -c configs/default.yaml
```

Call from Python:

```python
from rf_shap import DiagnosisPipeline, load_config
from rf_shap.synthetic import make_synthetic_cubes

x, y = make_synthetic_cubes()
cfg = load_config("configs/default.yaml")
result = DiagnosisPipeline(cfg).run(x=x, y=y)
result.save("outputs/run")
print(result.metrics)
```

You can also pass in-memory cubes and skip file I/O: `DiagnosisPipeline(cfg).run(x=x, y=y)`.

---

## Pipeline order

Everything is driven by YAML (`configs/default.yaml`) or a Python dict.

```
load X, Y
  → rename time/lat/lon
  → regrid + align time
  → preprocess X and Y independently
  → build X features (raw / lag / EOF-PCA)
  → flatten to SampleTable (drop NaNs)
  → optional table-level PCA
  → train/test split
  → feature selection on train
  → subsample train
  → fit model
  → predict test + metrics + restore cube
  → explanations (importance, SHAP, ALE, ICE/PDP)
  → restore SHAP cube + save artifacts
```

If `split.method` is `time_window`, the pipeline fits **one model per fold** and writes `outputs/.../fold_1`, `fold_2`, …

---

## 1. Data requirements

### 1.1 Cube geometry

Every variable must be a 3-D field:

| Dimension | Meaning |
| --- | --- |
| `time` | time steps (datetime or integer index) |
| `lat` | latitude |
| `lon` | longitude |

Arrays are stored internally as `(time, lat, lon)`. Coordinate names can differ in the file; set `data.time_name`, `data.lat_name`, `data.lon_name`.

X and Y must correspond **one-to-one** after alignment: same `time`, `lat`, and `lon`. Each valid grid cell at one time is one sample.

### 1.2 Input formats

`load_cube()` dispatches on path type.

#### NetCDF (`.nc`, `.nc4`, `.cdf`)

- Opened with xarray.
- Must contain time/lat/lon dimensions (or the names given in config).
- `data.x.variables` / `data.y.variables` select which fields to keep (e.g. `tas`, `pr`, `huss` vs `obs`).

#### NumPy directory

Folder containing:

- `time.npy`, `lat.npy`, `lon.npy`
- one `.npy` per variable, shape `(n_time, n_lat, n_lon)`

#### NPZ archive (`.npz`)

Same keys as the directory layout: coordinate arrays plus variable arrays.

#### Single `.npy` file

Must already be `(time, lat, lon)`. A dummy lat/lon grid is assigned if coordinates are not stored in the file. Prefer a directory or NPZ when you have real coordinates.

#### Tables (`.csv`, `.parquet`, `.pq`, `.txt`)

Must **already be arranged** with columns:

- `time`, `lat`, `lon`
- one column per variable

Duplicate `(time, lat, lon)` rows are averaged. The table is pivoted into a cube so the rest of the pipeline is unchanged.

### 1.3 Alignment (`data`)

| Key | Options | What it does |
| --- | --- | --- |
| `regrid` | `nearest`, `linear`, `none` | Spatial interpolation of the non-target cube onto the target grid |
| `join_time` | `inner`, `nearest` | `inner`: keep timestamps present in both cubes. `nearest`: reindex Y time to X (or vice versa) by nearest stamp |
| `target` | `y` (default), `x` | Whose grid is truth. Default `y` interpolates the **model** onto the **observation** grid (usual CMIP6-vs-obs setup) |

After alignment, X and Y have identical `time`, `lat`, `lon` sizes. Flattening then uses only cells that are finite in **all** X features and in Y.

### 1.4 Synthetic cubes

`rf_shap.synthetic.make_synthetic_cubes(n_time, n_lat, n_lon, seed)` builds a small CMIP6-like X (`tas`, `pr`, `huss`) and an observation Y (`obs`) for demos and tests.

---

## 2. Preprocessing (`preprocess`)

X and Y are preprocessed **separately**, but with the same operators. Each operator is applied per variable, along `time`, independently at every grid cell.

| Flag | Function | Behaviour |
| --- | --- | --- |
| `detrend` | Linear detrend | Fit `value ~ time_index` at each cell; keep residuals. Cells with fewer than 3 finite times are left as NaN. |
| `deseasonalize` | Remove climatology | Subtract the mean of each season bin. `climatology: month` uses calendar month; `dayofyear` uses day of year. If time is not a datetime, bins cycle every 12 (month) or `min(365, n_time)` steps. |
| `anomaly` | Remove time mean | Subtract the time-mean at each cell. |
| `standardize` | Z-score in time | `(x − mean_t) / std_t`. Zero std becomes NaN. |

Order if several flags are true: **detrend → deseasonalize → anomaly → standardize**.

These steps are the usual way to stop the model from fitting a shared trend or seasonal cycle instead of the X–Y mapping you want to diagnose.

---

## 3. X feature construction (`x_transform`)

The cube shape is preserved so X and Y stay collocated.

| `mode` | What is built |
| --- | --- |
| `raw` | Original predictor fields only. |
| `lag` | Original fields **plus** time-shifted copies. `lags: [1, 2]` creates `tas_lag1`, `tas_lag2`, … The first `max(lags)` time steps are dropped so lags are defined. |
| `pca` | Spatial EOF compression **per variable**: PCA on the flattened spatial field, keep `pca_n_components` components (integer count when used as EOF rank; default 8 if a float is passed), reconstruct the field. Cube shape unchanged; high-frequency spatial noise is reduced. |
| `lag_pca` | `lag` first, then EOF reconstruction on the lagged cube. |

Additional switch:

| Key | Function |
| --- | --- |
| `table_pca: true` | After flattening to a table, run sklearn PCA on the **feature columns** (not space). Replaces columns with `pc1`, `pc2`, … `pca_n_components` may be an integer or a variance fraction (e.g. `0.95`). `pca_whiten` optional. |

Use cube EOF (`mode: pca`) when you want spatially smoothed fields still on lat/lon. Use `table_pca` when you want a compact feature basis after stacking variables.

---

## 4. Flattening and restoring cubes

`cube_to_table` stacks `(time, lat, lon)` into rows. Each row is one sample with columns = X variables and `y`. Invalid rows (any NaN in X or Y) are dropped.

`SampleTable` stores:

- `X`, `y`, `feature_names`
- a MultiIndex `(time, lat, lon)` for every sample
- `extras["template"]`: original Y cube, used to scatter results back

`table_to_cube` writes a 1-D vector of sample values back onto the template. Unused cells stay NaN. This is used for **predictions** and **per-feature SHAP fields**.

---

## 5. Feature selection (`feature_selection`)

Methods run **in list order** on the **training** table only. The test table is then restricted to the kept columns.

### 5.1 `collinearity`

Three filters, in this order:

1. **Correlation.** Pairwise `|r|`. If two features exceed `corr_threshold` (default `0.9`), keep the one with larger variance, drop the other.
2. **VIF.** For each feature, regress it on all remaining features; `VIF = 1 / (1 − R²)`. Iteratively drop the largest VIF until all VIF ≤ `vif_threshold` (default `10`).
3. **Linear redundancy.** If a feature is predicted by the others with `R² ≥ linreg_r2_threshold` (default `0.95`), drop it (it carries almost no unique linear information).

### 5.2 `pairwise_importance`

If `|corr(i, j)| > corr_threshold` (default `0.8`), fit a small ExtraTrees model and **drop the less important** of the pair. This is the “two related variables, keep the stronger one” rule.

### 5.3 `permutation`

Iterative **permutation-importance elimination**:

- Fit ExtraTrees.
- Shuffle each feature, measure the drop in score (`n_repeats`).
- Drop the lowest-importance feature.
- Repeat up to `max_rounds`.
- Stop when every remaining importance is ≥ `drop_below` (default `0`).

Slower than collinearity; enable only when you need model-based pruning.

### 5.4 `rfe`

Recursive Feature Elimination.

- If `n_features_to_select` is `null` and `cv` is set: **RFECV** (cross-validated number of features, scoring `r2`).
- Otherwise: **RFE** down to `n_features_to_select` (or half the features if unset).
- `step` is how many features to drop per round.

Selection history is stored in `result.artifacts["selection"]`.

---

## 6. Train / test split (`split`)

`test_size` is the approximate test fraction where a fraction is needed.

### 6.1 `time`

Chronological split. The last `test_size` of unique times is test; earlier times are train.

Optional `time.train_end`: explicit cutoff timestamp. Everything `≤ train_end` is train.

Use this to test whether the X–Y mapping generalizes **forward in time**.

### 6.2 `space`

Holds out a geographic region.

| `space.mode` | Behaviour |
| --- | --- |
| `lon_band` | Highest longitudes (fraction `test_size`) are test. |
| `lat_band` | Highest latitudes are test. |
| `checkerboard` | Tile lat/lon into `n_blocks × n_blocks`; even tiles train, odd tiles test. |
| `block` | Same tiling; randomly assign a fraction of blocks to test (`n_blocks`, `test_size`). |

Use this to test **spatial** generalization (no leakage from neighbouring cells in the same block, depending on mode).

### 6.3 `time_window` (alias `sliding`)

Rolling or expanding time folds. Returns **several** splits.

| Key | Meaning |
| --- | --- |
| `n_splits` | Number of folds |
| `test_span` | Length of each test window, in time steps |
| `expanding` | `true`: train = all times before the test window. `false`: train = a window of length `test_span` immediately before the test window |

### 6.4 `random`

Shuffle samples (grid cells × times) into train/test with fraction `test_size`. This **mixes** time and space; optimistic if nearby cells are correlated.

### 6.5 `extreme`

Define tails of Y with `quantile` (default `0.9`): values ≤ `(1 − q)` quantile or ≥ `q` quantile are “extreme”.

| `hold_extremes_as` | Meaning |
| --- | --- |
| `test` | Train on non-extremes, test on extremes (can the mapping predict tails?). |
| `train` | Opposite: train on extremes, test on the bulk. |

---

## 7. Training-set sampling (`sampling`)

Applied **after** the split, **on train only**. `n_samples: null` keeps every training row.

These options cover the usual random draw plus **feature-space coverage** designs used in hydrological ML (conditioned Latin hypercube, Kennard–Stone, FSCS-style clustering), including the sampling-design literature you pointed to.

| `method` | Aliases | What it does |
| --- | --- | --- |
| `random` | | Uniform sample without replacement. |
| `stratified` | | Bin Y into `n_bins` quantile bins; draw from each bin in proportion so rare Y values are not missed. |
| `fscs` | `cluster`, `coverage` | **Feature-space coverage sampling**: standardize X, MiniBatch K-means with `k = n_samples`, keep the point closest to each centroid. Covers predictor space instead of oversampling common climates. |
| `clhs` | `lhs` | Simplified **conditioned Latin hypercube**: each X column is split into `n_samples` strata; swap samples to put about one point in each stratum (greedy / annealing-style). |
| `kennard_stone` | `ks` | Start from a random point plus its farthest neighbour; repeatedly add the point with largest minimum distance to the selected set (space-filling in X). |
| `spatial` | | Same clustering as FSCS but on `(lat, lon)` so the subset is geographically spread. |
| `systematic` | | Evenly spaced indices with a random start (like a 1-D systematic sample). |
| `none` / `all` | | No downsampling. |

`n_bins` is used by stratified sampling. Cluster count for FSCS equals `n_samples` (not `n_clusters` in the current implementation).

---

## 8. Models (`model`)

Set `model.name` and optional sklearn/xgboost/lightgbm `params`. Unspecified hyperparameters use the defaults below. `seed` from the top-level config is used as `random_state` unless you override it.

| `name` | Aliases | Backend | Role |
| --- | --- | --- | --- |
| `extra_trees` | `fast_rf`, `et` | `sklearn.ensemble.ExtraTreesRegressor` | **Default fast random forest.** Random split thresholds; usually much faster than sklearn RF, still gives `feature_importances_` and works with TreeSHAP. |
| `rf` | | `RandomForestRegressor` | Classical RF. Defaults: 300 trees, `n_jobs=-1`, `max_depth=16`, `min_samples_leaf=2`, `max_samples=0.7` (subsample rows per tree). |
| `hist_gbm` | `hgb` | `HistGradientBoostingRegressor` | Histogram boosting; strong and fast on large cubes. Not a forest; use Kernel/permutation SHAP if TreeSHAP is picky. |
| `xgboost` | | `XGBRegressor` | Gradient boosting, `tree_method=hist`. Requires `xgboost`. |
| `lightgbm` | | `LGBMRegressor` | Gradient boosting. Requires `lightgbm`. Often the fastest TreeSHAP among boosters. |
| `lightgbm_rf` | | LightGBM with `boosting_type=rf` | **Random-forest mode** in LightGBM (`bagging_freq=1`, `bagging_fraction=0.8`, `feature_fraction=0.8`). Fast RF-like ensemble. |

Python stand-ins for SoftwareX 2026 **fru** (Rust RF in R): ExtraTrees, LightGBM-RF, and histogram methods. `fru` itself has no stable Python API here.

---

## 9. Prediction (`predict`)

After `model.predict` on the test table:

### 9.1 Metrics (`regression_metrics`)

Computed on finite test pairs:

| Name | Definition |
| --- | --- |
| `n` | Number of finite test samples |
| `r2` | Coefficient of determination |
| `rmse` | Root mean squared error |
| `mae` | Mean absolute error |
| `bias` | Mean(`pred − obs`) |
| `ubrmse` | Unbiased RMSE: `sqrt(max(rmse² − bias², 0))` — scatter after removing mean bias |
| `nse` | Nash–Sutcliffe efficiency (same formula as R² against the observed mean) |
| `kge` | Kling–Gupta efficiency from Pearson r, variability ratio, and bias ratio |
| `pearson_r` | Pearson correlation |
| `mape` | Mean absolute percentage error (undefined where Y = 0) |
| `n_train` / `n_test` / `n_features` | Sizes after sampling and selection |

### 9.2 Restore cube

If `predict.restore_cube: true` (default), test predictions are scattered onto the original `(time, lat, lon)` template → `prediction.nc`. Train cells and dropped NaNs remain NaN.

---

## 10. Explanation (`explain`)

Each block can be turned off independently.

### 10.1 Impurity / gain importance (`rf_importance`)

Uses `model.feature_importances_` (RF, ExtraTrees, XGBoost, LightGBM). Ranking of how often a feature reduces error in the trees. Fast, but biased toward high-cardinality or correlated features.

Saved as `rf_importance.csv`.

### 10.2 Permutation importance (`permutation_importance`)

Shuffle one feature at a time on the **test** set, `permutation_n_repeats` times, and record the drop in score. Model-agnostic; more reliable than impurity importance when predictors are correlated, but slower.

Saved as `permutation_importance.csv`.

### 10.3 SHAP (`explain.shap`)

Local attributions: for each explained sample, `sum(SHAP) + expected_value ≈ prediction`.

| `method` | Algorithm | When to use |
| --- | --- | --- |
| `auto` | FastTreeSHAP if installed, else path-dependent TreeSHAP | Default. |
| `fasttreeshap` | LinkedIn FastTreeSHAP (`algorithm`: `v1`, `v2`, `auto`) | Fastest tree SHAP. v1 ≈ 1.5× TreeSHAP, same memory. v2 ≈ 2.5×, precomputes tree terms, more RAM. |
| `treeshap` | `shap.TreeExplainer` path-dependent (no background) | Exact tree SHAP; usually the fastest **exact** option without FastTreeSHAP. |
| `kernel` | KernelSHAP | Any model; slow. Keep `background_size` and `max_samples` small. |
| `permutation` | `shap.PermutationExplainer` | Model-agnostic permutation SHAP. |
| `sampling` | `shap.SamplingExplainer` | Model-agnostic sampling approximation. |

Speed controls:

| Key | Effect |
| --- | --- |
| `max_samples` | Explain only this many test rows (random). Other cells are NaN in the SHAP cube. `null` = all test samples. |
| `background_size` | Size of the k-means (fallback: random) background for interventional / model-agnostic SHAP. Smaller = faster, noisier. |
| `n_jobs` | Passed to FastTreeSHAP. |
| `algorithm` | FastTreeSHAP only: `v0` (original), `v1`, `v2`, `auto`. |

Other practical speed-ups: shallower trees (`max_depth`), ExtraTrees or LightGBM instead of deep sklearn RF, and GPU TreeSHAP if your `shap` build supports it (same `TreeExplainer` path).

**Restore to cube:** `shap_to_cube` writes one data variable per feature, dims `(time, lat, lon)`, onto the Y template. Attributes store `shap_method` and `expected_value`. Files: `shap.nc` (cube) and `shap_samples.csv` (long table with time/lat/lon).

### 10.4 ALE (`ale`)

**Accumulated Local Effects**, one curve per feature:

- Bin the feature; in each bin compare predictions when that feature is set to the left vs right edge.
- Accumulate those local differences and center the curve.

ALE is more robust than PDP when features are correlated. Result: `result.artifacts["ale"][feature]` with columns `x`, `ale`.

### 10.5 ICE + PDP (`ice_pdp`)

For each feature:

- `pdp_grid`: quantile grid (default 20 points).
- ICE: pick `ice_n_samples` rows, vary only that feature along the grid, record individual curves.
- PDP: average of the ICE curves.

Result: `result.artifacts["ice_pdp"][feature]` with `grid`, `pdp`, `ice`, `ice_index`.

---

## 11. Outputs

`PipelineResult.save(output_dir)` writes:

| File | Content |
| --- | --- |
| `metrics.csv` | All scores in section 9.1 |
| `prediction.nc` | Predicted Y cube |
| `shap.nc` | SHAP cube, one variable per feature |
| `rf_importance.csv` | Tree importance |
| `permutation_importance.csv` | Permutation importance |
| `shap_samples.csv` | SHAP values for explained samples plus coordinates |

In memory you also get `result.model`, `result.table`, `result.split`, `result.prediction`, `result.artifacts` (`features`, `selection`, `sampled_index`, `shap_result`, `ale`, `ice_pdp`).

---

## 12. Configuration reference

See [`configs/default.yaml`](configs/default.yaml). Top-level keys:

| Key | Role |
| --- | --- |
| `seed` | RNG for split, sampling, models, SHAP subset |
| `data` | Paths, variable lists, coordinate names, regrid, time join, target grid |
| `preprocess` | Detrend / season / anomaly / z-score for X and Y |
| `x_transform` | raw / lag / pca / lag_pca, lags, PCA settings, table_pca |
| `feature_selection` | Ordered methods and their thresholds |
| `split` | Train/test protocol |
| `sampling` | Downsample train |
| `model` | Estimator name and params |
| `predict` | Restore prediction cube |
| `explain` | Importance, SHAP, ALE, ICE/PDP |
| `output_dir` | Where `save()` writes |

CLI: `python -m rf_shap.cli -c path/to/config.yaml`.

---

## 13. Package layout

```
src/rf_shap/
  pipeline.py          # DiagnosisPipeline, PipelineResult, run_pipeline
  config.py            # YAML / dict config
  cli.py               # command line
  synthetic.py         # demo cubes
  cube/                # I/O, align, preprocess, lag/PCA, flatten/restore
  features/            # collinearity, pairwise, permutation drop, RFE
  models/              # model factory, sampling, split
  predict/             # metrics, cube reconstruction
  explain/             # importance, SHAP, ALE, ICE/PDP
configs/default.yaml
examples/run_demo.py
tests/
```

Public API: `load_config`, `DiagnosisPipeline`, `run_pipeline`.
