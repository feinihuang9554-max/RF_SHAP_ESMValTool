# rf_shap: a cube-native random-forest and SHAP diagnostic for Earth-system model versus observation relationships

**Manuscript type.** Development and technical paper (Geoscientific Model Development)

**Authors.** Feini Huang<sup>1</sup>  
<sup>1</sup>University of Valencia, Valencia, Spain (feini.huang@uv.es)

**Correspondence.** Feini Huang (feini.huang@uv.es)

**Code.** https://github.com/feinihuang9554-max/RF_SHAP_ESMValTool

---

## Abstract

Diagnosing why an Earth-system model (ESM) field agrees or disagrees with observations requires more than a scalar skill score. The relationship between collocated model predictors \(X\) and an observed target \(Y\) is often nonlinear, spatially heterogeneous, and temporally lagged. We present **rf_shap**, a Python diagnostic that treats CMIP-style NetCDF cubes as first-class objects, learns \(Y = f(X)\) with tree ensembles, and returns explanations that can be mapped back onto the original `(time, latitude, longitude)` grid. The workflow (i) loads model and observation cubes, (ii) remaps them onto a common grid and time axis, (iii) optionally removes trend and seasonality, (iv) constructs lagged or EOF-based predictors, (v) selects features, (vi) splits and subsamples training records, (vii) fits a random forest or related ensemble, and (viii) computes skill metrics, impurity and permutation importance, SHAP, accumulated local effects (ALE), and individual conditional expectation / partial dependence (ICE/PDP). Predictions and per-feature SHAP values are restored as data cubes, which is the form required by ESMValTool-style diagnostics. Fast training (histogram ExtraTrees, LightGBM random-forest mode, optional MABSplit FastForest) and fast TreeSHAP (FastTreeSHAP v1/v2, linear-time SHAP for scikit-learn trees) are provided so that global monthly cubes remain tractable. The design is intended as a reusable ESMValTool diagnostic once a recipe selects the CMOR variables, domain, and period.

## 1 Introduction

ESM evaluation against observations is central to CMIP and to community tools such as ESMValTool (Eyring et al., 2020). Standard diagnostics report bias, RMSE, correlation, and pattern statistics. Those scores do not say *which predictors* drive the residual, *where* they matter, or *how* the mapping changes in the tails of the distribution.

Tree ensembles (Breiman, 2001) are widely used for this mapping because they capture interactions without a prescribed functional form. SHAP values (Lundberg and Lee, 2017; Lundberg et al., 2020) then attribute each prediction to features in a locally additive way. In climate applications the difficulty is not the algorithm in isolation but the **data geometry**: CMIP6 output and satellite or reanalysis products are four-dimensional cubes; samples are grid cells \(\times\) times; missingness, seasonality, spatial dependence, and grid mismatch dominate the workflow.

rf_shap is written around that geometry. Every scientific step is a module that either operates on an `xarray` cube or on a table that still carries a `(time, lat, lon)` index, so that predictions and SHAP fields can be scattered back onto the parent grid. The package is the scientific kernel of a planned ESMValTool diagnostic: the recipe would supply CMOR names, a region, and a time window; rf_shap would return NetCDF skill and explanation cubes.

This paper describes the logic of every component, its scientific purpose, and how the pieces compose. It is a method and software description, not a CMIP evaluation of a particular variable.

## 2 Design principles

1. **Cubes in, cubes out.** Predictors \(X\) and the target \(Y\) are fields with dimensions `(time, lat, lon)`. Flattening to a regression table is an internal step, not the user-facing representation.
2. **One-to-one collocation.** After alignment, each finite cell at one time is one sample. No interpolation across unmatched times is hidden inside the forest.
3. **Train-only selection and sampling.** Feature pruning and training-set designs never see the test fold.
4. **Configurable, not a single black box.** YAML (or a Python dict) chooses regridding, lags, split protocol, sampler, model, and explainer.
5. **Explanations as fields.** SHAP for feature \(j\) is a cube \(S_j(t,\varphi,\lambda)\), so maps and Hovmöller-style summaries are immediate.

The orchestrator is `DiagnosisPipeline` (`src/rf_shap/pipeline.py`). A run is:

```
load X, Y
  → optional 1° area-weighted regrid
  → spatial and temporal alignment
  → preprocess X and Y independently
  → lag / EOF features
  → flatten (drop NaNs)
  → optional table-level PCA
  → train/test split
  → feature selection on train
  → subsample train
  → fit tree model
  → predict, metrics, restore prediction cube
  → importance, SHAP, ALE, ICE/PDP
  → restore SHAP cube and write NetCDF/CSV
```

If the split is a sliding time window, one model is fit per fold.

## 3 Data cubes, I/O, and collocation

### 3.1 Required geometry

Every variable must be a three-dimensional field `(time, lat, lon)`. Coordinate names in files may differ (`latitude`/`longitude`, `valid_time`); they are renamed internally. A sample is one valid grid cell at one time. This matches CMOR-style CMIP6 and most observationally based monthly products.

### 3.2 Loaders (`cube/io.py`)

| Input | Purpose |
| --- | --- |
| NetCDF | Native ESM/observation format; selected by variable list. |
| Directory of `.npy` or a `.npz` archive | Research arrays with explicit `time`, `lat`, `lon` keys. |
| CSV / parquet | Station or already-tidied tables with `time, lat, lon` columns, pivoted to a cube (duplicate coordinates are averaged). |

`load_cube` dispatches on path type so a recipe can point at either CMIP files or a prepared table.

### 3.3 Area-weighted 1° remapping (`cube/regrid.py`)

CMIP models and reanalyses rarely share a grid. `regrid_to_1deg` remaps a regular lat/lon field onto a global 1° grid with a first-order **area weight**: overlap in longitude (including the dateline wrap) times \(\sin\phi\) overlap in latitude. Two target grids are provided: ERA5-like centres (`90, 89, \ldots, -90`; `0, 1, \ldots, 359`) and cell centres (`±89.5`, `0.5\ldots359.5`). The source file is never overwritten (`regrid_file_to_1deg`). Missing source cells contribute zero weight, so land/ocean masks are preserved as NaN on the target.

This step is the conservative, ESMValTool-adjacent alternative to nearest-neighbour interpolation when products must share a common analysis grid before the forest is fit.

### 3.4 Alignment (`cube/align.py`)

After (optional) 1° remapping, `align_xy` puts \(X\) and \(Y\) on one grid and one time axis.

- **Spatial target.** Default `target: y` interpolates the **model onto the observation grid** (usual observation-centric diagnosis). `target: x` does the reverse.
- **Interpolator.** `nearest`, `linear`, or `none`.
- **Time.** `inner` keeps timestamps present in both cubes; `nearest` reindexes by nearest stamp.

Empty overlap is an error. Subsequent flattening drops any cell that is non-finite in any \(X\) feature or in \(Y\).

### 3.5 Synthetic cubes (`synthetic.py`)

`make_synthetic_cubes` builds a small seasonal \(X\) (`tas`, `pr`, `huss`) and a nonlinear observed \(Y\) for tests and the demo notebook. It is not a climate simulation; it only guarantees that the pipeline can be executed without CMIP files.

## 4 Preprocessing and predictor construction

### 4.1 Time-wise filters (`cube/preprocess.py`)

Applied independently to \(X\) and \(Y\), per grid cell, along time, in this order if several flags are on:

1. **Linear detrend** — residuals of \(y \sim t\). Cells with fewer than three finite times stay NaN.
2. **Deseasonalise** — subtract the climatology of calendar month or day-of-year (or a cyclic bin if time is not datetime).
3. **Anomaly** — subtract the time mean.
4. **Standardise** — divide by the time standard deviation.

Purpose: prevent the forest from spending splits on a shared trend or seasonal cycle when the scientific question is the *anomaly* relationship (e.g. VPD versus GPP residuals).

### 4.2 Feature modes (`cube/features.py`)

The cube shape is preserved so collocation is not broken.

| Mode | Logic | Purpose |
| --- | --- | --- |
| `raw` | Original fields. | Baseline contemporaneous mapping. |
| `lag` | Original plus `shift(time=k)` copies; leading times dropped. | Memory (soil moisture, vegetation lag). |
| `pca` | Per-variable spatial EOF, reconstruct with the leading components. | Spatial smoothing / reduced rank while staying on lat/lon. |
| `lag_pca` | Lag then EOF. | Lagged large-scale modes. |
| `table_pca` | After flatten, sklearn PCA on columns. | Compact feature basis when many correlated CMIP variables are stacked. |

## 5 From cubes to a regression table and back

`cube/tabular.py` is the hinge of the design.

- `cube_to_table` stacks `(time, lat, lon)` into rows, stores a `pandas` MultiIndex, and keeps the original \(Y\) cube as `extras["template"]`.
- `SampleTable` supports `subset`, `take`, `select_columns`, and `to_frame`.
- `table_to_cube` / `values_to_cubes` scatter a vector or a matrix of attributions onto the template. Unused cells remain NaN. This is how **predictions** and **SHAP** become ESMValTool-writable NetCDF.

## 6 Feature selection (`features/`)

Methods run **in list order on the training table only**. Test columns are then restricted to the survivors. The purpose is to drop redundant CMIP predictors before SHAP, which otherwise splits credit among collinear fields.

| Method | Logic | Purpose |
| --- | --- | --- |
| `collinearity` | (i) drop the lower-variance member of pairs with \(\lvert r\rvert\) above a threshold; (ii) iterative VIF; (iii) drop features linearly predicted by the others at high \(R^2\). | Remove near-duplicates (e.g. two humidity variables). |
| `pairwise_importance` | If two features are highly correlated, drop the one with lower ExtraTrees importance. | Keep the physically stronger of two related drivers. |
| `permutation` | Iteratively drop the lowest permutation importance. | Model-based pruning. |
| `rfe` / RFECV | Recursive feature elimination, optionally with \(R^2\) cross-validation. | Automatic subset size. |

History is stored so a recipe can report which CMOR names were kept.

## 7 Independence of training and testing (`models/split.py`)

Climate samples are not i.i.d.: nearby cells and consecutive months are dependent. The split module therefore offers protocols that match different scientific claims.

| Protocol | Logic | What it tests |
| --- | --- | --- |
| `time` | Last fraction of unique times (or an explicit `train_end`) held out. | Forward-in-time generalisation. |
| `space` | Longitude or latitude band; checkerboard tiles; random spatial blocks. | Spatial transfer. |
| `time_window` | Expanding or rolling folds of length `test_span`. | Stability across decades. |
| `random` | i.i.d. shuffle of cells \(\times\) times. | Optimistic upper bound (leakage if neighbours are correlated). |
| `extreme` | Tails of \(Y\) (quantile) held out as test or used as train. | Skill on extremes versus the bulk. |

## 8 Training-set designs (`models/sampling.py`)

Global monthly cubes easily exceed \(10^7\) rows. Fitting on every cell is neither necessary nor always desirable: common climates dominate, deserts and high latitudes are over-represented on an equal-angle grid, and extremes are rare. `sample_train` selects **existing rows** (it never invents covariate combinations) and, for some designs, returns inverse-probability weights so later summaries can be put back on the population.

| Method | Logic | Purpose |
| --- | --- | --- |
| `random` | Uniform without replacement. | Baseline. |
| `stratified` | Quantile bins of \(Y\). | Do not miss rare target values. |
| `fscs` / `cluster` | \(k\)-means in standardised \(X\); keep the point nearest each centroid (Brus, 2019). | Cover predictor space. |
| `cluster_medoid` / `cluster_within` | Medoid or within-cluster draws. | Same idea, different representatives. |
| `clhs` | Conditioned Latin hypercube (Minasny and McBratney, 2006): strata occupancy, optional class proportions, correlation structure, swap annealing. | Match the multivariate histogram of \(X\). |
| `aclhs` | cLHS terms plus a classical variogram of \(Y\) in time or space (Le and Vargas, 2024, adapted to pool rows). | Preserve spatial/temporal dependence of the target. |
| `kennard_stone` / `maximin` | Greedy farthest-point in \(X\). | Space-filling designs. |
| `spatial` | Clustering on `(lat, lon)`. | Geographic spread. |
| `systematic` | Equal stride with a random start. | Simple even coverage of the index. |
| `area_weighted` / `pps` | Probability proportional to \(\cos\phi\) (and optional time weights). | Correct equal-angle area bias (Hansen and Hurwitz, 1943; Cochran, 1977). |
| `process_stratified` | Strata from season, categorical columns, or binned \(X\). | Sample by climate process, not by map area. |
| `extreme` / `threshold` | Enrich tails of \(Y\) or a chosen column; optional PPS weights. | Stress the mapping where impacts matter. |
| `active` | Pool-based uncertainty / committee sampling (Settles, 2009). | Add rows the current forest is least sure about. |

## 9 Tree models (`models/registry.py`, `models/fast_forest.py`)

`build_model` is a factory. Unspecified hyperparameters receive defaults aimed at large cubes (depth caps, row subsampling, many cores).

| Name | Backend | Role in the diagnostic |
| --- | --- | --- |
| `rf` | scikit-learn `RandomForestRegressor` | Reference Breiman forest; TreeSHAP and linear TreeSHAP apply. |
| `extra_trees` | ExtraTrees | Default **fast RF**: random thresholds, usually much faster than sklearn RF, still impurity importance and TreeSHAP. |
| `hist_gbm` | `HistGradientBoostingRegressor` | Histogram boosting for large \(n\). |
| `xgboost` | `XGBRegressor` (`tree_method=hist`) | Gradient boosting; native TreeSHAP. |
| `lightgbm` | `LGBMRegressor` | Fast boosting; often the cheapest TreeSHAP among boosters. |
| `lightgbm_rf` | LightGBM `boosting_type=rf` | RF-mode bagging inside LightGBM. |
| `fastforest` | ThrunGroup FastForest / MABSplit (Tiwari et al., 2022) | Histogram forest that allocates split evaluations with a multi-armed bandit. Optional; repository on `PYTHONPATH`. Trees are **not** sklearn/XGBoost/LightGBM, so TreeSHAP does not apply and `compute_shap` falls back to a model-agnostic explainer. |

The scientific default for an ESMValTool recipe is ExtraTrees or sklearn RF plus FastTreeSHAP: interpretable trees, parallel SHAP, no extra native library beyond scikit-learn and `fasttreeshap`.

## 10 Prediction and skill (`predict/`)

`regression_metrics` reports, on finite test pairs:

- sample size \(n\)
- \(R^2\), RMSE, MAE
- **bias** and **ubRMSE** \(\sqrt{\max(\mathrm{RMSE}^2-\mathrm{bias}^2,0)}\) (scatter after removing mean bias, standard in soil-moisture and flux evaluation)
- Nash–Sutcliffe efficiency
- Kling–Gupta efficiency (correlation, variability ratio, bias ratio)
- Pearson \(r\), MAPE

`reconstruct_prediction` writes \(\hat Y(t,\varphi,\lambda)\) to NetCDF. Train cells and masked locations remain missing, which is the correct cube for difference maps \(\hat Y - Y_{\mathrm{obs}}\).

## 11 Explanation (`explain/`)

### 11.1 Global importance

- **Impurity / gain** (`model_importance`): fast ranking from the forest; biased under collinearity.
- **Permutation** (`permutation_scores`): shuffle each feature on the **test** set; more reliable when CMIP predictors covary.

### 11.2 SHAP (`explain/shap_explain.py`)

For an explained sample, \(\sum_j \phi_j + \mathbb{E}[f] \approx \hat y\). `shap_to_cube` writes one cube per feature.

| Method | Logic | When to use |
| --- | --- | --- |
| `auto` | FastTreeSHAP v2 if installed, else path-dependent TreeSHAP. | Default. |
| `fasttreeshap` (`fast_tree_shap`) | LinkedIn FastTreeSHAP (Yang, 2021). v1 \(\approx 1.5\times\) TreeSHAP at equal memory; v2 \(\approx 2.5\times\) with precomputation; large extra gains on sklearn RF because stock SHAP is single-core. Path-dependent (no background) or interventional (small k-means / random background). | Production SHAP on RF/ExtraTrees/LightGBM/XGBoost. |
| `treeshap` | `shap.TreeExplainer`, path-dependent. | Exact tree SHAP without FastTreeSHAP. |
| `linear_treeshap` | yupbank/linear_tree_shap, \(O(LD)\) per sklearn tree; forest = mean of trees; v2 uses Chebyshev interpolation for deep trees. | Exact linear-time SHAP **only** for scikit-learn trees/forests. |
| `kernel` / `permutation` / `sampling` | Model-agnostic SHAP on a small background. | FastForest or any estimator without TreeSHAP. |

`max_samples` explains a random subset; other cells are NaN in the SHAP cube. That is the intended way to map attributions on a global grid without explaining every cell.

### 11.3 ALE and ICE/PDP

- **ALE** (`ale_1d`): bin a feature, accumulate the local change in prediction when that feature is moved from the left to the right edge of the bin, then centre the curve. More robust than PDP when features are correlated (Apley and Zhu, 2020).
- **ICE + PDP** (`ice_pdp`): individual curves for a subset of samples and their mean. Shows heterogeneity (e.g. VPD effect that saturates only in dry regions).

### 11.4 Plotting (`plot.py`)

Helpers for maps, domain-mean series, histograms, correlation, split coverage, sampling maps, predicted-versus-observed scatter, error maps, importance bars, SHAP beeswarm and SHAP cubes, ALE, and ICE/PDP. Used by `examples/run_demo.ipynb`.

## 12 Configuration, API, and outputs

A YAML file (`configs/default.yaml`) or `PipelineConfig.from_dict` controls every switch listed above. `run_pipeline` / `rf-shap -c` load, run, and save:

| File | Content |
| --- | --- |
| `metrics.csv` | Skill scores |
| `prediction.nc` | \(\hat Y\) cube |
| `shap.nc` | One SHAP field per feature |
| `rf_importance.csv`, `permutation_importance.csv` | Global rankings |
| `shap_samples.csv` | Long table with coordinates |

Public imports: `DiagnosisPipeline`, `load_config`, `fast_tree_shap`, `linear_tree_shap`, `fast_forest`, `regrid_to_1deg`, `regrid_file_to_1deg`.

## 13 Demonstration (not a CMIP result)

`examples/run_demo.py` and `examples/run_demo.ipynb` run the full chain on the synthetic cube (lag-1 \(X\), chronological split, FSCS subsample, ExtraTrees, TreeSHAP restored to a cube). Additional notebooks (`examples/obs_vpd_gpp_benchmark.ipynb`, `examples/time_rf_shap.ipynb`) exercise observational VPD–GPP style settings and timing of SHAP backends. Unit tests cover cube round-trip, regridding, sampling, FastForest wrapping, pipeline smoke tests, and SHAP restore.

These runs establish **software correctness and tractability**. They are not a substitute for a peer-reviewed evaluation on a named CMIP experiment and in situ or satellite product; that evaluation belongs in a follow-up application paper and in the ESMValTool recipe once WP4 fixes \(Y\), the CMOR list, the domain, and the period.

## 14 Intended use inside ESMValTool

rf_shap is the numerical kernel. An ESMValTool recipe would:

1. select CMOR variables for \(X\) and an observational dataset for \(Y\);
2. extract a region and time window;
3. (optional) call `regrid_to_1deg` or ESMValTool’s own regridder;
4. invoke `DiagnosisPipeline` with a documented YAML;
5. publish `prediction.nc`, `shap.nc`, and metric tables as diagnostic products.

Until that wrapper exists, the GitHub repository is the citable implementation of the method.

## 15 Conclusions

rf_shap turns a collocated ESM-versus-observation question into a cube-native tree-and-SHAP diagnostic. The contribution is not a new learning algorithm but a **complete, configurable chain** that respects climate data geometry: alignment and area-weighted regridding, seasonality control, lags and EOFs, collinearity-aware feature selection, climate-aware splits and sampling, fast forests, fast TreeSHAP, and restoration of both predictions and attributions to `(time, lat, lon)` NetCDF. That chain is the prerequisite for an ESMValTool recipe that WP4 can run on CMIP6 and observations without flattening the science into a spreadsheet.

## Code and data availability

Source code is available at https://github.com/feinihuang9554-max/RF_SHAP_ESMValTool under the repository licence. The package requires Python \(\ge 3.9\), numpy, pandas, xarray, netCDF4, scikit-learn, scipy, PyYAML, and matplotlib. Optional extras: xgboost, lightgbm, shap, fasttreeshap, and (for MABSplit FastForest) the ThrunGroup/FastForest repository on `PYTHONPATH`. Synthetic cubes for the demo are generated in memory; no external observational product is redistributed in the repository.

## Author contributions

FH designed the diagnostic, implemented the software, and wrote the manuscript draft.

## Competing interests

The author declares that there are no competing interests.

## Acknowledgements

This work is intended as a contribution to an ESMValTool-style explainable-AI diagnostic. Community review via a GitHub issue with the ESMValTool technical and user-engagement teams is the planned next step for the recipe wrapper.

## References

Apley, D. W. and Zhu, J.: Visualizing the effects of predictor variables in black box supervised learning models, J. R. Stat. Soc. B, 82, 1059–1086, 2020.

Breiman, L.: Random forests, Mach. Learn., 45, 5–32, 2001.

Brus, D. J.: Sampling for digital soil mapping: a tutorial supported by R scripts, Geoderma, 338, 464–480, 2019.

Cochran, W. G.: Sampling Techniques, 3rd ed., Wiley, 1977.

Eyring, V., et al.: Earth System Model Evaluation Tool (ESMValTool) v2.0, Geosci. Model Dev., 13, 1179–1197, 2020.

Hansen, M. H. and Hurwitz, W. N.: On the theory of sampling from finite populations, Ann. Math. Statist., 14, 333–362, 1943.

Kennard, R. W. and Stone, L. A.: Computer aided design of experiments, Technometrics, 11, 137–148, 1969.

Le, N. D. and Vargas, R.: acLHS, Comput. Geosci., 184, 105539, 2024.

Lundberg, S. M. and Lee, S.-I.: A unified approach to interpreting model predictions, NeurIPS, 2017.

Lundberg, S. M., et al.: From local explanations to global understanding with explainable AI for trees, Nat. Mach. Intell., 2, 56–67, 2020.

Minasny, B. and McBratney, A. B.: A conditioned Latin hypercube method for sampling in the presence of ancillary information, Comput. Geosci., 32, 1378–1388, 2006.

Settles, B.: Active learning literature survey, Univ. Wisconsin–Madison TR 1648, 2009.

Tiwari, M., et al.: Bandit-based estimation of distribution algorithms for discrete optimization, FastForest / MABSplit, NeurIPS, 2022.

Yang, J.: Fast TreeSHAP: accelerating SHAP value computation for trees, arXiv:2109.09847, 2021.

---

## Appendix A Mapping from scientific step to source file

| Scientific step | Module |
| --- | --- |
| Config / CLI | `config.py`, `cli.py` |
| Orchestration | `pipeline.py` |
| Load cubes | `cube/io.py` |
| 1° area regrid | `cube/regrid.py` |
| Collocate \(X,Y\) | `cube/align.py` |
| Detrend, season, z-score | `cube/preprocess.py` |
| Lag, EOF, table PCA | `cube/features.py` |
| Flatten / restore | `cube/tabular.py` |
| Feature selection | `features/*.py` |
| Split | `models/split.py` |
| Sampling | `models/sampling.py` |
| Model factory | `models/registry.py` |
| MABSplit FastForest | `models/fast_forest.py` |
| Metrics / prediction cube | `predict/metrics.py`, `predict/reconstruct.py` |
| Importance, SHAP, ALE, ICE | `explain/*.py` |
| Figures | `plot.py` |
| Synthetic demo | `synthetic.py`, `examples/run_demo.*` |

## Appendix B Notation

- \(X(t,\varphi,\lambda)\): vector of model (or lagged/EOF) predictors at time \(t\), latitude \(\varphi\), longitude \(\lambda\).
- \(Y(t,\varphi,\lambda)\): observed target.
- \(f\): tree ensemble trained on collocated samples.
- \(\phi_j\): SHAP value of feature \(j\); \(\hat Y = \mathbb{E}[f] + \sum_j \phi_j\) on explained samples.
- \(S_j(t,\varphi,\lambda)\): \(\phi_j\) scattered back onto the parent cube.
