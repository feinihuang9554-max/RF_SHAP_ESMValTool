# rf_shap

English documentation (every feature, in detail): [`README_EN.md`](README_EN.md)

面向 **模式数据（CMIP6 一类的 data cube）与观测数据** 的输入–输出关系诊断工具包。  
数据保持 `time × lat × lon` 立方体；建模在摊平后的样本上进行；预测值和 SHAP 可以还原回原来的 cube。

## 能做什么

1. **数据**：nc / npy / npz / csv / parquet → 统一成 cube，并按时空对齐 X 与 Y。
2. **X 选项**：原始值、时间 lag、空间 EOF/PCA；X 和 Y 都可去趋势、去季节、距平、标准化。
3. **特征工程**：共线性（相关、VIF、线性回归）、相关变量对中去掉重要性低的、置换重要性剔除、RFE。
4. **建模**：RF / ExtraTrees（快速 RF）/ HistGBM / XGBoost / LightGBM / LightGBM-RF；多种训练采样和训练/测试划分。
5. **预测**：R²、RMSE、ubRMSE、NSE、KGE 等，并还原 cube。
6. **解释**：树模型重要性、置换重要性、TreeSHAP / FastTreeSHAP / Kernel / Permutation SHAP、ALE、ICE+PDP；SHAP 还原为 cube。

## 安装

```bash
pip install -e ".[all]"
```

至少需要 `numpy pandas xarray netCDF4 scikit-learn scipy pyyaml matplotlib`。  
解释与 boosting 模型建议同时安装 `shap xgboost lightgbm`。  
若要进一步加速树 SHAP：`pip install fasttreeshap`。

## 快速开始

内存中的合成立方体：

```bash
PYTHONPATH=src python examples/run_demo.py
PYTHONPATH=src jupyter notebook examples/run_demo.ipynb   # 逐步可视化版本
```

配置文件：

```bash
PYTHONPATH=src python -m rf_shap.cli -c configs/default.yaml
```

代码调用：

```python
from rf_shap import DiagnosisPipeline, load_config
from rf_shap.synthetic import make_synthetic_cubes

x, y = make_synthetic_cubes()
cfg = load_config("configs/default.yaml")
result = DiagnosisPipeline(cfg).run(x=x, y=y)
result.save("outputs/run")
print(result.metrics)
```

## 数据约定

每个变量都应是 cube：`time, lat, lon`。X 与 Y 必须能对齐到同一套时空格点。

| 格式 | 要求 |
| --- | --- |
| NetCDF | 含 `time/lat/lon` 维，或在配置里改名 |
| npy 目录 | `time.npy lat.npy lon.npy` + 各变量 `(T,Y,X)` |
| npz | 同上键名 |
| 表格 | 已排好，含 `time,lat,lon` 和变量列，会先透视成 cube |

对齐默认把模式场插值到观测网格（`data.target: y`），时间取交集。

### 预处理与 X 变换

`preprocess.x` / `preprocess.y`：`detrend`、`deseasonalize`、`anomaly`、`standardize`。

`x_transform.mode`：

- `raw`：原始预测因子
- `lag`：原变量 + `lags`
- `pca`：各变量空间 EOF 重建（保持 cube，X/Y 仍逐格点对应）
- `lag_pca`：先 lag 再 EOF
- 若还要在表格特征空间做 PCA，设 `table_pca: true`

## 特征选择（按配置顺序执行）

- `collinearity`：高相关、VIF、被其他变量线性回归几乎完全解释的特征
- `pairwise_importance`：两变量高相关时丢掉树重要性更低的那个
- `permutation`：递归丢掉置换重要性最低的特征
- `rfe`：递归特征消除 / RFECV

## 采样与划分

训练采样（水文机器学习里常用的“覆盖特征空间”思路，对应你给的 Journal of Hydrology 类工作，以及 WRR 2025 的 FSCS/cLHS 比较）：

- `random` 简单随机
- `stratified` 按 Y 分箱分层
- `fscs` / `cluster` 特征空间覆盖（标准化后聚类，取距中心最近样本）
- `clhs` 条件拉丁超立方
- `kennard_stone` Kennard–Stone
- `spatial` 在 lat/lon 上覆盖
- `systematic` 等间隔

划分：

- `time` 时间切末段（或 `train_end`）
- `space`：`lon_band` / `lat_band` / `block` / `checkerboard`
- `time_window` 滑动/扩张窗口
- `random`
- `extreme` 把 Y 两端极端值放到测试或训练

## 快速随机森林

对应 SoftwareX 2026 的 `fru`（Rust 高速 RF）在 Python 侧提供可替换后端：

| 名称 | 说明 |
| --- | --- |
| `extra_trees` | 默认快速 RF 替代，通常明显快于 sklearn RF |
| `rf` | sklearn RandomForest，多核 + `max_samples` |
| `lightgbm_rf` | LightGBM 的 RF boosting 模式 |
| `hist_gbm` | 直方图梯度提升，大样本很快 |
| `xgboost` / `lightgbm` | 常规 boosting |

R 的 `fru` 没有稳定 Python 绑定，因此这里用 ExtraTrees / LightGBM-RF / 直方图方法覆盖“又快又能做置换重要性”的需求。

## SHAP 加速与还原

`explain.shap.method`：

- `auto`：有 `fasttreeshap` 就用 v1/v2，否则用 path-dependent TreeSHAP
- `fasttreeshap`：LinkedIn FastTreeSHAP（v1 约 1.5×，v2 约 2.5×，内存略增）
- `treeshap`：精确树路径 SHAP，不抽背景，通常是树模型最快的精确算法
- `kernel` / `permutation` / `sampling`：模型无关；请把 `background_size` 设小（k-means 背景）

其他加速：

- `max_samples`：只解释子集，其余格点在 cube 里为 NaN
- `background_size`：干预式 SHAP 的背景样本数
- 用 `extra_trees` / `lightgbm`：树更浅、SHAP 更快
- GPU TreeSHAP：若本机 shap 编译了 GPU，可继续走 TreeExplainer

SHAP 矩阵按样本的 `(time, lat, lon)` 写回 `shap.nc`，每个特征一个变量。

## 配置要点

见 `configs/default.yaml`。把 `data.x.path` / `data.y.path` 换成你的 CMIP6 与观测文件即可。

输出目录默认写入：

- `metrics.csv`
- `prediction.nc`
- `shap.nc`
- `rf_importance.csv`
- `permutation_importance.csv`
- `shap_samples.csv`

## 测试

```bash
PYTHONPATH=src pytest -q
```
