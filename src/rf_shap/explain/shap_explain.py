from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
import xarray as xr

from rf_shap.cube.tabular import SampleTable, values_to_cubes


@dataclass
class ShapResult:
    values: np.ndarray
    expected_value: float | np.ndarray
    method: str
    index: pd.MultiIndex
    feature_names: list[str]
    sample_positions: np.ndarray

    def to_frame(self) -> pd.DataFrame:
        frame = pd.DataFrame(self.values, columns=self.feature_names, index=self.index[self.sample_positions])
        return frame.reset_index()


def _try_import_shap():
    try:
        import shap
        return shap
    except ImportError as exc:
        raise ImportError("Install shap to compute SHAP values: pip install shap") from exc


def _try_import_fasttreeshap():
    try:
        import fasttreeshap
        return fasttreeshap
    except ImportError:
        return None


def _is_tree_model(model) -> bool:
    name = type(model).__name__.lower()
    tree_bits = (
        "forest",
        "tree",
        "xgb",
        "lgbm",
        "lightgbm",
        "histgradient",
        "extratrees",
        "gbm",
        "catboost",
    )
    return any(bit in name for bit in tree_bits)


def _background(x: np.ndarray, size: int, seed: int) -> np.ndarray:
    shap = _try_import_shap()
    n = min(size, len(x))
    if n <= 0:
        return x
    try:
        return shap.kmeans(x, n).data
    except Exception:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(x), size=n, replace=False)
        return x[idx]


def _subset(x: np.ndarray, max_samples: int | None, seed: int) -> np.ndarray:
    if max_samples is None or max_samples >= len(x):
        return np.arange(len(x))
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(len(x), size=int(max_samples), replace=False))


def _fast_treeshap(model, x_explain, background, algorithm: str, n_jobs: int):
    fast = _try_import_fasttreeshap()
    if fast is None:
        return None
    explainer = fast.TreeExplainer(
        model,
        data=background,
        algorithm=algorithm if algorithm in {"v0", "v1", "v2", "auto"} else "auto",
        n_jobs=n_jobs if n_jobs is not None else -1,
    )
    values = explainer.shap_values(x_explain)
    expected = explainer.expected_value
    return np.asarray(values), expected, "fasttreeshap"


def _tree_shap(model, x_explain, background, n_jobs: int):
    shap = _try_import_shap()
    kwargs = {}
    if background is None:
        explainer = shap.TreeExplainer(model)
    else:
        explainer = shap.TreeExplainer(model, data=background, feature_perturbation="interventional")
    if hasattr(explainer, "shap_values"):
        try:
            values = explainer.shap_values(x_explain, check_additivity=False)
        except TypeError:
            values = explainer.shap_values(x_explain)
    else:
        values = explainer(x_explain).values
    return np.asarray(values), explainer.expected_value, "treeshap"


def _kernel_shap(model, x_explain, background):
    shap = _try_import_shap()
    explainer = shap.KernelExplainer(model.predict, background)
    values = explainer.shap_values(x_explain, nsamples="auto")
    return np.asarray(values), explainer.expected_value, "kernel"


def _permutation_shap(model, x_explain, background):
    shap = _try_import_shap()
    explainer = shap.PermutationExplainer(model.predict, background)
    explanation = explainer(x_explain)
    return np.asarray(explanation.values), explanation.base_values, "permutation"


def _sampling_shap(model, x_explain, background):
    shap = _try_import_shap()
    explainer = shap.SamplingExplainer(model.predict, background)
    values = explainer.shap_values(x_explain)
    return np.asarray(values), explainer.expected_value, "sampling"


def _normalize_values(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values)
    if arr.ndim == 3:
        # classification-style (n, features, outputs) or (outputs, n, features)
        if arr.shape[-1] == 1:
            arr = arr[..., 0]
        elif arr.shape[0] == 1:
            arr = arr[0]
        else:
            arr = arr[..., -1]
    return arr


def compute_shap(
    model,
    table: SampleTable,
    method: str = "auto",
    algorithm: str = "v2",
    max_samples: int | None = 2000,
    background_size: int = 100,
    n_jobs: int = -1,
    seed: int = 42,
    background: np.ndarray | None = None,
) -> ShapResult:
    """Compute SHAP with several speed/quality trade-offs.

    Efficiency options:
      auto/fasttreeshap : LinkedIn FastTreeSHAP v1/v2 when installed
      treeshap          : path-dependent TreeSHAP (no background, fastest exact tree path)
      treeshap+background: interventional TreeSHAP
      kernel/permutation/sampling : model-agnostic, use tiny k-means background
      max_samples       : only explain a subset, then restore those cells to the cube
    """
    x = np.asarray(table.X, dtype=np.float64)
    positions = _subset(x, max_samples, seed)
    x_explain = x[positions]
    bg = background if background is not None else _background(x, background_size, seed)
    method = method.lower()
    values = expected = used = None

    if method in {"auto", "fasttreeshap"} and _is_tree_model(model):
        values_pack = _fast_treeshap(model, x_explain, bg, algorithm, n_jobs)
        if values_pack is not None:
            values, expected, used = values_pack
        elif method == "fasttreeshap":
            values, expected, used = _tree_shap(model, x_explain, bg, n_jobs)

    if values is None and method in {"auto", "treeshap"} and _is_tree_model(model):
        # path-dependent TreeSHAP is typically the fastest exact option
        use_bg = None if method == "treeshap" else bg
        if method == "auto":
            use_bg = None
        values, expected, used = _tree_shap(model, x_explain, use_bg, n_jobs)

    if values is None:
        if method in {"kernel"}:
            values, expected, used = _kernel_shap(model, x_explain, bg)
        elif method in {"sampling"}:
            values, expected, used = _sampling_shap(model, x_explain, bg)
        else:
            values, expected, used = _permutation_shap(model, x_explain, bg)

    values = _normalize_values(values)
    if values.shape[0] != len(positions):
        raise ValueError(f"SHAP values have unexpected shape {values.shape}")
    return ShapResult(
        values=values,
        expected_value=expected,
        method=used or method,
        index=table.index,
        feature_names=list(table.feature_names),
        sample_positions=positions,
    )


def shap_to_cube(result: ShapResult, table: SampleTable) -> xr.Dataset:
    template = table.extras.get("template")
    if template is None:
        raise ValueError("SampleTable is missing a cube template; cannot restore SHAP")
    index = table.index[result.sample_positions]
    ds = values_to_cubes(result.values, index, template, result.feature_names)
    ds.attrs["shap_method"] = result.method
    ds.attrs["expected_value"] = str(result.expected_value)
    return ds
