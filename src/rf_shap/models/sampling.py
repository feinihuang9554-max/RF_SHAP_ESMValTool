from __future__ import annotations

import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import StandardScaler


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def _cap(n_samples: int | None, n: int) -> int:
    if n_samples is None or n_samples >= n:
        return n
    if n_samples <= 0:
        raise ValueError("n_samples must be positive")
    return int(n_samples)


def random_sample(n: int, n_samples: int | None, seed: int) -> np.ndarray:
    k = _cap(n_samples, n)
    return _rng(seed).choice(n, size=k, replace=False)


def stratified_sample(y: np.ndarray, n_samples: int | None, n_bins: int = 10, seed: int = 42) -> np.ndarray:
    n = len(y)
    k = _cap(n_samples, n)
    bins = np.digitize(y, np.nanquantile(y, np.linspace(0, 1, n_bins + 1)[1:-1]))
    rng = _rng(seed)
    chosen = []
    remaining = k
    groups = [np.where(bins == b)[0] for b in np.unique(bins)]
    groups = [g for g in groups if len(g)]
    for i, group in enumerate(groups):
        take = remaining if i == len(groups) - 1 else max(1, int(round(k * len(group) / n)))
        take = min(take, len(group), remaining)
        chosen.append(rng.choice(group, size=take, replace=False))
        remaining -= take
        if remaining <= 0:
            break
    idx = np.concatenate(chosen) if chosen else random_sample(n, k, seed)
    if len(idx) < k:
        extra = np.setdiff1d(np.arange(n), idx)
        idx = np.concatenate([idx, rng.choice(extra, size=min(k - len(idx), len(extra)), replace=False)])
    return idx[:k]


def cluster_sample(x: np.ndarray, n_samples: int | None, seed: int = 42) -> np.ndarray:
    n = len(x)
    k = _cap(n_samples, n)
    if k == n:
        return np.arange(n)
    scaled = StandardScaler().fit_transform(x)
    km = MiniBatchKMeans(n_clusters=k, random_state=seed, batch_size=min(2048, n), n_init=3)
    labels = km.fit_predict(scaled)
    centers = km.cluster_centers_
    chosen = []
    for c in range(k):
        members = np.where(labels == c)[0]
        if len(members) == 0:
            continue
        dist = np.linalg.norm(scaled[members] - centers[c], axis=1)
        chosen.append(members[int(np.argmin(dist))])
    return np.unique(chosen)


def fs_coverage_sample(x: np.ndarray, n_samples: int | None, seed: int = 42) -> np.ndarray:
    """Feature-space coverage sampling (cluster centroids in standardized X).

    Motivated by hydrological ML sampling studies that prefer covering the
    predictor space rather than drawing i.i.d. pixels, e.g. FSCS / cLHS
    comparisons in WRR 2025 and related Journal of Hydrology sampling work.
    """
    return cluster_sample(x, n_samples, seed)


def kennard_stone(x: np.ndarray, n_samples: int | None, seed: int = 42) -> np.ndarray:
    n = len(x)
    k = _cap(n_samples, n)
    if k == n:
        return np.arange(n)
    scaled = StandardScaler().fit_transform(x)
    rng = _rng(seed)
    start = int(rng.integers(0, n))
    d0 = np.linalg.norm(scaled - scaled[start], axis=1)
    selected = [start, int(np.argmax(d0))]
    remaining = set(range(n)) - set(selected)
    min_dist = np.minimum(
        np.linalg.norm(scaled - scaled[selected[0]], axis=1),
        np.linalg.norm(scaled - scaled[selected[1]], axis=1),
    )
    while len(selected) < k and remaining:
        nxt = max(remaining, key=lambda i: min_dist[i])
        selected.append(nxt)
        remaining.remove(nxt)
        min_dist = np.minimum(min_dist, np.linalg.norm(scaled - scaled[nxt], axis=1))
    return np.asarray(selected)


def clhs_sample(x: np.ndarray, n_samples: int | None, seed: int = 42, n_iter: int = 2000) -> np.ndarray:
    """Simplified conditioned Latin Hypercube sampling."""
    n, p = x.shape
    k = _cap(n_samples, n)
    if k == n:
        return np.arange(n)
    rng = _rng(seed)
    ranks = np.argsort(np.argsort(x, axis=0), axis=0)
    strata = np.minimum((ranks * k) // n, k - 1)
    idx = rng.choice(n, size=k, replace=False)

    def objective(choice: np.ndarray) -> float:
        occ = np.zeros((p, k), dtype=int)
        for j in range(p):
            for s in strata[choice, j]:
                occ[j, s] += 1
        return float(np.sum((occ - 1) ** 2))

    best = idx.copy()
    best_obj = objective(best)
    pool = np.setdiff1d(np.arange(n), best)
    for _ in range(n_iter):
        if len(pool) == 0:
            break
        drop_i = int(rng.integers(0, k))
        add_i = int(rng.integers(0, len(pool)))
        cand = best.copy()
        cand[drop_i] = pool[add_i]
        obj = objective(cand)
        if obj <= best_obj:
            out_val = best[drop_i]
            best = cand
            best_obj = obj
            pool[add_i] = out_val
            if obj == 0:
                break
    return best


def spatial_sample(lat: np.ndarray, lon: np.ndarray, n_samples: int | None, seed: int = 42) -> np.ndarray:
    coords = np.column_stack([lat, lon])
    return cluster_sample(coords, n_samples, seed)


def systematic_sample(n: int, n_samples: int | None, seed: int = 42) -> np.ndarray:
    k = _cap(n_samples, n)
    step = n / k
    start = float(_rng(seed).uniform(0, step))
    idx = (start + step * np.arange(k)).astype(int)
    return np.unique(np.clip(idx, 0, n - 1))


def sample_train(
    x: np.ndarray,
    y: np.ndarray,
    method: str = "random",
    n_samples: int | None = None,
    n_bins: int = 10,
    seed: int = 42,
    lat: np.ndarray | None = None,
    lon: np.ndarray | None = None,
) -> np.ndarray:
    method = method.lower()
    n = len(y)
    if method in {"none", "all"}:
        return np.arange(n)
    if method == "random":
        return random_sample(n, n_samples, seed)
    if method == "stratified":
        return stratified_sample(y, n_samples, n_bins, seed)
    if method in {"fscs", "cluster", "coverage"}:
        return fs_coverage_sample(x, n_samples, seed)
    if method in {"clhs", "lhs"}:
        return clhs_sample(x, n_samples, seed)
    if method in {"kennard_stone", "ks"}:
        return kennard_stone(x, n_samples, seed)
    if method == "spatial":
        if lat is None or lon is None:
            raise ValueError("spatial sampling needs lat/lon")
        return spatial_sample(lat, lon, n_samples, seed)
    if method == "systematic":
        return systematic_sample(n, n_samples, seed)
    raise ValueError(f"Unknown sampling method: {method}")
