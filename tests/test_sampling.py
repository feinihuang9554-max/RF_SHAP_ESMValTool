import numpy as np

from rf_shap.models.sampling import maximin_sample, sample_train


def _xy(n=80, p=3, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, p))
    y = x[:, 0] - 0.3 * x[:, 1]
    return x, y


def test_existing_methods_still_return_unique_indices():
    x, y = _xy()
    for method in ("random", "stratified", "fscs", "cluster", "clhs", "kennard_stone", "systematic"):
        idx = sample_train(x, y, method=method, n_samples=12, seed=0, n_iter=40)
        assert len(idx) <= 12
        assert len(np.unique(idx)) == len(idx)


def test_area_weight_never_draws_a_zero_weight_row():
    x, y = _xy(n=6)
    idx = sample_train(
        x,
        y,
        method="area",
        n_samples=3,
        seed=1,
        area_weights=np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0]),
    )
    assert set(idx).issubset({0, 1, 2})
    chosen, weights = sample_train(
        x,
        y,
        method="area",
        n_samples=3,
        seed=1,
        area_weights=np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0]),
        return_weights=True,
    )
    assert weights is not None and len(weights) == len(chosen)


def test_process_strata_cover_both_labels():
    x, y = _xy(n=20)
    strata = np.array([0] * 10 + [1] * 10)
    idx = sample_train(x, y, method="process", n_samples=4, seed=0, strata=strata, balance="equal")
    assert len(idx) == 4
    assert set(strata[idx]) == {0, 1}


def test_maximin_on_a_line_starts_at_the_endpoints():
    x = np.arange(20, dtype=float).reshape(-1, 1)
    y = x[:, 0]
    idx = maximin_sample(x, n_samples=2, seed=0)
    assert set(idx.tolist()) == {0, 19}


def test_extreme_fraction_draws_only_from_the_upper_tail():
    x = np.zeros((100, 2))
    y = np.arange(100, dtype=float)
    idx, weights = sample_train(
        x,
        y,
        method="extreme",
        n_samples=5,
        seed=0,
        extreme_fraction=1.0,
        extreme_quantile=0.9,
        extreme_side="high",
        return_weights=True,
    )
    assert np.all(y[idx] >= 90)
    assert np.all(weights > 1)


def test_cluster_within_and_medoid_are_real_rows():
    x, y = _xy(n=60)
    within = sample_train(x, y, method="cluster_within", n_samples=12, seed=0, n_clusters=4)
    medoid = sample_train(x, y, method="cluster_medoid", n_samples=6, seed=0)
    assert len(np.unique(within)) == len(within) <= 12
    assert len(np.unique(medoid)) == len(medoid) <= 6


def test_active_and_aclhs_select_existing_rows():
    x, y = _xy(n=40, seed=2)
    time = np.arange(40)
    active = sample_train(x, y, method="active", n_samples=10, seed=0, active_rounds=2)
    aclhs = sample_train(
        x,
        y,
        method="aclhs",
        n_samples=8,
        seed=0,
        time=time,
        dependence="time",
        n_iter=20,
    )
    assert len(np.unique(active)) == 10
    assert len(np.unique(aclhs)) == 8
    assert set(active).issubset(set(range(40)))
    assert set(aclhs).issubset(set(range(40)))
