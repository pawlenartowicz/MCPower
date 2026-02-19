"""Precision boundary tests for C++ LME solvers.

Tests solver behavior at extreme parameter values and edge cases.
"""

import numpy as np
import pytest


def _native_available():
    try:
        from mcpower.backends.native import is_native_available
        return is_native_available()
    except ImportError:
        return False


pytestmark = [
    pytest.mark.lme,
    pytest.mark.skipif(
        not _native_available(), reason="Native C++ backend not available"
    ),
]


def _gen_q1(K=20, n=1000, p=2, seed=42, tau=0.5):
    rng = np.random.RandomState(seed)
    obs = n // K
    cids = np.repeat(np.arange(K), obs).astype(np.int32)
    na = len(cids)
    X = rng.randn(na, p)
    Xi = np.column_stack([np.ones(na), X])
    b = rng.randn(K) * tau
    y = Xi @ np.array([1.0] + [0.5] * p) + b[cids] + rng.randn(na)
    return X, y, cids, K


def _gen_general(K=30, n=1500, p=2, q=2, seed=42):
    rng = np.random.RandomState(seed)
    obs = n // K
    cids = np.repeat(np.arange(K), obs).astype(np.int32)
    na = len(cids)
    X = rng.randn(na, p)
    Xi = np.column_stack([np.ones(na), X])
    Z = np.column_stack([np.ones(na), X[:, 0]])
    G = np.array([[0.25, 0.05], [0.05, 0.1]])
    b = rng.randn(K, q) @ np.linalg.cholesky(G).T
    y = Xi @ np.array([1.0] + [0.5] * p) + np.sum(Z * b[cids], axis=1) + rng.randn(na)
    return X, y, Z, cids, K, q


def _gen_nested(Kp=10, Kc=30, n=1500, p=2, seed=42, tp=0.4, tc=0.3):
    rng = np.random.RandomState(seed)
    cpp = Kc // Kp
    opc = n // Kc
    pids, cids = [], []
    c2p = np.zeros(Kc, dtype=np.int32)
    for pj in range(Kp):
        for cl in range(cpp):
            cj = pj * cpp + cl
            c2p[cj] = pj
            pids.extend([pj] * opc)
            cids.extend([cj] * opc)
    pids = np.array(pids, dtype=np.int32)
    cids = np.array(cids, dtype=np.int32)
    na = len(pids)
    X = rng.randn(na, p)
    Xi = np.column_stack([np.ones(na), X])
    bp = rng.randn(Kp) * tp
    bc = rng.randn(Kc) * tc
    y = Xi @ np.array([1.0] + [0.5] * p) + bp[pids] + bc[cids] + rng.randn(na)
    return X, y, pids, cids, Kp, Kc, c2p


class TestPrecisionBoundariesGeneral:
    """Precision boundary tests for general q>1 solver."""

    def test_extreme_ICC_low(self):
        """Solver works with very low ICC (0.01)."""
        import mcpower.backends.mcpower_native as _native
        from mcpower.stats.lme_solver import compute_lme_critical_values

        rng = np.random.RandomState(42)
        K, n, p, q = 30, 1500, 2, 2
        obs = n // K
        cids = np.repeat(np.arange(K), obs).astype(np.int32)
        na = len(cids)
        X = rng.randn(na, p)
        Xi = np.column_stack([np.ones(na), X])
        Z = np.column_stack([np.ones(na), X[:, 0]])
        # Very low ICC: tiny random effects
        G = np.array([[0.01, 0.001], [0.001, 0.005]])
        b = rng.randn(K, q) @ np.linalg.cholesky(G).T
        y = Xi @ np.array([1.0, 0.5, 0.3]) + np.sum(Z * b[cids], axis=1) + rng.randn(na)

        target_indices = np.arange(p, dtype=np.int32)
        chi2_crit, z_crit, crits = compute_lme_critical_values(0.05, p, p, 0)
        result = np.array(_native.lme_analysis_general(
            np.ascontiguousarray(X, dtype=np.float64),
            np.ascontiguousarray(y, dtype=np.float64),
            np.ascontiguousarray(Z, dtype=np.float64),
            np.ascontiguousarray(cids, dtype=np.int32),
            K, q, target_indices, chi2_crit, z_crit,
            np.ascontiguousarray(crits, dtype=np.float64), 0,
            np.empty(0, dtype=np.float64)))
        assert len(result) > 0

    def test_extreme_ICC_high(self):
        """Solver works with high ICC (0.7)."""
        import mcpower.backends.mcpower_native as _native
        from mcpower.stats.lme_solver import compute_lme_critical_values

        rng = np.random.RandomState(42)
        K, n, p, q = 30, 1500, 2, 2
        obs = n // K
        cids = np.repeat(np.arange(K), obs).astype(np.int32)
        na = len(cids)
        X = rng.randn(na, p)
        Xi = np.column_stack([np.ones(na), X])
        Z = np.column_stack([np.ones(na), X[:, 0]])
        # High ICC
        G = np.array([[2.33, 0.1], [0.1, 0.5]])
        b = rng.randn(K, q) @ np.linalg.cholesky(G).T
        y = Xi @ np.array([1.0, 0.5, 0.3]) + np.sum(Z * b[cids], axis=1) + rng.randn(na)

        target_indices = np.arange(p, dtype=np.int32)
        chi2_crit, z_crit, crits = compute_lme_critical_values(0.05, p, p, 0)
        result = np.array(_native.lme_analysis_general(
            np.ascontiguousarray(X, dtype=np.float64),
            np.ascontiguousarray(y, dtype=np.float64),
            np.ascontiguousarray(Z, dtype=np.float64),
            np.ascontiguousarray(cids, dtype=np.int32),
            K, q, target_indices, chi2_crit, z_crit,
            np.ascontiguousarray(crits, dtype=np.float64), 0,
            np.empty(0, dtype=np.float64)))
        assert len(result) > 0

    def test_numerical_stability_large_N(self):
        """Solver is numerically stable with N=10000."""
        import mcpower.backends.mcpower_native as _native
        from mcpower.stats.lme_solver import compute_lme_critical_values

        X, y, Z, cids, K, q = _gen_general(K=50, n=10000, seed=42)
        p = X.shape[1]
        target_indices = np.arange(p, dtype=np.int32)
        chi2_crit, z_crit, crits = compute_lme_critical_values(0.05, p, p, 0)
        result = np.array(_native.lme_analysis_general(
            np.ascontiguousarray(X, dtype=np.float64),
            np.ascontiguousarray(y, dtype=np.float64),
            np.ascontiguousarray(Z, dtype=np.float64),
            np.ascontiguousarray(cids, dtype=np.int32),
            K, q, target_indices, chi2_crit, z_crit,
            np.ascontiguousarray(crits, dtype=np.float64), 0,
            np.empty(0, dtype=np.float64)))
        assert len(result) > 0
        assert np.all(np.isfinite(result))


class TestPrecisionBoundariesNested:
    """Precision boundary tests for nested solver."""

    def test_extreme_ICC_low_nested(self):
        """Nested solver works with very low ICCs."""
        import mcpower.backends.mcpower_native as _native
        from mcpower.stats.lme_solver import compute_lme_critical_values

        X, y, pids, cids, Kp, Kc, c2p = _gen_nested(tp=0.1, tc=0.1, seed=42)
        p = X.shape[1]
        target_indices = np.arange(p, dtype=np.int32)
        chi2_crit, z_crit, crits = compute_lme_critical_values(0.05, p, p, 0)
        result = np.array(_native.lme_analysis_nested(
            np.ascontiguousarray(X, dtype=np.float64),
            np.ascontiguousarray(y, dtype=np.float64),
            np.ascontiguousarray(pids, dtype=np.int32),
            np.ascontiguousarray(cids, dtype=np.int32),
            Kp, Kc,
            np.ascontiguousarray(c2p, dtype=np.int32),
            target_indices, chi2_crit, z_crit,
            np.ascontiguousarray(crits, dtype=np.float64), 0,
            np.empty(0, dtype=np.float64)))
        assert len(result) > 0

    def test_numerical_stability_large_N_nested(self):
        """Nested solver stable with N=10000."""
        import mcpower.backends.mcpower_native as _native
        from mcpower.stats.lme_solver import compute_lme_critical_values

        X, y, pids, cids, Kp, Kc, c2p = _gen_nested(Kp=20, Kc=60, n=6000, seed=42)
        p = X.shape[1]
        target_indices = np.arange(p, dtype=np.int32)
        chi2_crit, z_crit, crits = compute_lme_critical_values(0.05, p, p, 0)
        result = np.array(_native.lme_analysis_nested(
            np.ascontiguousarray(X, dtype=np.float64),
            np.ascontiguousarray(y, dtype=np.float64),
            np.ascontiguousarray(pids, dtype=np.int32),
            np.ascontiguousarray(cids, dtype=np.int32),
            Kp, Kc,
            np.ascontiguousarray(c2p, dtype=np.int32),
            target_indices, chi2_crit, z_crit,
            np.ascontiguousarray(crits, dtype=np.float64), 0,
            np.empty(0, dtype=np.float64)))
        assert len(result) > 0
        assert np.all(np.isfinite(result))

    def test_convergence_rate_nested(self):
        """Nested solver converges on 90%+ of datasets."""
        import mcpower.backends.mcpower_native as _native
        from mcpower.stats.lme_solver import compute_lme_critical_values

        n_datasets = 50
        converged = 0
        for seed in range(n_datasets):
            X, y, pids, cids, Kp, Kc, c2p = _gen_nested(seed=seed + 500)
            p = X.shape[1]
            target_indices = np.arange(p, dtype=np.int32)
            chi2_crit, z_crit, crits = compute_lme_critical_values(0.05, p, p, 0)
            result = np.array(_native.lme_analysis_nested(
                np.ascontiguousarray(X, dtype=np.float64),
                np.ascontiguousarray(y, dtype=np.float64),
                np.ascontiguousarray(pids, dtype=np.int32),
                np.ascontiguousarray(cids, dtype=np.int32),
                Kp, Kc,
                np.ascontiguousarray(c2p, dtype=np.int32),
                target_indices, chi2_crit, z_crit,
                np.ascontiguousarray(crits, dtype=np.float64), 0,
                np.empty(0, dtype=np.float64)))
            if len(result) > 0:
                converged += 1
        assert converged >= n_datasets * 0.9, f"Only {converged}/{n_datasets} converged"
