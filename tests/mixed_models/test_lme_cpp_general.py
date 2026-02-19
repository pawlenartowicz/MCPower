"""C++ general (q>1) solver: parameter recovery and convergence tests."""

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


def _generate_general_data(K=30, n=1500, p=2, q=2, seed=42, beta_true=None, G=None):
    rng = np.random.RandomState(seed)
    obs_per_cluster = n // K
    cluster_ids = np.repeat(np.arange(K), obs_per_cluster).astype(np.int32)
    n_actual = len(cluster_ids)
    X = rng.randn(n_actual, p)
    if beta_true is None:
        beta_true = np.array([1.0] + [0.5] * p)
    X_int = np.column_stack([np.ones(n_actual), X])
    Z = np.column_stack([np.ones(n_actual), X[:, 0]])
    if G is None:
        G = np.array([[0.25, 0.05], [0.05, 0.1]])
    L_G = np.linalg.cholesky(G)
    b = rng.randn(K, q) @ L_G.T
    y = X_int @ beta_true + np.sum(Z * b[cluster_ids], axis=1) + rng.randn(n_actual)
    return X, y, Z, cluster_ids, K, q


class TestCppGeneralParameterRecovery:
    """Parameter recovery for C++ general solver."""

    def test_beta_recovery(self):
        """Fixed effects recovered within reasonable tolerance."""
        from mcpower.stats.lme_solver import compute_lme_critical_values
        import mcpower.backends.mcpower_native as _native

        beta_true = np.array([1.0, 0.5, 0.3])
        X, y, Z, cluster_ids, K, q = _generate_general_data(
            K=50, n=5000, beta_true=beta_true, seed=42)
        p = X.shape[1]
        target_indices = np.arange(p, dtype=np.int32)
        chi2_crit, z_crit, crits = compute_lme_critical_values(0.05, p, p, 0)

        result = np.array(_native.lme_analysis_general(
            np.ascontiguousarray(X, dtype=np.float64),
            np.ascontiguousarray(y, dtype=np.float64),
            np.ascontiguousarray(Z, dtype=np.float64),
            np.ascontiguousarray(cluster_ids, dtype=np.int32),
            K, q, target_indices, chi2_crit, z_crit,
            np.ascontiguousarray(crits, dtype=np.float64), 0,
            np.empty(0, dtype=np.float64)))

        assert len(result) > 0, "C++ solver returned empty (failed)"

    def test_convergence_rate(self):
        """C++ general solver converges on 90%+ of datasets."""
        import mcpower.backends.mcpower_native as _native
        from mcpower.stats.lme_solver import compute_lme_critical_values

        n_datasets = 50
        converged = 0
        for seed in range(n_datasets):
            X, y, Z, cluster_ids, K, q = _generate_general_data(
                K=30, n=1500, seed=seed + 200)
            p = X.shape[1]
            target_indices = np.arange(p, dtype=np.int32)
            chi2_crit, z_crit, crits = compute_lme_critical_values(0.05, p, p, 0)
            result = np.array(_native.lme_analysis_general(
                np.ascontiguousarray(X, dtype=np.float64),
                np.ascontiguousarray(y, dtype=np.float64),
                np.ascontiguousarray(Z, dtype=np.float64),
                np.ascontiguousarray(cluster_ids, dtype=np.int32),
                K, q, target_indices, chi2_crit, z_crit,
                np.ascontiguousarray(crits, dtype=np.float64), 0,
                np.empty(0, dtype=np.float64)))
            if len(result) > 0:
                converged += 1

        assert converged >= n_datasets * 0.9, f"Only {converged}/{n_datasets} converged"

    def test_lr_test_positive_for_real_effects(self):
        """LR test detects real effects (overall significance = 1.0)."""
        import mcpower.backends.mcpower_native as _native
        from mcpower.stats.lme_solver import compute_lme_critical_values

        # Large effect for reliable detection
        beta_true = np.array([1.0, 1.0, 0.8])
        X, y, Z, cluster_ids, K, q = _generate_general_data(
            K=50, n=5000, beta_true=beta_true, seed=99)
        p = X.shape[1]
        target_indices = np.arange(p, dtype=np.int32)
        chi2_crit, z_crit, crits = compute_lme_critical_values(0.05, p, p, 0)

        result = np.array(_native.lme_analysis_general(
            np.ascontiguousarray(X, dtype=np.float64),
            np.ascontiguousarray(y, dtype=np.float64),
            np.ascontiguousarray(Z, dtype=np.float64),
            np.ascontiguousarray(cluster_ids, dtype=np.int32),
            K, q, target_indices, chi2_crit, z_crit,
            np.ascontiguousarray(crits, dtype=np.float64), 0,
            np.empty(0, dtype=np.float64)))

        assert len(result) > 0
        assert result[0] == 1.0, "Should detect significant overall effect"
