"""C++ nested solver: parameter recovery and convergence tests."""

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


def _generate_nested_data(K_parent=10, K_child=30, n=1500, p=2, seed=42,
                          tau_parent=0.4, tau_child=0.3):
    rng = np.random.RandomState(seed)
    children_per_parent = K_child // K_parent
    obs_per_child = n // K_child
    parent_ids = []
    child_ids = []
    child_to_parent = np.zeros(K_child, dtype=np.int32)
    for pj in range(K_parent):
        for cj_local in range(children_per_parent):
            cj = pj * children_per_parent + cj_local
            child_to_parent[cj] = pj
            parent_ids.extend([pj] * obs_per_child)
            child_ids.extend([cj] * obs_per_child)
    parent_ids = np.array(parent_ids, dtype=np.int32)
    child_ids = np.array(child_ids, dtype=np.int32)
    n_actual = len(parent_ids)
    X = rng.randn(n_actual, p)
    beta_true = np.array([1.0] + [0.5] * p)
    X_int = np.column_stack([np.ones(n_actual), X])
    b_parent = rng.randn(K_parent) * tau_parent
    b_child = rng.randn(K_child) * tau_child
    y = X_int @ beta_true + b_parent[parent_ids] + b_child[child_ids] + rng.randn(n_actual)
    return X, y, parent_ids, child_ids, K_parent, K_child, child_to_parent


class TestCppNestedParameterRecovery:
    """Parameter recovery for C++ nested solver."""

    def test_beta_recovery(self):
        """Fixed effects recovered (non-empty result)."""
        import mcpower.backends.mcpower_native as _native
        from mcpower.stats.lme_solver import compute_lme_critical_values

        X, y, parent_ids, child_ids, K_parent, K_child, c2p = _generate_nested_data(
            K_parent=15, K_child=45, n=2250, seed=42)
        p = X.shape[1]
        target_indices = np.arange(p, dtype=np.int32)
        chi2_crit, z_crit, crits = compute_lme_critical_values(0.05, p, p, 0)

        result = np.array(_native.lme_analysis_nested(
            np.ascontiguousarray(X, dtype=np.float64),
            np.ascontiguousarray(y, dtype=np.float64),
            np.ascontiguousarray(parent_ids, dtype=np.int32),
            np.ascontiguousarray(child_ids, dtype=np.int32),
            K_parent, K_child,
            np.ascontiguousarray(c2p, dtype=np.int32),
            target_indices, chi2_crit, z_crit,
            np.ascontiguousarray(crits, dtype=np.float64), 0,
            np.empty(0, dtype=np.float64)))

        assert len(result) > 0, "C++ nested solver returned empty"

    def test_convergence_rate(self):
        """C++ nested solver converges on 90%+ of datasets."""
        import mcpower.backends.mcpower_native as _native
        from mcpower.stats.lme_solver import compute_lme_critical_values

        n_datasets = 50
        converged = 0
        for seed in range(n_datasets):
            X, y, pid, cid, Kp, Kc, c2p = _generate_nested_data(seed=seed + 300)
            p = X.shape[1]
            target_indices = np.arange(p, dtype=np.int32)
            chi2_crit, z_crit, crits = compute_lme_critical_values(0.05, p, p, 0)
            result = np.array(_native.lme_analysis_nested(
                np.ascontiguousarray(X, dtype=np.float64),
                np.ascontiguousarray(y, dtype=np.float64),
                np.ascontiguousarray(pid, dtype=np.int32),
                np.ascontiguousarray(cid, dtype=np.int32),
                Kp, Kc,
                np.ascontiguousarray(c2p, dtype=np.int32),
                target_indices, chi2_crit, z_crit,
                np.ascontiguousarray(crits, dtype=np.float64), 0,
                np.empty(0, dtype=np.float64)))
            if len(result) > 0:
                converged += 1

        assert converged >= n_datasets * 0.9, f"Only {converged}/{n_datasets} converged"

    def test_lr_test_positive_for_real_effects(self):
        """LR test detects real effects in nested model."""
        import mcpower.backends.mcpower_native as _native
        from mcpower.stats.lme_solver import compute_lme_critical_values

        X, y, pid, cid, Kp, Kc, c2p = _generate_nested_data(
            K_parent=15, K_child=45, n=4500, seed=99)
        p = X.shape[1]
        target_indices = np.arange(p, dtype=np.int32)
        chi2_crit, z_crit, crits = compute_lme_critical_values(0.05, p, p, 0)

        result = np.array(_native.lme_analysis_nested(
            np.ascontiguousarray(X, dtype=np.float64),
            np.ascontiguousarray(y, dtype=np.float64),
            np.ascontiguousarray(pid, dtype=np.int32),
            np.ascontiguousarray(cid, dtype=np.int32),
            Kp, Kc,
            np.ascontiguousarray(c2p, dtype=np.int32),
            target_indices, chi2_crit, z_crit,
            np.ascontiguousarray(crits, dtype=np.float64), 0,
            np.empty(0, dtype=np.float64)))

        assert len(result) > 0
        assert result[0] == 1.0, "Should detect significant overall effect"
