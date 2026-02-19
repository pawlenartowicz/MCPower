"""C++ vs Python parity tests for all 3 LME solver types.

Tests that the C++ native backend produces statistically equivalent results
to the Python profiled-deviance solver for q=1, general (q>1), and nested models.
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


def _generate_q1_data(K=20, n=600, p=2, seed=42):
    rng = np.random.RandomState(seed)
    obs_per_cluster = n // K
    cluster_ids = np.repeat(np.arange(K), obs_per_cluster).astype(np.int32)
    n_actual = len(cluster_ids)
    X = rng.randn(n_actual, p)
    beta_true = rng.randn(p + 1) * 0.5
    X_int = np.column_stack([np.ones(n_actual), X])
    tau = 0.5
    b = rng.randn(K) * tau
    y = X_int @ beta_true + b[cluster_ids] + rng.randn(n_actual)
    return X, y, cluster_ids, K


def _generate_general_data(K=30, n=1500, p=2, q=2, seed=42):
    rng = np.random.RandomState(seed)
    obs_per_cluster = n // K
    cluster_ids = np.repeat(np.arange(K), obs_per_cluster).astype(np.int32)
    n_actual = len(cluster_ids)
    X = rng.randn(n_actual, p)
    beta_true = np.array([1.0] + [0.5] * p)
    X_int = np.column_stack([np.ones(n_actual), X])
    Z = np.column_stack([np.ones(n_actual), X[:, 0]])
    G = np.array([[0.25, 0.05], [0.05, 0.1]])
    L_G = np.linalg.cholesky(G)
    b = rng.randn(K, q) @ L_G.T
    y = X_int @ beta_true + np.sum(Z * b[cluster_ids], axis=1) + rng.randn(n_actual)
    return X, y, Z, cluster_ids, K, q


def _generate_nested_data(K_parent=10, K_child=30, n=1500, p=2, seed=42):
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
    b_parent = rng.randn(K_parent) * 0.4
    b_child = rng.randn(K_child) * 0.3
    y = X_int @ beta_true + b_parent[parent_ids] + b_child[child_ids] + rng.randn(n_actual)
    return X, y, parent_ids, child_ids, K_parent, K_child, child_to_parent


class TestQ1Parity:
    """C++ vs Python parity for q=1 random intercept."""

    def test_q1_analysis_parity(self):
        """Same data -> same result array (within 1e-6)."""
        from mcpower.stats.lme_solver import compute_lme_critical_values, lme_analysis_full
        import mcpower.backends.mcpower_native as _native

        X, y, cluster_ids, K = _generate_q1_data()
        p = X.shape[1]
        n_targets = p
        target_indices = np.arange(n_targets, dtype=np.int32)
        chi2_crit, z_crit, correction_z_crits = compute_lme_critical_values(0.05, p, n_targets, 0)

        py_result = lme_analysis_full(X, y, cluster_ids, K, target_indices,
                                       chi2_crit, z_crit, correction_z_crits, 0)
        cpp_result = np.array(_native.lme_analysis(
            np.ascontiguousarray(X, dtype=np.float64),
            np.ascontiguousarray(y, dtype=np.float64),
            np.ascontiguousarray(cluster_ids, dtype=np.int32),
            K, target_indices, chi2_crit, z_crit,
            np.ascontiguousarray(correction_z_crits, dtype=np.float64), 0, -1.0))

        assert py_result is not None
        assert len(cpp_result) > 0
        # Binary decisions should match exactly
        np.testing.assert_array_equal(py_result[:1 + n_targets], cpp_result[:1 + n_targets])

    def test_q1_multiple_seeds(self):
        """Statistical equivalence across 50 datasets."""
        from mcpower.stats.lme_solver import compute_lme_critical_values, lme_analysis_full
        import mcpower.backends.mcpower_native as _native

        agree_count = 0
        n_tests = 50
        for seed in range(n_tests):
            X, y, cluster_ids, K = _generate_q1_data(seed=seed + 100)
            p = X.shape[1]
            target_indices = np.arange(p, dtype=np.int32)
            chi2_crit, z_crit, correction_z_crits = compute_lme_critical_values(0.05, p, p, 0)

            py_result = lme_analysis_full(X, y, cluster_ids, K, target_indices,
                                           chi2_crit, z_crit, correction_z_crits, 0)
            cpp_result = np.array(_native.lme_analysis(
                np.ascontiguousarray(X, dtype=np.float64),
                np.ascontiguousarray(y, dtype=np.float64),
                np.ascontiguousarray(cluster_ids, dtype=np.int32),
                K, target_indices, chi2_crit, z_crit,
                np.ascontiguousarray(correction_z_crits, dtype=np.float64), 0, -1.0))

            if py_result is not None and len(cpp_result) > 0:
                if np.allclose(py_result, cpp_result, atol=1e-6):
                    agree_count += 1
        # At least 90% should agree exactly
        assert agree_count >= n_tests * 0.9, f"Only {agree_count}/{n_tests} agreed"


class TestGeneralParity:
    """C++ vs Python parity for general q>1."""

    def test_general_analysis_parity(self):
        """Same data -> same result array for q>1."""
        from mcpower.stats.lme_solver import compute_lme_critical_values, lme_analysis_general
        import mcpower.backends.mcpower_native as _native

        X, y, Z, cluster_ids, K, q = _generate_general_data()
        p = X.shape[1]
        n_targets = p
        target_indices = np.arange(n_targets, dtype=np.int32)
        chi2_crit, z_crit, correction_z_crits = compute_lme_critical_values(0.05, p, n_targets, 0)

        py_result = lme_analysis_general(X, y, cluster_ids, K, q, Z,
                                          target_indices, chi2_crit, z_crit,
                                          correction_z_crits, 0)
        cpp_result = np.array(_native.lme_analysis_general(
            np.ascontiguousarray(X, dtype=np.float64),
            np.ascontiguousarray(y, dtype=np.float64),
            np.ascontiguousarray(Z, dtype=np.float64),
            np.ascontiguousarray(cluster_ids, dtype=np.int32),
            K, q, target_indices, chi2_crit, z_crit,
            np.ascontiguousarray(correction_z_crits, dtype=np.float64), 0,
            np.empty(0, dtype=np.float64)))

        assert py_result is not None
        assert len(cpp_result) > 0
        # Binary decisions should match
        np.testing.assert_array_equal(py_result[:1 + n_targets], cpp_result[:1 + n_targets])

    def test_general_corrections_parity(self):
        """Bonferroni corrections produce matching results."""
        from mcpower.stats.lme_solver import compute_lme_critical_values, lme_analysis_general
        import mcpower.backends.mcpower_native as _native

        X, y, Z, cluster_ids, K, q = _generate_general_data()
        p = X.shape[1]
        n_targets = p
        target_indices = np.arange(n_targets, dtype=np.int32)
        chi2_crit, z_crit, correction_z_crits = compute_lme_critical_values(0.05, p, n_targets, 1)

        py_result = lme_analysis_general(X, y, cluster_ids, K, q, Z,
                                          target_indices, chi2_crit, z_crit,
                                          correction_z_crits, 1)
        cpp_result = np.array(_native.lme_analysis_general(
            np.ascontiguousarray(X, dtype=np.float64),
            np.ascontiguousarray(y, dtype=np.float64),
            np.ascontiguousarray(Z, dtype=np.float64),
            np.ascontiguousarray(cluster_ids, dtype=np.int32),
            K, q, target_indices, chi2_crit, z_crit,
            np.ascontiguousarray(correction_z_crits, dtype=np.float64), 1,
            np.empty(0, dtype=np.float64)))

        assert py_result is not None and len(cpp_result) > 0
        np.testing.assert_array_equal(py_result, cpp_result)


class TestNestedParity:
    """C++ vs Python parity for nested random intercepts."""

    def test_nested_analysis_parity(self):
        """Same data -> same result array for nested."""
        from mcpower.stats.lme_solver import compute_lme_critical_values, lme_analysis_nested
        import mcpower.backends.mcpower_native as _native

        X, y, parent_ids, child_ids, K_parent, K_child, child_to_parent = _generate_nested_data()
        p = X.shape[1]
        n_targets = p
        target_indices = np.arange(n_targets, dtype=np.int32)
        chi2_crit, z_crit, correction_z_crits = compute_lme_critical_values(0.05, p, n_targets, 0)

        py_result = lme_analysis_nested(X, y, parent_ids, child_ids,
                                         K_parent, K_child, child_to_parent,
                                         target_indices, chi2_crit, z_crit,
                                         correction_z_crits, 0)
        cpp_result = np.array(_native.lme_analysis_nested(
            np.ascontiguousarray(X, dtype=np.float64),
            np.ascontiguousarray(y, dtype=np.float64),
            np.ascontiguousarray(parent_ids, dtype=np.int32),
            np.ascontiguousarray(child_ids, dtype=np.int32),
            K_parent, K_child,
            np.ascontiguousarray(child_to_parent, dtype=np.int32),
            target_indices, chi2_crit, z_crit,
            np.ascontiguousarray(correction_z_crits, dtype=np.float64), 0,
            np.empty(0, dtype=np.float64)))

        assert py_result is not None
        assert len(cpp_result) > 0
        np.testing.assert_array_equal(py_result[:1 + n_targets], cpp_result[:1 + n_targets])

    def test_nested_corrections_parity(self):
        """Holm corrections produce matching results."""
        from mcpower.stats.lme_solver import compute_lme_critical_values, lme_analysis_nested
        import mcpower.backends.mcpower_native as _native

        X, y, parent_ids, child_ids, K_parent, K_child, child_to_parent = _generate_nested_data()
        p = X.shape[1]
        n_targets = p
        target_indices = np.arange(n_targets, dtype=np.int32)
        chi2_crit, z_crit, correction_z_crits = compute_lme_critical_values(0.05, p, n_targets, 3)

        py_result = lme_analysis_nested(X, y, parent_ids, child_ids,
                                         K_parent, K_child, child_to_parent,
                                         target_indices, chi2_crit, z_crit,
                                         correction_z_crits, 3)
        cpp_result = np.array(_native.lme_analysis_nested(
            np.ascontiguousarray(X, dtype=np.float64),
            np.ascontiguousarray(y, dtype=np.float64),
            np.ascontiguousarray(parent_ids, dtype=np.int32),
            np.ascontiguousarray(child_ids, dtype=np.int32),
            K_parent, K_child,
            np.ascontiguousarray(child_to_parent, dtype=np.int32),
            target_indices, chi2_crit, z_crit,
            np.ascontiguousarray(correction_z_crits, dtype=np.float64), 3,
            np.empty(0, dtype=np.float64)))

        assert py_result is not None and len(cpp_result) > 0
        np.testing.assert_array_equal(py_result, cpp_result)
