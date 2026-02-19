"""
Tests for data generation utilities.
"""

import numpy as np


def _generate_X_via_backend(n_samples, n_vars, corr, var_types, var_params, upload_normal, upload_data, seed):
    """Helper: call NativeBackend.generate_X with proper dtypes."""
    from mcpower.backends.native import NativeBackend

    backend = NativeBackend()
    return backend.generate_X(
        n_samples,
        n_vars,
        np.ascontiguousarray(corr, dtype=np.float64),
        np.ascontiguousarray(var_types, dtype=np.int32),
        np.ascontiguousarray(var_params, dtype=np.float64),
        np.ascontiguousarray(upload_normal, dtype=np.float64),
        np.ascontiguousarray(upload_data, dtype=np.float64),
        seed,
    )


class TestGenerateX:
    """Test generate_X via NativeBackend."""

    def test_basic_generation(self):
        n_samples = 100
        n_vars = 3
        corr = np.eye(n_vars)
        var_types = np.zeros(n_vars, dtype=np.int32)
        var_params = np.zeros(n_vars, dtype=np.float64)
        upload_normal = np.zeros((2, 2))
        upload_data = np.zeros((2, 2))

        X = _generate_X_via_backend(n_samples, n_vars, corr, var_types, var_params, upload_normal, upload_data, seed=42)

        assert X.shape == (n_samples, n_vars)

    def test_reproducibility_with_seed(self):
        n_samples = 50
        n_vars = 2
        corr = np.eye(n_vars)
        var_types = np.zeros(n_vars, dtype=np.int32)
        var_params = np.zeros(n_vars, dtype=np.float64)
        upload_normal = np.zeros((2, 2))
        upload_data = np.zeros((2, 2))

        X1 = _generate_X_via_backend(n_samples, n_vars, corr, var_types, var_params, upload_normal, upload_data, seed=42)
        X2 = _generate_X_via_backend(n_samples, n_vars, corr, var_types, var_params, upload_normal, upload_data, seed=42)

        assert np.allclose(X1, X2)

    def test_different_seeds(self):
        n_samples = 50
        n_vars = 2
        corr = np.eye(n_vars)
        var_types = np.zeros(n_vars, dtype=np.int32)
        var_params = np.zeros(n_vars, dtype=np.float64)
        upload_normal = np.zeros((2, 2))
        upload_data = np.zeros((2, 2))

        X1 = _generate_X_via_backend(n_samples, n_vars, corr, var_types, var_params, upload_normal, upload_data, seed=42)
        X2 = _generate_X_via_backend(n_samples, n_vars, corr, var_types, var_params, upload_normal, upload_data, seed=123)

        assert not np.allclose(X1, X2)

    def test_correlation_structure(self):
        n_samples = 5000  # Large sample for correlation estimation
        n_vars = 2
        target_corr = np.array([[1.0, 0.7], [0.7, 1.0]])
        var_types = np.zeros(n_vars, dtype=np.int32)
        var_params = np.zeros(n_vars, dtype=np.float64)
        upload_normal = np.zeros((2, 2))
        upload_data = np.zeros((2, 2))

        X = _generate_X_via_backend(n_samples, n_vars, target_corr, var_types, var_params, upload_normal, upload_data, seed=42)

        # Empirical correlation should be close to target
        empirical_corr = np.corrcoef(X.T)
        assert abs(empirical_corr[0, 1] - 0.7) < 0.05

    def test_binary_variable(self):
        n_samples = 1000
        n_vars = 1
        corr = np.eye(n_vars)
        var_types = np.array([1], dtype=np.int32)  # binary
        var_params = np.array([0.3], dtype=np.float64)  # 30% proportion
        upload_normal = np.zeros((2, 2))
        upload_data = np.zeros((2, 2))

        X = _generate_X_via_backend(n_samples, n_vars, corr, var_types, var_params, upload_normal, upload_data, seed=42)

        # Binary variables are standardized, so check for exactly 2 unique values
        unique_vals = np.unique(X)
        assert len(unique_vals) == 2

        # The lower value should be more common (70%) for 0.3 proportion
        lower_val = unique_vals[0]
        proportion_lower = np.mean(X == lower_val)
        assert abs(proportion_lower - 0.7) < 0.05  # ~70% should be the "0" group


class TestGenerateFactors:
    """Test _generate_factors function."""

    def test_basic_factor(self):
        from mcpower.stats.data_generation import _generate_factors

        n_samples = 100
        specs = [{"n_levels": 3, "proportions": [1 / 3, 1 / 3, 1 / 3]}]

        X = _generate_factors(n_samples, specs, seed=42)

        assert X.shape == (n_samples, 2)  # n_levels - 1 dummies

    def test_factor_dummy_coding(self):
        from mcpower.stats.data_generation import _generate_factors

        n_samples = 100
        specs = [{"n_levels": 3, "proportions": [1 / 3, 1 / 3, 1 / 3]}]

        X = _generate_factors(n_samples, specs, seed=42)

        # Each row should have at most one 1 (or all zeros for reference)
        for row in X:
            assert sum(row) <= 1

    def test_factor_proportions(self):
        from mcpower.stats.data_generation import _generate_factors

        n_samples = 3000
        specs = [{"n_levels": 3, "proportions": [0.5, 0.3, 0.2]}]

        X = _generate_factors(n_samples, specs, seed=42)

        # Reference category (level 1) = all zeros
        reference_count = np.sum(np.all(X == 0, axis=1))
        level2_count = np.sum(X[:, 0] == 1)
        level3_count = np.sum(X[:, 1] == 1)

        # Check proportions (with tolerance)
        assert abs(reference_count / n_samples - 0.5) < 0.05
        assert abs(level2_count / n_samples - 0.3) < 0.05
        assert abs(level3_count / n_samples - 0.2) < 0.05

    def test_multiple_factors(self):
        from mcpower.stats.data_generation import _generate_factors

        n_samples = 100
        specs = [{"n_levels": 3, "proportions": [1 / 3, 1 / 3, 1 / 3]}, {"n_levels": 2, "proportions": [0.5, 0.5]}]

        X = _generate_factors(n_samples, specs, seed=42)

        # 2 dummies from first + 1 dummy from second = 3 columns
        assert X.shape == (n_samples, 3)

    def test_no_factors(self):
        from mcpower.stats.data_generation import _generate_factors

        n_samples = 100
        specs = []

        X = _generate_factors(n_samples, specs, seed=42)

        assert X.shape == (n_samples, 0)


class TestCreateUploadedLookupTables:
    """Test create_uploaded_lookup_tables function."""

    def test_basic_creation(self):
        from mcpower.stats.data_generation import create_uploaded_lookup_tables

        np.random.seed(42)
        data = np.random.randn(100, 2)

        normal_vals, uploaded_vals = create_uploaded_lookup_tables(data)

        assert normal_vals.shape == (2, 100)
        assert uploaded_vals.shape == (2, 100)

    def test_sorted_uploaded_values(self):
        from mcpower.stats.data_generation import create_uploaded_lookup_tables

        np.random.seed(42)
        data = np.random.randn(50, 2)

        normal_vals, uploaded_vals = create_uploaded_lookup_tables(data)

        # Uploaded values should be sorted
        for i in range(2):
            assert np.all(np.diff(uploaded_vals[i]) >= 0)

    def test_normal_quantiles_range(self):
        from mcpower.stats.data_generation import create_uploaded_lookup_tables

        np.random.seed(42)
        data = np.random.randn(100, 2)

        normal_vals, uploaded_vals = create_uploaded_lookup_tables(data)

        # Normal quantiles should span reasonable range
        for i in range(2):
            assert normal_vals[i].min() < -2
            assert normal_vals[i].max() > 2


class TestDistributionTypes:
    """Test different distribution types via NativeBackend."""

    def test_normal_distribution(self):
        n_samples = 5000
        n_vars = 1
        corr = np.eye(n_vars)
        var_types = np.array([0], dtype=np.int32)  # normal
        var_params = np.zeros(n_vars, dtype=np.float64)
        upload_normal = np.zeros((2, 2))
        upload_data = np.zeros((2, 2))

        X = _generate_X_via_backend(n_samples, n_vars, corr, var_types, var_params, upload_normal, upload_data, seed=42)

        # Should be approximately standard normal
        assert abs(np.mean(X)) < 0.1
        assert abs(np.std(X) - 1.0) < 0.1

    def test_right_skewed_distribution(self):
        from scipy.stats import skew

        n_samples = 5000
        n_vars = 1
        corr = np.eye(n_vars)
        var_types = np.array([2], dtype=np.int32)  # right_skewed
        var_params = np.zeros(n_vars, dtype=np.float64)
        upload_normal = np.zeros((2, 2))
        upload_data = np.zeros((2, 2))

        X = _generate_X_via_backend(n_samples, n_vars, corr, var_types, var_params, upload_normal, upload_data, seed=42)

        # Should have positive skewness
        assert skew(X.flatten()) > 0.5

    def test_uniform_distribution(self):
        n_samples = 5000
        n_vars = 1
        corr = np.eye(n_vars)
        var_types = np.array([5], dtype=np.int32)  # uniform
        var_params = np.zeros(n_vars, dtype=np.float64)
        upload_normal = np.zeros((2, 2))
        upload_data = np.zeros((2, 2))

        X = _generate_X_via_backend(n_samples, n_vars, corr, var_types, var_params, upload_normal, upload_data, seed=42)

        # Uniform should have values roughly evenly distributed
        # Skewness should be near zero
        from scipy.stats import skew

        assert abs(skew(X.flatten())) < 0.2

    def test_left_skewed_distribution(self):
        from scipy.stats import skew

        n_samples = 5000
        n_vars = 1
        corr = np.eye(n_vars)
        var_types = np.array([3], dtype=np.int32)  # left_skewed
        var_params = np.zeros(n_vars, dtype=np.float64)
        upload_normal = np.zeros((2, 2))
        upload_data = np.zeros((2, 2))

        X = _generate_X_via_backend(n_samples, n_vars, corr, var_types, var_params, upload_normal, upload_data, seed=42)

        # Should have negative skewness
        assert skew(X.flatten()) < -0.5

    def test_high_kurtosis_distribution(self):
        from scipy.stats import kurtosis

        n_samples = 5000
        n_vars = 1
        corr = np.eye(n_vars)
        var_types = np.array([4], dtype=np.int32)  # high_kurtosis (t(3))
        var_params = np.zeros(n_vars, dtype=np.float64)
        upload_normal = np.zeros((2, 2))
        upload_data = np.zeros((2, 2))

        X = _generate_X_via_backend(n_samples, n_vars, corr, var_types, var_params, upload_normal, upload_data, seed=42)

        # t(3)/sqrt(3) should have higher kurtosis than normal (excess kurtosis > 0)
        assert kurtosis(X.flatten()) > 1.0

    def test_uploaded_data_distribution(self):
        from mcpower.stats.data_generation import create_uploaded_lookup_tables

        np.random.seed(42)
        # Create uploaded data with known shape
        raw_data = np.random.exponential(2, (100, 1))
        normal_vals, uploaded_vals = create_uploaded_lookup_tables(raw_data)

        n_samples = 500
        n_vars = 1
        corr = np.eye(n_vars)
        var_types = np.array([99], dtype=np.int32)  # uploaded_data
        var_params = np.zeros(n_vars, dtype=np.float64)

        X = _generate_X_via_backend(n_samples, n_vars, corr, var_types, var_params, normal_vals, uploaded_vals, seed=42)

        assert X.shape == (n_samples, n_vars)
        # Values should not all be identical
        assert np.std(X) > 0


class TestGenerateClusterEffects:
    """Test _generate_cluster_effects function."""

    def test_basic_cluster_effects(self):
        from mcpower.stats.data_generation import _generate_cluster_effects

        sample_size = 100
        cluster_specs = {"school": {"n_clusters": 10, "cluster_size": 10, "tau_squared": 0.5}}
        result = _generate_cluster_effects(sample_size, cluster_specs, sim_seed=42)
        assert result.shape == (sample_size, 1)

    def test_cluster_effects_shape_multiple(self):
        from mcpower.stats.data_generation import _generate_cluster_effects

        sample_size = 60
        cluster_specs = {
            "school": {"n_clusters": 3, "cluster_size": 20, "tau_squared": 0.3},
        }
        result = _generate_cluster_effects(sample_size, cluster_specs, sim_seed=42)
        assert result.shape == (sample_size, 1)

    def test_cluster_effects_variance(self):
        from mcpower.stats.data_generation import _generate_cluster_effects

        sample_size = 1000
        tau_sq = 1.0
        cluster_specs = {"school": {"n_clusters": 100, "cluster_size": 10, "tau_squared": tau_sq}}
        result = _generate_cluster_effects(sample_size, cluster_specs, sim_seed=42)
        # Unique values should be ~100 (one per cluster)
        unique_vals = np.unique(result)
        assert len(unique_vals) == 100

    def test_cluster_effects_seed_reproducibility(self):
        from mcpower.stats.data_generation import _generate_cluster_effects

        sample_size = 50
        specs = {"g": {"n_clusters": 5, "cluster_size": 10, "tau_squared": 0.5}}
        r1 = _generate_cluster_effects(sample_size, specs, sim_seed=42)
        r2 = _generate_cluster_effects(sample_size, specs, sim_seed=42)
        assert np.allclose(r1, r2)

    def test_cluster_effects_empty_specs(self):
        from mcpower.stats.data_generation import _generate_cluster_effects

        result = _generate_cluster_effects(100, {}, sim_seed=42)
        assert result.shape == (100, 0)

    def test_cluster_effects_trim_to_sample_size(self):
        from mcpower.stats.data_generation import _generate_cluster_effects

        # cluster_size * n_clusters > sample_size → should be trimmed
        sample_size = 25
        specs = {"g": {"n_clusters": 3, "cluster_size": 10, "tau_squared": 0.5}}
        result = _generate_cluster_effects(sample_size, specs, sim_seed=42)
        assert result.shape == (sample_size, 1)


class TestCholeskyEdgeCases:
    """Test Cholesky edge cases in C++ generate_X."""

    def test_near_singular_matrix(self):
        # Near-singular correlation matrix
        corr = np.array([[1.0, 0.999], [0.999, 1.0]])
        n_samples = 100
        n_vars = 2
        var_types = np.zeros(n_vars, dtype=np.int32)
        var_params = np.zeros(n_vars, dtype=np.float64)
        upload_normal = np.zeros((2, 2))
        upload_data = np.zeros((2, 2))

        # Should not raise
        X = _generate_X_via_backend(n_samples, n_vars, corr, var_types, var_params, upload_normal, upload_data, seed=42)
        assert X.shape == (n_samples, n_vars)
