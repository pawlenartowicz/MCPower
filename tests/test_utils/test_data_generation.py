"""
Tests for data generation utilities.
"""

import pytest
import numpy as np
import pandas as pd
from mcpower.utils.data_generation import (
    _generate_X, create_uploaded_lookup_tables
)


class TestGenerateX:
    """Test X matrix generation."""
    
    def test_basic_generation(self):
        """Test basic X matrix generation."""
        X = _generate_X(
            sample_size=100,
            n_vars=3,
            correlation_matrix=np.eye(3),
            var_types=np.array([0, 0, 0]),  # all normal
            var_params=np.array([0.5, 0.5, 0.5]),
            seed=42
        )
        
        assert X.shape == (100, 3)
        assert not np.any(np.isnan(X))
        assert not np.any(np.isinf(X))
    
    def test_reproducibility(self):
        """Test same seed gives same results."""
        params = {
            'sample_size': 50,
            'n_vars': 2,
            'correlation_matrix': np.eye(2),
            'var_types': np.array([0, 1]),
            'var_params': np.array([0.5, 0.5])
        }
        
        X1 = _generate_X(**params, seed=123)
        X2 = _generate_X(**params, seed=123)
        
        np.testing.assert_array_equal(X1, X2)
    
    def test_different_seeds(self):
        """Test different seeds give different results."""
        params = {
            'sample_size': 50,
            'n_vars': 2,
            'correlation_matrix': np.eye(2),
            'var_types': np.array([0, 0]),
            'var_params': np.array([0.5, 0.5])
        }
        
        X1 = _generate_X(**params, seed=123)
        X2 = _generate_X(**params, seed=456)
        
        assert not np.array_equal(X1, X2)


class TestVariableTypes:
    """Test different variable type generation."""
    
    def test_normal_variables(self):
        """Test normal distribution generation."""
        X = _generate_X(
            sample_size=1000,
            n_vars=2,
            correlation_matrix=np.eye(2),
            var_types=np.array([0, 0]),  # normal
            var_params=np.array([0.5, 0.5]),
            seed=42
        )
        
        # Check approximately normal
        for col in range(2):
            mean = np.mean(X[:, col])
            std = np.std(X[:, col])
            assert abs(mean) < 0.2  # Close to 0
            assert abs(std - 1) < 0.2  # Close to 1
    
    def test_binary_variables(self):
        """Test binary variable generation."""
        X = _generate_X(
            sample_size=1000,
            n_vars=2,
            correlation_matrix=np.eye(2),
            var_types=np.array([1, 1]),  # binary
            var_params=np.array([0.3, 0.7]),  # different proportions
            seed=42
        )
        
        # Check binary values (after centering)
        for col in range(2):
            unique_vals = np.unique(X[:, col])
            assert len(unique_vals) <= 3  # At most 3 unique values after centering
    
    def test_skewed_variables(self):
        """Test skewed distribution generation."""
        X = _generate_X(
            sample_size=1000,
            n_vars=2,
            correlation_matrix=np.eye(2),
            var_types=np.array([2, 3]),  # right_skewed, left_skewed
            var_params=np.array([0.5, 0.5]),
            seed=42
        )
        
        assert X.shape == (1000, 2)
        assert not np.any(np.isnan(X))
    
    def test_high_kurtosis(self):
        """Test high kurtosis distribution."""
        X = _generate_X(
            sample_size=1000,
            n_vars=1,
            correlation_matrix=np.eye(1),
            var_types=np.array([4]),  # high_kurtosis
            var_params=np.array([0.5]),
            seed=42
        )
        
        assert X.shape == (1000, 1)
        assert not np.any(np.isnan(X))
    
    def test_uniform_variables(self):
        """Test uniform distribution generation."""
        X = _generate_X(
            sample_size=1000,
            n_vars=1,
            correlation_matrix=np.eye(1),
            var_types=np.array([5]),  # uniform
            var_params=np.array([0.5]),
            seed=42
        )
        
        assert X.shape == (1000, 1)
        # Uniform should be bounded
        assert np.all(X >= -2)  # Rough bounds
        assert np.all(X <= 2)


class TestCorrelations:
    """Test correlation handling."""
    
    def test_identity_correlation(self):
        """Test uncorrelated variables."""
        X = _generate_X(
            sample_size=1000,
            n_vars=3,
            correlation_matrix=np.eye(3),
            var_types=np.array([0, 0, 0]),
            var_params=np.array([0.5, 0.5, 0.5]),
            seed=42
        )
        
        # Check correlations are close to 0
        corr_matrix = np.corrcoef(X.T)
        off_diagonal = corr_matrix[np.triu_indices_from(corr_matrix, k=1)]
        assert np.all(np.abs(off_diagonal) < 0.15)  # Loose tolerance
    
    def test_positive_correlation(self):
        """Test positive correlations."""
        corr_matrix = np.array([
            [1.0, 0.5, 0.3],
            [0.5, 1.0, 0.2],
            [0.3, 0.2, 1.0]
        ])
        
        X = _generate_X(
            sample_size=2000,
            n_vars=3,
            correlation_matrix=corr_matrix,
            var_types=np.array([0, 0, 0]),
            var_params=np.array([0.5, 0.5, 0.5]),
            seed=42
        )
        
        # Check correlations are in right direction
        empirical_corr = np.corrcoef(X.T)
        assert empirical_corr[0, 1] > 0.2  # Should be positive
        assert empirical_corr[0, 2] > 0.1
    
    def test_negative_correlation(self):
        """Test negative correlations."""
        corr_matrix = np.array([
            [1.0, -0.6],
            [-0.6, 1.0]
        ])
        
        X = _generate_X(
            sample_size=1000,
            n_vars=2,
            correlation_matrix=corr_matrix,
            var_types=np.array([0, 0]),
            var_params=np.array([0.5, 0.5]),
            seed=42
        )
        
        empirical_corr = np.corrcoef(X.T)
        assert empirical_corr[0, 1] < -0.3  # Should be negative


class TestUploadedData:
    """Test uploaded data functionality."""
    
    def test_uploaded_data_type(self):
        """Test uploaded data variable type."""
        # Create dummy uploaded data matrices
        normal_vals = np.array([[0.0, 1.0, 2.0]])  # 1 variable, 3 points
        data_vals = np.array([[-1.0, 0.0, 1.0]])
        
        X = _generate_X(
            sample_size=100,
            n_vars=1,
            correlation_matrix=np.eye(1),
            var_types=np.array([99]),  # uploaded_data
            var_params=np.array([0.5]),
            normal_values=normal_vals,
            uploaded_values=data_vals,
            seed=42
        )
        
        assert X.shape == (100, 1)
        assert not np.any(np.isnan(X))
    
    def test_mixed_uploaded_synthetic(self):
        """Test mix of uploaded and synthetic variables."""
        normal_vals = np.array([[0.0, 1.0]])  # 1 variable
        data_vals = np.array([[-1.0, 1.0]])
        
        X = _generate_X(
            sample_size=100,
            n_vars=2,
            correlation_matrix=np.eye(2),
            var_types=np.array([99, 0]),  # uploaded, normal
            var_params=np.array([0.5, 0.5]),
            normal_values=normal_vals,
            uploaded_values=data_vals,
            seed=42
        )
        
        assert X.shape == (100, 2)


class TestCreateUploadedLookupTables:
    """Test lookup table creation for uploaded data."""
    
    def test_basic_lookup_creation(self):
        """Test basic lookup table creation."""
        data_matrix = np.array([
            [1, 4],
            [2, 5], 
            [3, 6]
        ])  # 3 samples, 2 variables
        
        normal_vals, data_vals = create_uploaded_lookup_tables(data_matrix)
        
        assert normal_vals.shape[0] == 2  # 2 variables
        assert data_vals.shape[0] == 2
        assert normal_vals.shape[1] == 3  # 3 samples
        assert data_vals.shape[1] == 3
    
    def test_single_variable(self):
        """Test single variable lookup."""
        data_matrix = np.array([[1, 2, 3, 4, 5]]).T  # 5 samples, 1 variable
        
        normal_vals, data_vals = create_uploaded_lookup_tables(data_matrix)
        
        assert normal_vals.shape == (1, 5)
        assert data_vals.shape == (1, 5)
        
        # Data should be normalized and sorted
        assert np.all(np.diff(data_vals[0]) >= 0)  # Sorted
    
    def test_normalization(self):
        """Test data normalization in lookup tables."""
        # Create data with known mean/std
        raw_data = np.array([[10, 20, 30, 40, 50]]).T
        
        normal_vals, data_vals = create_uploaded_lookup_tables(raw_data)
        
        # Normalized data should have mean ≈ 0, std ≈ 1
        normalized = data_vals[0]
        assert abs(np.mean(normalized)) < 1e-10
        assert abs(np.std(normalized) - 1) < 1e-10


class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_small_sample_size(self):
        """Test very small sample sizes."""
        X = _generate_X(
            sample_size=2,
            n_vars=1,
            correlation_matrix=np.eye(1),
            var_types=np.array([0]),
            var_params=np.array([0.5]),
            seed=42
        )
        
        assert X.shape == (2, 1)
    
    def test_single_variable(self):
        """Test single variable generation."""
        X = _generate_X(
            sample_size=100,
            n_vars=1,
            correlation_matrix=np.eye(1),
            var_types=np.array([1]),  # binary
            var_params=np.array([0.7]),
            seed=42
        )
        
        assert X.shape == (100, 1)
    
    def test_extreme_correlations(self):
        """Test near-perfect correlations."""
        corr_matrix = np.array([
            [1.0, 0.99],
            [0.99, 1.0]
        ])
        
        X = _generate_X(
            sample_size=100,
            n_vars=2,
            correlation_matrix=corr_matrix,
            var_types=np.array([0, 0]),
            var_params=np.array([0.5, 0.5]),
            seed=42
        )
        
        assert X.shape == (100, 2)
        assert not np.any(np.isnan(X))
    
    def test_binary_extreme_proportions(self):
        """Test binary variables with extreme proportions."""
        X = _generate_X(
            sample_size=1000,
            n_vars=2,
            correlation_matrix=np.eye(2),
            var_types=np.array([1, 1]),
            var_params=np.array([0.01, 0.99]),  # Very unbalanced
            seed=42
        )
        
        assert X.shape == (1000, 2)
        assert not np.any(np.isnan(X))


class TestPerformance:
    """Test performance-related aspects."""
    
    def test_large_sample_size(self):
        """Test large sample size generation."""
        X = _generate_X(
            sample_size=10000,
            n_vars=5,
            correlation_matrix=np.eye(5),
            var_types=np.array([0, 1, 2, 3, 4]),
            var_params=np.array([0.5, 0.3, 0.5, 0.5, 0.5]),
            seed=42
        )
        
        assert X.shape == (10000, 5)
        assert not np.any(np.isnan(X))
    
    def test_many_variables(self):
        """Test many variables."""
        n_vars = 20
        X = _generate_X(
            sample_size=100,
            n_vars=n_vars,
            correlation_matrix=np.eye(n_vars),
            var_types=np.zeros(n_vars, dtype=int),
            var_params=np.full(n_vars, 0.5),
            seed=42
        )
        
        assert X.shape == (100, n_vars)


class TestDefaultParameters:
    """Test default parameter handling."""
    
    def test_none_correlation_matrix(self):
        """Test None correlation matrix."""
        X = _generate_X(
            sample_size=100,
            n_vars=2,
            var_types=np.array([0, 0]),
            var_params=np.array([0.5, 0.5]),
            seed=42
        )
        
        assert X.shape == (100, 2)
    
    def test_none_uploaded_data(self):
        """Test None uploaded data matrices."""
        X = _generate_X(
            sample_size=100,
            n_vars=2,
            correlation_matrix=np.eye(2),
            var_types=np.array([0, 0]),
            var_params=np.array([0.5, 0.5]),
            seed=42
        )
        
        assert X.shape == (100, 2)
    
    def test_no_seed(self):
        """Test without explicit seed."""
        X = _generate_X(
            sample_size=50,
            n_vars=2,
            correlation_matrix=np.eye(2),
            var_types=np.array([0, 0]),
            var_params=np.array([0.5, 0.5])
        )
        
        assert X.shape == (50, 2)