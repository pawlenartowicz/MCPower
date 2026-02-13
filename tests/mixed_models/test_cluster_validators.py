"""
Tests for cluster configuration validators.

Tests validation of:
- ICC ranges (0 or 0.1-0.9)
- Minimum observations per cluster (>= 25)
- Cluster size constraints
"""

import pytest

from mcpower import MCPower

pytestmark = pytest.mark.lme


class TestICCValidation:
    """Test ICC validation rules."""

    def test_icc_zero_allowed(self):
        """ICC=0 should be allowed (no clustering)."""
        model = MCPower("y ~ x + (1|cluster)")
        model.set_cluster("cluster", ICC=0.0, n_clusters=5)
        # Should not raise
        assert True

    def test_icc_valid_range(self):
        """ICC between 0.1 and 0.9 should be allowed."""
        for icc in [0.1, 0.2, 0.5, 0.8, 0.9]:
            model = MCPower("y ~ x + (1|cluster)")
            model.set_cluster("cluster", ICC=icc, n_clusters=5)
            # Should not raise
        assert True

    def test_icc_too_low_rejected(self):
        """ICC between 0 and 0.1 (exclusive) should be rejected."""
        for icc in [0.01, 0.05, 0.09]:
            with pytest.raises(ValueError, match="ICC must be 0.*or between 0.1 and 0.9"):
                model = MCPower("y ~ x + (1|cluster)")
                model.set_cluster("cluster", ICC=icc, n_clusters=5)

    def test_icc_too_high_rejected(self):
        """ICC above 0.9 should be rejected."""
        for icc in [0.91, 0.95, 0.99]:
            with pytest.raises(ValueError, match="ICC must be 0.*or between 0.1 and 0.9"):
                model = MCPower("y ~ x + (1|cluster)")
                model.set_cluster("cluster", ICC=icc, n_clusters=5)

    def test_icc_negative_rejected(self):
        """Negative ICC should be rejected."""
        with pytest.raises(ValueError, match="ICC must be between 0 and 1"):
            model = MCPower("y ~ x + (1|cluster)")
            model.set_cluster("cluster", ICC=-0.1, n_clusters=5)

    def test_icc_one_rejected(self):
        """ICC=1 should be rejected (singularity)."""
        with pytest.raises(ValueError, match="ICC must be between 0 and 1"):
            model = MCPower("y ~ x + (1|cluster)")
            model.set_cluster("cluster", ICC=1.0, n_clusters=5)


class TestClusterSizeValidation:
    """Test cluster size validation rules."""

    def test_cluster_size_minimum(self):
        """cluster_size >= 15 should be allowed."""
        for size in [15, 20, 50]:
            model = MCPower("y ~ x + (1|cluster)")
            model.set_cluster("cluster", ICC=0.2, cluster_size=size)
            # Should not raise
        assert True

    def test_cluster_size_too_small_rejected(self):
        """cluster_size < 15 should be rejected."""
        for size in [1, 5, 10, 14]:
            with pytest.raises(ValueError, match="cluster_size must be.*>= 15"):
                model = MCPower("y ~ x + (1|cluster)")
                model.set_cluster("cluster", ICC=0.2, cluster_size=size)

    def test_n_clusters_minimum(self):
        """n_clusters >= 2 should be allowed."""
        for n in [2, 5, 10]:
            model = MCPower("y ~ x + (1|cluster)")
            model.set_cluster("cluster", ICC=0.2, n_clusters=n)
            # Should not raise
        assert True

    def test_n_clusters_too_small_rejected(self):
        """n_clusters < 2 should be rejected."""
        for n in [0, 1]:
            with pytest.raises(ValueError, match="n_clusters must be.*>= 2"):
                model = MCPower("y ~ x + (1|cluster)")
                model.set_cluster("cluster", ICC=0.2, n_clusters=n)


class TestSampleSizeValidation:
    """Test sample size validation with cluster configuration."""

    def test_sufficient_observations_per_cluster(self):
        """sample_size / n_clusters >= 25 should be allowed."""
        model = MCPower("y ~ x + (1|cluster)")
        model.set_cluster("cluster", ICC=0.2, n_clusters=5)
        model.set_effects("x=0.5")
        model.set_simulations(10)
        model.apply()

        # 250 / 5 = 50 (meets both minimum cluster size and complexity requirements)
        result = model.find_power(sample_size=250, return_results=True)
        assert result is not None

        # 300 / 5 = 60 (above minimum)
        result = model.find_power(sample_size=300, return_results=True)
        assert result is not None

    def test_insufficient_observations_per_cluster_rejected(self):
        """sample_size / n_clusters < 25 should be rejected."""
        model = MCPower("y ~ x + (1|cluster)")
        model.set_cluster("cluster", ICC=0.2, n_clusters=5)
        model.set_effects("x=0.5")
        model.set_simulations(10)
        model.apply()

        # 120 / 5 = 24 (below minimum)
        with pytest.raises(ValueError, match="Insufficient observations per cluster"):
            model.find_power(sample_size=120)

        # 100 / 5 = 20 (well below minimum)
        with pytest.raises(ValueError, match="Insufficient observations per cluster"):
            model.find_power(sample_size=100)

    def test_validation_message_suggestions(self):
        """Validation error should provide helpful suggestions."""
        model = MCPower("y ~ x + (1|cluster)")
        model.set_cluster("cluster", ICC=0.2, n_clusters=10)
        model.set_effects("x=0.5")
        model.set_simulations(10)
        model.apply()

        with pytest.raises(ValueError) as exc_info:
            model.find_power(sample_size=100)  # 100/10 = 10 < 25

        error_msg = str(exc_info.value)
        # Should suggest increasing sample_size
        assert "250" in error_msg  # 10 * 25 = 250
        # Should suggest reducing n_clusters
        assert "4" in error_msg  # 100 // 25 = 4


class TestValidatorIntegration:
    """Test validator integration with actual power analysis."""

    def test_valid_config_runs_successfully(self):
        """Valid configuration should run power analysis without issues."""
        model = MCPower("y ~ x + (1|cluster)")  # Single predictor for stability
        model.set_cluster("cluster", ICC=0.2, n_clusters=5)
        model.set_effects("x=0.5")
        model.set_simulations(10)
        model.apply()

        result = model.find_power(sample_size=250, return_results=True)  # 50 per cluster (meets complexity req)

        assert result is not None
        assert "results" in result
        assert "n_simulations_failed" in result["results"]

    def test_edge_case_exactly_50_per_cluster(self):
        """Exactly 50 observations per cluster should work (meets complexity requirement)."""
        model = MCPower("y ~ x + (1|cluster)")
        model.set_cluster("cluster", ICC=0.2, n_clusters=4)
        model.set_effects("x=0.5")
        model.set_simulations(10)
        model.apply()

        result = model.find_power(sample_size=200, return_results=True)  # 200/4 = 50

        assert result is not None
        assert result["results"]["n_simulations_failed"] <= 5  # Allow some failures at edge

    def test_icc_zero_no_convergence_issues(self):
        """ICC=0 should behave like OLS with no convergence issues."""
        model = MCPower("y ~ x + (1|cluster)")
        model.set_cluster("cluster", ICC=0.0, n_clusters=5)
        model.set_effects("x=0.5")
        model.set_simulations(20)
        model.apply()

        result = model.find_power(sample_size=250, return_results=True)

        assert result is not None
        # ICC=0 means no random effects, so should have minimal failures
        assert result["results"]["n_simulations_failed"] <= 2
