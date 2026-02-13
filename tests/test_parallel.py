"""
Tests for parallel execution in MCPower.
"""

import pytest


def _joblib_available():
    """Check if joblib is available."""
    import importlib.util

    return importlib.util.find_spec("joblib") is not None


pytestmark = pytest.mark.skipif(not _joblib_available(), reason="joblib not installed")


class TestParallelExecution:
    """Test parallel execution for find_sample_size."""

    def test_parallel_results_match_sequential(self, suppress_output):
        """Parallel and sequential find_sample_size produce identical power values."""
        from mcpower import MCPower

        # Create and configure model
        model = MCPower("y = x1 + x2")
        model.set_effects("x1=0.3, x2=0.2")
        model.set_seed(42)

        # Run sequential analysis
        model.set_parallel(False)
        result_seq = model.find_sample_size(from_size=50, to_size=150, by=50, return_results=True)

        # Run parallel analysis with same seed
        model.set_seed(42)
        model.set_parallel(True, n_cores=2)
        result_par = model.find_sample_size(from_size=50, to_size=150, by=50, return_results=True)

        # Compare power values (should be identical with same seed)
        seq_sizes = result_seq["results"]["sample_sizes_tested"]
        par_sizes = result_par["results"]["sample_sizes_tested"]
        assert seq_sizes == par_sizes

        seq_powers = result_seq["results"]["powers_by_test"]["overall"]
        par_powers = result_par["results"]["powers_by_test"]["overall"]

        for seq_pow, par_pow in zip(seq_powers, par_powers, strict=True):
            # Power values should be very close (allowing tiny floating point differences)
            assert abs(seq_pow - par_pow) < 1e-10

    def test_parallel_with_scenarios(self, suppress_output):
        """find_sample_size with scenarios=True works correctly in parallel."""
        from mcpower import MCPower

        # Create and configure model
        model = MCPower("y = x1 + x2")
        model.set_effects("x1=0.3, x2=0.2")
        model.set_seed(42)

        # Run sequential with scenarios
        model.set_parallel(False)
        result_seq = model.find_sample_size(from_size=50, to_size=100, by=50, scenarios=True, return_results=True)

        # Run parallel with scenarios
        model.set_seed(42)
        model.set_parallel(True, n_cores=2)
        result_par = model.find_sample_size(from_size=50, to_size=100, by=50, scenarios=True, return_results=True)

        # Both should have scenario results
        assert "scenarios" in result_seq
        assert "scenarios" in result_par
        assert "optimistic" in result_seq["scenarios"]
        assert "doomer" in result_seq["scenarios"]
        assert "realistic" in result_seq["scenarios"]
        assert "optimistic" in result_par["scenarios"]
        assert "doomer" in result_par["scenarios"]
        assert "realistic" in result_par["scenarios"]

        # Power values should match across scenarios
        for scenario in ["optimistic", "doomer", "realistic"]:
            seq_data = result_seq["scenarios"][scenario]["results"]["powers_by_test"]["overall"]
            par_data = result_par["scenarios"][scenario]["results"]["powers_by_test"]["overall"]

            assert len(seq_data) == len(par_data)
            for seq_pow, par_pow in zip(seq_data, par_data, strict=True):
                assert abs(seq_pow - par_pow) < 1e-10

    def test_parallel_with_interactions(self, suppress_output):
        """Interaction terms work correctly in parallel execution."""
        from mcpower import MCPower

        # Create model with interaction
        model = MCPower("y = a + b + a:b")
        model.set_effects("a=0.4, b=0.3, a:b=0.2")
        model.set_seed(42)

        # Run sequential
        model.set_parallel(False)
        result_seq = model.find_sample_size(from_size=50, to_size=100, by=50, return_results=True)

        # Run parallel
        model.set_seed(42)
        model.set_parallel(True, n_cores=2)
        result_par = model.find_sample_size(from_size=50, to_size=100, by=50, return_results=True)

        # Results should match
        seq_sizes = result_seq["results"]["sample_sizes_tested"]
        par_sizes = result_par["results"]["sample_sizes_tested"]
        assert seq_sizes == par_sizes

        # Verify all effects (a, b, a:b, overall) are present in results and match
        seq_powers_by_test = result_seq["results"]["powers_by_test"]
        par_powers_by_test = result_par["results"]["powers_by_test"]

        assert set(seq_powers_by_test.keys()) == set(par_powers_by_test.keys())

        for test_name in seq_powers_by_test:
            seq_powers = seq_powers_by_test[test_name]
            par_powers = par_powers_by_test[test_name]
            for seq_pow, par_pow in zip(seq_powers, par_powers, strict=True):
                assert abs(seq_pow - par_pow) < 1e-10

    def test_parallel_fallback_on_failure(self, suppress_output, monkeypatch):
        """Graceful fallback to sequential when parallel execution fails."""
        from mcpower import MCPower

        # Create and configure model
        model = MCPower("y = x1 + x2")
        model.set_effects("x1=0.3, x2=0.2")
        model.set_seed(42)
        model.set_parallel(True, n_cores=2)

        # Mock joblib.Parallel to raise an exception
        def mock_parallel(*args, **kwargs):
            raise RuntimeError("Simulated parallel failure")

        import joblib

        monkeypatch.setattr(joblib, "Parallel", mock_parallel)

        # Should still work via fallback
        result = model.find_sample_size(from_size=50, to_size=100, by=50, return_results=True)

        # Should have valid results
        assert "results" in result
        assert "sample_sizes_tested" in result["results"]
        assert len(result["results"]["sample_sizes_tested"]) > 0

        # Verify we got the expected sample sizes
        assert result["results"]["sample_sizes_tested"] == [50, 100]

    def test_find_power_ignores_parallel(self, suppress_output):
        """find_power() works correctly with parallel=True (single sample size)."""
        from mcpower import MCPower

        # Create and configure model
        model = MCPower("y = x1 + x2")
        model.set_effects("x1=0.3, x2=0.2")
        model.set_seed(42)

        # Run with parallel=False
        model.set_parallel(False)
        result_seq = model.find_power(sample_size=100, return_results=True)

        # Run with parallel=True (should work fine for single sample size)
        model.set_seed(42)
        model.set_parallel(True, n_cores=2)
        result_par = model.find_power(sample_size=100, return_results=True)

        # Results should be identical
        seq_powers = result_seq["results"]["individual_powers"]
        par_powers = result_par["results"]["individual_powers"]

        assert set(seq_powers.keys()) == set(par_powers.keys())

        for test_name in seq_powers:
            assert abs(seq_powers[test_name] - par_powers[test_name]) < 1e-10


class TestParallelConfiguration:
    """Test parallel configuration settings."""

    def test_set_parallel_enables_parallel(self, suppress_output):
        """set_parallel(True) enables parallel execution."""
        from mcpower import MCPower

        model = MCPower("y = x1 + x2")
        assert model.parallel == "mixedmodels"  # Default is "mixedmodels"

        model.set_parallel(True)
        assert model.parallel is True

    def test_set_parallel_with_n_cores(self, suppress_output):
        """set_parallel can configure number of cores."""
        from unittest.mock import patch

        from mcpower import MCPower

        model = MCPower("y = x1 + x2")
        with patch("multiprocessing.cpu_count", return_value=8):
            model.set_parallel(True, n_cores=4)

        assert model.parallel is True
        assert model.n_cores == 4

    def test_parallel_default_n_cores(self, suppress_output):
        """Parallel uses (cpu_count // 2) by default."""
        import multiprocessing as mp

        from mcpower import MCPower

        model = MCPower("y = x1 + x2")
        model.set_parallel(True)

        expected_cores = max(1, mp.cpu_count() // 2)
        assert model.n_cores == expected_cores
