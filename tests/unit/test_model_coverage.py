"""Tests for model.py — parallel fallback, Tukey validation, NaN under Tukey correction."""

import warnings
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from mcpower import MCPower


class TestTukeyWithoutPosthoc:
    """Tukey correction without posthoc specs should raise ValueError."""

    def test_tukey_without_posthoc_raises(self):
        model = MCPower("y = x1 + x2")
        model.set_effects("x1=0.5, x2=0.3")

        with pytest.raises(ValueError, match="Tukey correction requires"):
            model.find_power(
                sample_size=100,
                correction="tukey",
                print_results=False,
            )


class TestTukeyNaNification:
    """Non-posthoc tests should be NaN-ified under Tukey correction."""

    def test_non_posthoc_tests_nan_under_tukey(self):
        model = MCPower("y = group + x1")
        model.set_variable_type("group=(factor,3)")
        model.set_effects("group[2]=0.5, group[3]=0.4, x1=0.3")
        model.n_simulations = 50
        model.seed = 42

        result = model.find_sample_size(
            target_test="all, all-posthoc",
            correction="tukey",
            from_size=30,
            to_size=60,
            by=30,
            print_results=False,
            return_results=True,
        )

        assert result is not None
        results = result["results"]
        corrected = results.get("powers_by_test_corrected", {})

        # Post-hoc comparisons should have real power values
        # Non-posthoc tests (like "x1", "group[2]", "group[3]", "overall")
        # should have NaN values
        posthoc_labels = {s.label for s in model._posthoc_specs}
        for test_name, powers in corrected.items():
            if test_name not in posthoc_labels:
                assert all(isinstance(v, float) and np.isnan(v) for v in powers), \
                    f"Expected NaN for non-posthoc test '{test_name}', got {powers}"

        # first_achieved_corrected for non-posthoc should be -1
        for test_name, n in results.get("first_achieved_corrected", {}).items():
            if test_name not in posthoc_labels:
                assert n == -1, f"Expected -1 for '{test_name}', got {n}"


class TestParallelFallback:
    """Parallel execution falls back to sequential on exception."""

    def test_parallel_exception_falls_back(self, capsys):
        model = MCPower("y = x1 + x2")
        model.set_effects("x1=0.5, x2=0.3")
        model.parallel = True
        model.n_simulations = 50
        model.seed = 42

        # Parallel is imported inside the function via `from joblib import Parallel`,
        # so we patch it at the joblib module level.
        with patch("joblib.Parallel", side_effect=RuntimeError("joblib broken")):
            # Should still complete via sequential fallback
            result = model.find_sample_size(
                from_size=30,
                to_size=60,
                by=30,
                print_results=False,
                return_results=True,
            )
            assert result is not None
            captured = capsys.readouterr()
            assert "Falling back to sequential" in captured.out


class TestIsParallelEffective:
    """Test _is_parallel_effective resolution."""

    def test_true_always_parallel(self):
        model = MCPower("y = x1 + x2")
        model.parallel = True
        assert model._is_parallel_effective() is True

    def test_false_never_parallel(self):
        model = MCPower("y = x1 + x2")
        model.parallel = False
        assert model._is_parallel_effective() is False

    def test_mixedmodels_with_clusters(self):
        model = MCPower("y ~ x1 + (1|school)")
        model.set_cluster("school", ICC=0.2, n_clusters=20)
        model.set_effects("x1=0.5")
        model._apply()  # cluster_specs are deferred until apply()
        model.parallel = "mixedmodels"
        assert model._is_parallel_effective() is True

    def test_mixedmodels_without_clusters(self):
        model = MCPower("y = x1 + x2")
        model.set_effects("x1=0.5, x2=0.3")
        model.parallel = "mixedmodels"
        assert model._is_parallel_effective() is False
