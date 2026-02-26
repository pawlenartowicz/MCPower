"""
End-to-end integration tests for the test_formula feature.

The test_formula feature generates data using one model formula but fits a
different (reduced) model for statistical testing, enabling model
misspecification analysis (e.g. omitted variable bias).
"""

import numpy as np
import pandas as pd
import pytest

from mcpower import MCPower

N_SIMS = 200
SEED = 42


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _power_model(formula, effects, *, n_sims=N_SIMS, seed=SEED, **kwargs):
    """Create a configured MCPower model ready for find_power."""
    model = MCPower(formula)

    # Apply optional configuration before effects
    if "variable_types" in kwargs:
        model.set_variable_type(kwargs.pop("variable_types"))
    if "correlations" in kwargs:
        model.set_correlations(kwargs.pop("correlations"))
    if "cluster" in kwargs:
        cluster_cfg = kwargs.pop("cluster")
        model.set_cluster(**cluster_cfg)
    if "max_failed" in kwargs:
        model.set_max_failed_simulations(kwargs.pop("max_failed"))
    if "upload_data" in kwargs:
        model.upload_data(kwargs.pop("upload_data"))

    model.set_effects(effects)
    model.set_simulations(n_sims)
    model.set_seed(seed)
    return model


def _run_power(model, sample_size, **kwargs):
    """Run find_power with standard test defaults."""
    return model.find_power(
        sample_size,
        print_results=False,
        return_results=True,
        progress_callback=False,
        **kwargs,
    )


def _individual_powers(result):
    """Extract individual_powers dict from a result."""
    return result["results"]["individual_powers"]


# ===========================================================================
# Class 1: TestOLSSubset
# ===========================================================================


class TestOLSSubset:
    """Test basic OLS test_formula subsetting scenarios."""

    def test_omitted_variable_reduces_power(self):
        """Omitting x3 from test formula excludes it from results."""
        model = _power_model(
            "y = x1 + x2 + x3",
            "x1=0.5, x2=0.3, x3=0.5",
        )
        result = _run_power(model, 100, test_formula="y = x1 + x2")

        powers = _individual_powers(result)
        assert "x1" in powers
        assert "x2" in powers
        assert "x3" not in powers

    def test_omitted_interaction(self):
        """Omitting interaction from test formula excludes it from results."""
        model = _power_model(
            "y = x1 + x2 + x1:x2",
            "x1=0.5, x2=0.3, x1:x2=0.2",
        )
        result = _run_power(model, 100, test_formula="y = x1 + x2")

        powers = _individual_powers(result)
        assert "x1" in powers
        assert "x2" in powers
        assert "x1:x2" not in powers

    def test_single_variable_test(self):
        """Testing only x1 from a 3-variable generation model."""
        model = _power_model(
            "y = x1 + x2 + x3",
            "x1=0.5, x2=0.3, x3=0.2",
        )
        result = _run_power(model, 100, test_formula="y = x1")

        powers = _individual_powers(result)
        assert "x1" in powers
        assert "overall" in powers
        assert "x2" not in powers
        assert "x3" not in powers

    def test_same_formula_matches_no_test_formula(self):
        """Using test_formula identical to generation gives same powers."""
        model_a = _power_model("y = x1 + x2", "x1=0.5, x2=0.3")
        result_a = _run_power(model_a, 100, test_formula="y = x1 + x2")

        model_b = _power_model("y = x1 + x2", "x1=0.5, x2=0.3")
        result_b = _run_power(model_b, 100)

        powers_a = _individual_powers(result_a)
        powers_b = _individual_powers(result_b)

        for key in powers_b:
            assert abs(powers_a[key] - powers_b[key]) < 0.01, (
                f"Power mismatch for {key}: {powers_a[key]} vs {powers_b[key]}"
            )

    def test_empty_test_formula_uses_generation(self):
        """Empty test_formula string uses the generation formula (default)."""
        model_a = _power_model("y = x1 + x2", "x1=0.5, x2=0.3")
        result_a = _run_power(model_a, 100, test_formula="")

        model_b = _power_model("y = x1 + x2", "x1=0.5, x2=0.3")
        result_b = _run_power(model_b, 100)

        powers_a = _individual_powers(result_a)
        powers_b = _individual_powers(result_b)

        for key in powers_b:
            assert abs(powers_a[key] - powers_b[key]) < 0.01, (
                f"Power mismatch for {key}: {powers_a[key]} vs {powers_b[key]}"
            )


# ===========================================================================
# Class 2: TestFactorVariables
# ===========================================================================


class TestFactorVariables:
    """Test test_formula with factor (categorical) variables."""

    def test_omitted_factor(self):
        """Omitting a factor variable from test formula excludes its dummies."""
        model = _power_model(
            "y = x1 + x2",
            "x1=0.5, x2[2]=0.3, x2[3]=0.4",
            variable_types="x2=(factor,3)",
        )
        result = _run_power(model, 150, test_formula="y = x1")

        powers = _individual_powers(result)
        assert "x1" in powers
        # Factor dummies should not be in results
        assert "x2[2]" not in powers
        assert "x2[3]" not in powers

    def test_factor_kept_continuous_dropped(self):
        """Keeping factor but dropping continuous variable."""
        model = _power_model(
            "y = x1 + x2",
            "x1=0.5, x2[2]=0.3, x2[3]=0.4",
            variable_types="x2=(factor,3)",
        )
        result = _run_power(model, 150, test_formula="y = x2")

        powers = _individual_powers(result)
        # x1 excluded
        assert "x1" not in powers
        # Factor dummies should be present
        assert "x2[2]" in powers
        assert "x2[3]" in powers


# ===========================================================================
# Class 3: TestCorrelationStructures
# ===========================================================================


class TestCorrelationStructures:
    """Test test_formula with correlated predictors."""

    def test_correlated_variables_subset(self):
        """Subsetting correlated variables runs without error."""
        model = _power_model(
            "y = x1 + x2",
            "x1=0.5, x2=0.3",
            correlations="(x1,x2)=0.5",
        )
        result = _run_power(model, 100, test_formula="y = x1")

        assert result is not None
        powers = _individual_powers(result)
        assert "x1" in powers
        assert "x2" not in powers


# ===========================================================================
# Class 4: TestResultsStructure
# ===========================================================================


class TestResultsStructure:
    """Test that result dict contains correct test_formula metadata."""

    def test_results_contain_both_formulas(self):
        """Result should have data_formula and test_formula fields."""
        model = _power_model(
            "y = x1 + x2 + x3",
            "x1=0.5, x2=0.3, x3=0.2",
        )
        result = _run_power(model, 100, test_formula="y = x1 + x2")

        assert "data_formula" in result["model"]
        assert "test_formula" in result["model"]
        # data_formula should be the generation formula
        assert "x3" in result["model"]["data_formula"]
        # test_formula should be the reduced formula
        assert result["model"]["test_formula"] == "y = x1 + x2"

    def test_target_tests_reflect_test_formula(self):
        """target_tests in results should not contain excluded effects."""
        model = _power_model(
            "y = x1 + x2 + x3",
            "x1=0.5, x2=0.3, x3=0.2",
        )
        result = _run_power(model, 100, test_formula="y = x1 + x2")

        target_tests = result["model"]["target_tests"]
        assert "x1" in target_tests
        assert "x2" in target_tests
        assert "x3" not in target_tests


# ===========================================================================
# Class 5: TestValidation
# ===========================================================================


class TestValidation:
    """Test validation errors for invalid test_formula usage."""

    def test_nonexistent_variable_raises(self):
        """test_formula with unknown variable raises ValueError."""
        model = _power_model(
            "y = x1 + x2",
            "x1=0.5, x2=0.3",
        )
        with pytest.raises(ValueError, match="not found"):
            _run_power(model, 100, test_formula="y = x1 + x99")

    def test_ols_to_lme_raises(self):
        """test_formula with random effects on OLS model raises ValueError.

        When the grouping variable (school) is not in the generation model,
        validation fails with 'not found'. When it is present but has no
        cluster config, it fails with 'random effects'.
        """
        # Case 1: grouping var not in model at all -> "not found"
        model = _power_model(
            "y = x1 + x2",
            "x1=0.5, x2=0.3",
        )
        with pytest.raises(ValueError, match="not found"):
            _run_power(model, 100, test_formula="y = x1 + (1|school)")

    def test_ols_with_cluster_var_but_no_cluster_config_raises(self):
        """test_formula with random effects when var exists but no cluster config.

        When the generation model knows about 'school' as a variable but has
        no cluster specification, the random effects check triggers.
        """
        # This would require a model that has 'school' as a predictor but
        # no set_cluster call. The generation model includes school as a
        # fixed effect, so it's a known variable.
        model = _power_model(
            "y = x1 + school",
            "x1=0.5, school=0.3",
        )
        with pytest.raises(ValueError, match="random effects"):
            _run_power(model, 100, test_formula="y = x1 + (1|school)")


# ===========================================================================
# Class 6: TestFindSampleSize
# ===========================================================================


class TestFindSampleSize:
    """Test test_formula with find_sample_size."""

    def test_subset_via_find_sample_size(self):
        """find_sample_size with test_formula excludes omitted variable."""
        model = _power_model(
            "y = x1 + x2 + x3",
            "x1=0.5, x2=0.3, x3=0.2",
        )
        result = model.find_sample_size(
            target_test="x1",
            from_size=30,
            to_size=100,
            by=10,
            test_formula="y = x1 + x2",
            print_results=False,
            return_results=True,
            progress_callback=False,
        )

        assert result is not None
        powers_by_test = result["results"]["powers_by_test"]
        assert "x1" in powers_by_test
        assert "x3" not in powers_by_test


# ===========================================================================
# Class 7: TestMixedModelCross (LME)
# ===========================================================================


@pytest.mark.lme
class TestMixedModelCross:
    """Test test_formula across mixed model boundaries."""

    def test_lme_gen_ols_test(self):
        """Generate with LME, test with OLS (drop random effects)."""
        model = _power_model(
            "y ~ x1 + x2 + (1|school)",
            "x1=0.5, x2=0.3",
            cluster={"grouping_var": "school", "ICC": 0.2, "n_clusters": 20},
            max_failed=0.10,
        )
        result = _run_power(model, 1000, test_formula="y ~ x1 + x2")

        powers = _individual_powers(result)
        assert "x1" in powers
        assert "x2" in powers

    def test_lme_gen_lme_subset(self):
        """Generate with LME full model, test with LME subset (drop x2)."""
        model = _power_model(
            "y ~ x1 + x2 + (1|school)",
            "x1=0.5, x2=0.3",
            cluster={"grouping_var": "school", "ICC": 0.2, "n_clusters": 20},
            max_failed=0.10,
        )
        result = _run_power(model, 1000, test_formula="y ~ x1 + (1|school)")

        powers = _individual_powers(result)
        assert "x1" in powers
        assert "x2" not in powers


# ===========================================================================
# Class 8: TestUploadedData
# ===========================================================================


class TestUploadedData:
    """Test test_formula with uploaded empirical data."""

    def test_upload_with_test_formula(self):
        """Uploaded data with test_formula excludes omitted variable."""
        np.random.seed(SEED)
        data = pd.DataFrame({
            "x1": np.random.normal(0, 1, 50),
            "x2": np.random.normal(0, 1, 50),
            "x3": np.random.normal(0, 1, 50),
        })

        model = _power_model(
            "y = x1 + x2 + x3",
            "x1=0.5, x2=0.3, x3=0.2",
            upload_data=data,
        )
        result = _run_power(model, 100, test_formula="y = x1 + x2")

        powers = _individual_powers(result)
        assert "x1" in powers
        assert "x2" in powers
        assert "x3" not in powers
