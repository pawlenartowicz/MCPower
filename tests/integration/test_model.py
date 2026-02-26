"""
Tests for MCPower model class.
"""

import numpy as np
import pytest


class TestMCPowerInit:
    """Test MCPower initialization."""

    def test_simple_equation(self, suppress_output):
        from mcpower import MCPower

        model = MCPower("y = x1 + x2")
        assert model.equation == "y = x1 + x2"
        assert model._registry.dependent == "y"
        assert set(model._registry.predictor_names) == {"x1", "x2"}

    def test_tilde_equation(self, suppress_output):
        from mcpower import MCPower

        model = MCPower("y ~ x1 + x2")
        assert model._registry.dependent == "y"

    def test_interaction_equation(self, suppress_output):
        from mcpower import MCPower

        model = MCPower("y = a + b + a:b")
        effects = model._registry.effect_names
        assert "a" in effects
        assert "b" in effects
        assert "a:b" in effects

    def test_star_interaction(self, suppress_output):
        from mcpower import MCPower

        model = MCPower("y = a * b")
        effects = model._registry.effect_names
        assert "a" in effects
        assert "b" in effects
        assert "a:b" in effects

    def test_default_values(self, suppress_output):
        from mcpower import MCPower

        model = MCPower("y = x1")
        assert model.alpha == 0.05
        assert model.power == 80.0
        assert model.n_simulations == 1600
        assert model.seed == 2137
        assert not model._applied


class TestMCPowerProperties:
    """Test MCPower properties."""

    def test_model_type(self, simple_model):
        # model_type now returns combined generation and test formula info
        assert "linear_regression" in simple_model.model_type.lower()
        assert "generation" in simple_model.model_type.lower() or "test" in simple_model.model_type.lower()


class TestSetMethods:
    """Test set_* methods store pending and set _applied=False."""

    def test_set_effects(self, simple_model):
        simple_model.set_effects("x1=0.3, x2=0.2")
        assert simple_model._pending_effects == "x1=0.3, x2=0.2"
        assert simple_model._applied is False

    def test_set_correlations_string(self, simple_model):
        simple_model.set_correlations("(x1,x2)=0.5")
        assert simple_model._pending_correlations == "(x1,x2)=0.5"
        assert simple_model._applied is False

    def test_set_correlations_matrix(self, simple_model, correlation_matrix_2x2):
        simple_model.set_correlations(correlation_matrix_2x2)
        assert np.array_equal(simple_model._pending_correlations, correlation_matrix_2x2)

    def test_set_variable_type(self, suppress_output):
        from mcpower import MCPower

        model = MCPower("y = group + x1")
        model.set_variable_type("group=(factor,3)")
        assert model._pending_variable_types == "group=(factor,3)"
        assert model._applied is False

    def test_upload_data_dict(self, simple_model, sample_data):
        simple_model.upload_data(sample_data)
        assert simple_model._pending_data is not None
        assert simple_model._applied is False

    def test_upload_data_numpy(self, simple_model):
        data = np.random.randn(50, 2)
        simple_model.upload_data(data, columns=["x1", "x2"])
        assert simple_model._pending_data is not None

    def test_set_alpha(self, simple_model):
        simple_model.set_alpha(0.01)
        assert simple_model.alpha == 0.01

    def test_set_power(self, simple_model):
        simple_model.set_power(90.0)
        assert simple_model.power == 90.0

    def test_set_simulations(self, simple_model):
        simple_model.set_simulations(2000)
        assert simple_model.n_simulations == 2000

    def test_set_seed(self, simple_model):
        simple_model.set_seed(12345)
        assert simple_model.seed == 12345

    def test_method_chaining(self, simple_model):
        result = simple_model.set_effects("x1=0.3, x2=0.2").set_alpha(0.01)
        assert result is simple_model


class TestApply:
    """Test apply() method."""

    def test_apply_sets_flag(self, configured_model):
        configured_model._apply()
        assert configured_model._applied is True

    def test_apply_processes_effects(self, simple_model):
        simple_model.set_effects("x1=0.5, x2=0.3")
        simple_model._apply()
        effect_sizes = simple_model._registry.get_effect_sizes()
        assert effect_sizes[0] == 0.5
        assert effect_sizes[1] == 0.3

    def test_apply_processes_variable_types(self, suppress_output):
        from mcpower import MCPower

        model = MCPower("y = group + x1")
        model.set_variable_type("group=(factor,3)")
        model.set_effects("group[2]=0.4, group[3]=0.3, x1=0.2")
        model._apply()
        assert len(model._registry.factor_names) == 1
        assert len(model._registry.dummy_names) == 2

    def test_apply_processes_correlations(self, simple_model):
        simple_model.set_effects("x1=0.3, x2=0.2")
        simple_model.set_correlations("(x1,x2)=0.5")
        simple_model._apply()
        corr = simple_model.correlation_matrix
        assert corr[0, 1] == 0.5

    def test_apply_order_independence(self, suppress_output):
        """Test that set_* methods can be called in any order."""
        from mcpower import MCPower

        # Order 1: effects, variable_type, correlations
        m1 = MCPower("y = group + x1 + x2")
        m1.set_effects("group[2]=0.4, group[3]=0.3, x1=0.2, x2=0.1")
        m1.set_variable_type("group=(factor,3)")
        m1.set_correlations("(x1,x2)=0.5")
        m1._apply()

        # Order 2: variable_type, correlations, effects
        m2 = MCPower("y = group + x1 + x2")
        m2.set_variable_type("group=(factor,3)")
        m2.set_correlations("(x1,x2)=0.5")
        m2.set_effects("group[2]=0.4, group[3]=0.3, x1=0.2, x2=0.1")
        m2._apply()

        # Both should have same effect sizes
        assert np.allclose(m1._registry.get_effect_sizes(), m2._registry.get_effect_sizes())


class TestFindPower:
    """Test find_power() method."""

    def test_auto_apply(self, configured_model):
        """Test that find_power auto-applies if not applied."""
        assert configured_model._applied is False
        configured_model.find_power(100, print_results=False)
        assert configured_model._applied is True

    def test_returns_result(self, configured_model):
        result = configured_model.find_power(100, print_results=False, return_results=True)
        assert result is not None
        assert "results" in result
        assert "individual_powers" in result["results"]

    def test_power_in_range(self, configured_model):
        result = configured_model.find_power(100, print_results=False, return_results=True)
        power = result["results"]["individual_powers"]["overall"]
        # x1=0.3, x2=0.2, n=100: ~88.9 (C++) / ~90.2 (Python)
        assert 85 <= power <= 93

    def test_target_test_single(self, configured_model):
        result = configured_model.find_power(100, target_test="x1", print_results=False, return_results=True)
        assert "x1" in result["results"]["individual_powers"]

    def test_target_test_multiple(self, configured_model):
        result = configured_model.find_power(100, target_test="x1,x2", print_results=False, return_results=True)
        powers = result["results"]["individual_powers"]
        assert "x1" in powers
        assert "x2" in powers

    def test_correction_bonferroni(self, configured_model):
        result = configured_model.find_power(100, correction="bonferroni", print_results=False, return_results=True)
        assert "individual_powers_corrected" in result["results"]

    def test_find_power_runs(self, configured_model):
        """Test power analysis runs successfully."""
        result = configured_model.find_power(100, print_results=False, return_results=True)
        assert result is not None
        power = result["results"]["individual_powers"]["overall"]
        # x1=0.3, x2=0.2, n=100: ~88.9 (C++) / ~90.2 (Python)
        assert 85 <= power <= 93


class TestFindSampleSize:
    """Test find_sample_size() method."""

    def test_returns_result(self, configured_model):
        result = configured_model.find_sample_size(from_size=50, to_size=100, by=25, print_results=False, return_results=True)
        assert result is not None
        assert "results" in result

    def test_sample_sizes_tested(self, configured_model):
        result = configured_model.find_sample_size(from_size=50, to_size=100, by=25, print_results=False, return_results=True)
        assert result["results"]["sample_sizes_tested"] == [50, 75, 100]

    def test_first_achieved(self, configured_model):
        result = configured_model.find_sample_size(from_size=50, to_size=200, by=50, print_results=False, return_results=True)
        assert "first_achieved" in result["results"]

    def test_find_sample_size_runs(self, configured_model):
        """Test sample size analysis runs successfully."""
        result = configured_model.find_sample_size(from_size=50, to_size=100, by=50, print_results=False, return_results=True)
        assert result is not None


class TestErrors:
    """Test error handling."""

    def test_invalid_effect_name(self, simple_model):
        simple_model.set_effects("invalid=0.3")
        with pytest.raises(ValueError, match="not found"):
            simple_model._apply()

    def test_missing_effects(self, simple_model):
        with pytest.raises(ValueError, match="Effect sizes must be set"):
            simple_model.find_power(100, print_results=False)

    def test_invalid_alpha(self, simple_model):
        with pytest.raises(ValueError):
            simple_model.set_alpha(0.5)  # > 0.25

    def test_invalid_power(self, simple_model):
        with pytest.raises(ValueError):
            simple_model.set_power(150)  # > 100

    def test_upload_data_auto_columns(self, simple_model):
        """Numpy array without columns auto-generates column_1, column_2, ..."""
        simple_model.upload_data(np.random.randn(50, 2))
        assert simple_model._pending_data["columns"] == ["column_1", "column_2"]

    def test_upload_data_too_few_samples(self, simple_model):
        with pytest.raises(ValueError, match="at least 25 samples"):
            simple_model.upload_data({"x1": [1, 2, 3]})


class TestSetFactorLevels:
    """Tests for set_factor_levels() method."""

    def test_basic_named_levels(self):
        from mcpower import MCPower

        model = MCPower("y = treatment + x1")
        model.set_factor_levels("treatment=placebo,drug_a,drug_b")
        model.set_effects("treatment[drug_a]=0.5, treatment[drug_b]=0.8, x1=0.3")
        model._apply()
        assert "treatment" in model._registry.factor_names
        assert "treatment[drug_a]" in model._registry.dummy_names
        assert "treatment[drug_b]" in model._registry.dummy_names
        assert "treatment[placebo]" not in model._registry.dummy_names

    def test_multiple_factors(self):
        from mcpower import MCPower

        model = MCPower("y = group + dose")
        model.set_factor_levels("group=control,treatment; dose=low,medium,high")
        model.set_effects("group[treatment]=0.5, dose[medium]=0.3, dose[high]=0.6")
        model._apply()
        assert "group[treatment]" in model._registry.dummy_names
        assert "dose[medium]" in model._registry.dummy_names
        assert "dose[high]" in model._registry.dummy_names

    def test_unknown_variable_raises(self):
        from mcpower import MCPower

        model = MCPower("y = x1")
        with pytest.raises(ValueError, match="not found"):
            model.set_factor_levels("unknown=a,b,c")
            model._apply()

    def test_single_level_raises(self):
        from mcpower import MCPower

        model = MCPower("y = x1")
        with pytest.raises(ValueError, match="at least 2"):
            model.set_factor_levels("x1=only_one")
            model._apply()

    def test_find_power_with_named_levels(self):
        """End-to-end: find_power works with set_factor_levels."""
        from mcpower import MCPower

        model = MCPower("y = treatment + x1")
        model.set_factor_levels("treatment=placebo,drug_a,drug_b")
        model.set_effects("treatment[drug_a]=0.5, treatment[drug_b]=0.8, x1=0.3")
        result = model.find_power(
            sample_size=100, print_results=False, return_results=True,
            progress_callback=False,
        )
        assert result is not None
