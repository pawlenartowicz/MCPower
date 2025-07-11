"""
Tests for MCPowerBase class.
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock

import mcpower
from mcpower.base import MCPowerBase


class TestMCPowerBaseInitialization:
    """Test model initialization and equation parsing."""
    
    def test_valid_equation_formats(self):
        """Test various valid equation formats."""
        equations = [
            "y = x1 + x2",
            "outcome ~ treatment + age", 
            "y = x1 * x2",
            "response = a + b + a:b",
            "y ~ x1 + x2 + x1:x2"
        ]
        
        for eq in equations:
            model = mcpower.LinearRegression(eq)
            assert model.equation == eq.strip()
            assert 'variable_0' in model.variables  # dependent variable
            assert len(model.effects) > 0  # has predictors
    
    def test_invalid_equations(self):
        """Test invalid equation formats raise errors."""
        # These should raise ValueError due to empty formula_part
        invalid_equations = [
            "y = ",
            "y ~ ",
            "y =   ",
            "outcome ~ "
        ]
        
        for eq in invalid_equations:
            with pytest.raises(ValueError, match="Equation cannot be empty"):
                mcpower.LinearRegression(eq)
        
        # Test completely empty string
        with pytest.raises(ValueError):
            mcpower.LinearRegression("")
    
    def test_equation_parsing_components(self):
        """Test equation parsing extracts correct components."""
        model = mcpower.LinearRegression("satisfaction = treatment + age + treatment:age")
        
        # Check dependent variable
        assert model.variables['variable_0']['name'] == 'satisfaction'
        
        # Check predictors
        predictor_names = [info['name'] for key, info in model.variables.items() if key != 'variable_0']
        assert 'treatment' in predictor_names
        assert 'age' in predictor_names
        
        # Check effects
        effect_names = [info['name'] for info in model.effects.values()]
        assert 'treatment' in effect_names
        assert 'age' in effect_names
        assert 'treatment:age' in effect_names
    
    def test_interaction_parsing(self):
        """Test interaction term parsing."""
        model = mcpower.LinearRegression("y = x1*x2*x3")
        
        effect_names = [info['name'] for info in model.effects.values()]
        
        # Should have main effects
        assert 'x1' in effect_names
        assert 'x2' in effect_names  
        assert 'x3' in effect_names
        
        # Should have 2-way interactions
        assert 'x1:x2' in effect_names
        assert 'x1:x3' in effect_names
        assert 'x2:x3' in effect_names
        
        # Should have 3-way interaction
        assert 'x1:x2:x3' in effect_names


class TestEffectSetting:
    """Test effect size configuration."""
    
    def test_valid_effect_setting(self, sample_model):
        """Test setting valid effect sizes."""
        sample_model.set_effects("x1=0.5, x2=0.3, x1:x2=0.2")
        
        assert sample_model.effect_sizes_initiated is True
        
        # Check effects were set
        effects = {info['name']: info.get('effect_size') for info in sample_model.effects.values()}
        assert effects['x1'] == 0.5
        assert effects['x2'] == 0.3
        assert effects['x1:x2'] == 0.2
    
    def test_invalid_effect_strings(self, sample_model):
        """Test invalid effect string formats."""
        invalid_strings = [
            "",
            "x1=invalid",
            "nonexistent=0.5",
            "x1=",
            "=0.5"
        ]
        
        for invalid in invalid_strings:
            with pytest.raises(ValueError):
                sample_model.set_effects(invalid)
    
    def test_effect_type_validation(self, sample_model):
        """Test non-string input raises TypeError."""
        with pytest.raises(TypeError):
            sample_model.set_effects(123)
        
        with pytest.raises(TypeError):
            sample_model.set_effects(['x1=0.5'])


class TestVariableTypes:
    """Test variable type configuration."""
    
    def test_valid_variable_types(self, simple_model):
        """Test setting valid variable types."""
        simple_model.set_variable_type("treatment=binary")
        
        # Find treatment variable
        treatment_var = None
        for var_info in simple_model.variables.values():
            if var_info['name'] == 'treatment':
                treatment_var = var_info
                break
        
        assert treatment_var is not None
        assert treatment_var['type'] == 'binary'
        assert treatment_var['proportion'] == 0.5  # default
    
    def test_binary_with_proportion(self, simple_model):
        """Test binary variable with custom proportion."""
        simple_model.set_variable_type("treatment=(binary,0.3)")
        
        treatment_var = None
        for var_info in simple_model.variables.values():
            if var_info['name'] == 'treatment':
                treatment_var = var_info
                break
        
        assert treatment_var['type'] == 'binary'
        assert treatment_var['proportion'] == 0.3
    
    def test_supported_distributions(self, simple_model):
        """Test all supported distribution types."""
        distributions = ['normal', 'binary', 'right_skewed', 'left_skewed', 'high_kurtosis', 'uniform']
        
        for dist in distributions:
            simple_model.set_variable_type(f"treatment={dist}")
            
            treatment_var = None
            for var_info in simple_model.variables.values():
                if var_info['name'] == 'treatment':
                    treatment_var = var_info
                    break
            
            assert treatment_var['type'] == dist
    
    def test_invalid_variable_types(self, simple_model):
        """Test invalid variable types raise errors."""
        with pytest.raises(ValueError):
            simple_model.set_variable_type("treatment=invalid_distribution")
        
        with pytest.raises(ValueError):
            simple_model.set_variable_type("nonexistent_var=binary")


class TestCorrelations:
    """Test correlation configuration."""
    
    def test_string_correlation_format(self, sample_model):
        """Test string-based correlation specification."""
        sample_model.set_variable_type("")  # Initialize variable types
        sample_model.set_correlations("corr(x1, x2)=0.5")
        
        # Check correlation matrix was created
        assert sample_model.correlation_matrix is not None
        assert sample_model.correlation_matrix[0, 1] == 0.5
        assert sample_model.correlation_matrix[1, 0] == 0.5
    
    def test_matrix_correlation_format(self, sample_model):
        """Test numpy matrix correlation specification."""
        sample_model.set_variable_type("")
        
        corr_matrix = np.array([
            [1.0, 0.3],
            [0.3, 1.0]
        ])
        
        sample_model.set_correlations(corr_matrix)
        np.testing.assert_array_equal(sample_model.correlation_matrix, corr_matrix)
    
    def test_invalid_correlations(self, sample_model):
        """Test invalid correlation specifications."""
        sample_model.set_variable_type("")
        
        # Invalid correlation values
        with pytest.raises(ValueError):
            sample_model.set_correlations("corr(x1, x2)=1.5")  # > 1
        
        with pytest.raises(ValueError):
            sample_model.set_correlations("corr(x1, x2)=-1.1")  # < -1
        
        # Invalid matrix
        invalid_matrix = np.array([
            [1.0, 1.5],
            [1.5, 1.0]
        ])
        
        with pytest.raises(ValueError):
            sample_model.set_correlations(invalid_matrix)


class TestParameterValidation:
    """Test parameter validation methods."""
    
    def test_power_validation(self, simple_model):
        """Test power parameter validation."""
        # Valid values
        simple_model.set_power(80)
        assert simple_model.power == 80.0
        
        simple_model.set_power(95.5)
        assert simple_model.power == 95.5
        
        # Invalid values
        with pytest.raises(ValueError):
            simple_model.set_power(-5)
        
        with pytest.raises(ValueError):
            simple_model.set_power(105)
    
    def test_alpha_validation(self, simple_model):
        """Test alpha parameter validation."""
        # Valid values
        simple_model.set_alpha(0.05)
        assert simple_model.alpha == 0.05
        
        simple_model.set_alpha(0.01)
        assert simple_model.alpha == 0.01
        
        # Invalid values
        with pytest.raises(ValueError):
            simple_model.set_alpha(-0.01)
        
        with pytest.raises(ValueError):
            simple_model.set_alpha(0.3)
    
    def test_simulations_validation(self, simple_model):
        """Test simulation count validation."""
        # Valid values
        simple_model.set_simulations(1000)
        assert simple_model.n_simulations == 1000
        
        # Float gets rounded
        simple_model.set_simulations(1500.7)
        assert simple_model.n_simulations == 1501
        
        # Invalid values
        with pytest.raises(ValueError):
            simple_model.set_simulations(0)
        
        with pytest.raises(ValueError):
            simple_model.set_simulations(-100)
    
    def test_heterogeneity_validation(self, simple_model):
        """Test heterogeneity parameter validation."""
        # Valid values
        simple_model.set_heterogeneity(0.2)
        assert simple_model.heterogeneity == 0.2
        
        simple_model.set_heterogeneity(0.0)
        assert simple_model.heterogeneity == 0.0
        
        # Invalid values
        with pytest.raises(ValueError):
            simple_model.set_heterogeneity(-0.1)
        
        with pytest.raises(TypeError):
            simple_model.set_heterogeneity("0.2")
    
    def test_heteroskedasticity_validation(self, simple_model):
        """Test heteroskedasticity parameter validation."""
        # Valid values
        simple_model.set_heteroskedasticity(0.3)
        assert simple_model.heteroskedasticity == 0.3
        
        simple_model.set_heteroskedasticity(-0.5)
        assert simple_model.heteroskedasticity == -0.5
        
        # Invalid values
        with pytest.raises(ValueError):
            simple_model.set_heteroskedasticity(1.5)
        
        with pytest.raises(ValueError):
            simple_model.set_heteroskedasticity(-1.1)
    
    def test_seed_validation(self, simple_model):
        """Test seed setting."""
        simple_model.set_seed(42)
        assert simple_model.seed == 42
        
        simple_model.set_seed(None)
        assert simple_model.seed is None
        
        with pytest.raises(ValueError):
            simple_model.set_seed(-1)
        
        with pytest.raises(TypeError):
            simple_model.set_seed("42")


class TestDataUpload:
    """Test data upload functionality."""
    
    def test_upload_valid_data(self, simple_model, sample_data):
        """Test uploading valid data."""
        # Create data with treatment column
        data_with_treatment = sample_data.copy()
        data_with_treatment['treatment'] = np.random.choice([0, 1], len(sample_data))
        
        simple_model.upload_own_data(data_with_treatment)
        
        assert simple_model.data_uploaded is True
        assert 'treatment' in simple_model.uploaded_data
    
    def test_upload_invalid_data_type(self, simple_model):
        """Test uploading non-DataFrame raises error."""
        with pytest.raises(TypeError):
            simple_model.upload_own_data([1, 2, 3])
        
        with pytest.raises(TypeError):
            simple_model.upload_own_data("not a dataframe")
    
    def test_upload_empty_data(self, simple_model):
        """Test uploading empty DataFrame raises error."""
        empty_df = pd.DataFrame()
        
        with pytest.raises(ValueError):
            simple_model.upload_own_data(empty_df)


class TestParallelSettings:
    """Test parallel processing configuration."""
    
    @patch('joblib.Parallel')
    @patch('multiprocessing.cpu_count')
    def test_enable_parallel(self, mock_cpu_count, mock_parallel, simple_model):
        """Test enabling parallel processing."""
        mock_cpu_count.return_value = 8
        
        simple_model.set_parallel(True)
        
        assert simple_model.parallel is True
        assert simple_model.n_cores == 7  # cpu_count - 1
    
    @patch('multiprocessing.cpu_count')
    def test_custom_cores(self, mock_cpu_count, simple_model):
        """Test setting custom number of cores."""
        mock_cpu_count.return_value = 8
        
        simple_model.set_parallel(True, n_cores=4)
        
        assert simple_model.parallel is True
        assert simple_model.n_cores == 4
    
    def test_disable_parallel(self, simple_model):
        """Test disabling parallel processing."""
        simple_model.set_parallel(False)
        
        assert simple_model.parallel is False
        assert simple_model.n_cores == 1
    
    def test_parallel_fallback_when_joblib_missing(self, simple_model):
        """Test fallback when joblib is not available."""
        # Temporarily remove joblib from sys.modules if it exists
        import sys
        joblib_backup = sys.modules.get('joblib')
        if 'joblib' in sys.modules:
            del sys.modules['joblib']
        
        # Mock joblib to not be importable
        sys.modules['joblib'] = None
        
        try:
            simple_model.set_parallel(True)
            assert simple_model.parallel is False
        finally:
            # Restore original state
            if joblib_backup is not None:
                sys.modules['joblib'] = joblib_backup
            elif 'joblib' in sys.modules:
                del sys.modules['joblib']


class TestAbstractMethods:
    """Test abstract method implementation requirements."""
    
    def test_abstract_methods_exist(self):
        """Test that abstract methods are properly defined."""
        # MCPowerBase should not be instantiable
        with pytest.raises(TypeError):
            MCPowerBase("y = x")
    
    def test_subclass_implements_abstract_methods(self):
        """Test that LinearRegression implements required methods."""
        model = mcpower.LinearRegression("y = x")
        
        # These should not raise NotImplementedError
        assert hasattr(model, 'model_type')
        assert hasattr(model, '_run_statistical_analysis')
        assert hasattr(model, '_generate_dependent_variable')


class TestScenarioConfigs:
    """Test scenario configuration."""
    
    def test_default_scenario_configs(self, simple_model):
        """Test default scenario configurations."""
        configs = simple_model.get_scenario_configs()
        
        assert 'realistic' in configs
        assert 'doomer' in configs
        assert 'heterogeneity' in configs['realistic']
        assert 'heteroskedasticity' in configs['realistic']
    
    def test_custom_scenario_configs(self, simple_model):
        """Test setting custom scenario configurations."""
        custom_configs = {
            'realistic': {'heterogeneity': 0.3},
            'doomer': {'heterogeneity': 0.5}
        }
        
        simple_model.set_scenario_configs(custom_configs)
        
        configs = simple_model.get_scenario_configs()
        assert configs['realistic']['heterogeneity'] == 0.3
        assert configs['doomer']['heterogeneity'] == 0.5
    
    def test_invalid_scenario_configs(self, simple_model):
        """Test invalid scenario configuration types."""
        with pytest.raises(TypeError):
            simple_model.set_scenario_configs("not a dict")


class TestMethodChaining:
    """Test method chaining functionality."""
    
    def test_method_chaining(self):
        """Test that configuration methods return self for chaining."""
        model = (mcpower.LinearRegression("y = x1 + x2")
                .set_effects("x1=0.5, x2=0.3")
                .set_variable_type("x1=binary")
                .set_power(90)
                .set_alpha(0.01)
                .set_simulations(500))
        
        assert model.power == 90
        assert model.alpha == 0.01
        assert model.n_simulations == 500
        assert model.effect_sizes_initiated is True


class TestRepr:
    """Test string representation."""
    
    def test_repr_format(self, simple_model):
        """Test __repr__ returns expected format."""
        repr_str = repr(simple_model)
        
        assert 'LinearRegression' in repr_str
        assert simple_model.equation in repr_str