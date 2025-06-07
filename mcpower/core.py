import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import skewnorm, t
from sklearn.linear_model import LinearRegression
from scipy import stats
from typing import Dict, List, Any

class MCPower:
    def __init__(self, equation: str):
        """
        Initialize MCPower with R-style equation string.
        
        Args:
            equation: R-style equation like "y = x1 + x2 + x3*m" or "x1 + x2 + x3*m"
        """
        self.equation = equation.strip()
        self.variables = {}
        self.effects = {}
        self.effect_sizes_initiated = False
        self.variable_types_initiated = False
        self.power = 80.0  # Default 80%
        self.alpha = 0.05  # Default 0.05
        self.sample_size = None  # No default
        self.n_simulations = 1000  # Default for Monte Carlo
        self._parse_equation()
    
    def _parse_equation(self):
        """Parse the equation string into variables dictionary."""
        equation = self.equation.replace(' ', '')
        
        # Check if there's a dependent variable (y = ...)
        if '=' in equation:
            left_side, right_side = equation.split('=', 1)
            # Add dependent variable as variable_0
            self.variables['variable_0'] = {'name': left_side.strip()}
            formula_part = right_side
        else:
            # No dependent variable specified, create default
            self.variables['variable_0'] = {'name': 'explained_variable'}
            formula_part = equation
        
        # Parse independent variables and effects from right side of equation
        self._parse_independent_variables(formula_part)
    
    def _parse_independent_variables(self, formula: str):
        """Extract independent variables and effects from formula string."""
        
        # Split by + and - to get individual terms
        terms = re.split(r'[+\-]', formula)
        
        variable_counter = 1
        effect_counter = 1
        seen_variables = set()
        seen_effects = set()
        
        # First pass: identify all variables and effects
        for term in terms:
            term = term.strip()
            if not term:
                continue
            
            # Check for interaction terms (* or :)
            if '*' in term or ':' in term:
                # This is an interaction term
                interaction_vars = re.findall(r'[a-zA-Z][a-zA-Z0-9_]*', term)
                
                # Add individual variables if not seen
                for var in interaction_vars:
                    if var not in seen_variables:
                        self.variables[f'variable_{variable_counter}'] = {'name': var}
                        seen_variables.add(var)
                        variable_counter += 1
                
                # Add main effects for each variable in interaction FIRST
                for var in interaction_vars:
                    if var not in seen_effects:
                        self.effects[f'effect_{effect_counter}'] = {
                            'name': var, 
                            'type': 'main'
                        }
                        seen_effects.add(var)
                        effect_counter += 1
                
                # Add interaction effect AFTER main effects
                interaction_name = ':'.join(interaction_vars)
                if interaction_name not in seen_effects:
                    self.effects[f'effect_{effect_counter}'] = {
                        'name': interaction_name, 
                        'type': 'interaction'
                    }
                    seen_effects.add(interaction_name)
                    effect_counter += 1
            else:
                # This is a main effect term
                variables_in_term = re.findall(r'[a-zA-Z][a-zA-Z0-9_]*', term)
                
                for var in variables_in_term:
                    # Add variable if not seen
                    if var not in seen_variables:
                        self.variables[f'variable_{variable_counter}'] = {'name': var}
                        seen_variables.add(var)
                        variable_counter += 1
                    
                    # Add main effect if not seen
                    if var not in seen_effects:
                        self.effects[f'effect_{effect_counter}'] = {
                            'name': var, 
                            'type': 'main'
                        }
                        seen_effects.add(var)
                        effect_counter += 1
    
    def set_effects(self, effects_string):
        """Set effect sizes using string format like 'x1=0.5, x2:x3=0.3'"""
        if not self.effect_sizes_initiated:
            self.effect_sizes_initiated = True
        
        # Parse the string
        assignments = [assignment.strip() for assignment in effects_string.split(',')]
        for assignment in assignments:
            if '=' in assignment:
                effect_name, effect_size = assignment.split('=', 1)
                effect_name = effect_name.strip()
                effect_size = float(effect_size.strip())
                
                # Find the effect by name and update its size
                for effect_key, effect_info in self.effects.items():
                    if effect_info['name'] == effect_name:
                        effect_info['effect_size'] = effect_size
                        break
    
    def set_variable_type(self, **kwargs):
        """Set variable types (e.g., binary, linear). Default is linear."""
        if not self.variable_types_initiated:
            # Initialize all variables to normal
            for var_key, var_info in self.variables.items():
                var_info['type'] = 'normal'
            self.variable_types_initiated = True
        
        # Update specified variable types
        for var_name, var_type_info in kwargs.items():
            # Find the variable by name and update its type
            for var_key, var_info in self.variables.items():
                if var_info['name'] == var_name:
                    if isinstance(var_type_info, str):
                        # Simple format: x1="binary"
                        var_info['type'] = var_type_info
                    elif isinstance(var_type_info, tuple) and len(var_type_info) == 2:
                        # Complex format: x1=("binary", 0.2)
                        var_type, proportion = var_type_info
                        if var_type == "binary":
                            var_info['type'] = var_type
                            var_info['proportion'] = proportion
                        else:
                            raise ValueError(f"Proportion only valid for binary variables, not {var_type}")
                    else:
                        raise ValueError(f"Invalid format for {var_name}. Use string or tuple (type, proportion)")
                    break
    
    def set_power(self, power: float):
        """Set power level (0-100%)."""
        if 0 <= power <= 100:
            self.power = power
        else:
            raise ValueError("Power must be between 0 and 100")
    
    def set_alpha(self, alpha: float):
        """Set alpha level (0-1)."""
        if 0 <= alpha <= 1:
            self.alpha = alpha
        else:
            raise ValueError("Alpha must be between 0 and 1")
    
    def set_sample_size(self, sample_size):
        """Set sample size (positive integer)."""
        if sample_size <= 0:
            raise ValueError("Sample size must be positive")
        
        original_value = sample_size
        rounded_value = int(round(sample_size))
        
        if original_value != rounded_value:
            print(f"Warning: Sample size rounded from {original_value} to {rounded_value}")
        
        self.sample_size = rounded_value
    
    def set_simulations(self, n_simulations):
        """Set number of Monte Carlo simulations (positive integer)."""
        if n_simulations <= 0:
            raise ValueError("Number of simulations must be positive")
        
        original_value = n_simulations
        rounded_value = int(round(n_simulations))
        
        if original_value != rounded_value:
            print(f"Warning: Number of simulations rounded from {original_value} to {rounded_value}")
        
        self.n_simulations = rounded_value
    
    def _normal_dist(self, n=1000, seed=None):
        """Generate normalized standard normal distribution"""
        np.random.seed(seed)
        data = np.random.normal(0, 1, n)
        return (data - np.mean(data)) / np.std(data)
    
    def _right_skewed_dist(self, n=1000, seed=None):
        """Generate normalized right-skewed distribution (skew ≈ 1)"""
        np.random.seed(seed)
        data = skewnorm.rvs(a=10, loc=0, scale=1, size=n)
        return (data - np.mean(data)) / np.std(data)
    
    def _left_skewed_dist(self, n=1000, seed=None):
        """Generate normalized left-skewed distribution (skew ≈ -1)"""
        np.random.seed(seed)
        data = skewnorm.rvs(a=-10, loc=0, scale=1, size=n)
        return (data - np.mean(data)) / np.std(data)
    
    def _high_kurtosis_dist(self, n=1000, seed=None):
        """Generate normalized high-kurtosis distribution (kurt ≈ 6)"""
        np.random.seed(seed)
        data = t.rvs(df=3, loc=0, scale=1, size=n)
        return (data - np.mean(data)) / np.std(data)
    
    def _uniform_dist(self, n=1000, seed=None):
        """Generate normalized uniform distribution (mean=0, sd=1)"""
        np.random.seed(seed)
        data = np.random.uniform(-np.sqrt(3), np.sqrt(3), n)
        return (data - np.mean(data)) / np.std(data)
    
    def _binary_dist(self, n=1000, proportion=0.5, seed=None):
        """Generate binary distribution (0s and 1s)"""
        np.random.seed(seed)
        return np.random.choice([0, 1], size=n, p=[1-proportion, proportion])
    
    def generate_dataset(self, seed=None):
        """Generate dataset based on variable types and sample size"""
        if self.sample_size is None:
            raise ValueError("Sample size must be set before generating dataset")
        
        if not self.variable_types_initiated:
            # Initialize default types if not set
            self.set_variable_type()
        
        np.random.seed(seed)
        dataset = {}
        
        # Generate independent variables first
        for var_key, var_info in self.variables.items():
            if var_key == 'variable_0':  # Skip dependent variable
                continue
                
            var_name = var_info['name']
            var_type = var_info.get('type', 'normal')
            
            if var_type == 'binary':
                proportion = var_info.get('proportion', 0.5)
                dataset[var_name] = self._binary_dist(n=self.sample_size, proportion=proportion, seed=seed)
            elif var_type == 'normal':
                dataset[var_name] = self._normal_dist(n=self.sample_size, seed=seed)
            elif var_type == 'right_skewed':
                dataset[var_name] = self._right_skewed_dist(n=self.sample_size, seed=seed)
            elif var_type == 'left_skewed':
                dataset[var_name] = self._left_skewed_dist(n=self.sample_size, seed=seed)
            elif var_type == 'high_kurtosis':
                dataset[var_name] = self._high_kurtosis_dist(n=self.sample_size, seed=seed)
            elif var_type == 'uniform':
                dataset[var_name] = self._uniform_dist(n=self.sample_size, seed=seed)
            else:
                raise ValueError(f"Unknown variable type: {var_type}")
        
        # Generate dependent variable using effects (betas)
        dep_var = self.variables['variable_0']['name']
        y = np.zeros(self.sample_size)
        
        # Add effects based on effect sizes
        for effect_key, effect_info in self.effects.items():
            effect_name = effect_info['name']
            effect_size = effect_info.get('effect_size', 0.0)
            
            if ':' in effect_name:
                # Interaction effect
                var_names = effect_name.split(':')
                interaction_term = np.ones(self.sample_size)
                for var in var_names:
                    interaction_term *= dataset[var]
                y += effect_size * interaction_term
            else:
                # Main effect
                y += effect_size * dataset[effect_name]
        
        # Add error term (normal with sd=1)
        error = np.random.normal(0, 1, self.sample_size)
        y += error
        
        dataset[dep_var] = y
        
        return dataset
    
    def run_regression(self, dataset, debug=False):
        """
        Perform linear regression and return significance results.
        
        Args:
            dataset: Dictionary with variable data
            debug: If True, prints detailed regression table
        
        Returns:
            dict: Contains 'overall_significant' (F-test) and individual variable significance
        """
        # Convert to DataFrame for easier handling
        df = pd.DataFrame(dataset)
        
        # Get dependent variable
        dep_var = self.variables['variable_0']['name']
        
        # Build design matrix including interactions
        X_columns = []
        X_names = []
        
        # Add all effects (main and interactions) to the design matrix
        for effect_key, effect_info in self.effects.items():
            effect_name = effect_info['name']
            
            if ':' in effect_name:
                # Interaction term
                var_names = effect_name.split(':')
                interaction_term = np.ones(len(df))
                for var in var_names:
                    interaction_term *= df[var].values
                X_columns.append(interaction_term)
                X_names.append(effect_name)
            else:
                # Main effect
                X_columns.append(df[effect_name].values)
                X_names.append(effect_name)
        
        # Prepare data
        y = df[dep_var].values
        X = np.column_stack(X_columns)
        
        # Fit regression
        model = LinearRegression()
        model.fit(X, y)
        
        # Calculate F-statistic for overall model
        y_pred = model.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        
        n = len(y)
        p = len(X_names)
        
        # F-test for overall significance
        f_stat = ((ss_tot - ss_res) / p) / (ss_res / (n - p - 1))
        f_p_value = 1 - stats.f.cdf(f_stat, p, n - p - 1)
        
        # T-tests for individual variables
        mse = ss_res / (n - p - 1)
        
        # Calculate standard errors
        X_with_intercept = np.column_stack([np.ones(n), X])
        try:
            cov_matrix = mse * np.linalg.inv(X_with_intercept.T @ X_with_intercept)
            std_errors = np.sqrt(np.diag(cov_matrix)[1:])  # Skip intercept
            
            # Calculate t-statistics and p-values
            t_stats = model.coef_ / std_errors
            t_p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), n - p - 1))
            
        except np.linalg.LinAlgError:
            # Handle singular matrix (perfect multicollinearity)
            t_p_values = [1.0] * len(X_names)
        
        # Build results
        results: Dict[str, bool] = {
            'overall_significant': bool(f_p_value < self.alpha)
        }
        
        # Add individual variable significance
        for i, effect_name in enumerate(X_names):
            results[f'{effect_name}_significant'] = bool(t_p_values[i] < self.alpha)
        
        # Add detailed statistics
        results['statistics'] = {
            'f_statistic': float(f_stat),
            'f_p_value': float(f_p_value),
            'r_squared': float(1 - ss_res/ss_tot),
            'coefficients': {},
            't_statistics': {},
            'p_values': {},
            'standard_errors': {}
        }
        
        # Add individual statistics
        try:
            for i, effect_name in enumerate(X_names):
                results['statistics']['coefficients'][effect_name] = float(model.coef_[i])
                results['statistics']['t_statistics'][effect_name] = float(t_stats[i])
                results['statistics']['p_values'][effect_name] = float(t_p_values[i])
                results['statistics']['standard_errors'][effect_name] = float(std_errors[i])
        except:
            # Handle multicollinearity case
            for i, effect_name in enumerate(X_names):
                results['statistics']['coefficients'][effect_name] = 0.0
                results['statistics']['t_statistics'][effect_name] = 0.0
                results['statistics']['p_values'][effect_name] = 1.0
                results['statistics']['standard_errors'][effect_name] = 0.0
        
        # Print debug table if requested
        if debug:
            print("\n" + "="*70)
            print("REGRESSION RESULTS")
            print("="*70)
            print(f"Dependent Variable: {dep_var}")
            print(f"R-squared: {1 - ss_res/ss_tot:.4f}")
            print(f"F-statistic: {f_stat:.4f}, p-value: {f_p_value:.4f}")
            print(f"Overall significant: {results['overall_significant']}")
            print("\n" + "-"*70)
            print(f"{'Variable':<15} {'Coeff':<10} {'Std Err':<10} {'t-stat':<10} {'p-value':<10} {'Signif':<8}")
            print("-"*70)
            
            try:
                for i, effect_name in enumerate(X_names):
                    coef = model.coef_[i]
                    std_err = std_errors[i]
                    t_stat = t_stats[i]
                    p_val = t_p_values[i]
                    signif = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
                    
                    print(f"{effect_name:<15} {coef:<10.4f} {std_err:<10.4f} {t_stat:<10.4f} {p_val:<10.4f} {signif:<8}")
            except:
                print("Error displaying coefficients (possible multicollinearity)")
            
            print("-"*70)
            print("Significance codes: *** p<0.001, ** p<0.01, * p<0.05")
            print("="*70)
        
        return results
    
    def find_power(self, target_test='overall', test_formula=None, seed=None, correction=None):
        """
        Run Monte Carlo power analysis for one or more tests.
        
        Args:
            target_test: Test(s) to analyze power for. Options:
                        - Single test: 'overall', 'x1', 'x2:x3', etc.
                        - Multiple tests: 'x1, x2, x2:x3' (comma-separated string)
                        - All tests: 'all' (includes overall + all effects)
                        - Default: 'overall' for F-test
            test_formula: Optional different formula for testing (default: use init formula)
            seed: Random seed for reproducibility
            correction: Multiple comparison correction (placeholder for future)
        
        Returns:
            dict: Power analysis results
        """
        if self.sample_size is None:
            raise ValueError("Sample size must be set using set_sample_size()")
        
        if not self.effect_sizes_initiated:
            raise ValueError("Effect sizes must be set using set_effects()")
        
        # Parse target_test string into list
        if isinstance(target_test, str):
            if target_test.strip().lower() == 'all':
                # Get all available effects plus overall
                target_test = ['overall'] + [effect_info['name'] for effect_info in self.effects.values()]
            elif ',' in target_test:
                target_test = [test.strip() for test in target_test.split(',')]
            else:
                target_test = [target_test.strip()]
        
        # Handle dependent variable name as overall test
        dep_var_name = self.variables['variable_0']['name']
        target_test = ['overall' if test == dep_var_name else test for test in target_test]
        
        # Validate target_tests
        valid_effects = [effect_info['name'] for effect_info in self.effects.values()]
        for test in target_test:
            if test != 'overall' and test not in valid_effects:
                raise ValueError(f"Target test '{test}' not found. Valid options: 'overall', {valid_effects}")
        
        # Handle test formula
        if test_formula:
            try:
                temp_model = MCPower(test_formula)
                temp_model.sample_size = self.sample_size
                temp_model.alpha = self.alpha
                temp_model.n_simulations = self.n_simulations
                temp_model.power = self.power
                
                # Copy variable types
                if self.variable_types_initiated:
                    temp_model.variable_types_initiated = True
                    for var_key, var_info in self.variables.items():
                        for temp_var_key, temp_var_info in temp_model.variables.items():
                            if temp_var_info['name'] == var_info['name']:
                                temp_model.variables[temp_var_key]['type'] = var_info.get('type', 'normal')
                                if 'proportion' in var_info:
                                    temp_model.variables[temp_var_key]['proportion'] = var_info['proportion']
                                break
                
                # Validate that all target_tests exist in test formula
                test_effects = [effect_info['name'] for effect_info in temp_model.effects.values()]
                for test in target_test:
                    if test != 'overall' and test not in test_effects:
                        raise ValueError(f"Target test '{test}' not found in test formula. Available: {test_effects}")
            
            except Exception as e:
                raise ValueError(f"Invalid test formula '{test_formula}': {str(e)}")
        else:
            temp_model = self
        
        # Initialize tracking
        individual_significant_counts = {test: 0 for test in target_test}
        combined_significant_counts = {i: 0 for i in range(len(target_test) + 1)}  # 0, 1, 2, ..., all
        all_results = []
        
        # Run Monte Carlo simulations
        for i in range(self.n_simulations):
            try:
                sim_seed = seed + i if seed is not None else None
                dataset = self.generate_dataset(seed=sim_seed)
                
                # Validate dataset
                required_vars = [var_info['name'] for var_info in temp_model.variables.values()]
                missing_vars = [var for var in required_vars if var not in dataset]
                if missing_vars:
                    raise ValueError(f"Generated dataset missing variables: {missing_vars}")
                
                # Run regression
                results = temp_model.run_regression(dataset, debug=False)
                all_results.append(results)
                
                # Track individual test significance
                sim_significant = []
                for test in target_test:
                    if test == 'overall':
                        is_sig = results['overall_significant']
                    else:
                        test_key = f'{test}_significant'
                        is_sig = results.get(test_key, False)
                    
                    sim_significant.append(is_sig)
                    if is_sig:
                        individual_significant_counts[test] += 1
                
                # Count how many tests were significant in this simulation
                num_significant = sum(sim_significant)
                combined_significant_counts[num_significant] += 1
                        
            except Exception as e:
                raise RuntimeError(f"Error in simulation {i+1}: {str(e)}")
        
        # Calculate individual powers
        individual_powers = {test: (count / self.n_simulations) * 100 
                           for test, count in individual_significant_counts.items()}
        
        # Calculate combined probabilities
        combined_probabilities = {f'exactly_{i}_significant': (count / self.n_simulations) * 100 
                                for i, count in combined_significant_counts.items()}
        
        # Calculate cumulative probabilities
        cumulative_probabilities = {}
        for i in range(len(target_test) + 1):
            cumulative_count = sum(combined_significant_counts[j] for j in range(i, len(target_test) + 1))
            cumulative_probabilities[f'at_least_{i}_significant'] = (cumulative_count / self.n_simulations) * 100
        
        # Collect detailed statistics for each test
        detailed_stats = {}
        for test in target_test:
            if test == 'overall':
                p_values = [r['statistics']['f_p_value'] for r in all_results]
                test_stats = [r['statistics']['f_statistic'] for r in all_results]
            else:
                p_values = [r['statistics']['p_values'].get(test, 1.0) for r in all_results]
                test_stats = [r['statistics']['t_statistics'].get(test, 0.0) for r in all_results]
            
            detailed_stats[test] = {
                'power': individual_powers[test],
                'mean_p_value': np.mean(p_values),
                'mean_test_statistic': np.mean(test_stats),
                'power_achieved': individual_powers[test] >= self.power
            }
        
        # Print results
        print(f"\n{'='*80}")
        print("MONTE CARLO POWER ANALYSIS RESULTS")
        print(f"{'='*80}")
        print(f"Data formula: {self.equation}")
        print(f"Test formula: {test_formula if test_formula else self.equation}")
        print(f"Sample size: {self.sample_size}, Alpha: {self.alpha}, Simulations: {self.n_simulations}")
        if correction:
            print(f"Multiple comparison correction: {correction}")
        
        print(f"\n{'Individual Test Powers:':<30}")
        print(f"{'Test':<20} {'Power (%)':<12} {'Target (%)':<12} {'Achieved':<10}")
        print(f"{'-'*54}")
        for test in target_test:
            power = individual_powers[test]
            achieved = "✓" if power >= self.power else "✗"
            print(f"{test:<20} {power:<12.2f} {self.power:<12.1f} {achieved:<10}")
        
        print(f"\n{'Combined Test Probabilities:'}")
        print(f"{'Outcome':<25} {'Probability (%)':<15}")
        print(f"{'-'*40}")
        for key, prob in combined_probabilities.items():
            print(f"{key.replace('_', ' ').title():<25} {prob:<15.2f}")
        
        print(f"\n{'Cumulative Probabilities:'}")
        print(f"{'Outcome':<25} {'Probability (%)':<15}")
        print(f"{'-'*40}")
        for key, prob in cumulative_probabilities.items():
            print(f"{key.replace('_', ' ').title():<25} {prob:<15.2f}")
        
        print(f"{'='*80}")
        
        # Prepare return results
        power_results = {
            'target_tests': target_test,
            'test_formula': test_formula if test_formula else self.equation,
            'data_formula': self.equation,
            'sample_size': self.sample_size,
            'alpha': self.alpha,
            'n_simulations': self.n_simulations,
            'correction': correction,
            'individual_powers': individual_powers,
            'detailed_statistics': detailed_stats,
            'combined_probabilities': combined_probabilities,
            'cumulative_probabilities': cumulative_probabilities,
            'overall_power_achieved': all(stats['power_achieved'] for stats in detailed_stats.values())
        }
        
        return power_results
    
    def get_sample_size(self, target_test='overall', from_size=30, to_size=200, by=5, 
                       test_formula=None, seed=None, correction=None, plot=True):
        """
        Find sample sizes needed to achieve target power through Monte Carlo analysis.
        
        Args:
            target_test: Test(s) to analyze. Same format as find_power()
                        - Single: 'overall', 'x1', 'x2:x3'
                        - Multiple: 'x1, x2, x2:x3'
                        - All: 'all'
            from_size: Starting sample size (default: 30)
            to_size: Ending sample size (default: 200)  
            by: Step size (default: 5)
            test_formula: Optional different formula for testing
            seed: Random seed for reproducibility
            correction: Multiple comparison correction (placeholder)
            plot: Whether to create plots (default: True)
        
        Returns:
            dict: Sample size analysis results with plots
        """
        if not self.effect_sizes_initiated:
            raise ValueError("Effect sizes must be set using set_effects()")
        
        # Store original sample size
        original_sample_size = self.sample_size
        
        # Parse target_test (reuse logic from find_power)
        if isinstance(target_test, str):
            if target_test.strip().lower() == 'all':
                target_test = ['overall'] + [effect_info['name'] for effect_info in self.effects.values()]
            elif ',' in target_test:
                target_test = [test.strip() for test in target_test.split(',')]
            else:
                target_test = [target_test.strip()]
        
        # Handle dependent variable name as overall test
        dep_var_name = self.variables['variable_0']['name']
        target_test = ['overall' if test == dep_var_name else test for test in target_test]
        
        # Validate target_tests
        valid_effects = [effect_info['name'] for effect_info in self.effects.values()]
        for test in target_test:
            if test != 'overall' and test not in valid_effects:
                raise ValueError(f"Target test '{test}' not found. Valid options: 'overall', {valid_effects}")
        
        # Generate sample size range
        sample_sizes = list(range(from_size, to_size + 1, by))
        
        # Store results for each sample size
        results_by_size = {}
        powers_by_test = {test: [] for test in target_test}
        first_achieved = {test: None for test in target_test}
        
        print(f"\n{'='*80}")
        print("SAMPLE SIZE ANALYSIS")
        print(f"{'='*80}")
        print(f"Target power: {self.power}%")
        print(f"Testing: {', '.join(target_test)}")
        print(f"Sample size range: {from_size} to {to_size} by {by}")
        print(f"{'='*80}")
        
        # Run power analysis for each sample size

        achieved_count = 0
        for sample_size in sample_sizes:
            print(f"Testing sample size: {sample_size}", end=" ... ")
            
            # Set sample size temporarily
            self.set_sample_size(sample_size)
            
            # Run power analysis (suppress output)
            import sys
            from io import StringIO
            
            # Capture print output from find_power
            old_stdout = sys.stdout
            sys.stdout = StringIO()
            
            try:
                power_result = self.find_power(
                    target_test=', '.join(target_test),
                    test_formula=test_formula,
                    seed=seed,
                    correction=correction
                )
                
                # Restore stdout
                sys.stdout = old_stdout
                
                results_by_size[sample_size] = power_result
                
                # Extract individual powers

                for test in target_test:
                    power = power_result['individual_powers'][test]
                    powers_by_test[test].append(power)
                    
                    # Check if this is first time achieving target power
                    if power >= self.power and first_achieved[test] is None:
                        first_achieved[test] = sample_size
                        achieved_count += 1
                
                print(f"Done ({achieved_count}/{len(target_test)} achieved)")
                
            except Exception as e:
                sys.stdout = old_stdout
                print(f"Error: {str(e)}")
                # Fill with zeros for failed sample sizes
                for test in target_test:
                    powers_by_test[test].append(0.0)
        
        # Restore original sample size
        if original_sample_size is not None:
            self.set_sample_size(original_sample_size)
        else:
            self.sample_size = None
        
        # Create plots if requested
        if plot:
            self._create_sample_size_plots(sample_sizes, powers_by_test, first_achieved, target_test)
        
        # Print summary
        print(f"\n{'Sample Size Requirements:'}")
        print(f"{'Test':<20} {'Required N':<12} {'Status':<15}")
        print(f"{'-'*47}")
        for test in target_test:
            req_n = first_achieved[test] if first_achieved[test] else f">{to_size}"
            status = "Achieved" if first_achieved[test] else "Not achieved"
            print(f"{test:<20} {req_n:<12} {status:<15}")
        
        print(f"{'='*80}")
        
        # Prepare results
        sample_size_results = {
            'target_tests': target_test,
            'target_power': self.power,
            'sample_sizes_tested': sample_sizes,
            'powers_by_test': powers_by_test,
            'first_achieved': first_achieved,
            'all_results': results_by_size,
        }
        
        return sample_size_results
    
    def _create_sample_size_plots(self, sample_sizes, powers_by_test, first_achieved, target_test):
        """Create plots for sample size analysis."""
        # Set up the plot
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Plot power curves for each test
        colors = plt.cm.Set1(np.linspace(0, 1, len(target_test)))
        
        for i, test in enumerate(target_test):
            powers = powers_by_test[test]
            ax.plot(sample_sizes, powers, 'o-', color=colors[i], 
                   label=test, linewidth=2, markersize=4)
            
            # Mark first achievement point
            if first_achieved[test] is not None:
                achieved_idx = sample_sizes.index(first_achieved[test])
                achieved_power = powers[achieved_idx]
                ax.plot(first_achieved[test], achieved_power, 's', 
                       color=colors[i], markersize=10, markerfacecolor='white',
                       markeredgewidth=2, markeredgecolor=colors[i])
                
                # Add annotation
                ax.annotate(f'N={first_achieved[test]}', 
                           xy=(first_achieved[test], achieved_power),
                           xytext=(10, 10), textcoords='offset points',
                           bbox=dict(boxstyle='round,pad=0.3', facecolor=colors[i], alpha=0.3),
                           arrowprops=dict(arrowstyle='->', color=colors[i]))
        
        # Add target power line
        ax.axhline(y=self.power, color='red', linestyle='--', linewidth=2, 
                  label=f'Target Power ({self.power}%)')
        
        # Formatting
        ax.set_xlabel('Sample Size', fontsize=12)
        ax.set_ylabel('Power (%)', fontsize=12)
        ax.set_title('Power Analysis: Sample Size Requirements', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.set_ylim(0, 105)
        
        # Set nice tick marks
        ax.set_xticks(range(min(sample_sizes), max(sample_sizes)+1, 
                           max(10, (max(sample_sizes) - min(sample_sizes)) // 10)))
        
        plt.tight_layout()
        plt.show()
    
    def __repr__(self):
        return f'equation: {self.equation}' + '\n' + f'effects: {self.effects}'