![Tests](https://github.com/pawlenartowicz/MCPower/workflows/Tests/badge.svg)
![PyPI](https://img.shields.io/pypi/v/MCPower)
![Python](https://img.shields.io/pypi/pyversions/mcpower)
![License: GPL v3](https://img.shields.io/badge/License-GPLv3-green.svg)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.16502735.svg)](https://doi.org/10.5281/zenodo.16502735)


```
███╗   ███╗  ██████╗ ██████╗ 
████╗ ████║ ██╔════╝ ██╔══██╗ ██████╗ ██╗    ██╗███████╗██████╗ 
██╔████╔██║ ██║      ██║  ██║██╔═══██╗██║    ██║██╔════╝██╔══██╗
██║╚██╔╝██║ ██║      ██████╔╝██║   ██║██║ █╗ ██║█████╗  ██████╔╝
██║ ╚═╝ ██║ ██║      ██╔═══╝ ██║   ██║██║███╗██║██╔══╝  ██╔══██╗
██║     ██║ ╚██████╗ ██║     ╚██████╔╝╚███╔███╔╝███████╗██║  ██║
╚═╝     ╚═╝  ╚═════╝ ╚═╝      ╚═════╝  ╚══╝╚══╝ ╚══════╝╚═╝  ╚═╝
```
# MCPower

**Simple Monte Carlo power analysis for complex models.** Find the sample size you need or check if your study has enough power - even with complex models that traditional power analysis can't handle.

## Why MCPower?

**Traditional power analysis breaks down** with interactions, correlated predictors, categorical variables, or non-normal data. MCPower uses simulation instead of formulas - it generates thousands of datasets exactly like yours, then sees how often your analysis finds real effects.

✅ **Works with complexity**: Interactions, correlations, factors, any distribution  
✅ **R-style formulas**: `outcome = treatment + covariate + treatment*covariate`  
✅ **Categorical variables**: Multi-level factors automatically handled  
✅ **Two simple commands**: Find sample size or check power  
✅ **Scenario analysis**: Test robustness under realistic conditions  
✅ **Minimal math required**: Just specify your model and effects

## Get Started in 2 Minutes

### Install
```bash
pip install mcpower
```

### Update to the latest version (every few days).
```bash
pip install --upgrade mcpower
```

### Your First Power Analysis
```python

# 0. Import installed package
import mcpower

# 1. Define your model (just like R)
model = mcpower.LinearRegression("satisfaction = treatment + motivation")

# 2. Set effect sizes (how big you expect effects to be)
model.set_effects("treatment=0.5, motivation=0.3")

# 3. Change the treatment to "binary" (people receive treatment or not).
model.set_variable_type("treatment=binary")

# 4. Find the sample size you need
model.find_sample_size(target_test="treatment", from_size=50, to_size=200, summary="long")
```
**Output**: "You need N=75 for 80% power to detect the treatment effect"

That's it! 🎉

## 🎯 Scenario Analysis: Test Your Assumptions

**Real studies rarely match perfect assumptions.** MCPower's scenario analysis tests how robust your power calculations are under realistic conditions.

```python
# Test robustness with scenario analysis
model.find_sample_size(
    target_test="treatment", 
    from_size=50, to_size=300,
    scenarios=True  # 🔥 The magic happens here
)
```

**Output:**
```
SCENARIO SUMMARY
================================================================================

Uncorrected Sample Sizes:
Test                                     Optimistic   Realistic    Doomer      
-------------------------------------------------------------------------------
treatment                                75           85           100         
================================================================================
```

**What each scenario means:**
- **Optimistic**: Your ideal conditions (original settings)
- **Realistic**: Moderate real-world complications (small effect variations, mild assumption violations)
- **Doomer**: Conservative estimate (larger effect variations, stronger assumption violations)

**💡 Pro tip**: Use the **Realistic** scenario for planning. If **Doomer** is acceptable, you're really safe!

## Understanding Effect Sizes

**Effect sizes tell you how much the outcome changes when predictors change.**

- **Effect size = 0.5** means the outcome increases by **0.5 standard deviations** when:
  - **Continuous variables**: Predictor increases by 1 standard deviation  
  - **Binary variables**: Predictor changes from 0 to 1 (e.g., control → treatment)
  - **Factor variables**: Each level compared to reference level (first level)

**Practical examples:**
```python
model.set_effects("treatment=0.5, age=0.3, income=0.2")
```

- **`treatment=0.5`**: Treatment increases outcome by 0.5 SD (medium-large effect)
- **`age=0.3`**: Each 1 SD increase in age → 0.3 SD increase in outcome  
- **`income=0.2`**: Each 1 SD increase in income → 0.2 SD increase in outcome

**Effect size guidelines:**
- **0.1** = Small effect (detectable but modest)
- **0.25** = Medium effect (clearly noticeable) 
- **0.4** = Large effect (substantial impact)

**Effect size guidelines (binary variables):**
- **0.2** = Small effect (detectable but modest)
- **0.5** = Medium effect (clearly noticeable) 
- **0.8** = Large effect (substantial impact)

**Your uploaded data is automatically standardized** (mean=0, SD=1) so effect sizes work the same way whether you use synthetic or real data.

## Copy-Paste Examples for Common Studies

### Randomized Controlled Trial
```python
import mcpower

# RCT with treatment + control variables
model = mcpower.LinearRegression("outcome = treatment + age + baseline_score")
model.set_effects("treatment=0.6, age=0.2, baseline_score=0.8")
model.set_variable_type("treatment=binary")  # 0/1 treatment

# Find sample size for treatment effect with scenario analysis
model.find_sample_size(target_test="treatment", from_size=100, to_size=500, 
                      by=50, scenarios=True)
```

### A/B Test with Interaction
```python
import mcpower

# Test if treatment effect depends on user type
model = mcpower.LinearRegression("conversion = treatment + user_type + treatment*user_type")
model.set_effects("treatment=0.4, user_type=0.3, treatment:user_type=0.5")
model.set_variable_type("treatment=binary, user_type=binary")

# Check power robustness for the interaction
model.find_power(sample_size=400, target_test="treatment:user_type", scenarios=True)
```

### Multi-Group Study with Categorical Variables
```python
import mcpower

# Study with 3 treatment groups and 4 education levels
model = mcpower.LinearRegression("wellbeing = treatment + education + age")
model.set_variable_type("treatment=(factor,3), education=(factor,4)")

# Set effects for each factor level (vs. reference level 1)
model.set_effects("treatment[2]=0.4, treatment[3]=0.6, education[2]=0.3, education[3]=0.5, education[4]=0.7, age=0.2")

# Find sample size for treatment effects
model.find_sample_size(target_test="treatment[2], treatment[3]", scenarios=True)
```

### Survey with Correlated Predictors
```python
import mcpower

# Predictors are often correlated in real data
model = mcpower.LinearRegression("wellbeing = income + education + social_support")
model.set_effects("income=0.4, education=0.3, social_support=0.6")
model.set_correlations("corr(income, education)=0.5, corr(income, social_support)=0.3")

# Find sample size for any effect
model.find_sample_size(target_test="all", from_size=200, to_size=800, 
                      by=100, scenarios=True)
```

## Customize for Your Study

### Different Variable Types
```python
# Binary, factors, skewed, or other distributions
model.set_variable_type("treatment=binary, condition=(factor,3), income=right_skewed, age=normal")

# Binary with custom proportions (30% get treatment)
model.set_variable_type("treatment=(binary,0.3)")

# Factors with custom group sizes (20%, 50%, 30%)
model.set_variable_type("condition=(factor,0.2,0.5,0.3)")
```

### Working with Factors (Categorical Variables)
```python
# Factors automatically create dummy variables
model = mcpower.LinearRegression("outcome = treatment + education")
model.set_variable_type("treatment=(factor,3), education=(factor,4)")

# Set effects for specific levels (level 1 is always reference)
model.set_effects("treatment[2]=0.5, treatment[3]=0.7, education[2]=0.3, education[3]=0.4, education[4]=0.6")
```

### Your Own Data
```python
import pandas as pd

# Use your pilot data for realistic simulations
df = pd.read_csv('examples/cars.csv')
model.upload_own_data(df)  # Automatically preserves correlations
```

### Multiple Testing
```python
# Testing multiple effects? Control false positives
model.find_power(
    sample_size=200, 
    target_test="treatment,covariate,treatment:covariate",
    correction="Benjamini-Hochberg",
    scenarios=True  # Test robustness too!
)
```

### Test the single violation of assumptions.
```python
# Customize how much "messiness" to add in scenarios
model.set_heterogeneity(0.2)        # Effect sizes vary between people
model.set_heteroskedasticity(0.15)  # Violation of equal variance assumption

# Then run scenario analysis
model.find_sample_size(target_test="treatment", scenarios=False)
```

### More precision
```python
# To make a more precise estimation, consider increasing the number of simulations.
model.set_simulations(10000)

# MCPower is already heavily optimized, but there is old code that allows for parallelization. Use it to speed up your largest simulations.
model.set_parallel(True)

```

## Quick Reference

| **Want to...** | **Use this** |
|-----------------|--------------|
| Find required sample size | `model.find_sample_size(target_test="effect_name")` |
| Check power for specific N | `model.find_power(sample_size=150, target_test="effect_name")` |
|**Test robustness** | **Add `scenarios=True` to either method** |
|**Detailed output with plots**  | **Add `summary="long"` to either method** |
| Test overall model | `target_test="overall"` |
| Test multiple effects | `target_test="effect1, effect2"` or `"all"` |
| Binary variables | `model.set_variable_type("var=binary")` |
| **Factor variables** | **`model.set_variable_type("var=(factor,3)")`** |
| **Factor effects** | **`model.set_effects("var[2]=0.5, var[3]=0.7")`** |
| Correlated predictors | `model.set_correlations("corr(var1, var2)=0.4")` |
| Multiple testing correction | Add `correction="FDR", or "Holm" pr "Bonferroni"`|

## When to Use MCPower

**✅ Use MCPower when you have:**
- Interaction terms (`treatment*covariate`)
- **Categorical variables with multiple levels**
- Binary or non-normal variables
- Correlated predictors
- Multiple effects to test
- **Need to test assumption robustness**
- Complex models where traditional power analysis fails

**✅ Use Scenario Analysis when:**
- Planning important studies (grants, dissertations)
- Working with messy real-world data
- Effect sizes are uncertain
- Want conservative sample size estimates
- Stakeholders need confidence in your numbers

**❌ Use traditional power analysis for:**
- For models that are not yet implemented
- When all assumptions are clearly met

## What Makes Scenarios Different? (Be careful, unvalidated, preliminary scenarios)

**Traditional power analysis assumes perfect conditions.** MCPower's scenarios add realistic "messiness":

| **Scenario** | **What's Different** | **When to Use** |
|-------------|---------------------|------------------|
| **Optimistic** | Your exact settings | Best-case planning |
| **Realistic** | Mild effect variations, small assumption violations | **Recommended for most studies** |
| **Doomer** | Larger effect variations, stronger assumption violations | Conservative/worst-case planning |

**Behind the scenes**, scenarios randomly vary:
- Effect sizes between participants
- Correlation strengths  
- Variable distributions
- Assumption violations

This gives you a **range of realistic outcomes** instead of a single optimistic estimate.
⚠️ **Important**: Scenario analysis and uploaded data features are experimental. 
Use with caution for critical decisions.

<details>
<summary><strong>📚 Advanced Features (Click to expand)</strong></summary>

## Advanced Options

### All Variable Types
```python
model.set_variable_type("""
    treatment=binary,           # 0/1 with 50% split
    ses=(binary,0.3),          # 0/1 with 30% split  
    condition=(factor,3),       # 3-level factor (equal proportions)
    education=(factor,0.2,0.5,0.3), # 3-level factor (custom proportions)
    age=normal,                # Standard normal (default)
    income=right_skewed,       # Positively skewed
    depression=left_skewed,    # Negatively skewed
    response_time=high_kurtosis, # Heavy-tailed
    rating=uniform             # Uniform distribution
""")
```

### Factor Variables in Detail
```python
# Factor variables are categorical with multiple levels
model = mcpower.LinearRegression("outcome = treatment + education")

# Create factors
model.set_variable_type("treatment=(factor,3), education=(factor,4)")

# This creates dummy variables automatically:
# treatment[2], treatment[3] (treatment[1] is reference)
# education[2], education[3], education[4] (education[1] is reference)

# Set effects for specific levels
model.set_effects("treatment[2]=0.5, treatment[3]=0.7, education[2]=0.3")

# Or set same effect for all levels of a factor
model.set_effects("treatment=0.5")  # Applies to treatment[2] and treatment[3]

# Important: Factors cannot be used in correlations
# This will error: model.set_correlations("corr(treatment, education)=0.3")
# Use continuous variables only: model.set_correlations("corr(age, income)=0.3")
```

### Complex Correlation Structures
```python
import numpy as np

# Full correlation matrix for 3 CONTINUOUS variables only
# (Factors are excluded from correlation matrices)
corr_matrix = np.array([
    [1.0, 0.4, 0.6],    # Variable 1 with others
    [0.4, 1.0, 0.2],    # Variable 2 with others
    [0.6, 0.2, 1.0]     # Variable 3 with others
])
model.set_correlations(corr_matrix)
```

### Performance Tuning
```python
# Adjust for your needs
model.set_power(90)           # Target 90% power instead of 80%
model.set_alpha(0.01)         # Stricter significance (p < 0.01)
model.set_simulations(10000)  # High precision (slower)
```

### Formula Syntax
```python
# These are equivalent:
"y = x1 + x2 + x1*x2"        # Assignment style
"y ~ x1 + x2 + x1*x2"        # R-style formula  
"x1 + x2 + x1*x2"            # Predictors only

# Interactions:
"x1*x2"         # Main effects + interaction (x1 + x2 + x1:x2)
"x1:x2"         # Interaction only
"x1*x2*x3"      # All main effects + all interactions
```

### Correlation Syntax (Continuous Variables Only)
```python
# String format (recommended)
model.set_correlations("corr(x1, x2)=0.3, corr(x1, x3)=-0.2")

# Shorthand format  
model.set_correlations("(x1, x2)=0.3, (x1, x3)=-0.2")

# Note: Factor variables cannot be correlated
# Only use continuous/binary variables in correlations
```

</details>

## Requirements

- Python ≥ 3.7
- NumPy, SciPy, scikit-learn, matplotlib
- joblib (optional, for parallel processing)

## Need Help?

- **Issues**: [GitHub Issues](https://github.com/pawlenartowicz/MCPower/issues)
- **Questions**: pawellenartowicz@europe.com

## Aim for future (waiting for suggestions)
- ✅ Linear Regression
- ✅ Scenarios, robustness analysis
- ✅ Factor variables (categorical predictors)
- 🚧 Logistic Regression (coming soon)
- 🚧 ANOVA (coming soon)
- 🚧 Guide about methods, corrections (coming soon)
- 📋 Rewriting to Cython (backend change)
- 📋 Mixed Effects Models
- 📋 2 groups comparision with alternative tests
- 📋 Robust regression methods


## License & Citation

GPL v3. If you use MCPower in research, please cite:

Lenartowicz, P. (2025). MCPower: Monte Carlo Power Analysis for Statistical Models [Computer software]. Zenodo. https://doi.org/10.5281/zenodo.16502735

```bibtex
@software{mcpower2025,
  author = {Pawel Lenartowicz},
  title = {MCPower: Monte Carlo Power Analysis for Statistical Models},
  year = {2025},
  publisher = {Zenodo},
  doi = {10.5281/zenodo.16502735},
  url = {https://doi.org/10.5281/zenodo.16502735}
}
```

---

**🚀 Ready to start?** Copy one of the examples above and adapt it to your study!