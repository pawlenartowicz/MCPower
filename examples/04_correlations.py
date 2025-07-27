"""
Correlated Predictors Example
=============================

This example shows how to handle correlated predictors in power analysis.
Real-world predictors are often correlated, which affects statistical power.
"""

import mcpower
import numpy as np

# Example: Social science study with correlated predictors
# Research question: What factors predict life satisfaction?

print("=" * 60)
print("CORRELATED PREDICTORS EXAMPLE")
print("=" * 60)

# 1. Define model with multiple predictors
model = mcpower.LinearRegression(
    "life_satisfaction = income + education + social_support + health"
)

# 2. Set effect sizes
model.set_effects("income=0.3, education=0.25, social_support=0.4, health=0.5")

print("Model with correlated predictors:")
print(f"Formula: {model.equation}")
print("Effects: income=0.3, education=0.25, social_support=0.4, health=0.5")

# 3. Set correlations between predictors (string format)
print("\n" + "=" * 60)
print("CORRELATION SPECIFICATIONS")
print("=" * 60)

print("\n1. STRING-BASED CORRELATIONS:")
model.set_correlations(
    """
    corr(income, education)=0.6, 
    corr(income, health)=0.4, 
    corr(education, health)=0.3,
    corr(social_support, health)=0.5
"""
)

# Power analysis with correlations
print("\n2. POWER WITH CORRELATIONS:")
corr_power = model.find_power(
    sample_size=200, target_test="all", scenarios=False, summary="short"
)

print("\n3. ROBUST ANALYSIS WITH CORRELATIONS:")
robust_corr = model.find_power(
    sample_size=200,
    target_test="all",
    scenarios=True,  # Correlation patterns may vary
    summary="short",
)

# 4. Matrix-based correlation specification
print("\n" + "=" * 60)
print("MATRIX-BASED CORRELATIONS")
print("=" * 60)

# Create new model for matrix example
matrix_model = mcpower.LinearRegression("wellbeing = stress + exercise + sleep")
matrix_model.set_effects("stress=-0.4, exercise=0.3, sleep=0.5")

# Define full correlation matrix
correlation_matrix = np.array(
    [
        [1.0, -0.3, 0.4],  # stress with others
        [-0.3, 1.0, 0.2],  # exercise with others
        [0.4, 0.2, 1.0],  # sleep with others
    ]
)

matrix_model.set_correlations(correlation_matrix)

print("Full correlation matrix specified:")
print("stress-exercise: -0.3 (negative: less stress, more exercise)")
print("stress-sleep: 0.4 (positive: more stress, worse sleep)")
print("exercise-sleep: 0.2 (positive: more exercise, better sleep)")

# Sample size with correlation matrix
matrix_n = matrix_model.find_sample_size(
    target_test="all",
    from_size=100,
    to_size=500,
    by=50,
    scenarios=True,
    summary="short",
)

# 5. Compare uncorrelated vs correlated models
print("\n" + "=" * 60)
print("CORRELATION IMPACT COMPARISON")
print("=" * 60)

# Model without correlations
uncorr_model = mcpower.LinearRegression("outcome = x1 + x2 + x3")
uncorr_model.set_effects("x1=0.4, x2=0.3, x3=0.5")
# No correlation specification = independent predictors

print("\n1. INDEPENDENT PREDICTORS:")
uncorr_power = uncorr_model.find_power(
    sample_size=150, target_test="all", scenarios=False, summary="short"
)

# Same model with strong correlations
corr_model = mcpower.LinearRegression("outcome = x1 + x2 + x3")
corr_model.set_effects("x1=0.4, x2=0.3, x3=0.5")
corr_model.set_correlations("corr(x1,x2)=0.7, corr(x1,x3)=0.6, corr(x2,x3)=0.8")

print("\n2. HIGHLY CORRELATED PREDICTORS:")
high_corr_power = corr_model.find_power(
    sample_size=150, target_test="all", scenarios=False, summary="short"
)

# 6. Sample size comparison
print("\n3. SAMPLE SIZE IMPACT:")
uncorr_n = uncorr_model.find_sample_size(
    target_test="x2", from_size=50, to_size=300, by=25, scenarios=True, summary="short"
)

corr_n = corr_model.find_sample_size(
    target_test="x2", from_size=50, to_size=400, by=25, scenarios=True, summary="short"
)

print("\n" + "=" * 60)
print("CORRELATION GUIDELINES")
print("=" * 60)
print(
    """
Key insights about correlated predictors:

1. POWER IMPACT:
   - High correlations may reduce power to detect individual effects
   - Overall model power may remain similar
   - Huge impact on interactions

2. CORRELATION SIZES:
   - 0.1-0.3: Small correlation
   - 0.3-0.7: Moderate correlation  
   - >0.7: Strong correlation (multicollinearity concern)

3. DESIGN STRATEGIES:
   - Measure actual correlations in pilot data
   - Use scenarios=True to test correlation uncertainty
   - Consider dropping one of highly correlated predictors
   - Focus on theoretically important predictors

4. SPECIFICATION OPTIONS:
   - String format: Good for sparse correlations
   - Matrix format: Good for complete correlation structure
   - Upload data: Preserves empirical correlations
"""
)
