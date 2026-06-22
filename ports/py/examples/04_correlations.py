"""Correlated predictors reduce statistical power; shows set_correlations via string or matrix and compares the independent vs correlated case using MCPower's own per-test tables."""

from mcpower import MCPower
import numpy as np

# Example: Social-science study.
# Research question: What predicts life satisfaction?

# 1. Model with several predictors.
model = MCPower("life_satisfaction = income + education + social_support + health")
model.set_effects("income=0.25, education=0.2, social_support=0.3, health=0.35")

# 2. String-based correlations — good for a sparse set of pairs.
model.set_correlations(
    "corr(income, education)=0.6, "
    "corr(income, health)=0.4, "
    "corr(education, health)=0.3, "
    "corr(social_support, health)=0.5"
)

# 3. Power across all predictors — long form. The joint distribution in
#    .summary() shows how correlation erodes the chance of detecting every
#    effect together.
print(">>> result = model.find_power(sample_size=150, target_test='all', verbose=False)")
print(">>> print(result.summary())")
result = model.find_power(sample_size=150, target_test="all", verbose=False)
print(result.summary())

# 4. Matrix-based correlations — good for a full correlation structure.
#    Short form (printed automatically).
matrix_model = MCPower("wellbeing = stress + exercise + sleep")
matrix_model.set_effects("stress=-0.3, exercise=0.25, sleep=0.3")
correlation_matrix = np.array(
    [
        [1.0, -0.3, 0.4],   # stress with exercise, sleep
        [-0.3, 1.0, 0.2],   # exercise with stress, sleep
        [0.4, 0.2, 1.0],    # sleep with stress, exercise
    ]
)
matrix_model.set_correlations(correlation_matrix)
print("\n>>> matrix_model.find_power(sample_size=150, target_test='all')")
matrix_model.find_power(sample_size=150, target_test="all")

# 5. Independent vs correlated, side by side. Same effects, same N — the only
#    difference is the correlation structure. Compare the two power tables:
#    every predictor loses power once they overlap.
independent = MCPower("outcome = x1 + x2 + x3")
independent.set_effects("x1=0.3, x2=0.25, x3=0.35")
print("\n>>> independent.find_power(sample_size=130, target_test='all')  # no correlations")
independent.find_power(sample_size=130, target_test="all")

correlated = MCPower("outcome = x1 + x2 + x3")
correlated.set_effects("x1=0.3, x2=0.25, x3=0.35")
correlated.set_correlations("corr(x1, x2)=0.7, corr(x1, x3)=0.6, corr(x2, x3)=0.8")
print("\n>>> correlated.find_power(sample_size=130, target_test='all')  # strong correlations")
correlated.find_power(sample_size=130, target_test="all")
