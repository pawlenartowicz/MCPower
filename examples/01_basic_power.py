"""
Basic Power Analysis Example
============================

This example demonstrates basic power calculation for a simple treatment study.
Shows how to check if your planned sample size provides enough statistical power.
"""

import mcpower

# Example: Clinical trial testing new therapy vs control
# Research question: Does the new therapy improve patient outcomes?

print("=" * 60)
print("BASIC POWER ANALYSIS EXAMPLE")
print("=" * 60)

# 1. Define your model using R-style formula
model = mcpower.LinearRegression("patient_outcome = treatment + baseline_score")

# 2. Set expected effect sizes
# treatment = 0.5 means therapy improves outcomes by 0.5 standard deviations
# baseline_score = 0.7 means baseline strongly predicts outcome
model.set_effects("treatment=0.5, baseline_score=0.7")

# 3. Define variable types
# treatment is binary (0=control, 1=therapy)
model.set_variable_type("treatment=binary")

print("\nModel setup complete:")
print(f"Formula: {model.equation}")
print("Expected effects: treatment=0.5, baseline_score=0.7")
print("Treatment is binary variable")

# 4. Calculate power for a specific sample size
print("\n" + "=" * 60)
print("POWER ANALYSIS RESULTS")
print("=" * 60)

# Basic power calculation
print("\n1. BASIC POWER (Optimistic Assumptions):")
power_result = model.find_power(
    sample_size=160,
    target_test="treatment",  # Focus on treatment effect
    scenarios=False,          # No robustness testing
    summary='short'           # Concise output
)

# 5. Test robustness with scenario analysis
print("\n2. ROBUST POWER (Realistic Conditions):")
robust_result = model.find_power(
    sample_size=160,
    target_test="treatment",
    scenarios=True,           # Test under realistic violations
    summary='short'
)

# 6. Detailed analysis with all information
print("\n3. DETAILED ANALYSIS:")
detailed_result = model.find_power(
    sample_size=160,
    target_test="all",        # Test all effects
    scenarios=True,
    summary='long'            # Comprehensive output with plots
)

print("\n" + "=" * 60)
print("INTERPRETATION GUIDE")
print("=" * 60)
print("""
Key takeaways:
- Use 'scenarios=True' for realistic planning
- 'summary=long' provides detailed tables and plots
- Focus on 'Realistic' scenario results for conservative estimates
- If Doomer scenario is acceptable, you're very safe

Next steps:
- If power is too low, increase sample size
- If power is adequate, you're ready to proceed
- Consider multiple testing if testing several effects
""")