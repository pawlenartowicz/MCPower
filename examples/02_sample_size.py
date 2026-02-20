"""
Sample Size Calculation Example
===============================

This example shows how to find the minimum sample size needed to detect
your expected effects with adequate statistical power.
"""

from mcpower import MCPower

# Example: Educational intervention study
# Research question: What sample size do we need to detect the intervention effect?

print("=" * 60)
print("SAMPLE SIZE CALCULATION EXAMPLE")
print("=" * 60)

# 1. Define your study model
model = MCPower("test_score = intervention + prior_knowledge + motivation")

# 2. Set expected effect sizes based on literature/pilot data
# intervention = 0.4 means a medium-sized improvement from the intervention
model.set_effects("intervention=0.4, prior_knowledge=0.6, motivation=0.3")

# 3. Set variable types
model.set_variable_type("intervention=binary")  # 0=control, 1=intervention

print("\nStudy setup:")
print(f"Formula: {model.equation}")
print("Expected effects: intervention=0.4, prior_knowledge=0.6, motivation=0.3")
print("Target: 80% power to detect intervention effect")

# 4. Basic sample size calculation
print("\n" + "=" * 60)
print("SAMPLE SIZE ANALYSIS")
print("=" * 60)

print("\n1. BASIC CALCULATION (Optimistic Assumptions):")
basic_result = model.find_sample_size(
    target_test="intervention",  # Focus on intervention effect
    from_size=30,  # Start searching from N=30
    to_size=200,  # Up to N=200
    by=10,  # Test every 10 participants
    scenarios=False,  # Ideal conditions
    summary="short",
)

# 5. Robust sample size with scenario analysis
print("\n2. ROBUST CALCULATION (Realistic Conditions):")
robust_result = model.find_sample_size(
    target_test="intervention",
    from_size=30,
    to_size=300,  # Extended range for robustness
    by=10,
    scenarios=True,  # Test assumption violations
    summary="short",
)

# 6. Comprehensive analysis for all effects
print("\n3. COMPREHENSIVE ANALYSIS:")
comprehensive_result = model.find_sample_size(
    target_test="all",  # Find N for all effects
    from_size=50,
    to_size=300,
    by=10,
    scenarios=True,
    summary="long",  # Detailed output with power curves
)

# 7. Custom power target
print("\n4. HIGH POWER REQUIREMENT (90% power):")
model.set_power(90)  # Increase from default 80% to 90%
high_power_result = model.find_sample_size(
    target_test="intervention",
    from_size=50,
    to_size=400,
    by=10,
    scenarios=True,
    summary="short",
)

print("\n" + "=" * 60)
print("PLANNING RECOMMENDATIONS")
print("=" * 60)
print(
    """
Sample size decision framework:

1. BASIC CALCULATION: Use for initial estimates
2. ROBUST CALCULATION: Use for realistic planning
   - Plan with 'Realistic' scenario results
   - 'Doomer' scenario shows worst-case needs

3. Add 10-20% buffer for:
   - Dropout/missing data
   - Unforeseen complications
   - Conservative planning

4. Consider constraints:
   - Budget limitations
   - Time constraints
   - Population availability
"""
)
