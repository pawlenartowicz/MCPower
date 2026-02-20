"""
Interactions Analysis Example
=============================

This example demonstrates how to analyze interaction effects - when the effect
of one variable depends on the level of another variable.
"""

from mcpower import MCPower

# Example: Marketing study with interaction
# Research question: Does the effect of advertising depend on customer age?

print("=" * 60)
print("INTERACTION EFFECTS EXAMPLE")
print("=" * 60)

# 1. Define model with interaction term
# advertising*age expands to: advertising + age + advertising:age
model = MCPower("sales = advertising * age")

# 2. Set effect sizes for all terms
# advertising:age = 0.3 means the advertising effect varies by age
model.set_effects("advertising=0.4, age=0.2, advertising:age=0.3")

# 3. Set variable types
model.set_variable_type("advertising=binary")  # 0=no ads, 1=ads shown

print("\nModel with interaction:")
print(f"Formula: {model.equation}")
print("Effects: advertising=0.4, age=0.2, advertising:age=0.3")
print("Interaction means: advertising effect varies by customer age")

# 4. Test power for interaction effect specifically
print("\n" + "=" * 60)
print("INTERACTION POWER ANALYSIS")
print("=" * 60)

print("\n1. POWER FOR INTERACTION EFFECT:")
interaction_power = model.find_power(
    sample_size=200,
    target_test="advertising:age",  # Focus on interaction
    scenarios=False,
    summary="short",
)

print("\n2. ROBUST INTERACTION TESTING:")
robust_interaction = model.find_power(
    sample_size=200,
    target_test="advertising:age",
    scenarios=True,  # More conservative
    summary="short",
)

# 5. Sample size for detecting interaction
print("\n3. SAMPLE SIZE FOR INTERACTION:")
interaction_n = model.find_sample_size(
    target_test="advertising:age",
    from_size=100,
    to_size=600,
    by=50,
    scenarios=True,
    summary="short",
)

# 6. Test all effects simultaneously
print("\n4. COMPREHENSIVE ANALYSIS:")
all_effects = model.find_power(
    sample_size=300,
    target_test="all",  # Test main effects + interaction
    scenarios=True,
    summary="long",  # Detailed output with plots
)

# 7. Complex interaction example
print("\n" + "=" * 60)
print("THREE-WAY INTERACTION EXAMPLE")
print("=" * 60)

# More complex model with three-way interaction
complex_model = MCPower("outcome = treatment * gender * age")

# Set effects (three-way interactions need large samples)
complex_model.set_effects(
    "treatment=0.5, gender=0.2, age=0.3, treatment:gender=0.2, treatment:age=0.1, gender:age=0.1, treatment:gender:age=0.4"
)

complex_model.set_variable_type("treatment=binary, gender=binary")

print("Complex model with three-way interaction:")
print(f"Formula: {complex_model.equation}")

# Sample size for three-way interaction (typically requires large N)
complex_n = complex_model.find_sample_size(
    target_test="treatment:gender:age",
    from_size=200,
    to_size=1000,
    by=100,
    scenarios=True,
    summary="short",
)

print("\n" + "=" * 60)
print("INTERACTION ANALYSIS GUIDELINES")
print("=" * 60)
print(
    """
Key insights about interactions:

1. SAMPLE SIZE: Interactions need larger samples than main effects
   - 2-way interactions: ~2-4x larger N
   - 3-way interactions: ~4-8x larger N

2. EFFECT SIZES: Interaction effects are often smaller
   - Main effects: 0.3-0.8 typical
   - Interactions: 0.1-0.4 typical

3. INTERPRETATION: 
   - Significant interaction = effect varies by subgroup
   - Use scenarios=True for robust planning
   - Consider if interaction is theoretically meaningful

4. DESIGN TIPS:
   - Balance groups when possible
   - Pre-register interaction hypotheses
   - You could include interaction you are not sure, 
      loss in overall power is small at worst
      
5. ATTENTION!
   - Most variability in power for replication is from correlation (next example)
"""
)
