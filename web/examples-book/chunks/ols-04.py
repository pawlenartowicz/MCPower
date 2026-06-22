from mcpower import MCPower

# Continuous-by-continuous moderation: does the effect of income on well_being depend on social_support?
# '*' expands to the two main effects plus their interaction (income + social_support + income:social_support).
model = MCPower("well_being = income * social_support")

# Standardised effects. Main effects are moderate; the interaction (the test of
# interest) is smaller, as moderation effects usually are.
model.set_effects("income=0.30, social_support=0.25, income:social_support=0.15")

# Power for the interaction term at N=200.
model.find_power(sample_size=200, target_test="income:social_support")
