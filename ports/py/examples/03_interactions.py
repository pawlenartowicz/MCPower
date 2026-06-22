"""Interaction effects in power analysis; `a*b` expands to `a + b + a:b`, `:` is interaction-only. Interaction effect-size benchmarks are only directly comparable to main-effect benchmarks when both components are uncorrelated continuous predictors."""

from mcpower import MCPower

# 1. The `*` operator — factorial shorthand. It expands to advertising + age +
#    advertising:age, so all three terms take an effect below.
model = MCPower("sales = advertising * age")
model.set_effects("advertising=0.4, age=0.2, advertising:age=0.3")
model.set_variable_type("advertising=binary")

# 2. Power for the interaction term — short form. Target the `a:b` term directly.
#    At N=200 the interaction sits well below 80%: interactions are expensive.
print(">>> model.find_power(sample_size=200, target_test='advertising:age')")
model.find_power(sample_size=200, target_test="advertising:age")

# 3. The `:` operator — write the interaction explicitly. This formula is
#    equivalent to `advertising * age` above (use `:` when you want to spell out
#    exactly which terms are in the model), so the interaction power matches.
#    Long form via .summary() for the full picture at a larger N.
explicit = MCPower("sales = advertising + age + advertising:age")
explicit.set_effects("advertising=0.4, age=0.2, advertising:age=0.3")
explicit.set_variable_type("advertising=binary")
print("\n>>> result = explicit.find_power(sample_size=300, target_test='all', verbose=False)")
print(">>> print(result.summary())")
result = explicit.find_power(sample_size=300, target_test="all", verbose=False)
print(result.summary())

# 4. Robustness — every term under each scenario, with the Δ drop column.
#    The interaction degrades fastest once assumptions are violated.
print("\n>>> model.find_power(sample_size=250, target_test='all', scenarios=True)")
model.find_power(sample_size=250, target_test="all", scenarios=True)
