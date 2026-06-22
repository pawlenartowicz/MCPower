"""Find the minimum sample size that reaches target power; tours the find_sample_size result: short form, .summary(), and save_plot."""

from mcpower import MCPower

# Example: Educational intervention study.
# Research question: What N do we need to detect the intervention effect?

# 1. Define the model.
model = MCPower("test_score = intervention + prior_knowledge + motivation")

# 2. Expected effects from literature / pilot data.
model.set_effects("intervention=0.4, prior_knowledge=0.35, motivation=0.3")

# 3. Variable types — intervention is binary (0=control, 1=intervention).
model.set_variable_type("intervention=binary")

# 4. Short form (printed automatically). find_sample_size sweeps a grid. The
#    headline is the model-based fitted N (isotonic crossing of the power curve,
#    atom-ceiled to the cluster size) when the fit succeeded; otherwise the
#    grid's first_achieved value.
#    The long-form summary() adds a "Required N & 95% CI" table with Wilson
#    band-inversion bounds, rounded outward to integers.
print(">>> model.find_sample_size(target_test='intervention', from_size=30, to_size=300, by=10)")
model.find_sample_size(target_test="intervention", from_size=30, to_size=300, by=10)

# 5. Long form for all effects, plus the joint "≥ k of n" required-N breakdown.
print("\n>>> result = model.find_sample_size(target_test='all', from_size=30, to_size=300, by=10, verbose=False)")
print(">>> print(result.summary())")
result = model.find_sample_size(
    target_test="all",
    from_size=30,
    to_size=300,
    by=10,
    verbose=False,
)
print(result.summary())

# save_plot() renders the power-vs-N curve to a file (png/svg/pdf/html); the
# same view shown inline in Jupyter. Requires: pip install mcpower[plot]
# result.save_plot("curve.png")

# 6. Robustness — the optimistic / realistic / doomer sweep (see 01) applied to
#    the sample-size search: the required N under each scenario.
print("\n>>> model.find_sample_size(target_test='intervention', from_size=30, to_size=400, by=20, scenarios=True)")
model.find_sample_size(
    target_test="intervention",
    from_size=30,
    to_size=400,
    by=20,
    scenarios=True,
)

# 7. Higher power requirement (90% instead of the 80% default).
print("\n>>> model.set_power(90)")
model.set_power(90)
print(">>> model.find_sample_size(target_test='intervention', from_size=50, to_size=400, by=20)")
model.find_sample_size(
    target_test="intervention",
    from_size=50,
    to_size=400,
    by=20,
)
