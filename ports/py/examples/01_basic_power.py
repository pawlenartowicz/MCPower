"""Power calculation for a simple treatment study; tours the find_power result: short form, .summary(), .to_dataframe(), and save_plot."""

from mcpower import MCPower

# Example: Clinical trial testing a new therapy vs control.
# Research question: Does the new therapy improve patient outcomes?

# 1. Define the model with an R-style formula.
model = MCPower("patient_outcome = treatment + baseline_score")

# 2. Expected effect sizes (standardised).
#    treatment=0.5      → therapy shifts outcomes by 0.5 SD (a medium effect).
#    baseline_score=0.3 → baseline moderately predicts the outcome.
model.set_effects("treatment=0.5, baseline_score=0.3")

# 3. Variable types — treatment is binary (0=control, 1=therapy).
model.set_variable_type("treatment=binary")

# 4. Short form (printed automatically). One row per effect with Power,
#    95% CI, and a ✓/✗ marker against the target power. Here treatment lands
#    just under 80% at N=120 — the sample-size example finds the N that fixes it.
print(">>> model.find_power(sample_size=120, target_test='treatment')")
model.find_power(sample_size=120, target_test="treatment")

# 5. Long form via .summary(). verbose=False suppresses the auto short form;
#    target_test="all" adds the omnibus "Overall F" row and the joint-
#    significance distribution.
print("\n>>> result = model.find_power(sample_size=120, target_test='all', verbose=False)")
print(">>> print(result.summary())")
result = model.find_power(sample_size=120, target_test="all", verbose=False)
print(result.summary())

# 6. Robustness — rerun the analysis under three assumption scenarios:
#      optimistic — all assumptions hold (what you computed above)
#      realistic  — how your data might actually look
#      doomer     — a pessimistic (but still plausible) version of your data
#    Δ = power drop vs the optimistic baseline.
print("\n>>> model.find_power(sample_size=120, target_test='all', scenarios=True)")
model.find_power(sample_size=120, target_test="all", scenarios=True)

# 7. Use the result programmatically — .to_dataframe() gives a tidy
#    (test × scenario) frame for downstream code.
print("\n>>> result.to_dataframe()")
df = result.to_dataframe()
print(df.to_string(index=False))

# 8. save_plot() renders the chart to a file (png/svg/pdf/html); the same view
#    shown inline in Jupyter. Requires: pip install mcpower[plot]
# result.save_plot("power.png")
