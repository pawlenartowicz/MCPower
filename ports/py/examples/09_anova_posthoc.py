"""One-way ANOVA with post-hoc pairwise contrasts.

The omnibus F asks "do the group means differ at all?"; the post-hoc table asks
"which specific pairs differ?". `target_test="overall, all-contrasts"` reports
both. `correction="tukey"` applies Tukey HSD across the pairwise family; running
it once without and once with correction shows the family-wise power cost.
"""

from mcpower import MCPower

# Three-arm trial: does pain reduction differ across placebo / low / high dose?
model = MCPower("pain_reduction = dose_group")

# 3 roughly equal arms. No uploaded data -> integer-labelled levels 1, 2, 3
# (level 1 = reference). With uploaded data the labels are the data values.
model.set_variable_type("dose_group=(factor,0.34,0.33,0.33)")

# Per-arm effects vs the reference (placebo). The 2-vs-3 pairwise contrast is the
# 0.3 gap between these - the comparison only a full post-hoc view exposes.
model.set_effects("dose_group[2]=0.5, dose_group[3]=0.8")

# 1. Omnibus F + every pairwise contrast, UNCORRECTED.
print(">>> model.find_power(sample_size=130, target_test='overall, all-contrasts')")
model.find_power(sample_size=130, target_test="overall, all-contrasts")

# 2. Same design, Tukey HSD across the pairwise family. Compare the post-hoc
#    table's 'corrected' column to step 1 to see the family-wise power cost.
print("\n>>> model.find_power(sample_size=130, target_test='overall, all-contrasts', correction='tukey')")
model.find_power(sample_size=130, target_test="overall, all-contrasts", correction="tukey")

# 3. Long form adds CIs and the joint-significance distribution.
print("\n>>> print(model.find_power(sample_size=130, target_test='overall, all-contrasts',")
print("...       correction='tukey', verbose=False).summary())")
result = model.find_power(sample_size=130, target_test="overall, all-contrasts",
                          correction="tukey", verbose=False)
print(result.summary())
