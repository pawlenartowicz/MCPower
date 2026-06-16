"""Multiple-testing corrections (Bonferroni, Holm, BH) and the joint-significance distribution for answering "how many effects will I detect?"."""

from mcpower import MCPower

# Example: Medical study screening biomarkers.
# Research question: Which biomarkers predict treatment response?

# 1. Five predictors — two are true nulls (effect = 0).
model = MCPower(
    "treatment_response = biomarker1 + biomarker2 + biomarker3 + biomarker4 + age"
)
model.set_effects(
    "biomarker1=0.3, biomarker2=0.0, biomarker3=0.25, biomarker4=0.0, age=0.2"
)

# 2. The uncorrected baseline — long form. .summary() shows the joint
#    distribution ("exactly k" / "at least k" significant), the headline number
#    for a screening study: how many discoveries should you expect?
print(">>> uncorrected = model.find_power(sample_size=140, target_test='all', correction=None, verbose=False)")
print(">>> print(uncorrected.summary())")
uncorrected = model.find_power(
    sample_size=140, target_test="all", correction=None, verbose=False
)
print(uncorrected.summary())

# 3. Now apply each correction — short form, one table each. Compare against the
#    uncorrected baseline above to watch power traded for false-positive control;
#    the null biomarkers (biomarker2, biomarker4) stay near α whatever you pick.
for correction in ("Bonferroni", "Holm", "Benjamini-Hochberg"):
    print(f"\n>>> model.find_power(sample_size=140, target_test='all', correction={correction!r})")
    model.find_power(sample_size=140, target_test="all", correction=correction)

# 4. Sample size to reach target power under a correction.
print("\n>>> model.find_sample_size(target_test='all', from_size=50, to_size=400, by=20, correction='Bonferroni')")
model.find_sample_size(
    target_test="all",
    from_size=50,
    to_size=400,
    by=20,
    correction="Bonferroni",
)

# 5. Focused, pre-registered hypotheses only — correct across just the tests
#    you committed to, not every coefficient.
print("\n>>> model.find_power(sample_size=140, target_test='biomarker1, biomarker3', correction='Bonferroni')")
model.find_power(
    sample_size=140,
    target_test="biomarker1, biomarker3",  # only the promising biomarkers
    correction="Bonferroni",
)
