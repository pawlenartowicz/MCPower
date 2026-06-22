from mcpower import MCPower

# Continuous-by-continuous moderation on a binary outcome: does the effect of
# biomarker_level on whether a patient relapses (yes/no) depend on their age?
# '*' expands to both main effects plus the product term
# (biomarker_level + age + biomarker_level:age). family="logit" makes relapse a
# binary (0/1) outcome fitted by a logistic GLM.
model = MCPower("relapse = biomarker_level * age", family="logit")

# family="logit" needs the event rate at the reference level (all predictors at
# their mean): here 30% of patients relapse. This fixes the logistic
# intercept, so it must be set before find_power.
model.set_baseline_probability(0.3)

# Standardised effects on the continuous benchmark scale (0.10 / 0.25 / 0.40).
# Both main effects are moderate; the interaction (the test of interest) is
# smaller, as moderation effects usually are.
model.set_effects("biomarker_level=0.30, age=0.25, biomarker_level:age=0.15")

# Power for the interaction term at N=300.
model.find_power(sample_size=300, target_test="biomarker_level:age")
