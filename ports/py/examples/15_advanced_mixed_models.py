"""Four advanced mixed-model designs, one step beyond the basic random
intercept of example 08: random slopes, nested groupings, a clustered logistic
GLMM whose treatment is assigned at the cluster level, and random-effect
scenario stress-testing. Each section is self-contained — build a model, ask
for power. Run the whole file with: python 15_advanced_mixed_models.py"""

from mcpower import MCPower

# ===========================================================================
# 1. Random slopes — (1 + x | group)
# ===========================================================================
# A random INTERCEPT lets each group start at a different baseline; a random
# SLOPE lets the effect itself differ across groups. Here a repeated-measures
# study gives every subject several dose levels, and each subject responds to
# dose at their own rate. Declaring the slope as random in the formula
# (1 + dose|subject) and giving it a variance widens the SE of the *average*
# dose effect — pretending the slope is fixed would overstate power.

model = MCPower("rt = dose + (1 + dose|subject)", family="lme")
model.set_effects("dose=0.3")

# random_slopes names the predictor whose slope varies; slope_variance is the
# spread of those per-subject slopes and slope_intercept_corr their correlation
# with the random intercept. 30 subjects, ICC=0.20 between-subject baseline.
model.set_cluster("subject", ICC=0.20, n_clusters=30,
                  random_slopes=["dose"], slope_variance=0.15,
                  slope_intercept_corr=0.0)

print(">>> model.find_power(sample_size=300, target_test='dose')")
model.find_power(sample_size=300, target_test="dose")

# Long form via .summary() — the LME diagnostics earn their keep here:
# random-slope fits hit variance boundaries more often than intercept-only ones,
# so watch the boundary-hit rate, not just the power.
print("\n>>> result = model.find_power(sample_size=300, target_test='dose', verbose=False)")
print(">>> print(result.summary())")
result = model.find_power(sample_size=300, target_test="dose", verbose=False)
print(result.summary())


# ===========================================================================
# 2. Nested random effects — (1 | school/classroom)
# ===========================================================================
# Nesting stacks groupings: classrooms sit inside schools, students inside
# classrooms, and both levels absorb variance. Declare the nesting in the
# formula as (1|school/classroom) and call set_cluster TWICE — once for the
# outer grouping (school) and once for the nested child ("school:classroom"),
# where n_per_parent fixes how many classrooms each school contains.

print("\n\n>>> Nested random effects (1|school/classroom):")
model = MCPower("score = teaching_method + (1|school/classroom)", family="lme")
model.set_variable_type("teaching_method=binary")
model.set_effects("teaching_method=0.4")

# 10 schools × 3 classrooms each = 30 classrooms total. n_per_parent=3 is what
# makes the child grouping nested within school rather than crossed with it.
model.set_cluster("school", ICC=0.15, n_clusters=10)
model.set_cluster("school:classroom", ICC=0.10, n_clusters=30, n_per_parent=3)

print(">>> model.find_power(sample_size=300, target_test='teaching_method')")
model.find_power(sample_size=300, target_test="teaching_method")


# ===========================================================================
# 3. Clustered logistic GLMM with a cluster-level predictor
# ===========================================================================
# Combine a binary outcome (family="logit"), a random intercept (1|clinic), and
# a treatment assigned once per clinic (cluster_level_vars) — the design of a
# cluster-randomised trial. The treatment's effective sample size is the number
# of clinics, not patients, so it is far harder to power than the patient-level
# covariate measured on every individual.

print("\n\n>>> Clustered logistic GLMM, treatment assigned per clinic:")
model = MCPower(
    "recovered = treatment + baseline_severity + (1|clinic)", family="logit"
)
model.set_variable_type("treatment=binary")
model.set_baseline_probability(0.3)
model.set_effects("treatment=0.6, baseline_severity=0.25")

# treatment is declared cluster-level: drawn once per clinic, so its SE scales
# with n_clusters=40, not N=600.
model.set_cluster("clinic", ICC=0.10, n_clusters=40,
                  cluster_level_vars=["treatment"])

print(">>> model.find_power(sample_size=600, target_test='treatment, baseline_severity')")
model.find_power(sample_size=600, target_test="treatment, baseline_severity")

# Sample-size search for the cluster-level treatment. Its effective sample size is
# the number of clinics (40), not patients, so it crosses 80% only at a large
# patient count — ~2000, i.e. ~50 per clinic — far higher than the patient-level
# baseline_severity needs. Adding clinics is the more efficient lever (see example 12).
print("\n>>> model.find_sample_size(target_test='treatment', from_size=800, to_size=2400, by=200)")
model.find_sample_size(
    target_test="treatment", from_size=800, to_size=2400, by=200
)


# ===========================================================================
# 4. Random-effect scenario stress-testing
# ===========================================================================
# Mixed models add three scenario knobs on top of the OLS ones from example 10:
#   random_effect_dist — normal / heavy_tailed / right_skewed random effects
#   random_effect_df   — degrees of freedom for the heavy-tailed t
#   icc_noise_sd       — jitter on the ICC, for when it is only an educated guess
# A brand-new scenario name inherits "optimistic" and overrides only these keys.

print("\n\n>>> Random-effect scenario stress-test:")
model = MCPower("score = teaching_method + prior_gpa + (1|classroom)", family="lme")
model.set_variable_type("teaching_method=binary")
model.set_effects("teaching_method=0.4, prior_gpa=0.18")
model.set_cluster("classroom", ICC=0.15, n_clusters=30)
model.set_scenario_configs({
    "re_stress": {"random_effect_dist": "heavy_tailed",
                  "random_effect_df": 5, "icc_noise_sd": 0.07}
})

# Run the clean baseline next to the stressed one. Power that barely moves is
# reassuring: the design is robust to non-Gaussian random effects and an
# uncertain ICC.
print(">>> model.find_power(sample_size=300, target_test='teaching_method',")
print("...                  scenarios=['optimistic', 're_stress'])")
model.find_power(sample_size=300, target_test="teaching_method",
                 scenarios=["optimistic", "re_stress"])
