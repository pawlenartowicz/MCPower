suppressMessages(library(mcpower))

# One-way design with all pairwise post-hoc comparisons: pain reduction
# across three treatment arms. Research question goes past "do the arms differ?"
# to "which arms differ from which?" — every treatment-vs-treatment pair, tested directly.
model <- MCPower$new("pain_reduction ~ treatment")

# treatment is a categorical predictor with 3 levels -> 3 pairwise comparisons
# (placebo-drug_a, placebo-drug_b, drug_a-drug_b).
model$set_variable_type("treatment=(factor,3)")

# Standardised effect on the factor benchmark scale (0.20 / 0.50 / 0.80):
# a medium separation between the treatment means. A 3-level factor expands to two
# reference-coded dummies (treatment[2], treatment[3]) — set each one explicitly.
model$set_effects("treatment[2]=0.5, treatment[3]=0.5")

# Power at N=180 for every pairwise post-hoc comparison, with Holm correction
# to control the family-wise error rate across the three tests.
invisible(model$find_power(
  sample_size = 180,
  target_test = "all-contrasts",
  correction = "holm"
))
