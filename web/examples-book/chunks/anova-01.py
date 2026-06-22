# NOTE: the omnibus F-test cannot be requested on its own — the engine requires
# at least one marginal/contrast/post-hoc target alongside it. target_test="all"
# is the canonical way to get the omnibus: it reports the overall F PLUS every
# per-dummy coefficient. (Bare target_test="overall" is inexpressible and was
# replaced with "all".)
from mcpower import MCPower

# One-way ANOVA: plant biomass measured across three fertilizer regimes.
# Research question: do the fertilizer groups differ overall? -> the omnibus F-test.
model = MCPower("biomass = fertilizer")

# fertilizer is a categorical predictor with 3 levels (control / low / high).
model.set_variable_type("fertilizer=(factor,3)")

# Standardised effects on the factor benchmark scale (0.20 / 0.50 / 0.80):
# each non-reference level shifts biomass by a medium amount vs the control.
# A factor expands into one dummy per non-reference level, so effects are named
# per dummy (fertilizer[2], fertilizer[3]) — the bare factor name is not an effect.
model.set_effects("fertilizer[2]=0.5, fertilizer[3]=0.5")

# Power at N=120 for the omnibus F-test of overall fertilizer differences, reported
# alongside each per-dummy coefficient ("all" = overall F + every β).
model.find_power(sample_size=120, target_test="all")
