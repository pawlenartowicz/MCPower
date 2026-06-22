suppressMessages(library(mcpower))

# Moderation without water's own main effect: nitrogen predicts yield, and that slope is
# modulated by water (nitrogen:water), but water has no average effect of its own.
# `:` is interaction-only — it adds the product term without water's main effect.
model <- MCPower$new("yield ~ nitrogen + nitrogen:water")

# Both predictors are continuous (the default), so the interaction column is the
# product of two uncorrelated standard normals — unit variance, and the
# continuous benchmarks (0.10 / 0.25 / 0.40) apply to it directly.
#   nitrogen=0.40       -> large main slope.
#   nitrogen:water=0.25 -> medium moderation of that slope.
model$set_effects("nitrogen=0.40, nitrogen:water=0.25")

# Power for the interaction term — the hard-to-detect quantity here.
model$find_power(sample_size = 200, target_test = "nitrogen:water")
