# Logistic Regression Example
# ===========================
#
# Power for a binary outcome with family = "logit". set_baseline_probability()
# fixes the event rate at the reference level; effects are standardised log-odds
# shifts per SD (or per category for binary predictors).
#
# Run with:  Rscript 07_logistic_regression.R

suppressMessages(library(mcpower))

# Example: Observational study of a binary clinical event.
# Research question: Which factors predict whether a patient responds (yes/no)?

# 1. Declare a logistic model with family = "logit". The outcome is binary;
#    the engine fits a GLM (logit link) on every Monte Carlo iteration.
model <- MCPower$new("response ~ dose + age + sex", family = "logit")

# 2. sex is a binary predictor (0/1); dose and age are continuous (default).
model$set_variable_type("sex=binary")

# 3. Baseline probability — the event rate when every predictor is at its
#    reference value. Required for logit; it sets the model intercept via
#    log(p / (1 - p)). Here 30% of reference patients respond.
model$set_baseline_probability(0.3)

# 4. Effects are standardised log-odds shifts.
#    dose=0.4 -> a strong predictor; age=0.25 / sex=0.5 -> moderate.
model$set_effects("dose=0.4, age=0.25, sex=0.5")

# 5. Short form (printed automatically) — power per predictor at N=300.
#    Logistic models need more N than OLS for the same effect: at N=300 the
#    strong predictor (dose) clears 80% while the moderate age/sex sit mid-band.
cat(">>> model$find_power(sample_size = 300, target_test = 'all')\n")
invisible(model$find_power(sample_size = 300, target_test = "all"))

# 6. Long form via summary() for the joint-significance distribution and the
#    GLM diagnostics (convergence, realised baseline probability).
cat("\n>>> result <- model$find_power(sample_size = 300, target_test = 'all', verbose = FALSE)\n")
cat(">>> print(summary(result))\n")
result <- model$find_power(sample_size = 300, target_test = "all", verbose = FALSE)
print(summary(result))

# 7. Sample size for the primary predictor at the 80% default.
cat("\n>>> model$find_sample_size(target_test = 'dose', from_size = 100, to_size = 500, by = 25)\n")
invisible(model$find_sample_size(
  target_test = "dose", from_size = 100, to_size = 500, by = 25
))

# 8. Scenario sweep — works for logistic models too. Under realistic/doomer,
#    heterogeneity becomes log-odds jitter: each effect and the baseline odds
#    wobble by roughly ±20% / ±40% per observation, and the predictor-side
#    knobs (correlation noise, distribution swaps) apply as usual. Residual
#    and heteroskedasticity knobs have nothing to act on in a binary outcome
#    and are ignored.
cat("\n>>> model$find_power(sample_size = 300, target_test = 'all', scenarios = TRUE)\n")
invisible(model$find_power(sample_size = 300, target_test = "all", scenarios = TRUE))
