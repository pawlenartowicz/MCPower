# 06_own_data.R — Upload your own data to drive power analysis from real
# distributions and correlations rather than synthetic defaults.
#
# Mirrors ports/py/examples/06_own_data.py: same models, effects, modes and
# find_power / find_sample_size calls. upload_data accepts data in several forms
# — a built-in / in-memory object (data.frame / matrix / named list) or a path to
# a .csv / .tsv file. Both feed the same engine.
#
# Research question: scale up a pilot study to a full trial.

library(mcpower)

# ---------------------------------------------------------------------------
# 1. From the built-in mtcars dataset — base R's data frame, no file to find.
# ---------------------------------------------------------------------------
# R ships the classic 32-car mtcars dataset in the base `datasets` package, so
# this example runs anywhere with no CSV on disk. A file path works identically:
# model$upload_data("pilot.csv", mode = "partial").
#
# upload_data preserves the empirical distributions and correlations
# automatically. mode="partial" (default) reproduces each predictor's empirical
# marginal plus the measured correlation matrix, so the synthetic data mirrors
# the pilot's distributions and dependence structure. Only columns named in the
# formula are used; any others in the data are reported as "(extra)" and ignored.
model <- MCPower$new("mpg ~ hp + wt + am")
cat(">>> model$upload_data(mtcars, mode='partial')\n")
model$upload_data(mtcars, mode = "partial")   # <-- built-in data frame

# Borrow a starting point instead of guessing: when the outcome (mpg) is in the
# upload, get_effects_from_data fits the model and returns a ready-to-paste
# set_effects(...) string on the standardized scale. A first guess, not a target
# — never auto-applied; you read it, decide, and set it yourself.
cat(sprintf(">>> model$get_effects_from_data('mpg') -> %s\n", model$get_effects_from_data("mpg")))

model$set_effects("hp=-0.3, wt=-0.5, am=0.4")   # hp/wt decrease mpg, am increases it

# Power analysis with realistic data.
cat("\n>>> model$find_power(sample_size=100, target_test='hp')\n")
model$find_power(sample_size = 100, target_test = "hp")

cat("\n>>> model$find_power(sample_size=100, target_test='all', scenarios=TRUE)\n")
model$find_power(sample_size = 100, target_test = "all", scenarios = TRUE)

# Sample size calculation with pilot data.
cat("\n>>> model$find_sample_size(target_test='wt', from_size=50, to_size=150, scenarios=TRUE)\n")
model$find_sample_size(target_test = "wt", from_size = 50, to_size = 150, scenarios = TRUE)

# ---------------------------------------------------------------------------
# 2. From an in-memory object — no file needed.
# ---------------------------------------------------------------------------
# upload_data also takes a data.frame (or a matrix / named list) directly, so
# data you build or generate in code can be uploaded without ever writing a CSV.
# Here we synthesise a small pilot-like dataset and pass the data.frame in.
set.seed(2137)
generated <- data.frame(
  hp = rnorm(200, 150, 50),
  wt = rnorm(200, 3.2, 0.9),
  am = sample(0:1, 200, replace = TRUE)   # binary transmission
)

gen_model <- MCPower$new("mpg ~ hp + wt + am")
cat("\n>>> gen_model$upload_data(generated, mode='partial')  # data.frame, no CSV\n")
gen_model$upload_data(generated, mode = "partial")   # <-- in-memory object, no file
gen_model$set_effects("hp=-0.3, wt=-0.35, am=0.7")

cat("\n>>> gen_model$find_power(sample_size=60, target_test='all', scenarios=TRUE)\n")
gen_model$find_power(sample_size = 60, target_test = "all", scenarios = TRUE)

# ---------------------------------------------------------------------------
# 3. Mixed approach — some predictors from data, others synthetic.
# ---------------------------------------------------------------------------
# Upload only the columns you actually measured; any predictor in the formula
# that is NOT in the upload is generated synthetically. Here hp/wt come from the
# mtcars data and am/cyl are synthetic — we just pick the two columns we want.
mixed_model <- MCPower$new("mpg ~ hp + wt + am + cyl")
mixed_model$upload_data(mtcars[, c("hp", "wt")], mode = "partial")
mixed_model$set_effects("hp=-0.3, wt=-0.5, am=0.4, cyl=-0.2")
mixed_model$set_variable_type("am=binary")   # synthetic binary transmission

cat("\n>>> mixed_model$find_power(sample_size=250, target_test='all', scenarios=TRUE)\n")
mixed_model$find_power(sample_size = 250, target_test = "all", scenarios = TRUE)

# ---------------------------------------------------------------------------
# 4. mode="strict" — whole-row bootstrap.
# ---------------------------------------------------------------------------
#    Strict mode bootstraps complete rows from the pilot data, so the full
#    empirical joint distribution (all predictor-response dependencies) is
#    preserved in every simulated dataset — not just the marginals and a
#    correlation matrix. The cost is that each simulated dataset of size N is
#    drawn with replacement from U uploaded rows: when N approaches or exceeds U,
#    observations repeat. The "~g% of rows reused" line shows the expected
#    fraction of rows that appear more than once in a single bootstrap draw.
#    When N > 2*U the engine emits a warning suggesting mode='partial' or
#    mode='none', which are faster and more generalizable for large N.
#
#    mtcars has U=32 rows, so 2*U = 64.  The find_sample_size search below
#    spans N=30..120, which crosses that boundary and makes the warning visible:
#    at N=50 reuse is moderate (~47%) and below the threshold; at the achieved
#    N for 80% power the reuse climbs and triggers the advisory.

cat("\n# --- mode='strict' demonstration ---\n")
strict_model <- MCPower$new("mpg ~ hp + wt + am")
# Upload all 32 rows in strict mode (U=32) straight from the built-in dataset.
strict_model$upload_data(mtcars, mode = "strict")
# wt=-0.7 places the target N for 80% power just above 2*U = 64, making the
# crossover explicit: find_power at N=50 (below threshold) shows the reuse line
# only, while find_sample_size reaches N~70 and triggers the N>2*U warning.
strict_model$set_effects("hp=-0.2, wt=-0.7, am=0.35")

# 4a. find_power at N=50 — below 2*U=64, reuse line shows but no warning.
cat("\n>>> strict_model$find_power(sample_size=50, target_test='wt')\n")
cat("# [strict bootstrap] N=50, U=32: ~47% reuse — below the 2*U threshold, no warning.\n")
strict_model$find_power(sample_size = 50, target_test = "wt")

# 4b. find_sample_size — search [30, 120] spans 2*U=64; the achieved N lands
#     above it and triggers the N>2*U advisory.
cat("\n>>> strict_model$find_sample_size(target_test='wt', from_size=30, to_size=120)\n")
cat("# Expect: reuse line per achieved-N, and a warning when achieved N > 64.\n")
withCallingHandlers(
  strict_model$find_sample_size(target_test = "wt", from_size = 30, to_size = 120),
  warning = function(w) {
    cat(sprintf("  Warning: %s\n", conditionMessage(w)))
    invokeRestart("muffleWarning")
  }
)
