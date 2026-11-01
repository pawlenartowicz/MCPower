# Denominator df for mixed-model fixed-effect tests
# ==================================================
#
# df_method selects the test for mixed-model fixed effects. "auto" (the
# default) picks one per design: between-within denominator degrees of
# freedom for a single grouping factor without a random slope, Satterthwaite
# df (computed for each fit, as lmerTest does) for a linear mixed model with
# a random slope or with crossed or nested groupings, and Wald z for a binary
# or count model with crossed or nested groupings. "between_within",
# "satterthwaite" and "asymptotic" name one test each. A t test is the
# conservative choice when the number of clusters is small, since a t
# distribution with few df has heavier tails than the standard normal.
#
# Mirrors ports/py/examples/17_df_method.py.
#
# Run with:  Rscript 17_df_method.R

library(mcpower)

# Example 1: a few-cluster clinical trial -- only 6 clinics, each contributing
# many patients. A site-level treatment (assigned once per clinic) has just
# 6 information units, so the asymptotic Wald z overstates the precision of
# its test; between-within df (what "auto" runs here) corrects for this.

model <- MCPower$new("outcome ~ treat + baseline_score + (1|clinic)", family = "lme")
model$set_variable_type("treat=binary")
model$set_effects("treat=0.40, baseline_score=0.25")
model$set_cluster("clinic", ICC = 0.20, n_clusters = 6,
                  cluster_level_vars = "treat")

cat(">>> df_method='asymptotic' (Wald z)\n")
r_asym <- model$find_power(sample_size = 120, target_test = "treat",
                           df_method = "asymptotic", n_sims = 1000, seed = 2137)
cat(sprintf("power = %s\n", r_asym$power_uncorrected[[1]]))

cat("\n>>> df_method='auto' (default; here between-within df, the between-cluster df for the cluster-level predictor)\n")
r_auto <- model$find_power(sample_size = 120, target_test = "treat",
                           n_sims = 1000, seed = 2137)
cat(sprintf("power = %s\n", r_auto$power_uncorrected[[1]]))

# Example 2: a predictor with a random slope. Its effect varies across the 10
# schools, so its test rests on the number of schools, not on the 200 pupils.
# "auto" tests it at Satterthwaite df; between-within df would use the
# within-school df and test it almost like a Wald z.

slope_model <- MCPower$new("score ~ hours + (1 + hours|school)", family = "lme")
slope_model$set_effects("hours=0.30")
slope_model$set_cluster("school", ICC = 0.20, n_clusters = 10,
                        random_slopes = list(list(predictor = "hours", variance = 0.10)))

cat("\n>>> random slope, df_method='auto' (Satterthwaite df)\n")
r_satt <- slope_model$find_power(sample_size = 200, n_sims = 1000, seed = 2137)
cat(sprintf("power = %s\n", r_satt$power_uncorrected[[1]]))

cat("\n>>> random slope, df_method='between_within'\n")
r_bw <- slope_model$find_power(sample_size = 200, df_method = "between_within",
                               n_sims = 1000, seed = 2137)
cat(sprintf("power = %s\n", r_bw$power_uncorrected[[1]]))
