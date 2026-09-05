# Denominator df for mixed-model fixed-effect tests
# ==================================================
#
# df_method = "between_within" (the default) tests a mixed-model fixed effect
# against a t with between-within denominator degrees of freedom;
# df_method = "asymptotic" opts back into the Wald z. The t is the
# conservative choice when the number of clusters is small, since a t
# distribution with few df has heavier tails than the standard normal.
#
# Example: a few-cluster clinical trial -- only 6 clinics, each contributing
# many patients. A site-level treatment (assigned once per clinic) has just
# 6 information units, so the asymptotic Wald z overstates the precision of
# its test; between-within df corrects for this.
#
# Run with:  Rscript 17_df_method.R

library(mcpower)

model <- MCPower$new("outcome ~ treat + baseline_score + (1|clinic)", family = "lme")
model$set_variable_type("treat=binary")
model$set_effects("treat=0.40, baseline_score=0.25")
model$set_cluster("clinic", ICC = 0.20, n_clusters = 6,
                  cluster_level_vars = "treat")

cat(">>> df_method='asymptotic' (opt-in, Wald z)\n")
r_asym <- model$find_power(sample_size = 120, target_test = "treat",
                           df_method = "asymptotic", n_sims = 1000, seed = 2137)
power_asym <- r_asym$power_uncorrected[[1]]
cat(sprintf("power = %s\n", power_asym))

cat("\n>>> df_method='between_within' (default, t-test with the between-cluster df for the cluster-level predictor)\n")
r_bw <- model$find_power(sample_size = 120, target_test = "treat",
                         df_method = "between_within", n_sims = 1000, seed = 2137)
power_bw <- r_bw$power_uncorrected[[1]]
cat(sprintf("power = %s\n", power_bw))
