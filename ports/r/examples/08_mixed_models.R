# Mixed-Effects (LME) Example
# ===========================
#
# Power for clustered data with family = "lme". A (1|group) term plus
# set_cluster(ICC, n_clusters) builds a random-intercept design that accounts
# for within-cluster correlation, then searches for the required sample size.
#
# Run with:  Rscript 08_mixed_models.R

library(mcpower)

# Example: Education study where students are nested in classrooms.
# Research question: Does a teaching method raise test scores, accounting for
# the fact that students in the same classroom are correlated?

# 1. Declare a mixed model. The (1|classroom) term adds a random intercept per
#    classroom; family = "lme" fits it by maximum likelihood (MLE estimator).
model <- MCPower$new("score ~ teaching_method + prior_gpa + (1|classroom)",
                     family = "lme")
model$set_variable_type("teaching_method=binary")
model$set_effects("teaching_method=0.4, prior_gpa=0.18")

# 2. Describe the clustering: ICC=0.15 (15% of variance is between-classroom)
#    across 30 classrooms. At N=300 that is 10 students per classroom.
model$set_cluster("classroom", ICC = 0.15, n_clusters = 30)

# 3. Short form — target the fixed effects directly. (The omnibus test is not
#    reported for mixed models; ask for the coefficients you care about.)
cat(">>> model$find_power(sample_size = 300, target_test = 'teaching_method, prior_gpa')\n")
invisible(model$find_power(sample_size = 300, target_test = "teaching_method, prior_gpa"))

# 4. Long form via summary() — adds the joint-significance distribution and the
#    LME diagnostics (convergence rate, boundary-hit rate, tau-hat).
cat("\n>>> result <- model$find_power(sample_size = 300, target_test = 'teaching_method, prior_gpa', verbose = FALSE)\n")
cat(">>> print(summary(result))\n")
result <- model$find_power(
  sample_size = 300, target_test = "teaching_method, prior_gpa", verbose = FALSE
)
print(summary(result))

# 5. Sample-size search for the clustered design. With a fixed number of
#    classrooms (n_clusters=30) the search adds students per classroom; the
#    engine snaps each grid point to a whole number of observations per cluster,
#    so every reported N is a balanced design.
cat("\n>>> model$find_sample_size(target_test = 'teaching_method, prior_gpa', from_size = 120, to_size = 420, by = 30)\n")
invisible(model$find_sample_size(
  target_test = "teaching_method, prior_gpa", from_size = 120, to_size = 420, by = 30
))
