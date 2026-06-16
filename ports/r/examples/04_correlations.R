# Correlated Predictors Example
# =============================
#
# Real-world predictors are usually correlated, which changes statistical
# power. set_correlations accepts either a string spec or a full correlation
# matrix; this example uses both, and the result object to compare the
# independent vs correlated cases.
#
# Run with:  Rscript 04_correlations.R

suppressMessages(library(mcpower))

# 1. Model with several predictors.
model <- MCPower$new("life_satisfaction ~ income + education + social_support + health")
model$set_effects("income=0.25, education=0.2, social_support=0.3, health=0.35")

# 2. String-based correlations — good for a sparse set of pairs.
model$set_correlations(paste0(
  "corr(income, education)=0.6, ",
  "corr(income, health)=0.4, ",
  "corr(education, health)=0.3, ",
  "corr(social_support, health)=0.5"
))

# 3. Power across all predictors — the joint distribution in summary() shows
#    how correlation erodes the chance of detecting every effect together.
cat(">>> result <- model$find_power(sample_size = 150, target_test = 'all', verbose = FALSE)\n")
cat(">>> print(summary(result))\n")
result <- model$find_power(sample_size = 150, target_test = "all", verbose = FALSE)
print(summary(result))

# 4. Matrix-based correlations — good for a full correlation structure.
#    Rows/columns follow the non-factor predictors in formula order.
matrix_model <- MCPower$new("wellbeing ~ stress + exercise + sleep")
matrix_model$set_effects("stress=-0.3, exercise=0.25, sleep=0.3")
correlation_matrix <- matrix(c(
   1.0, -0.3, 0.4,   # stress with exercise, sleep
  -0.3,  1.0, 0.2,   # exercise with stress, sleep
   0.4,  0.2, 1.0    # sleep with stress, exercise
), nrow = 3, byrow = TRUE)
matrix_model$set_correlations(correlation_matrix)
cat("\n>>> matrix_model$find_power(sample_size = 150, target_test = 'all')\n")
invisible(matrix_model$find_power(sample_size = 150, target_test = "all"))

# 5. Compare independent vs highly-correlated predictors.
uncorr <- MCPower$new("outcome ~ x1 + x2 + x3")
uncorr$set_effects("x1=0.3, x2=0.25, x3=0.35")
uncorr_res <- uncorr$find_power(sample_size = 130, target_test = "all", verbose = FALSE)

corr <- MCPower$new("outcome ~ x1 + x2 + x3")
corr$set_effects("x1=0.3, x2=0.25, x3=0.35")
corr$set_correlations("corr(x1,x2)=0.7, corr(x1,x3)=0.6, corr(x2,x3)=0.8")
corr_res <- corr$find_power(sample_size = 130, target_test = "all", verbose = FALSE)

if (requireNamespace("tibble", quietly = TRUE)) {
  a <- tibble::as_tibble(uncorr_res)[, c("test", "power")]
  names(a)[2] <- "power_independent"
  b <- tibble::as_tibble(corr_res)[, c("test", "power")]
  names(b)[2] <- "power_correlated"
  cat("\n>>> merge(a, b, by = 'test')\n")
  print(merge(a, b, by = "test"))
}
