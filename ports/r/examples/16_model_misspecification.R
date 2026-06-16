# Model Misspecification
# ======================
#
# The model you TEST, not just the data you generate, decides what you can
# detect. One data-generating truth — studying raises the exam score, caffeine
# only rides along via their correlation (its real effect is 0) — fit three ways
# via test_formula: correct, confounded (omit a real cause), and over-specified
# (add a null covariate).
#
# Run with:  Rscript 16_model_misspecification.R

suppressMessages(library(mcpower))

# Data-generating truth. caffeine sits in the formula at effect 0 so it can shape
# the data (and the correlation) while staying out of the genuinely correct
# model, which is score ~ study.
model <- MCPower$new("score ~ study + caffeine")
model$set_effects("study=0.3, caffeine=0")
model$set_correlations("corr(study, caffeine)=0.6")

# test_formula fits a different model than the one that generated the data; every
# test term must already exist in the generation formula. find_power does not
# mutate the model, so one model is fit three ways.

# 1. Correct model — matches the truth. study is well-powered (~85%).
cat(">>> correct: test_formula = 'score ~ study'\n")
print(summary(model$find_power(sample_size = 100, target_test = "study",
                               test_formula = "score ~ study", verbose = FALSE)))

# 2. Mis-specified — omit the real cause, keep its correlated proxy. Direction A:
#    dropping a correlated true cause manufactures a spurious, significant effect
#    on the innocent proxy (omitted-variable confounding).
cat("\n>>> confounded: test_formula = 'score ~ caffeine'\n")
print(summary(model$find_power(sample_size = 100, target_test = "caffeine",
                               test_formula = "score ~ caffeine", verbose = FALSE)))

# 3. Over-specified — keep the real cause but pad the model with the null
#    covariate. Direction B: a correlated null predictor steals unique variance
#    from study, so study's power drops below the correct-model level while
#    caffeine sits at ~alpha.
cat("\n>>> over-specified: test_formula = 'score ~ study + caffeine'\n")
print(summary(model$find_power(sample_size = 100, target_test = "study, caffeine",
                               test_formula = "score ~ study + caffeine", verbose = FALSE)))
