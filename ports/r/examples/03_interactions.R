# Interactions Analysis Example
# =============================
#
# Interaction effects — when the effect of one variable depends on the level of
# another. advertising * age expands to advertising + age + advertising:age;
# use `:` for interaction-only.
#
# Effect-size caveat: an interaction column is the raw product of its component
# columns, so its scale depends on the components. Two uncorrelated standard-
# normal predictors give a unit-variance product (benchmarks apply), but
# binary/factor components or correlated predictors do not — treat the
# interaction effect size as approximate in those cases.
#
# Run with:  Rscript 03_interactions.R

suppressMessages(library(mcpower))

# 1. The `*` operator — factorial shorthand. It expands to advertising + age +
#    advertising:age, so all three terms take an effect below.
model <- MCPower$new("sales ~ advertising * age")
model$set_effects("advertising=0.4, age=0.2, advertising:age=0.3")
model$set_variable_type("advertising=binary")

# 2. Power for the interaction term — short form. Target the `a:b` term directly.
#    At N=200 the interaction sits well below 80%: interactions are expensive.
cat(">>> model$find_power(sample_size = 200, target_test = 'advertising:age')\n")
invisible(model$find_power(sample_size = 200, target_test = "advertising:age"))

# 3. The `:` operator — write the interaction explicitly. This formula is
#    equivalent to `advertising * age` above (use `:` when you want to spell out
#    exactly which terms are in the model), so the interaction power matches.
#    Long form via summary() for the full picture at a larger N.
explicit <- MCPower$new("sales ~ advertising + age + advertising:age")
explicit$set_effects("advertising=0.4, age=0.2, advertising:age=0.3")
explicit$set_variable_type("advertising=binary")
cat("\n>>> result <- explicit$find_power(sample_size = 300, target_test = 'all', verbose = FALSE)\n")
cat(">>> print(summary(result))\n")
result <- explicit$find_power(sample_size = 300, target_test = "all", verbose = FALSE)
print(summary(result))

# 4. Robustness — every term under each scenario. The interaction degrades
#    fastest once assumptions are violated.
cat("\n>>> model$find_power(sample_size = 250, target_test = 'all', scenarios = TRUE)\n")
invisible(model$find_power(sample_size = 250, target_test = "all", scenarios = TRUE))
