from mcpower import MCPower

# Three-habitat ecology study with a yes/no outcome: did the seedling survive?
# Research question: do the habitat types differ in seedling survival probability,
# tested as the two dummy contrasts against the reference habitat.
# family="logit" makes survived a binary (0/1) outcome fitted by a logistic GLM.
model = MCPower("survived = habitat", family="logit")

# habitat is a categorical predictor with 3 levels -> 2 dummy contrasts.
# A 3-level factor expands into per-level dummies habitat[2] and
# habitat[3] (level 1 is the reference); effects and tests address those
# dummy names directly, not the bare factor name.
model.set_variable_type("habitat=(factor,3)")

# Standardised effects on the factor benchmark scale (0.20 / 0.50 / 0.80):
# each non-reference habitat shifts the log-odds of survival by a medium amount
# relative to the reference habitat.
model.set_effects("habitat[2]=0.5, habitat[3]=0.5")

# family="logit" requires a baseline event rate: the survival probability in the
# reference habitat when all predictors are at their reference. It sets the model
# intercept (log-odds) for every Monte Carlo iteration; without it find_power
# raises. 0.30 = a 30% reference-habitat survival rate.
model.set_baseline_probability(0.30)

# Power at N=300 for both dummy contrasts (each habitat vs the reference habitat).
model.find_power(sample_size=300, target_test="habitat[2], habitat[3]")
