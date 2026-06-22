suppressMessages(library(mcpower))

# Three continuous predictors, additive (no interactions): each contributes
# independently to the outcome on the standardised effect scale.
model <- MCPower$new("cholesterol ~ age + bmi + exercise_hours")

# Standardised slopes: age medium (0.25), bmi small-to-medium (0.18), exercise_hours small (0.10).
model$set_effects("age=0.25, bmi=0.18, exercise_hours=0.10")

model$set_seed(2137)
model$set_simulations(1600)

invisible(model$find_power(sample_size = 200, target_test = "age, bmi, exercise_hours"))
