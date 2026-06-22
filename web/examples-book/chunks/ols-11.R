suppressMessages(library(mcpower))

# Does the slope of `experience_years` on the outcome differ between the two
# `gender` groups? `gender * experience_years` expands to gender +
# experience_years + gender:experience_years, so the interaction term carries
# the moderation. gender is binary; experience_years is continuous.
model <- MCPower$new("wage ~ gender * experience_years")
model$set_effects("gender=0.5, experience_years=0.25, gender:experience_years=0.2")
model$set_variable_type("gender=binary")

# Power for the interaction term — moderation is the question, so target it.
invisible(model$find_power(sample_size = 300, target_test = "gender:experience_years"))
