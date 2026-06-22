suppressMessages(library(mcpower))

# Three-habitat study with a continuous moderator.
# Research question: does the effect of rainfall on biomass differ across the
# three habitat types -- i.e. does habitat moderate the rainfall slope?
# '*' expands habitat * rainfall to habitat + rainfall + habitat:rainfall.
model <- MCPower$new("biomass ~ habitat * rainfall")

# habitat is a categorical predictor with 3 levels -> 2 dummy contrasts;
# rainfall is continuous (left at its default standardised distribution).
model$set_variable_type("habitat=(factor,3)")

# A 3-level factor expands to per-level dummies, so its effects and the
# interaction are named per level: habitat[2], habitat[3], and
# habitat[2]:rainfall, habitat[3]:rainfall (the bare names habitat / habitat:rainfall
# do not exist after expansion).
# Standardised effects:
#   habitat[2]=habitat[3]=0.5 -> factor benchmark (0.20/0.50/0.80), each
#                                 non-reference level shifts the outcome by
#                                 a medium amount.
#   rainfall=0.25              -> continuous benchmark (0.10/0.25/0.40), a
#                                 medium slope.
#   habitat[*]:rainfall=0.4   -> the moderation: how much each habitat's
#                                 rainfall slope departs from the reference slope.
model$set_effects(paste0(
  "habitat[2]=0.5, habitat[3]=0.5, rainfall=0.25, ",
  "habitat[2]:rainfall=0.4, habitat[3]:rainfall=0.4"
))

# Power at N=200 for the interaction dummies -- the moderation is the focal test.
invisible(model$find_power(
  sample_size = 200,
  target_test = "habitat[2]:rainfall, habitat[3]:rainfall"
))
