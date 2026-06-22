suppressMessages(library(mcpower))

# Two-way factorial ANOVA: hourly wage crossed by employment sector and gender.
# Research question -- does the sector wage gap depend on gender? -> the interaction.
# '*' expands sector * gender to sector + gender + sector:gender, so
# both main effects and the interaction are fitted explicitly.
model <- MCPower$new("hourly_wage ~ sector * gender")

# sector has 3 levels -> 2 dummy contrasts; gender has 2 levels -> 1 dummy.
# The interaction therefore contributes 2 cell-difference contrasts.
model$set_variable_type("sector=(factor,3), gender=(factor,2)")

# Effects are assigned per dummy contrast, not per base factor name: with no
# uploaded data the levels are integer-labelled, level 1 is the reference, and
# the expansion produces exactly these five terms. All on the factor benchmark
# scale (0.20 / 0.50 / 0.80) at medium = 0.50:
#   sector[2], sector[3]                -> sector's two main contrasts.
#   gender[2]                           -> gender's main contrast.
#   sector[2]:gender[2], sector[3]:gender[2] -> the two interaction cells.
model$set_effects(
  paste0(
    "sector[2]=0.50, sector[3]=0.50, gender[2]=0.50, ",
    "sector[2]:gender[2]=0.50, sector[3]:gender[2]=0.50"
  )
)

model$set_seed(2137)

# Power for one interaction cell -- the factorial design's focal test -- at
# N=240. (target_test names a single expanded effect; there is no bare
# 'sector:gender' term after dummy expansion.)
invisible(model$find_power(sample_size = 240, target_test = "sector[2]:gender[2]"))
