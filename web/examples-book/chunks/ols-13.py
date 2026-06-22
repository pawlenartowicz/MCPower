from mcpower import MCPower

# Covariate-adjusted association: the effect of `years_education` on `hourly_wage`,
# holding age, experience_years, and tenure constant. All four predictors are continuous.
model = MCPower("hourly_wage = years_education + age + experience_years + tenure")

# Standardised effect sizes (continuous benchmark scale).
#   years_education=0.25  -> the key predictor, a medium adjusted association.
#   age/experience_years/tenure -> nuisance covariates carried along for adjustment.
model.set_effects("years_education=0.25, age=0.20, experience_years=0.30, tenure=0.25")

# Covariates correlate with years_education — that confounding is the reason we
# adjust, and it changes the power for the years_education coefficient.
model.set_correlations(
    "corr(years_education, age)=0.3, "
    "corr(years_education, experience_years)=0.4, "
    "corr(experience_years, tenure)=0.5"
)

model.find_power(sample_size=200, target_test="years_education")
