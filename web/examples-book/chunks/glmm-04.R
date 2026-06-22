suppressMessages(library(mcpower))

# Species presence (yes/no) recorded repeatedly across temperature gradients at
# the same sites, where each site responds to temperature at its own rate.
# family = "logit" makes species_present binary (fit by GLM); (1 + temperature|site)
# adds a random intercept AND a random slope of temperature per site, so the
# average temperature effect is tested with the extra site-to-site slope spread
# folded into its standard error.
model <- MCPower$new("species_present ~ temperature + (1 + temperature|site)", family = "logit")

# Binary outcome needs its no-predictor base rate: 30% of surveys detect the
# species when temperature is at its average.
model$set_baseline_probability(0.3)

# Expected effect on the standardised benchmark scale (continuous predictor):
#   temperature=0.25 -> a medium association with the log-odds of species presence.
model$set_effects("temperature=0.25")

# Clustering: ICC=0.10 between sites across 40 sites. random_slopes is a list
# of one spec per random slope; each names the predictor whose slope varies
# (variance = the spread of those per-site temperature slopes, corr_with_intercept =
# their correlation with the random intercept). At N=400 that is 10 observations
# per site.
model$set_cluster("site", ICC = 0.10, n_clusters = 40,
                  random_slopes = list(
                    list(predictor = "temperature", variance = 0.15,
                         corr_with_intercept = 0.0)
                  ))

# Power at N=400 for the average temperature effect (GLM defaults: 1600 sims,
# alpha=0.05, seed=2137). The omnibus test is not reported for mixed models;
# target the coefficient directly.
invisible(model$find_power(sample_size = 400, target_test = "temperature"))
