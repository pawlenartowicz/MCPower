# Shared test helper: builds a LinearSpec JSON for a simple OLS model
# y ~ x1 (x1 is Normal), with a configurable x1 effect size.
# Matches the new wire contract:
#   - heterogeneity removed (scenario-only)
#   - heteroskedasticity: driver_var_index only (no ratio)
#   - residual: distribution only (no df)
#   - residual_dists: [4,2] (high_kurtosis=4, right_skewed=2)
# Used by test-find-power.R and downstream tasks.
.ols_spec_json <- function(formula, x1 = 0.5) {
  paste0(
    '{"formula":"', formula, '",',
    '"predictors":[{"name":"x1","kind":"normal"}],',
    '"effects":[{"name":"x1","size":', x1, '}],',
    '"correlations":[],',
    '"alpha":0.05,',
    '"correction":"none",',
    '"targets":["overall"],',
    '"report_overall":true,',
    '"contrast_pairs":[],',
    '"heteroskedasticity":{"driver_var_index":null},',
    '"residual":{"distribution":"normal"},',
    '"max_failed_fraction":1.0,',
    '"scenarios":[{"name":"optimistic","heterogeneity":0.0,"heteroskedasticity_ratio":1.0,',
    '"correlation_noise_sd":0.0,"distribution_change_prob":0.0,',
    '"new_distributions":[2,3,5],"residual_change_prob":0.0,',
    '"residual_dists":[4,2],"residual_df":10.0}]}'
  )
}
