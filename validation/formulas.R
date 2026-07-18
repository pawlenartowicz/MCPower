# formulas.R — concrete L3 validation cases (the full DGP per case).
#
# Hand-copied from the losf catalogue (never auto-generated).
# Each case is a fully-specified data-generating process: the formula, the
# numeric effects, the predictor distribution / correlations, factor levels,
# and (LME) the cluster structure. Per losf form we instantiate 2 numeric
# parameterizations (suffix _a/_b) so validation isn't overfit to one set of
# numbers.
#
# Coverage: losf 2-16. losf 1 (`y ~ 1`, intercept-only) is OMITTED -- the engine
# rejects predictor-free formulas ("expected identifier, got '1'").
#
# What MCPower actually generates (so the readable comments are honest):
#   - continuous predictors are sampled ~N(0, 1) (population mean 0, sd 1), not
#     re-standardized per draw, so a single draw scatters by ~1/sqrt(n);
#   - the OLS intercept is 0 (there is no set_intercept); the GLM "intercept"
#     is logit(baseline_probability);
#   - the default residual is N(0, 1);
#   - a 2-level factor is materialised as one binary column; a 3+-level factor as
#     k-1 dummy columns.
# Effect-size benchmarks: continuous 0.10/0.25/0.40 small/med/large;
# binary or factor 0.20/0.50/0.80.
#
# Field reference (maps to the MCPower R6 setters in common.R:build_model):
#   formula              R-style string ("*" expands a+b+a:b; ":" is interaction-only)
#   family               "ols" | "logit" | "lme"
#   effects              set_effects() string. MUST be listed in design-column
#                        order: main effects first (continuous, then factor
#                        dummies), then interactions -- true_beta_vector maps by
#                        position. Interaction terms named "x1:x2" / "x1:g[2]";
#                        3+-level factor dummies "g[2]", "g[3]"; a 2-level factor's
#                        single dummy is named by its bare variable name.
#   correlations         set_correlations() string, e.g. "corr(x1,x2)=0.3"   (optional)
#   variable_types       set_variable_type() string, or a character VECTOR with one
#                        entry per factor (a single string can't separate two)    (optional)
#   baseline_probability set_baseline_probability() numeric (logit only)      (optional)
#   residual             list(name=) for set_residual_distribution()            (optional; default N(0,1))
#   cluster              list(var=, ICC=, n_clusters=, cluster_size=)          (lme only)
#   n, seed              debug draw size and seed (n == n_clusters*cluster_size for lme)

# ---- OLS (continuous outcome) ------------------------------------------------

# losf 2 — one continuous predictor
ols_simple_a <- list(
  label = "ols_simple_a", losf = 2L,
  # y = 0.25*x1 + e
  formula = "y ~ x1", family = "ols",
  effects = "x1=0.25", n = 400L, seed = 2137
)
ols_simple_b <- list(
  label = "ols_simple_b", losf = 2L,
  # y = 0.40*x1 + e   (larger slope, different seed)
  formula = "y ~ x1", family = "ols",
  effects = "x1=0.40", n = 400L, seed = 2138
)

# losf 3 — second independent predictor
ols_two_a <- list(
  label = "ols_two_a", losf = 3L,
  # y = 0.25*x1 + 0.10*x2 + e ;  x1,x2 independent
  formula = "y ~ x1 + x2", family = "ols",
  effects = "x1=0.25, x2=0.10", n = 400L, seed = 2137
)
ols_two_b <- list(
  label = "ols_two_b", losf = 3L,
  # y = 0.40*x1 + 0.25*x2 + e
  formula = "y ~ x1 + x2", family = "ols",
  effects = "x1=0.40, x2=0.25", n = 400L, seed = 2138
)

# losf 3 — true-zero coefficient (null recovery centred on zero).
ols_zero_a <- list(
  label = "ols_zero_a", losf = 3L,
  # y = 0.25*x1 + 0*x2 + e — x2 is a true null; its recovered coefficient should be ~0.
  formula = "y ~ x1 + x2", family = "ols",
  effects = "x1=0.25, x2=0.0", n = 400L, seed = 2137
)

# losf 4 — predictor correlation (isolates correlation -> variance inflation)
ols_corr_a <- list(
  label = "ols_corr_a", losf = 4L,
  # same as ols_two_a but x1,x2 correlated 0.5
  formula = "y ~ x1 + x2", family = "ols",
  effects = "x1=0.25, x2=0.10", correlations = "corr(x1,x2)=0.5",
  n = 400L, seed = 2137
)
ols_corr_b <- list(
  label = "ols_corr_b", losf = 4L,
  # x1,x2 correlated 0.3
  formula = "y ~ x1 + x2", family = "ols",
  effects = "x1=0.40, x2=0.25", correlations = "corr(x1,x2)=0.3",
  n = 400L, seed = 2138
)

# losf 5 — two-way interaction (continuous x continuous)
ols_interaction_a <- list(
  label = "ols_interaction_a", losf = 5L,
  # y = 0.25*x1 + 0.10*x2 - 0.20*x1:x2 + e ;  x1,x2 corr 0.3
  formula = "y ~ x1*x2", family = "ols",
  effects = "x1=0.25, x2=0.10, x1:x2=-0.20", correlations = "corr(x1,x2)=0.3",
  n = 600L, seed = 2137
)
ols_interaction_b <- list(
  label = "ols_interaction_b", losf = 5L,
  # y = 0.40*x1 + 0.25*x2 + 0.15*x1:x2 + e   (independent, positive interaction)
  formula = "y ~ x1*x2", family = "ols",
  effects = "x1=0.40, x2=0.25, x1:x2=0.15", n = 600L, seed = 2138
)

# losf 6 — one categorical factor
ols_factor_a <- list(
  label = "ols_factor_a", losf = 6L,
  # y = 0.25*x1 + 0.50*g[2] + 0.80*g[3] + e ;  g 3-level (50/30/20%)
  formula = "y ~ x1 + g", family = "ols",
  variable_types = "g=(factor, 0.5, 0.3, 0.2)",
  effects = "x1=0.25, g[2]=0.50, g[3]=0.80", n = 600L, seed = 2137
)
ols_factor_b <- list(
  label = "ols_factor_b", losf = 6L,
  # y = 0.40*x1 + 0.20*g[2] + 0.50*g[3] + e ;  g 3-level (40/35/25%)
  formula = "y ~ x1 + g", family = "ols",
  variable_types = "g=(factor, 0.4, 0.35, 0.25)",
  effects = "x1=0.40, g[2]=0.20, g[3]=0.50", n = 600L, seed = 2138
)

# losf 7 — continuous x factor interaction
ols_cf_a <- list(
  label = "ols_cf_a", losf = 7L,
  # y = 0.30*x1 + 0.40*g[2] + 0.60*g[3] + 0.20*x1:g[2] + 0.30*x1:g[3] + e
  formula = "y ~ x1*g", family = "ols",
  variable_types = "g=(factor, 0.5, 0.3, 0.2)",
  effects = "x1=0.30, g[2]=0.40, g[3]=0.60, x1:g[2]=0.20, x1:g[3]=0.30",
  n = 800L, seed = 2137
)
ols_cf_b <- list(
  label = "ols_cf_b", losf = 7L,
  # y = 0.40*x1 + 0.50*g[2] + 0.80*g[3] + 0.25*x1:g[2] + 0.40*x1:g[3] + e
  formula = "y ~ x1*g", family = "ols",
  variable_types = "g=(factor, 0.4, 0.35, 0.25)",
  effects = "x1=0.40, g[2]=0.50, g[3]=0.80, x1:g[2]=0.25, x1:g[3]=0.40",
  n = 800L, seed = 2138
)

# losf 8 — factor x factor (2x2 cell-means / ANOVA). Both factors 2-level: g1
# contributes g1[2], g2 contributes g2[2]; interaction g1[2]:g2[2]. (Declared
# via separate set_variable_type entries, one per factor; the setters accumulate
# so both survive — see build_model in common.R.)
ols_ff_a <- list(
  label = "ols_ff_a", losf = 8L,
  # y = 0.50*g1[2] + 0.40*g2[2] + 0.30*g1[2]:g2[2] + e
  formula = "y ~ g1*g2", family = "ols",
  variable_types = c("g1=(factor, 0.5, 0.5)", "g2=(factor, 0.6, 0.4)"),
  effects = "g1[2]=0.50, g2[2]=0.40, g1[2]:g2[2]=0.30", n = 800L, seed = 2137
)
ols_ff_b <- list(
  label = "ols_ff_b", losf = 8L,
  # y = 0.20*g1[2] + 0.80*g2[2] + 0.50*g1[2]:g2[2] + e
  formula = "y ~ g1*g2", family = "ols",
  variable_types = c("g1=(factor, 0.5, 0.5)", "g2=(factor, 0.55, 0.45)"),
  effects = "g1[2]=0.20, g2[2]=0.80, g1[2]:g2[2]=0.50", n = 800L, seed = 2138
)

# ---- GLM — logistic (binary outcome) -----------------------------------------

# losf 9 — one continuous predictor on the logit scale
glm_simple_a <- list(
  label = "glm_simple_a", losf = 9L,
  # logit(P) = logit(0.3) + 0.5*x1
  formula = "y ~ x1", family = "logit",
  effects = "x1=0.5", baseline_probability = 0.3, n = 600L, seed = 2137
)
glm_simple_b <- list(
  label = "glm_simple_b", losf = 9L,
  # logit(P) = logit(0.5) + 0.8*x1
  formula = "y ~ x1", family = "logit",
  effects = "x1=0.8", baseline_probability = 0.5, n = 600L, seed = 2138
)

# losf 10 — multiple predictors on the logit scale
glm_two_a <- list(
  label = "glm_two_a", losf = 10L,
  # logit(P) = logit(0.3) + 0.5*x1 + 0.3*x2 ;  x1,x2 corr 0.2
  formula = "y ~ x1 + x2", family = "logit",
  effects = "x1=0.5, x2=0.3", baseline_probability = 0.3,
  correlations = "corr(x1,x2)=0.2", n = 800L, seed = 2137
)
glm_two_b <- list(
  label = "glm_two_b", losf = 10L,
  # logit(P) = logit(0.5) + 0.8*x1 + 0.5*x2 ;  independent
  formula = "y ~ x1 + x2", family = "logit",
  effects = "x1=0.8, x2=0.5", baseline_probability = 0.5, n = 800L, seed = 2138
)

# losf 11 — a factor on the logit scale
glm_factor_a <- list(
  label = "glm_factor_a", losf = 11L,
  # logit(P) = logit(0.3) + 0.5*x1 + 0.4*g[2] + 0.8*g[3]
  formula = "y ~ x1 + g", family = "logit",
  variable_types = "g=(factor, 0.5, 0.3, 0.2)",
  effects = "x1=0.5, g[2]=0.4, g[3]=0.8", baseline_probability = 0.3,
  n = 1000L, seed = 2137
)
glm_factor_b <- list(
  label = "glm_factor_b", losf = 11L,
  # logit(P) = logit(0.5) + 0.8*x1 + 0.5*g[2] + 0.8*g[3]
  formula = "y ~ x1 + g", family = "logit",
  variable_types = "g=(factor, 0.4, 0.35, 0.25)",
  effects = "x1=0.8, g[2]=0.5, g[3]=0.8", baseline_probability = 0.5,
  n = 1000L, seed = 2138
)

# losf 12 — interaction on the logit scale
glm_interaction_a <- list(
  label = "glm_interaction_a", losf = 12L,
  # logit(P) = logit(0.3) + 0.5*x1 + 0.3*x2 + 0.3*x1:x2
  formula = "y ~ x1*x2", family = "logit",
  effects = "x1=0.5, x2=0.3, x1:x2=0.3", baseline_probability = 0.3,
  n = 1000L, seed = 2137
)
glm_interaction_b <- list(
  label = "glm_interaction_b", losf = 12L,
  # logit(P) = logit(0.5) + 0.8*x1 + 0.5*x2 + 0.4*x1:x2
  formula = "y ~ x1*x2", family = "logit",
  effects = "x1=0.8, x2=0.5, x1:x2=0.4", baseline_probability = 0.5,
  n = 1000L, seed = 2138
)

# ---- GLM — probit (binary outcome, probit link) ------------------------------
# Same feature axes as the logit cases (losf 9-12) but on the probit scale:
# the intercept is Phi^-1(baseline_probability), betas are probit-scale.
# B-side oracle: glm(family = binomial(link = "probit")).

# probit — one continuous predictor
probit_simple_a <- list(
  label = "probit_simple_a", losf = 9L,
  # probit(P) = qnorm(0.3) + 0.5*x1
  formula = "y ~ x1", family = "probit",
  effects = "x1=0.5", baseline_probability = 0.3, n = 600L, seed = 2137
)
probit_simple_b <- list(
  label = "probit_simple_b", losf = 9L,
  # probit(P) = qnorm(0.5) + 0.8*x1
  formula = "y ~ x1", family = "probit",
  effects = "x1=0.8", baseline_probability = 0.5, n = 600L, seed = 2138
)

# probit — multiple predictors, correlated
probit_two_a <- list(
  label = "probit_two_a", losf = 10L,
  formula = "y ~ x1 + x2", family = "probit",
  effects = "x1=0.5, x2=0.3", baseline_probability = 0.3,
  correlations = "corr(x1,x2)=0.2", n = 800L, seed = 2137
)

# probit — a factor on the probit scale
probit_factor_a <- list(
  label = "probit_factor_a", losf = 11L,
  formula = "y ~ x1 + g", family = "probit",
  variable_types = "g=(factor, 0.5, 0.3, 0.2)",
  effects = "x1=0.5, g[2]=0.4, g[3]=0.8", baseline_probability = 0.3,
  n = 1000L, seed = 2137
)

# ---- GLM — Poisson (count outcome, log link) ---------------------------------
# Log-link count model. The intercept is log(baseline_rate); betas are
# log-rate-ratios. B-side oracle: glm(family = poisson()). No baseline
# probability — Poisson is sized by baseline_rate.

# poisson — one continuous predictor
poisson_simple_a <- list(
  label = "poisson_simple_a", losf = 9L,
  # log(lambda) = log(2.0) + 0.5*x1
  formula = "y ~ x1", family = "poisson",
  effects = "x1=0.5", baseline_rate = 2.0, n = 600L, seed = 2137
)
poisson_simple_b <- list(
  label = "poisson_simple_b", losf = 9L,
  # log(lambda) = log(5.0) + 0.3*x1 ; higher rate, weaker effect
  formula = "y ~ x1", family = "poisson",
  effects = "x1=0.3", baseline_rate = 5.0, n = 600L, seed = 2138
)

# poisson — multiple predictors, correlated
poisson_two_a <- list(
  label = "poisson_two_a", losf = 10L,
  formula = "y ~ x1 + x2", family = "poisson",
  effects = "x1=0.4, x2=0.2", baseline_rate = 2.0,
  correlations = "corr(x1,x2)=0.2", n = 800L, seed = 2137
)

# poisson — a factor on the log-rate scale
poisson_factor_a <- list(
  label = "poisson_factor_a", losf = 11L,
  formula = "y ~ x1 + g", family = "poisson",
  variable_types = "g=(factor, 0.5, 0.3, 0.2)",
  effects = "x1=0.4, g[2]=0.3, g[3]=0.5", baseline_rate = 2.0,
  n = 1000L, seed = 2137
)

# ---- Mixed / LME (clustered outcome) -----------------------------------------

# losf 13 — random intercept (set ICC)
lme_simple_a <- list(
  label = "lme_simple_a", losf = 13L,
  # y = 0.5*x1 + (1|grp) + e ;  ICC 0.2 ;  20 clusters x 30
  formula = "y ~ x1 + (1|grp)", family = "lme",
  effects = "x1=0.5",
  cluster = list(var = "grp", ICC = 0.2, n_clusters = 20L, cluster_size = 30L),
  n = 600L, seed = 2137
)
lme_simple_b <- list(
  label = "lme_simple_b", losf = 13L,
  # y = 0.3*x1 + (1|grp) + e ;  ICC 0.3 ;  20 clusters x 30
  formula = "y ~ x1 + (1|grp)", family = "lme",
  effects = "x1=0.3",
  cluster = list(var = "grp", ICC = 0.3, n_clusters = 20L, cluster_size = 30L),
  n = 600L, seed = 2138
)

# losf 14 — multiple fixed effects, clustered
lme_two_a <- list(
  label = "lme_two_a", losf = 14L,
  # y = 0.5*x1 + 0.3*x2 + (1|grp) + e ;  ICC 0.2 ;  25 clusters x 30
  formula = "y ~ x1 + x2 + (1|grp)", family = "lme",
  effects = "x1=0.5, x2=0.3",
  cluster = list(var = "grp", ICC = 0.2, n_clusters = 25L, cluster_size = 30L),
  n = 750L, seed = 2137
)
lme_two_b <- list(
  label = "lme_two_b", losf = 14L,
  # y = 0.3*x1 + 0.5*x2 + (1|grp) + e ;  ICC 0.3 ;  25 clusters x 30
  formula = "y ~ x1 + x2 + (1|grp)", family = "lme",
  effects = "x1=0.3, x2=0.5",
  cluster = list(var = "grp", ICC = 0.3, n_clusters = 25L, cluster_size = 30L),
  n = 750L, seed = 2138
)

# losf 15 — interaction with random intercept
lme_interaction_a <- list(
  label = "lme_interaction_a", losf = 15L,
  # y = 0.5*x1 + 0.3*x2 + 0.3*x1:x2 + (1|grp) + e ;  ICC 0.2 ;  25 clusters x 30
  formula = "y ~ x1*x2 + (1|grp)", family = "lme",
  effects = "x1=0.5, x2=0.3, x1:x2=0.3",
  cluster = list(var = "grp", ICC = 0.2, n_clusters = 25L, cluster_size = 30L),
  n = 750L, seed = 2137
)
lme_interaction_b <- list(
  label = "lme_interaction_b", losf = 15L,
  # y = 0.4*x1 + 0.3*x2 + 0.2*x1:x2 + (1|grp) + e ;  ICC 0.3 ;  30 clusters x 30
  formula = "y ~ x1*x2 + (1|grp)", family = "lme",
  effects = "x1=0.4, x2=0.3, x1:x2=0.2",
  cluster = list(var = "grp", ICC = 0.3, n_clusters = 30L, cluster_size = 30L),
  n = 900L, seed = 2138
)

# losf 16 — random intercept with a factor (probes ICC under factor variance).
# The set ICC is the CONDITIONAL (residual) ICC tau^2/(tau^2+sigma^2); the raw
# (marginal) ICC of the outcome is lower because the fixed part adds Var(Xbeta).
# _a is a small 2-level/modest-effect case (small gap); _b is a large
# 3-level/large-effect case (visibly larger gap). See the marginal-ICC row in
# validation_data_generation.rmd.
lme_factor_a <- list(
  label = "lme_factor_a", losf = 16L,
  # y = 0.30*x1 + 0.30*g[2] + (1|grp) + e ;  g 2-level 50/50 ;  ICC 0.2 ;  25 x 30
  formula = "y ~ x1 + g + (1|grp)", family = "lme",
  variable_types = "g=(factor, 0.5, 0.5)",
  effects = "x1=0.30, g[2]=0.30",
  cluster = list(var = "grp", ICC = 0.2, n_clusters = 25L, cluster_size = 30L),
  n = 750L, seed = 2137
)
lme_factor_b <- list(
  label = "lme_factor_b", losf = 16L,
  # y = 0.40*x1 + 0.50*g[2] + 0.80*g[3] + (1|grp) + e ;  g 3-level 50/30/20 ;
  # ICC 0.3 ;  30 x 30 ;  large fixed effects -> visibly larger marginal gap
  formula = "y ~ x1 + g + (1|grp)", family = "lme",
  variable_types = "g=(factor, 0.5, 0.3, 0.2)",
  effects = "x1=0.40, g[2]=0.50, g[3]=0.80",
  cluster = list(var = "grp", ICC = 0.3, n_clusters = 30L, cluster_size = 30L),
  n = 900L, seed = 2138
)

# ---- M2: crossed + nested groupings (general lmm path) ------------------------

# losf 17 — crossed random intercepts: y ~ x + (1|subj) + (1|item)
lmm_crossed_a <- list(
  label = "lmm_crossed_a", losf = 17L,
  # y = 0.30*x1 + 0.20*x2 + (1|grp) + (1|item) + e ; ICC .2 ; tau^2_item .15 ;
  # 20 subjects x 12 items (atom 240), n = 480
  formula = "y ~ x1 + x2 + (1|grp)", family = "lme",
  effects = "x1=0.30, x2=0.20",
  cluster = list(var = "grp", ICC = 0.2, n_clusters = 20L, cluster_size = 24L),
  extra_groupings = list(list(
    relation = list(Crossed = list(n_clusters = 12L)), tau_squared = 0.15)),
  lmer_re = "(1 | g_primary) + (1 | g_extra_1)",
  n = 480L, seed = 2200
)
lmm_crossed_b <- list(
  label = "lmm_crossed_b", losf = 17L,
  # y = 0.40*x1 + (1|grp) + (1|item) + e ; ICC .3 ; tau^2_item .05 ; 15 x 8, n = 360
  formula = "y ~ x1 + (1|grp)", family = "lme",
  effects = "x1=0.40",
  cluster = list(var = "grp", ICC = 0.3, n_clusters = 15L, cluster_size = 24L),
  extra_groupings = list(list(
    relation = list(Crossed = list(n_clusters = 8L)), tau_squared = 0.05)),
  lmer_re = "(1 | g_primary) + (1 | g_extra_1)",
  n = 360L, seed = 2201
)

# losf 18 — nested random intercepts: y ~ x + (1|site/class)
lmm_nested_a <- list(
  label = "lmm_nested_a", losf = 18L,
  # y = 0.30*x1 + (1|site) + (1|site:class) + e ; ICC .2 ; tau^2_class .10 ;
  # 12 sites x 3 classes (atom 36), n = 720
  formula = "y ~ x1 + (1|grp)", family = "lme",
  effects = "x1=0.30",
  cluster = list(var = "grp", ICC = 0.2, n_clusters = 12L, cluster_size = 60L),
  extra_groupings = list(list(
    relation = list(NestedWithin = list(n_per_parent = 3L)), tau_squared = 0.10)),
  lmer_re = "(1 | g_primary) + (1 | g_extra_1)",
  n = 720L, seed = 2210
)
lmm_nested_b <- list(
  label = "lmm_nested_b", losf = 18L,
  # y = 0.50*x1 + 0.30*x2 + nested ; ICC .3 ; tau^2_child .15 ; 10 x 4 (atom 40), n = 480
  formula = "y ~ x1 + x2 + (1|grp)", family = "lme",
  effects = "x1=0.50, x2=0.30",
  cluster = list(var = "grp", ICC = 0.3, n_clusters = 10L, cluster_size = 48L),
  extra_groupings = list(list(
    relation = list(NestedWithin = list(n_per_parent = 4L)), tau_squared = 0.15)),
  lmer_re = "(1 | g_primary) + (1 | g_extra_1)",
  n = 480L, seed = 2211
)

# losf 19 — crossed + nested combined
lmm_crossed_nested_a <- list(
  label = "lmm_crossed_nested_a", losf = 19L,
  # y = 0.30*x1 + (1|subj) + (1|subj:half) + (1|item) ; ICC .2 ; tau^2_child .08 ;
  # tau^2_item .15 ; 10 subj x 6 items x 2 (atom 120), n = 480
  formula = "y ~ x1 + (1|grp)", family = "lme",
  effects = "x1=0.30",
  cluster = list(var = "grp", ICC = 0.2, n_clusters = 10L, cluster_size = 48L),
  extra_groupings = list(
    list(relation = list(Crossed = list(n_clusters = 6L)), tau_squared = 0.15),
    list(relation = list(NestedWithin = list(n_per_parent = 2L)), tau_squared = 0.08)),
  lmer_re = "(1 | g_primary) + (1 | g_extra_1) + (1 | g_extra_2)",
  n = 480L, seed = 2220
)

# M2/M3/M4 cases join CASES in EXTRA_CASES / ALL_GEN_CASES (defined after M4_GLMM_CASES).
M2_LMM_CASES <- list(
  lmm_crossed_a, lmm_crossed_b, lmm_nested_a, lmm_nested_b, lmm_crossed_nested_a
)
names(M2_LMM_CASES) <- vapply(M2_LMM_CASES, function(c) c$label, character(1))

# ---- M3: random slopes (general lmm path) ------------------------------------
# Each case carries `lmer_re` (the lme4 RE term) and, for composition, `extra`
# (the intercept-only crossed/nested grouping). `corr_with` is the slope↔slope
# lower triangle (numeric(0) for the first slope).

# losf 20 — standalone single slope with correlation: y ~ x1 + (1 + x1 | g)
lmm_slope_a <- list(
  label = "lmm_slope_a", losf = 20L,
  # ICC .2 (τ₀²=.25) ; τ₁²=.10 ; ρ=+0.3 ; 24 clusters × 20, n = 480
  formula = "y ~ x1 + (1|grp)", family = "lme",
  effects = "x1=0.30",
  cluster = list(var = "grp", ICC = 0.2, n_clusters = 24L, cluster_size = 20L),
  slopes = list(list(column = 0L, variance = 0.10, corr_with_intercept = 0.3, corr_with = numeric(0))),
  lmer_re = "(1 + x1 | g_primary)",
  n = 480L, seed = 2300
)
lmm_slope_b <- list(
  label = "lmm_slope_b", losf = 20L,
  # stronger slope + negative correlation: τ₀²=.43 (ICC .3) ; τ₁²=.25 ; ρ=-0.5 ; n=480
  formula = "y ~ x1 + x2 + (1|grp)", family = "lme",
  effects = "x1=0.40, x2=0.20",
  cluster = list(var = "grp", ICC = 0.3, n_clusters = 20L, cluster_size = 24L),
  slopes = list(list(column = 0L, variance = 0.25, corr_with_intercept = -0.5, corr_with = numeric(0))),
  lmer_re = "(1 + x1 | g_primary)",
  n = 480L, seed = 2301
)

# losf 21 — MULTI-slope: y ~ x1 + x2 + (1 + x1 + x2 | g)
lmm_multislope <- list(
  label = "lmm_multislope", losf = 21L,
  # ICC .2 (τ₀²=.25) ; τ₁²=.10 ρ_01=+0.3 ; τ₂²=.08 ρ_02=+0.1 ρ_12=+0.2 ; 30 × 20, n=600
  formula = "y ~ x1 + x2 + (1|grp)", family = "lme",
  effects = "x1=0.30, x2=0.20",
  cluster = list(var = "grp", ICC = 0.2, n_clusters = 30L, cluster_size = 20L),
  slopes = list(
    list(column = 0L, variance = 0.10, corr_with_intercept = 0.3, corr_with = numeric(0)),
    list(column = 1L, variance = 0.08, corr_with_intercept = 0.1, corr_with = c(0.2))
  ),
  lmer_re = "(1 + x1 + x2 | g_primary)",
  n = 600L, seed = 2302
)

# losf 22 — COMPOSITION: (1 + x1 | grp) crossed with (1 | item)
lmm_slope_crossed <- list(
  label = "lmm_slope_crossed", losf = 22L,
  # grp ICC .2 (τ₀²=.25) τ₁²=.10 ρ=+0.3 ; item τ²=.16 ; 24 grp × 20, 6 items, n=480
  formula = "y ~ x1 + (1|grp)", family = "lme",
  effects = "x1=0.30",
  cluster = list(var = "grp", ICC = 0.2, n_clusters = 24L, cluster_size = 20L),
  slopes = list(list(column = 0L, variance = 0.10, corr_with_intercept = 0.3, corr_with = numeric(0))),
  extra = list(list(var = "item", kind = "crossed", n_clusters = 6L, tau_squared = 0.16)),
  lmer_re = "(1 + x1 | g_primary) + (1 | g_item)",
  n = 480L, seed = 2303
)

# losf 28 — EXTRA SLOPES (crossed): (1+x1|grp) + (1+x1|item); slopes on both groupings
lmm_slope_crossed_extra_slopes <- list(
  label = "lmm_slope_crossed_extra_slopes", losf = 28L,
  # grp ICC .2 (τ₀²=.25) τ₁²=.10 ρ=+0.3 ; item τ²=.16 τ₁²=.06 ρ=−0.2 ; 24 grp × 20, 6 items, n=480
  formula = "y ~ x1 + (1|grp)", family = "lme",
  effects = "x1=0.30",
  cluster = list(var = "grp", ICC = 0.2, n_clusters = 24L, cluster_size = 20L),
  slopes = list(list(column = 0L, variance = 0.10, corr_with_intercept = 0.3, corr_with = numeric(0))),
  extra = list(list(var = "item", kind = "crossed", n_clusters = 6L, tau_squared = 0.16,
                    slopes = list(list(column = 0L, variance = 0.06, corr_with_intercept = -0.2, corr_with = numeric(0))))),
  lmer_re = "(1 + x1 | g_primary) + (1 + x1 | g_item)",
  n = 480L, seed = 2304
)

# losf 29 — EXTRA SLOPES (nested): (1+x1|grp) + (1+x1|class); slopes on both groupings
lmm_slope_nested_extra_slopes <- list(
  label = "lmm_slope_nested_extra_slopes", losf = 29L,
  # grp ICC .2 (τ₀²=.25) τ₁²=.10 ρ=+0.3 ; class τ²=.10 τ₁²=.05 ρ=−0.2 ; 12 grp × 60, 3 nested, n=720
  formula = "y ~ x1 + (1|grp)", family = "lme",
  effects = "x1=0.30",
  cluster = list(var = "grp", ICC = 0.2, n_clusters = 12L, cluster_size = 60L),
  slopes = list(list(column = 0L, variance = 0.10, corr_with_intercept = 0.3, corr_with = numeric(0))),
  extra = list(list(var = "class", kind = "nested", n_clusters = 3L, tau_squared = 0.10,
                    slopes = list(list(column = 0L, variance = 0.05, corr_with_intercept = -0.2, corr_with = numeric(0))))),
  lmer_re = "(1 + x1 | g_primary) + (1 + x1 | g_class)",
  n = 720L, seed = 2305
)

M3_LMM_CASES <- list(lmm_slope_a, lmm_slope_b, lmm_multislope, lmm_slope_crossed,
                     lmm_slope_crossed_extra_slopes, lmm_slope_nested_extra_slopes)
names(M3_LMM_CASES) <- vapply(M3_LMM_CASES, function(c) c$label, character(1))

# Ordered case suite (losf complexity order). Extend as MCPower grows new forms.
CASES <- list(
  ols_simple_a, ols_simple_b, ols_two_a, ols_two_b, ols_zero_a,
  ols_corr_a, ols_corr_b, ols_interaction_a, ols_interaction_b,
  ols_factor_a, ols_factor_b, ols_cf_a, ols_cf_b, ols_ff_a, ols_ff_b,
  glm_simple_a, glm_simple_b, glm_two_a, glm_two_b,
  glm_factor_a, glm_factor_b, glm_interaction_a, glm_interaction_b,
  probit_simple_a, probit_simple_b, probit_two_a, probit_factor_a,
  poisson_simple_a, poisson_simple_b, poisson_two_a, poisson_factor_a,
  lme_simple_a, lme_simple_b, lme_two_a, lme_two_b,
  lme_interaction_a, lme_interaction_b,
  lme_factor_a, lme_factor_b
)
names(CASES) <- vapply(CASES, function(c) c$label, character(1))

# Continuous-main-effect cases for the get_effects round-trip gate (L3 G2,
# validation_get_effects.rmd + regression.R). Factors and interactions are out of
# scope for the settled convention. Correlated case (ols_corr_a) exercises the
# beta'Sigma beta oracle branch in expected_recovery.
GE_CASES <- list(CASES$ols_simple_a, CASES$ols_two_a, CASES$ols_corr_a,
                 CASES$glm_simple_a, CASES$glm_two_b,
                 CASES$lme_simple_a, CASES$lme_two_a)

# ---- Upload validation cases (A<->B, frame as reference) ---------------------
#
# These cases validate the NORTA-over-uploaded-frame path.  Each case carries
# an `upload_csv` path (relative to mcpower/validation/) and a
# `upload_cols` character vector naming which columns to pass to upload_data().
#
# The oracle is the frame itself:
#   - cont_cols   : continuous predictor names in the model; generated design
#                   columns must have mean ≈ 0, sd ≈ 1 (engine standardizes).
#   - binary_cols : binary predictor names; generated proportion must match
#                   the frame's empirical proportion (within DGP_TOL$moment_abs).
#
# Binary×anything correlations are NOT reproduced — correlation is
# continuous-only by design: binary/factor variables are generated from their
# marginals, independent of every other predictor.  The campaign asserts that
# the realized binary×continuous Pearson r is ≈ 0 (see validation_upload.rmd).
#
# Factor correlations are OUT OF SCOPE (discrete-correlation work not yet landed).
#
# Two cases from R's built-in mtcars dataset (columns hp, wt, am):
#   upload_cont_only  : two continuous predictors (hp, wt) — isolates
#                        continuous-marginal + continuous×continuous correlation
#                        reproduction under mode="partial".
#   upload_cont_binary : adds the binary predictor am — covers binary
#                         marginal reproduction and confirms the binary predictor
#                         is generated independently (no correlation reproduced).

upload_cont_only <- list(
  label       = "upload_cont_only",
  formula     = "mpg ~ hp + wt",
  family      = "ols",
  effects     = "hp=-0.30, wt=-0.50",
  upload_builtin = "mtcars",   # R built-in dataset (columns hp, wt, am)
  upload_cols = c("hp", "wt"),
  cont_cols   = c("hp", "wt"),
  binary_cols = character(0),
  n = 400L, seed = 2137
)

upload_cont_binary <- list(
  label       = "upload_cont_binary",
  formula     = "mpg ~ hp + wt + am",
  family      = "ols",
  effects     = "hp=-0.30, wt=-0.50, am=0.40",
  upload_builtin = "mtcars",
  upload_cols = c("hp", "wt", "am"),
  cont_cols   = c("hp", "wt"),
  binary_cols = c("am"),
  n = 400L, seed = 2137
)

# ---- Strict-mode upload case (P2-6 Step 1) -----------------------------------
#
# upload_strict_nonlinear validates the STRICT (whole-row bootstrap) path using a
# synthetic 150-row fixture where x2 = x1^2 exactly.  This nonlinear dependence
# cannot be reproduced by the NORTA/Gaussian-copula path (which only matches
# marginals + Pearson correlation), but strict bootstrap copies entire rows from
# the uploaded frame so every generated row satisfies x2 = x1^2.
#
# The fixture lives in validation/data/nonlinear_parabola.csv (x1 ∈ [0,3],
# x2 = x1^2; 150 rows).  The empirical Pearson r(x1,x2) ≈ 0.968, which NORTA
# would also match — the discriminating assertion is the parabolic joint, not
# the linear correlation alone.
#
# Fields specific to this case (beyond the shared upload-case fields):
#   upload_mode : "strict"  — triggers whole-row bootstrap path.
#   nonlinear   : TRUE  — signal to the report to run the parabola-preservation
#                 assertion (see validation_upload.rmd §Strict-mode case).
#
upload_strict_nonlinear <- list(
  label       = "upload_strict_nonlinear",
  formula     = "y ~ x1 + x2",
  family      = "ols",
  effects     = "x1=0.30, x2=0.20",
  upload_csv  = "data/nonlinear_parabola.csv",   # relative to mcpower/validation/
  upload_cols = c("x1", "x2"),
  cont_cols   = c("x1", "x2"),
  binary_cols = character(0),
  upload_mode = "strict",
  nonlinear   = TRUE,
  n = 150L, seed = 2137
)

UPLOAD_CASES <- list(upload_cont_only, upload_cont_binary, upload_strict_nonlinear)
names(UPLOAD_CASES) <- vapply(UPLOAD_CASES, function(c) c$label, character(1))

# ---- Crossing validation cases (validation_crossing.rmd + regression.R) -------
#
# Each case is built inline (no saved dataset), with a dense + coarse
# find_sample_size run. The dense run's fitted crossings are frozen to
# data/<label>.golden.rds; regression.R runs only the cheap coarse grid
# and compares to the frozen dense N within CROSSING_TOL$n_rel.
#
# cross_partial includes a small effect (x2=0.05) that will NOT reach 80%
# within [FROM_SIZE, TO_SIZE] — the not-fitted target is EXPECTED.
CROSS_CASES <- list(
  list(label = "cross_ols",
       formula = "y ~ x1 + x2", effects = "x1=0.40, x2=0.25",
       family = "ols", cluster = NULL),
  list(label = "cross_glm",
       formula = "y ~ x1 + x2", effects = "x1=0.50, x2=0.35",
       family = "logit", baseline_probability = 0.3, cluster = NULL),
  list(label = "cross_lme",
       formula = "y ~ x1 + (1|grp)", effects = "x1=0.45",
       family = "lme", cluster = list(var = "grp", ICC = 0.2,
                                      n_clusters = 10L, cluster_size = 20L)),
  list(label = "cross_partial",
       formula = "y ~ x1 + x2", effects = "x1=0.40, x2=0.05",
       family = "ols", cluster = NULL)   # x2 too small to reach 80% — not-fitted expected
)
names(CROSS_CASES) <- vapply(CROSS_CASES, function(c) c$label, character(1))

# ---- L5 scenario-perturbation cases (validation_scenarios.rmd) ----------------
#
# Each case probes ONE scenario knob (Tier A: knob on an otherwise-optimistic
# base design) or one knob interaction (Tier B / family gating). The oracle is
# the documented perturbation law, recovered by the probes in common.R; gates
# are the SE-of-mean z-band in tolerances.R (SCENARIO_TOL$z_c). Driven through
# MCPowerDebug$create_data() with .debug_n_sims = 1: every call draws scenario
# block 0, so seed + k gives K independent block-perturbation draws, and the
# same seed under two scenario names shares both RNG streams (paired draws).
#
# Fields beyond the shared L3 case fields (formula/family/effects/...):
#   scenario_configs  named list for set_scenario_configs() — the custom
#                     scenarios this case draws from (merged onto optimistic,
#                     so every unlisted knob is off: the Tier A isolation).
#   scenario          .debug_scenario name for single-scenario cases.
#   scenario_a/_b     the paired scenario names for paired-draw cases.
#   hsk_var           set_heteroskedasticity_driver() pin (ratio is always
#                     left to the scenario knob).
#   K                 number of block-perturbation draws (seed + 1..K).
#   n, seed           rows per draw; base seed (cases use disjoint 1000-wide
#                     seed blocks so their data streams never overlap).
#
# Knob magnitudes use the shipped preset values (realistic/doomer) so the
# validated points are the ones users actually run.

# -- Tier A: one knob, base design ----------------------------------------------

# He: slope-jitter SD. Probe: e²-on-x² slopes ≈ h²βⱼ² per predictor.
scen_he <- list(
  label = "scen_he", knob = "heterogeneity",
  formula = "y ~ x1 + x2", family = "ols",
  effects = "x1=0.40, x2=0.25",
  scenario_configs = list(t_he = list(heterogeneity = 0.4)),
  scenario = "t_he",
  n = 4000L, K = 200L, seed = 31000
)

# Hs: residual-variance ratio. Single predictor so the default lp driver IS x1
# (z = x1 exactly: lp = 0.25·x1 standardised by lp_pop_std = 0.25). Probe:
# log(e²)-on-x1 slope ≈ ln(λ)/4.
scen_hs <- list(
  label = "scen_hs", knob = "heteroskedasticity_ratio",
  formula = "y ~ x1", family = "ols",
  effects = "x1=0.25",
  scenario_configs = list(t_hs = list(heteroskedasticity_ratio = 4.0)),
  scenario = "t_hs",
  n = 4000L, K = 200L, seed = 32000
)

# Co (exact law, clamp negligible): per-block r mean ≈ ρ, SD ≈
# sqrt(s²/2 + (1-ρ²)²/n). ρ = 0.3, s = 0.15 → the ±0.8 clamp sits 4.7σ out.
scen_co_low <- list(
  label = "scen_co_low", knob = "correlation_noise_sd",
  formula = "y ~ x1 + x2", family = "ols",
  effects = "x1=0.25, x2=0.25", correlations = "corr(x1,x2)=0.3",
  scenario_configs = list(t_co = list(correlation_noise_sd = 0.15)),
  scenario = "t_co",
  n = 4000L, K = 400L, seed = 33000
)

# Co (clamp truncation): ρ = 0.75 puts the +0.8 clamp 0.47σ from the mean —
# the censored-normal law (clamp_normal_moments) is ~9σ from the naive ρ at
# this K, so the gate discriminates truncation-law vs no-truncation.
scen_co_high <- list(
  label = "scen_co_high", knob = "correlation_noise_sd",
  formula = "y ~ x1 + x2", family = "ols",
  effects = "x1=0.25, x2=0.25", correlations = "corr(x1,x2)=0.75",
  scenario_configs = list(t_co = list(correlation_noise_sd = 0.15)),
  scenario = "t_co",
  n = 2000L, K = 800L, seed = 34000
)

# Co (PSD repair, p = 3): repair cannot fire at p = 2, so this 3-var high-ρ
# case is where eigenvalue-floor + diagonal-renorm distortion lives. No
# analytic law — the empirical per-pair mean/SD are frozen as MCPower-golden.
scen_co_psd <- list(
  label = "scen_co_psd", knob = "correlation_noise_sd",
  formula = "y ~ x1 + x2 + x3", family = "ols",
  effects = "x1=0.25, x2=0.25, x3=0.25",
  correlations = "corr(x1,x2)=0.6, corr(x1,x3)=0.6, corr(x2,x3)=0.6",
  scenario_configs = list(t_co = list(correlation_noise_sd = 0.3)),
  scenario = "t_co",
  n = 2000L, K = 400L, seed = 35000
)

# Px (preset pool): swap frequency ≈ q per normal column per block; picks
# uniform over the pool; every candidate mean ≈ 0 / var ≈ 1. Independent
# predictors so the marginal law is exactly the documented transform (no
# NORTA coupling — that channel is B4's).
scen_px <- list(
  label = "scen_px", knob = "distribution_change_prob",
  formula = "y ~ x1 + x2", family = "ols",
  effects = "x1=0.25, x2=0.25",
  scenario_configs = list(t_px = list(distribution_change_prob = 0.5)),
  scenario = "t_px",
  n = 4000L, K = 400L, seed = 36000
)

# Px (custom pool, high_kurtosis): the only swappable kind outside the presets.
# q = 1 single-candidate pool → every block swapped; moment law from
# t3_table_moments (the censored, interpolated table IS the marginal).
scen_px_t3 <- list(
  label = "scen_px_t3", knob = "distribution_change_prob",
  formula = "y ~ x1", family = "ols",
  effects = "x1=0.25",
  scenario_configs = list(t_px_t3 = list(
    distribution_change_prob = 1.0,
    new_distributions = list("high_kurtosis")
  )),
  scenario = "t_px_t3",
  n = 4000L, K = 200L, seed = 37000
)

# Re (preset pool): swap frequency ≈ qᵣ; picks uniform over {high_kurtosis,
# right_skewed}; recovered df ≈ scenario residual_df via the shape laws
# (skew = sqrt(8/df), excess kurtosis = 6/(df-4)). df = 10 keeps the kurtosis
# estimator's CLT valid (t(10) has finite 8th moment).
scen_re <- list(
  label = "scen_re", knob = "residual_change_prob",
  formula = "y ~ x1", family = "ols",
  effects = "x1=0.25",
  scenario_configs = list(t_re = list(
    residual_change_prob = 0.5, residual_df = 10
  )),
  scenario = "t_re",
  n = 4000L, K = 400L, seed = 38000
)

# Re (pinned residual, no swap): set_residual_distribution("high_kurtosis")
# PINS the residual in the new API — the engine only swaps an unpinned
# default-normal residual, so residual_change_prob = 1.0 + residual_dists are
# inert here. LAW CHANGED under pin rule: this case no longer tests replacement
# (swap from one non-default to another); it now exercises a pinned high_kurtosis
# residual held fixed across scenarios. The scenario df knob (residual_df = 6) is
# still present in the config but does not fire. Re-freeze goldens after rebuilding.
scen_re_replace <- list(
  label = "scen_re_replace", knob = "residual_change_prob",
  formula = "y ~ x1", family = "ols",
  effects = "x1=0.25",
  residual = list(name = "high_kurtosis"),
  scenario_configs = list(t_re1 = list(
    residual_change_prob = 1.0, residual_dists = list("right_skewed"),
    residual_df = 6
  )),
  scenario = "t_re1",
  n = 4000L, K = 200L, seed = 39000
)

# Fa: factor-proportion sampling. FALSE (optimistic default): counts are a pure
# function of (n, p) — identical across draws, each within ~1 of n·p. TRUE:
# count_g ~ Binomial(n, p_g), variance n·p(1-p).
scen_fa_ols <- list(
  label = "scen_fa_ols", knob = "sampled_factor_proportions",
  formula = "y ~ x1 + g", family = "ols",
  variable_types = "g=(factor, 0.5, 0.3, 0.2)",
  effects = "x1=0.25, g[2]=0.50, g[3]=0.80",
  scenario_configs = list(t_fa = list(sampled_factor_proportions = TRUE)),
  scenario_a = "optimistic", scenario_b = "t_fa",
  n = 1000L, K = 400L, seed = 40000
)

# Fa under MLE — the one scenario knob the Mle estimator admits
# (is_optimistic() does not consult sampled_factor_proportions, so the batch.rs
# Mle rejection gate passes in either toggle position).
scen_fa_mle <- list(
  label = "scen_fa_mle", knob = "sampled_factor_proportions",
  formula = "y ~ x1 + g + (1|grp)", family = "lme",
  variable_types = "g=(factor, 0.5, 0.3, 0.2)",
  effects = "x1=0.30, g[2]=0.30, g[3]=0.50",
  cluster = list(var = "grp", ICC = 0.2, n_clusters = 20L, cluster_size = 50L),
  scenario_configs = list(t_fa = list(sampled_factor_proportions = TRUE)),
  scenario_a = "optimistic", scenario_b = "t_fa",
  n = 1000L, K = 200L, seed = 41000
)

# He under the new GLM families (probit latent / Poisson log-rate). The He
# per-study β-jitter is family-general (drawn once per study, slopes clipped at
# s_j = h|β_j|, intercept jitter symmetric), so each study fits a clean GLM at its
# own β_eff and the mean β̂ → E[β_eff] = glm_perstudy_beta(β, h). Gate the SLOPES
# against that per-study pseudo-true within glm_beta_abs (measured worst: probit
# 0.0075 / Poisson 0.0085 at h = 0.4). The intercept carries an ADDITIONAL
# link-nonlinearity (Jensen) shift ~0.025 the per-study oracle does not model —
# reported, not gated (mirrors scen_fg_glm_flip's Jensen note).
scen_he_probit <- list(
  label = "scen_he_probit", knob = "heterogeneity",
  formula = "y ~ x1 + x2", family = "probit",
  effects = "x1=0.5, x2=0.3", baseline_probability = 0.4,
  scenario_configs = list(t_he = list(heterogeneity = 0.4)),
  scenario = "t_he",
  n = 2000L, K = 300L, seed = 60000
)
scen_he_poisson <- list(
  label = "scen_he_poisson", knob = "heterogeneity",
  formula = "y ~ x1 + x2", family = "poisson",
  effects = "x1=0.5, x2=0.3", baseline_rate = 2.0,
  scenario_configs = list(t_he = list(heterogeneity = 0.4)),
  scenario = "t_he",
  n = 2000L, K = 300L, seed = 61000
)

# -- Tier B: joint --------------------------------------------------------------

# B0: optimistic ≡ baseline at the find_power level — the scenarios=FALSE
# single call vs the optimistic member of the scenarios=TRUE paired call,
# bit-identical PowerResult (DGP-level companion to P1's orchestrator test).
scen_b0 <- list(
  label = "scen_b0", knob = "B0",
  formula = "y ~ x1 + x2", family = "ols",
  effects = "x1=0.25, x2=0.10",
  n = 200L, n_sims = 400L, seed = 4242
)

# B1: β̂ unbiased (incl. the intercept — the mean-leak backstop) across the
# three shipped presets. OLS gated on the z-band against the true β (linear
# averaging keeps OLS exact under every knob). Logit gated on the absolute
# band against the PER-STUDY pseudo-true β (glm_perstudy_beta): the He β-jitter
# is drawn once per study, so each study fits a clean logit at its own β_eff and
# the mean β̂ → E[β_eff] — slopes get only the tiny clip nudge (×1.0008 at doomer
# h = 0.4), the intercept stays at β_0; no per-observation attenuation. The GLM
# gate is Tier A calibration + the flip rate, not β̂.
scen_b1_ols <- list(
  label = "scen_b1_ols", knob = "B1",
  formula = "y ~ x1 + x2", family = "ols",
  effects = "x1=0.40, x2=0.25", correlations = "corr(x1,x2)=0.3",
  presets = c("optimistic", "realistic", "doomer"),
  n = 2000L, K = 300L, seed = 43000
)
scen_b1_glm <- list(
  label = "scen_b1_glm", knob = "B1",
  formula = "y ~ x1 + x2", family = "logit",
  effects = "x1=0.5, x2=0.3", baseline_probability = 0.4,
  presets = c("optimistic", "realistic", "doomer"),
  n = 2000L, K = 300L, seed = 44000
)

# B2: He×Hs separation. λ driver pinned to x2 (set_heteroskedasticity_driver(),
# ratio left to the scenario), so the x1²-decomposed jitter variance is
# uncontaminated by the λ channel's cosh(γz) even component — assert it is
# λ-invariant across the pair {h=0.4, λ=1} vs {h=0.4, λ=4} (paired seeds).
scen_b2 <- list(
  label = "scen_b2", knob = "B2",
  formula = "y ~ x1 + x2", family = "ols",
  effects = "x1=0.40, x2=0.25",
  hsk_var = "x2",
  scenario_configs = list(
    b2_l1 = list(heterogeneity = 0.4),
    b2_l4 = list(heterogeneity = 0.4, heteroskedasticity_ratio = 4.0)
  ),
  scenario_a = "b2_l1", scenario_b = "b2_l4",
  n = 4000L, K = 200L, seed = 45000
)

# B3: Hs×Re preservation. Forced residual swap (qᵣ = 1, single-candidate pool)
# under λ = 4: the log-e² slope ≈ ln(λ)/4 is shape-blind, so the ±2σ ratio
# survives a t or chi² residual (tails amplified multiplicatively).
scen_b3 <- list(
  label = "scen_b3", knob = "B3",
  formula = "y ~ x1", family = "ols",
  effects = "x1=0.25",
  scenario_configs = list(
    b3_t   = list(heteroskedasticity_ratio = 4.0, residual_change_prob = 1.0,
                  residual_dists = list("high_kurtosis"), residual_df = 10),
    b3_chi = list(heteroskedasticity_ratio = 4.0, residual_change_prob = 1.0,
                  residual_dists = list("right_skewed"), residual_df = 10)
  ),
  scenarios = c("b3_t", "b3_chi"),
  n = 4000L, K = 200L, seed = 46000
)

# B4: Co×Px NORTA. Forced swap (q = 1, single-candidate pool) on a ρ = 0.5
# design: realized Pearson r matches the NORTA oracle (norta_r), not the
# latent spec value. Uniform pair closed form: (6/π)asin(ρ/2) ≈ 0.483; the
# censored-Exp(1) skewed pair has no closed form (norta_r Gauss–Hermite only).
scen_b4 <- list(
  label = "scen_b4", knob = "B4",
  formula = "y ~ x1 + x2", family = "ols",
  effects = "x1=0.25, x2=0.25", correlations = "corr(x1,x2)=0.5",
  scenario_configs = list(
    b4_logn = list(distribution_change_prob = 1.0,
                   new_distributions = list("right_skewed")),
    b4_unif = list(distribution_change_prob = 1.0,
                   new_distributions = list("uniform"))
  ),
  scenarios = c("b4_logn", "b4_unif"),
  n = 4000L, K = 200L, seed = 47000
)

# B5: P6 driver-moment staleness, isolated (correlation noise shifts the
# realized driver SD σ′ per block while het_coeffs stays anchored to the spec
# σ₀; He off, swaps off so the residual probe is clean). Gates: (1) mean raw
# log-e²-on-lp slope ≈ γ/σ₀ (magnitude law; note this mean alone cannot
# discriminate staleness — a per-block recompute predicts γ·E[1/σ′ₖ] ≈ γ/σ₀
# to 0.1%); (2) THE staleness discriminator: regression of the per-block
# measured λ̂′ₖ = exp(4·slopeₖ·σ̂′ₖ) on the moment-predicted
# λ′ₖ = exp(4γ σ̂′ₖ/σ₀) has slope ≈ 1 (stale anchor tracks σ̂′ₖ block by
# block; a recompute would pin λ̂′ₖ ≈ λ constant → slope 0, ~11σ away);
# (3) every λ′ₖ within the a-priori clamp-range drift bound (+ documented
# σ̂′ sampling slack). Presets quantified descriptively from σ̂′/σ₀ alone
# (He-immune — no residual probe needed).
scen_b5 <- list(
  label = "scen_b5", knob = "B5",
  formula = "y ~ x1 + x2", family = "ols",
  effects = "x1=0.40, x2=0.25", correlations = "corr(x1,x2)=0.3",
  scenario_configs = list(b5_iso = list(correlation_noise_sd = 0.3,
                                        heteroskedasticity_ratio = 4.0)),
  scenario = "b5_iso",
  presets = c("realistic", "doomer"),
  n = 4000L, K = 200L, seed = 48000
)

# -- Family gating ----------------------------------------------------------------

# GLM λ/residual inertness: paired bit-identity. apply_hsk requires a
# continuous outcome and consumes no RNG, so a λ toggle is an exact no-op
# under logit; the residual swap consumes only scenario-stream draws AFTER
# every other consumer, so outcomes stay bit-identical too.
scen_fg_glm_ident <- list(
  label = "scen_fg_glm_ident", knob = "family_gating",
  formula = "y ~ x1 + x2", family = "logit",
  effects = "x1=0.5, x2=0.3", baseline_probability = 0.4,
  scenario_configs = list(
    glm_l4   = list(heteroskedasticity_ratio = 4.0),
    glm_l4re = list(heteroskedasticity_ratio = 4.0, residual_change_prob = 1.0,
                    residual_dists = list("high_kurtosis"), residual_df = 10)
  ),
  # pair 1: optimistic vs λ=4 (λ inert); pair 2: λ=4 vs λ=4 + forced residual
  # swap (residual inert; both sides on the scenario path).
  pairs = list(c("optimistic", "glm_l4"), c("glm_l4", "glm_l4re")),
  n = 2000L, K = 20L, seed = 49000
)

# Probit / Poisson λ + residual-swap inertness — same family-gating premise as
# scen_fg_glm_ident but for the new links. apply_hsk requires a continuous
# Gaussian outcome and consumes no RNG, so a λ toggle is an exact no-op under
# probit (binary) and Poisson (count); the residual swap consumes only
# scenario-stream draws AFTER every other consumer, so outcomes stay
# bit-identical too. Both pairs verified bit-identical in-harness (run_fg_ident).
scen_fg_probit_ident <- list(
  label = "scen_fg_probit_ident", knob = "family_gating",
  formula = "y ~ x1 + x2", family = "probit",
  effects = "x1=0.5, x2=0.3", baseline_probability = 0.4,
  scenario_configs = list(
    g_l4   = list(heteroskedasticity_ratio = 4.0),
    g_l4re = list(heteroskedasticity_ratio = 4.0, residual_change_prob = 1.0,
                  residual_dists = list("high_kurtosis"), residual_df = 10)
  ),
  pairs = list(c("optimistic", "g_l4"), c("g_l4", "g_l4re")),
  n = 2000L, K = 20L, seed = 62000
)
scen_fg_poisson_ident <- list(
  label = "scen_fg_poisson_ident", knob = "family_gating",
  formula = "y ~ x1 + x2", family = "poisson",
  effects = "x1=0.5, x2=0.3", baseline_rate = 2.0,
  scenario_configs = list(
    g_l4   = list(heteroskedasticity_ratio = 4.0),
    g_l4re = list(heteroskedasticity_ratio = 4.0, residual_change_prob = 1.0,
                  residual_dists = list("high_kurtosis"), residual_df = 10)
  ),
  pairs = list(c("optimistic", "g_l4"), c("g_l4", "g_l4re")),
  n = 2000L, K = 20L, seed = 63000
)

# GLM He: the per-row log-odds jitter is latent (one Bernoulli draw), so the
# get is the paired h-toggle flip rate vs its Monte-Carlo prediction
# (glm_fliprate_pred). The Jensen mean-rate shift is reported, not gated.
scen_fg_glm_flip <- list(
  label = "scen_fg_glm_flip", knob = "family_gating",
  formula = "y ~ x1 + x2", family = "logit",
  effects = "x1=0.5, x2=0.3", baseline_probability = 0.4,
  scenario_configs = list(t_he = list(heterogeneity = 0.4)),
  scenario_a = "optimistic", scenario_b = "t_he",
  n = 2000L, K = 200L, seed = 50000
)

# Multi-grouping RE knobs (M2): one crossed extra beside the primary. The RE
# dist/df swap and the ICC jitter must hit BOTH groupings' draws (section 2.5
# uniformity — independent jitter per grouping); beta-hat unbiased is the tripwire.
scen_re_multi <- list(
  label = "scen_re_multi", knob = "random_effect_dist+icc_noise (multi-grouping)",
  formula = "y ~ x1 + (1|grp)", family = "lme",
  effects = "x1=0.30",
  cluster = list(var = "grp", ICC = 0.2, n_clusters = 20L, cluster_size = 24L),
  extra_groupings = list(list(
    relation = list(Crossed = list(n_clusters = 12L)), tau_squared = 0.15)),
  scenario_configs = list(re_t5 = list(random_effect_dist = "heavy_tailed",
                                       random_effect_df = 5,
                                       icc_noise_sd = 0.05)),
  scenario_a = "optimistic", scenario_b = "re_t5",
  n = 480L, K = 200L, seed = 52000
)

SCENARIO_CASES <- list(
  scen_he, scen_hs, scen_co_low, scen_co_high, scen_co_psd,
  scen_px, scen_px_t3, scen_re, scen_re_replace,
  scen_fa_ols, scen_fa_mle,
  scen_he_probit, scen_he_poisson,
  scen_b0, scen_b1_ols, scen_b1_glm, scen_b2, scen_b3, scen_b4, scen_b5,
  scen_fg_glm_ident, scen_fg_glm_flip,
  scen_fg_probit_ident, scen_fg_poisson_ident,
  scen_re_multi
)
names(SCENARIO_CASES) <- vapply(SCENARIO_CASES, function(c) c$label, character(1))

# ---- M4: clustered logistic GLMM (logit + random effects) ---------------------
# Each case carries `glmer_re` (the lme4 glmer RE term string) and, for
# composition, `extra` (the crossed/nested grouping). `slopes` mirrors M3's
# schema. `nagq` (if present) signals the nAGQ-level for the Laplace-bias cell
# (losf 27): cross-check that case against glmer(nAGQ=nagq), not Laplace.
# cluster$cluster_size is stored for reference but the engine derives the atom
# from n / n_clusters.

# losf 23 — random-intercept logistic GLMM: y ~ x1 + (1 | g), binary
glmm_intercept <- list(
  label = "glmm_intercept", losf = 23L,
  # logit(P) = logit(0.3) + 0.5*x1 + (1|grp) ;  ICC 0.2 ;  30 clusters x 20
  formula = "y ~ x1 + (1|grp)", family = "logit",
  effects = "x1=0.50", baseline_probability = 0.3,
  cluster = list(var = "grp", ICC = 0.2, n_clusters = 30L, cluster_size = 20L),
  glmer_re = "(1 | g_primary)",
  n = 600L, seed = 2400
)

# losf 24 — single random slope, logistic: y ~ x1 + (1 + x1 | g)
glmm_slope <- list(
  label = "glmm_slope", losf = 24L,
  # logit(P) = logit(0.3) + 0.5*x1 + (1 + x1|grp) ;  ICC 0.2 ;  τ₁²=.10 ;  30 x 20
  formula = "y ~ x1 + (1|grp)", family = "logit",
  effects = "x1=0.50", baseline_probability = 0.3,
  cluster = list(var = "grp", ICC = 0.2, n_clusters = 30L, cluster_size = 20L),
  slopes = list(list(column = 0L, variance = 0.10, corr_with_intercept = 0.3, corr_with = numeric(0))),
  glmer_re = "(1 + x1 | g_primary)",
  n = 600L, seed = 2401
)

# losf 25 — multi-slope logistic: y ~ x1 + x2 + (1 + x1 + x2 | g)
glmm_multislope <- list(
  label = "glmm_multislope", losf = 25L,
  # logit(P) = logit(0.3) + 0.5*x1 + 0.3*x2 + (1+x1+x2|grp) ;  ICC 0.2 ;  40 x 20
  formula = "y ~ x1 + x2 + (1|grp)", family = "logit",
  effects = "x1=0.50, x2=0.30", baseline_probability = 0.3,
  cluster = list(var = "grp", ICC = 0.2, n_clusters = 40L, cluster_size = 20L),
  slopes = list(
    list(column = 0L, variance = 0.10, corr_with_intercept = 0.3, corr_with = numeric(0)),
    list(column = 1L, variance = 0.08, corr_with_intercept = 0.1, corr_with = c(0.2))
  ),
  glmer_re = "(1 + x1 + x2 | g_primary)",
  n = 800L, seed = 2402
)

# losf 26 — composition: (1 + x1 | grp) crossed with (1 | item), logistic
glmm_slope_crossed <- list(
  label = "glmm_slope_crossed", losf = 26L,
  # logit(P) = logit(0.3) + 0.5*x1 + (1+x1|grp) + (1|item) ;  ICC 0.2 ;  24 x 20
  formula = "y ~ x1 + (1|grp)", family = "logit",
  effects = "x1=0.50", baseline_probability = 0.3,
  cluster = list(var = "grp", ICC = 0.2, n_clusters = 24L, cluster_size = 20L),
  slopes = list(list(column = 0L, variance = 0.10, corr_with_intercept = 0.3, corr_with = numeric(0))),
  extra = list(list(var = "item", kind = "crossed", n_clusters = 6L, tau_squared = 0.16)),
  glmer_re = "(1 + x1 | g_primary) + (1 | g_item)",
  n = 480L, seed = 2403
)

# losf 27 — Laplace-bias cell: large τ², tiny clusters (n_j < 5). Cross-checked
# against glmer(nAGQ = 7), NOT the Laplace glmer. τ² ≈ 1.0 (ICC 0.5 on logit
# scale) with cluster_size = 4 puts the Laplace bias at a visually detectable
# level; the gate is GLMM_TOL$laplace_bias_beta_abs (0.05), not the tight B↔C band.
glmm_laplace_bias <- list(
  label = "glmm_laplace_bias", losf = 27L, nagq = 7L,
  formula = "y ~ x1 + (1|grp)", family = "logit",
  effects = "x1=0.50", baseline_probability = 0.3,
  cluster = list(var = "grp", ICC = 0.5, n_clusters = 150L, cluster_size = 4L),
  glmer_re = "(1 | g_primary)",
  n = 600L, seed = 2404
)

# losf 30 — EXTRA SLOPES (crossed, logit): (1+x1|grp) + (1+x1|item); slopes on both groupings
glmm_slope_crossed_extra_slopes <- list(
  label = "glmm_slope_crossed_extra_slopes", losf = 30L,
  # logit(P) = logit(0.3) + 0.5*x1 + (1+x1|grp) + (1+x1|item) ; ICC 0.2 ; 24 grp × 20, 6 items, n=480
  formula = "y ~ x1 + (1|grp)", family = "logit",
  effects = "x1=0.50", baseline_probability = 0.3,
  cluster = list(var = "grp", ICC = 0.2, n_clusters = 24L, cluster_size = 20L),
  slopes = list(list(column = 0L, variance = 0.10, corr_with_intercept = 0.3, corr_with = numeric(0))),
  extra = list(list(var = "item", kind = "crossed", n_clusters = 6L, tau_squared = 0.16,
                    slopes = list(list(column = 0L, variance = 0.06, corr_with_intercept = -0.2, corr_with = numeric(0))))),
  glmer_re = "(1 + x1 | g_primary) + (1 + x1 | g_item)",
  n = 480L, seed = 2405
)

# losf 31 — EXTRA SLOPES (nested, logit): (1+x1|grp) + (1+x1|class); slopes on both groupings
glmm_slope_nested_extra_slopes <- list(
  label = "glmm_slope_nested_extra_slopes", losf = 31L,
  # logit(P) = logit(0.3) + 0.5*x1 + (1+x1|grp) + (1+x1|class) ; ICC 0.2 ; 12 grp × 60, 3 nested, n=720
  formula = "y ~ x1 + (1|grp)", family = "logit",
  effects = "x1=0.50", baseline_probability = 0.3,
  cluster = list(var = "grp", ICC = 0.2, n_clusters = 12L, cluster_size = 60L),
  slopes = list(list(column = 0L, variance = 0.10, corr_with_intercept = 0.3, corr_with = numeric(0))),
  extra = list(list(var = "class", kind = "nested", n_clusters = 3L, tau_squared = 0.10,
                    slopes = list(list(column = 0L, variance = 0.05, corr_with_intercept = -0.2, corr_with = numeric(0))))),
  glmer_re = "(1 + x1 | g_primary) + (1 + x1 | g_class)",
  n = 720L, seed = 2406
)

M4_GLMM_CASES <- list(glmm_intercept, glmm_slope, glmm_multislope, glmm_slope_crossed, glmm_laplace_bias,
                      glmm_slope_crossed_extra_slopes, glmm_slope_nested_extra_slopes)
names(M4_GLMM_CASES) <- vapply(M4_GLMM_CASES, function(c) c$label, character(1))

# C4: M2/M3/M4 now flow through data_generation.r's save loop, so they gain the
# same content-hash provenance + frozen golden as the CASES catalogue.
# glmm_laplace_bias is excluded: cluster_size=4 deliberately violates the
# validator (the point of that case); section 8.5 of the rmd handles it inline.
EXTRA_CASES   <- c(M2_LMM_CASES, M3_LMM_CASES,
                   M4_GLMM_CASES[names(M4_GLMM_CASES) != "glmm_laplace_bias"])
ALL_GEN_CASES <- c(CASES, EXTRA_CASES)
names(EXTRA_CASES)   <- vapply(EXTRA_CASES,   function(c) c$label, character(1))
names(ALL_GEN_CASES) <- vapply(ALL_GEN_CASES, function(c) c$label, character(1))
