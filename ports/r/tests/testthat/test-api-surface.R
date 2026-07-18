test_that("parse_formula returns structured terms as JSON", {
  j <- mcpower:::parse_formula("y ~ x1 + x2 + x1:x2")
  parsed <- jsonlite::fromJSON(j, simplifyVector = FALSE)

  expect_true(is.list(parsed))
  expect_equal(parsed$dependent, "y")
  expect_equal(parsed$predictors, list("x1", "x2"))

  # term[1]: main effect x1
  expect_equal(parsed$terms[[1]]$kind, "main")
  expect_equal(parsed$terms[[1]]$name, "x1")

  # term[2]: main effect x2
  expect_equal(parsed$terms[[2]]$kind, "main")
  expect_equal(parsed$terms[[2]]$name, "x2")

  # term[3]: interaction x1:x2
  expect_equal(parsed$terms[[3]]$kind, "interaction")
  expect_equal(parsed$terms[[3]]$vars, list("x1", "x2"))

  expect_equal(length(parsed$random_effects), 0L)
})

test_that("parse_assignments returns structured assignments as JSON", {
  j <- mcpower:::parse_assignments(
    "x1=0.5, x2=0.3",
    "effect",
    '{"predictors":["x1","x2"],"interaction_terms":[]}'
  )
  parsed <- jsonlite::fromJSON(j, simplifyVector = FALSE)

  expect_true(is.list(parsed))
  expect_equal(length(parsed$items), 2L)
  expect_equal(length(parsed$errors), 0L)

  # item[1]: key={name:"x1"}, value={effect:0.5}
  expect_equal(parsed$items[[1]]$key$name, "x1")
  expect_equal(parsed$items[[1]]$value$effect, 0.5)

  # item[2]: key={name:"x2"}, value={effect:0.3}
  expect_equal(parsed$items[[2]]$key$name, "x2")
  expect_equal(parsed$items[[2]]$value$effect, 0.3)
})

# --- RVariableRegistry (A8) — mirror of VariableRegistry + parsers reshaping ---

test_that("registry reshapes a simple formula like Python", {
  reg <- mcpower:::RVariableRegistry$new("y ~ x1 + x2")
  expect_equal(reg$non_factor_names, c("x1", "x2"))
  expect_equal(reg$factor_names, character(0))
  expect_equal(sort(reg$effect_names), c("x1", "x2"))
})

test_that("registry exposes equation and dependent", {
  reg <- mcpower:::RVariableRegistry$new("y ~ x1 + x2")
  expect_equal(reg$equation, "y ~ x1 + x2")
  expect_equal(reg$dependent, "y")
  expect_equal(reg$predictor_names, c("x1", "x2"))
})

test_that("registry reshapes interactions into effects", {
  reg <- mcpower:::RVariableRegistry$new("y ~ x1 + x2 + x1:x2")
  expect_equal(reg$effect_names, c("x1", "x2", "x1:x2"))
  eff <- reg$get_effect("x1:x2")
  expect_equal(eff$effect_type, "interaction")
  expect_equal(unlist(eff$var_names), c("x1", "x2"))
  expect_equal(eff$column_indices, c(0L, 1L))
})

test_that("registry tolerates = as ~ synonym", {
  reg <- mcpower:::RVariableRegistry$new("y = x1 + x2")
  expect_equal(reg$dependent, "y")
  expect_equal(reg$non_factor_names, c("x1", "x2"))
})

test_that("registry get_predictor exposes var_type and proportion", {
  reg <- mcpower:::RVariableRegistry$new("y ~ x1")
  p <- reg$get_predictor("x1")
  expect_equal(p$var_type, "normal")
  expect_equal(p$proportion, 0.5)
})

test_that("registry set_variable_types handles a binary predictor", {
  reg <- mcpower:::RVariableRegistry$new("y ~ x1 + x2")
  reg$set_variable_types("x1=(binary,0.3)")
  p <- reg$get_predictor("x1")
  expect_equal(p$var_type, "binary")
  expect_equal(p$proportion, 0.3)
})

test_that("registry handles a factor predictor", {
  reg <- mcpower:::RVariableRegistry$new("y ~ g")
  reg$set_variable_types("g=(factor,3)")
  expect_equal(reg$factor_names, "g")
  expect_equal(reg$`_factors`[["g"]]$n_levels, 3L)
  expect_equal(reg$`_factors`[["g"]]$reference_level, 1L)
  expect_false(reg$get_predictor("g") |> is.null())
  # non_factor_names excludes the factor.
  expect_equal(reg$non_factor_names, character(0))
})

test_that("registry expand_factors creates reference-coded dummies", {
  reg <- mcpower:::RVariableRegistry$new("y ~ x1 + g")
  reg$set_variable_types("g=(factor,3)")
  reg$expand_factors()
  expect_equal(reg$dummy_names, c("g[2]", "g[3]"))
  # Original factor predictor removed; x1 kept, dummies appended.
  expect_equal(reg$predictor_names, c("x1", "g[2]", "g[3]"))
  expect_true("g[2]" %in% reg$effect_names)
  expect_equal(reg$get_effect("g[2]")$column_index, 1L)
})

test_that("registry expand_factors expands factor interactions", {
  reg <- mcpower:::RVariableRegistry$new("y ~ x1 + g + x1:g")
  reg$set_variable_types("g=(factor,3)")
  reg$expand_factors()
  expect_true("x1:g[2]" %in% reg$effect_names)
  expect_true("x1:g[3]" %in% reg$effect_names)
})

test_that("registry set_effects assigns effect sizes in order", {
  reg <- mcpower:::RVariableRegistry$new("y ~ x1 + x2")
  reg$set_effects("x1=0.5, x2=0.3")
  expect_equal(reg$get_effect("x1")$effect_size, 0.5)
  expect_equal(reg$get_effect("x2")$effect_size, 0.3)
  expect_equal(reg$get_effect_sizes(), c(0.5, 0.3))
})

test_that("registry set_effects rejects unknown effects like Python", {
  reg <- mcpower:::RVariableRegistry$new("y ~ x1")
  expect_error(reg$set_effects("zzz=0.5"), "Effect validation failed")
})

test_that("registry builds correlation matrix from spec", {
  reg <- mcpower:::RVariableRegistry$new("y ~ x1 + x2 + x3")
  reg$set_correlation_spec("corr(x1,x2)=0.3")
  m <- reg$get_correlation_matrix()
  expect_equal(dim(m), c(3L, 3L))
  expect_equal(m[1, 2], 0.3)
  expect_equal(m[2, 1], 0.3)
  expect_equal(m[1, 3], 0.0)
  expect_equal(diag(m), c(1, 1, 1))
})

test_that("registry bare-parens correlation form normalises like Python", {
  reg <- mcpower:::RVariableRegistry$new("y ~ x1 + x2")
  reg$set_correlation_spec("(x1,x2)=0.4")
  m <- reg$get_correlation_matrix()
  expect_equal(m[1, 2], 0.4)
})

test_that("registry detects random intercept and cluster effects", {
  reg <- mcpower:::RVariableRegistry$new("y ~ x1 + (1|g)")
  expect_equal(length(reg$`_random_effects_parsed`), 1L)
  re <- reg$`_random_effects_parsed`[[1]]
  expect_equal(re$type, "random_intercept")
  expect_equal(re$grouping_var, "g")
  # Fixed effect still present.
  expect_equal(reg$effect_names, "x1")
})

test_that("registry detects nested random intercepts", {
  reg <- mcpower:::RVariableRegistry$new("y ~ x1 + (1|a/b)")
  res <- reg$`_random_effects_parsed`
  expect_equal(length(res), 2L)
  expect_equal(res[[1]]$grouping_var, "a")
  expect_null(res[[1]]$parent_var)
  expect_equal(res[[2]]$grouping_var, "a:b")
  expect_equal(res[[2]]$parent_var, "a")
})

test_that("registry rejects random-effects-only formula", {
  expect_error(mcpower:::RVariableRegistry$new("y ~ (1|g)"))
})

test_that("correction aliases normalize like Python", {
  expect_equal(mcpower:::.correction_for_rust("BH"), "benjamini_hochberg")
  expect_equal(mcpower:::.correction_for_rust("Benjamini-Hochberg"), "benjamini_hochberg")
  expect_equal(mcpower:::.correction_for_rust("fdr"), "benjamini_hochberg")
  expect_equal(mcpower:::.correction_for_rust(NULL), "none")
  expect_equal(mcpower:::.correction_for_rust("holm"), "holm")
})
test_that("residual codes cover all five canonical names; aliases removed", {
  # All five canonical names are valid.
  expect_equal(mcpower:::.residual_code("normal"), 0L)
  expect_equal(mcpower:::.residual_code("right_skewed"), 2L)
  expect_equal(mcpower:::.residual_code("left_skewed"), 3L)
  expect_equal(mcpower:::.residual_code("high_kurtosis"), 4L)
  expect_equal(mcpower:::.residual_code("uniform"), 5L)
  # Old aliases t/heavy_tailed/skewed are gone.
  expect_error(mcpower:::.residual_code("t"))
  expect_error(mcpower:::.residual_code("heavy_tailed"))
  expect_error(mcpower:::.residual_code("skewed"))
})

test_that("build_contract_from_spec returns names + raw contract bytes", {
  # Known-good LinearSpec JSON matching the new wire contract:
  # - heterogeneity removed (scenario-only)
  # - heteroskedasticity: driver_var_index only (no ratio)
  # - residual: distribution + optional pinned (no df)
  # - residual_dists: [4,2] (high_kurtosis=4, right_skewed=2, matching configs/scenarios.json defaults)
  spec_json <- paste0(
    '{"formula":"y ~ x1",',
    '"predictors":[{"name":"x1","kind":"normal"}],',
    '"effects":[{"name":"x1","size":0.5}],',
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

  out <- mcpower:::build_contract_from_spec(spec_json, "continuous", "canonical", "ols", 0.0, "[]")

  expect_type(out$contracts, "raw")
  expect_true(length(out$names) >= 1)
  expect_equal(out$names[[1]], "optimistic")
})

# --- .to_linear_spec_list + .scenario_dict + .encode_outcome_and_clusters (A9) ---

test_that("to_linear_spec_json emits the documented field set", {
  reg <- mcpower:::RVariableRegistry$new("y ~ x1"); reg$set_effects("x1=0.5")
  payload <- mcpower:::.to_linear_spec_list(reg, scenario_names = "optimistic",
               alpha = 0.05, correction = NULL, target_test = NULL,
               heteroskedasticity = list(driver_var_index = NULL),
               residual_name = "normal", max_failed = 1.0,
               test_formula = NULL)
  # heterogeneity removed from the top-level payload (scenario-only now).
  expect_setequal(names(payload), c("formula","predictors","effects","correlations",
    "alpha","correction","wald_se","nagq","targets","report_overall","contrast_pairs","posthoc_requests",
    "heteroskedasticity","residual","max_failed_fraction","scenarios"))
  expect_equal(payload$correction, "none")
  expect_equal(payload$targets, list("overall"))
  expect_true(payload$report_overall)
})

test_that("to_linear_spec_json emits the new wire shape for OLS y ~ x1", {
  reg <- mcpower:::RVariableRegistry$new("y ~ x1"); reg$set_effects("x1=0.5")
  payload <- mcpower:::.to_linear_spec_list(reg, scenario_names = "optimistic",
               alpha = 0.05, correction = NULL, target_test = NULL,
               heteroskedasticity = list(driver_var_index = NULL),
               residual_name = "normal", max_failed = 1.0,
               test_formula = NULL)
  j <- jsonlite::toJSON(payload, auto_unbox = TRUE, null = "null", digits = NA)
  reparsed <- jsonlite::fromJSON(j, simplifyVector = FALSE)
  # Scenario integer code lists survive as numbers (not strings).
  # residual_dists default = [4,2] (high_kurtosis=4, right_skewed=2).
  expect_equal(reparsed$scenarios[[1]]$new_distributions, list(2L, 3L, 5L))
  expect_equal(reparsed$scenarios[[1]]$residual_dists, list(4L, 2L))
  expect_equal(reparsed$scenarios[[1]]$residual_df, 10)
  expect_equal(reparsed$effects[[1]]$size, 0.5)
  # residual block: distribution only (no df); pinned absent when FALSE.
  expect_equal(reparsed$residual$distribution, "normal")
  expect_null(reparsed$residual[["df"]])
  expect_equal(reparsed$predictors[[1]], list(name = "x1", kind = "normal"))
  expect_equal(reparsed$contrast_pairs, list())
})

test_that("to_linear_spec_json reshapes a factor predictor into levels/proportions/reference", {
  reg <- mcpower:::RVariableRegistry$new("y ~ f1")
  reg$set_variable_types("f1=(factor,3)")
  reg$expand_factors()
  reg$set_effects("f1[2]=0.5, f1[3]=0.3")
  payload <- mcpower:::.to_linear_spec_list(reg, scenario_names = "optimistic",
               alpha = 0.05, correction = NULL, target_test = NULL,
               heteroskedasticity = list(driver_var_index = NULL),
               residual_name = "normal", max_failed = 1.0,
               test_formula = NULL)
  pred <- payload$predictors[[1]]
  expect_equal(pred$name, "f1")
  expect_equal(pred$kind, "factor")
  expect_equal(as.character(pred$levels), c("1", "2", "3"))
  expect_equal(as.numeric(pred$proportions), c(1/3, 1/3, 1/3))
  expect_equal(pred$reference, "1")
  # Effects are the expanded dummies, not the bare factor name.
  eff_names <- vapply(payload$effects, function(e) e$name, character(1))
  expect_equal(eff_names, c("f1[2]", "f1[3]"))
})

test_that("to_linear_spec_json emits non-zero correlations only", {
  reg <- mcpower:::RVariableRegistry$new("y ~ x1 + x2")
  reg$set_effects("x1=0.5, x2=0.3")
  reg$set_correlation_spec("(x1,x2)=0.4")
  payload <- mcpower:::.to_linear_spec_list(reg, scenario_names = "optimistic",
               alpha = 0.05, correction = NULL, target_test = NULL,
               heteroskedasticity = list(driver_var_index = NULL),
               residual_name = "normal", max_failed = 1.0,
               test_formula = NULL)
  expect_equal(length(payload$correlations), 1L)
  expect_equal(payload$correlations[[1]], list(a = "x1", b = "x2", value = 0.4))
})

test_that("public set_correlations accepts a matrix, matching the equivalent string spec", {
  M <- matrix(c(1.0, 0.3, 0.4,
                0.3, 1.0, 0.2,
                0.4, 0.2, 1.0), nrow = 3, byrow = TRUE)
  m <- MCPower$new("y ~ x1 + x2 + x3")
  m$set_effects("x1=0.3, x2=0.25, x3=0.35")
  m$set_correlations(M)
  m$.__enclos_env__$private$apply()
  expect_equal(m$.__enclos_env__$private$registry$get_correlation_matrix(), M)

  # The equivalent string spec must build the same matrix.
  s <- MCPower$new("y ~ x1 + x2 + x3")
  s$set_effects("x1=0.3, x2=0.25, x3=0.35")
  s$set_correlations("corr(x1,x2)=0.3, corr(x1,x3)=0.4, corr(x2,x3)=0.2")
  s$.__enclos_env__$private$apply()
  expect_equal(s$.__enclos_env__$private$registry$get_correlation_matrix(), M)
})

test_that("public set_correlations matrix validation mirrors the Python port", {
  mk <- function() {
    m <- MCPower$new("y ~ x1 + x2 + x3")
    m$set_effects("x1=0.3, x2=0.25, x3=0.35")
    m
  }
  bad_dim <- mk(); bad_dim$set_correlations(matrix(c(1, 0.2, 0.2, 1), nrow = 2))
  expect_error(bad_dim$.__enclos_env__$private$apply(),
               "Matrix shape \\(2, 2\\) doesn't match 3 non-factor variables")

  asym <- mk(); asym$set_correlations(matrix(c(1, 0.3, 0.4,
                                               0.1, 1, 0.2,
                                               0.4, 0.2, 1), nrow = 3, byrow = TRUE))
  expect_error(asym$.__enclos_env__$private$apply(), "must be symmetric")

  bad_diag <- mk(); bad_diag$set_correlations(matrix(c(0.9, 0.3, 0.4,
                                                       0.3, 1, 0.2,
                                                       0.4, 0.2, 1), nrow = 3, byrow = TRUE))
  expect_error(bad_diag$.__enclos_env__$private$apply(), "Diagonal elements")

  # Neither string nor matrix is rejected at set time.
  expect_error(mk()$set_correlations(5L), "string or matrix")
})

test_that(".scenario_dict encodes default scenarios with integer code lists", {
  s <- mcpower:::.scenario_dict("optimistic")
  expect_equal(s$name, "optimistic")
  expect_equal(as.integer(s$new_distributions), c(2L, 3L, 5L))
  # residual_dists default = [high_kurtosis=4, right_skewed=2] per configs/scenarios.json.
  expect_equal(as.integer(s$residual_dists), c(4L, 2L))
  expect_equal(s$residual_df, 10)
  # All five canonical residual names are valid in scenario pools.
  expect_no_error(mcpower:::.encode_residual_list(c("left_skewed"), "optimistic"))
  expect_no_error(mcpower:::.encode_residual_list(c("high_kurtosis"), "optimistic"))
  # Unknown names still error.
  expect_error(mcpower:::.encode_residual_list(c("zzz"), "optimistic"))
})

test_that(".scenario_dict carries sampled_factor_proportions default and override", {
  # Presets ride the engine-embedded configs/scenarios.json.
  expect_false(mcpower:::.scenario_dict("optimistic")$sampled_factor_proportions)
  expect_true(mcpower:::.scenario_dict("realistic")$sampled_factor_proportions)
  # Knob omitted in a custom config -> exact-allocation default (FALSE).
  d <- mcpower:::.scenario_dict("custom", list(custom = list()))
  expect_false(d$sampled_factor_proportions)
  # Explicit override flows through.
  d2 <- mcpower:::.scenario_dict("custom", list(custom = list(sampled_factor_proportions = TRUE)))
  expect_true(d2$sampled_factor_proportions)
})

test_that(".encode_outcome_and_clusters maps ols / logit / lme", {
  ols <- mcpower:::.encode_outcome_and_clusters("normal", "canonical", "ols", 0.0, list())
  expect_equal(ols$outcome_kind, "continuous")
  expect_equal(ols$estimator, "ols")
  expect_equal(ols$clusters_json, "[]")

  logit <- mcpower:::.encode_outcome_and_clusters("logit", "canonical", "glm", 0.0, list())
  expect_equal(logit$outcome_kind, "binary")
  expect_equal(logit$estimator, "glm")
  expect_equal(logit$clusters_json, "[]")

  lme <- mcpower:::.encode_outcome_and_clusters("lme", "canonical", "mle", 0.0,
            list(g = list(icc = 0.1, n_clusters = 20)))
  expect_equal(lme$outcome_kind, "continuous")
  expect_equal(lme$estimator, "mle")
  reparsed <- jsonlite::fromJSON(lme$clusters_json, simplifyVector = FALSE)
  expect_equal(reparsed[[1]]$sizing$FixedClusters$n_clusters, 20L)
  expect_equal(reparsed[[1]]$tau_squared, 0.1 / 0.9)
})

# ---------------------------------------------------------------------------
# MCPower R6 class — constructor, setters, summary (A10)
# ---------------------------------------------------------------------------

test_that("MCPower constructor sets defaults mirroring Python", {
  m <- MCPower$new("y ~ x1")
  expect_equal(m$family, "ols")
  expect_equal(m$estimator, "ols")
  expect_equal(m$outcome_kind, "continuous")
  expect_equal(m$seed, 2137)
  expect_equal(m$power, 80)
  expect_equal(m$alpha, 0.05)
  expect_equal(m$n_simulations, 1600)
})

test_that("MCPower estimator defaults derive from family", {
  expect_equal(MCPower$new("y ~ x1", family = "logit")$estimator, "glm")
  expect_equal(MCPower$new("y ~ x1 + (1|g)", family = "lme")$estimator, "mle")
  expect_equal(MCPower$new("y ~ x1", family = "ols")$estimator, "ols")
  # logit -> binary outcome
  expect_equal(MCPower$new("y ~ x1", family = "logit")$outcome_kind, "binary")
  # estimator override wins
  expect_equal(MCPower$new("y ~ x1 + (1|g)", family = "lme", estimator = "ols")$estimator, "ols")
  # solve_as alias
  expect_equal(MCPower$new("y ~ x1", family = "ols", solve_as = "glm")$estimator, "glm")
})

test_that("MCPower accepts an R formula object", {
  m <- MCPower$new(y ~ x1 + x2)
  expect_equal(m$equation, "y ~ x1 + x2")
})

test_that("MCPower setters store fields and chain", {
  m <- MCPower$new("y ~ x1")
  expect_identical(m$set_seed(99L), m)        # returns invisible(self)
  expect_equal(m$seed, 99L)
  m$set_power(90)
  expect_equal(m$power, 90)
  m$set_alpha(0.01)
  expect_equal(m$alpha, 0.01)
  m$set_simulations(500)
  expect_equal(m$n_simulations, 500)
  m$set_max_failed_simulations(0.2)
  expect_equal(m$max_failed_simulations, 0.2)
})

test_that("MCPower set_effects forwards to the registry", {
  m <- MCPower$new("y ~ x1 + x2")$set_effects("x1=0.5, x2=0.3")
  s <- m$summary()
  expect_equal(s$effects$x1, 0.5)
  expect_equal(s$effects$x2, 0.3)
})

test_that("MCPower set_residual_distribution validates and stores; all five canonical names accepted", {
  # All five canonical names are now valid.
  m <- MCPower$new("y ~ x1")$set_residual_distribution("high_kurtosis")
  s <- m$summary()
  expect_equal(s$residual_distribution, "high_kurtosis")
  expect_true(s$residual_pinned)
  # left_skewed is now also valid (no longer blocked).
  expect_no_error(MCPower$new("y ~ x1")$set_residual_distribution("left_skewed"))
  expect_no_error(MCPower$new("y ~ x1")$set_residual_distribution("uniform"))
  # Unknown names still error.
  expect_error(MCPower$new("y ~ x1")$set_residual_distribution("zzz"))
  # Setting "normal" explicitly marks as pinned.
  m2 <- MCPower$new("y ~ x1")$set_residual_distribution("normal")
  expect_true(m2$summary()$residual_pinned)
  # Default (unset) is not pinned.
  expect_false(MCPower$new("y ~ x1")$summary()$residual_pinned)
})

test_that("MCPower set_baseline_probability computes intercept = logit(p)", {
  m <- MCPower$new("y ~ x1", family = "logit")$set_baseline_probability(0.3)
  m$summary()  # forces apply
  expect_equal(m$intercept, log(0.3 / 0.7))
})

test_that("MCPower set_cluster shapes pending state for encode", {
  m <- MCPower$new("y ~ x1 + (1|g)", family = "lme")$set_cluster("g", ICC = 0.1, n_clusters = 20)
  pc <- m$.__enclos_env__$private$pending_clusters
  expect_equal(pc$g$icc, 0.1)
  expect_equal(pc$g$n_clusters, 20)
})

test_that("MCPower set_heteroskedasticity_driver builds the engine spec", {
  m <- MCPower$new("y ~ x1 + x2")
  m$set_heteroskedasticity_driver()  # off -> null driver (ratio is scenario-only)
  hs0 <- m$.__enclos_env__$private$heteroskedasticity
  expect_null(hs0$driver_var_index)
  # ratio key is gone (λ is scenario-only)
  expect_null(hs0[["ratio"]])
  m$set_heteroskedasticity_driver("x1")
  hs <- m$.__enclos_env__$private$heteroskedasticity
  expect_equal(hs$driver_var_index, 0L)
})

# Item #13 — set_heteroskedasticity_driver silent no-op for logit/LME
test_that("set_heteroskedasticity_driver is a no-op for logit: same power with and without", {
  # Logit model with explicit seed for determinism.
  make_logit <- function() {
    MCPower$new("y ~ x1", family = "logit")$
      set_effects("x1=0.5")$
      set_baseline_probability(0.3)$
      set_simulations(200)
  }
  res_plain <- make_logit()$find_power(200, seed = 2137, progress_callback = FALSE)
  res_het   <- make_logit()$
    set_heteroskedasticity_driver("x1")$
    find_power(200, seed = 2137, progress_callback = FALSE)
  # The engine ignores het for logit DGP: results must be byte-identical.
  expect_identical(res_plain$power_uncorrected, res_het$power_uncorrected)
})

test_that("set_heteroskedasticity_driver is a no-op for LME: same power with and without", {
  make_lme <- function() {
    MCPower$new("y ~ x1 + (1|g)", family = "lme")$
      set_effects("x1=0.5")$
      set_cluster("g", ICC = 0.1, n_clusters = 20)$
      set_simulations(100)
  }
  res_plain <- make_lme()$find_power(200, seed = 2137, progress_callback = FALSE)
  res_het   <- make_lme()$
    set_heteroskedasticity_driver("x1")$
    find_power(200, seed = 2137, progress_callback = FALSE)
  expect_identical(res_plain$power_uncorrected, res_het$power_uncorrected)
})

test_that("summary returns the documented keys", {
  m <- MCPower$new("y ~ x1")$set_effects("x1=0.5")
  s <- m$summary()
  # residual_df removed (df is scenario-only); residual_pinned added.
  expect_setequal(names(s), c("formula","family","outcome_kind","estimator","effects",
    "n_simulations","alpha","power_target","residual_distribution","residual_pinned","scenarios"))
})

# ── upload_data / get_effects_from_data tests (E2) ───────────────────────────

test_that("mode='strict' is accepted and stores the upload (Phase 2 — bootstrap whole rows)", {
  m <- MCPower$new("y ~ hp + wt")
  df <- data.frame(hp = rnorm(50), wt = rnorm(50))
  # strict mode must store pending_data with the strict mode + row count (was:
  # expect_no_error only — a silent no-op store would have passed).
  m$upload_data(df, mode = "strict", verbose = FALSE)
  pd <- m$.__enclos_env__$private$pending_data
  expect_false(is.null(pd))
  expect_equal(pd$uploaded_n, 50L)
  expect_equal(pd$mode, "strict")
})

test_that("mode='bad' raises", {
  m <- MCPower$new("y ~ hp + wt")
  df <- data.frame(hp = rnorm(50), wt = rnorm(50))
  expect_error(m$upload_data(df, mode = "bad"), regexp = "mode must be one of")
})

test_that("upload_data stores pending_data and the contract carries upload block", {
  set.seed(1)
  df <- data.frame(hp = rnorm(50), wt = rnorm(50), am = rbinom(50, 1, 0.5))
  m <- MCPower$new("y ~ hp + wt + am")
  m$upload_data(df, mode = "partial", verbose = FALSE)
  m$set_effects("hp=-0.3, wt=-0.5, am=0.4")

  # Drive the real upload_data → build_contract path by pulling pending_data from
  # the actual model private state and passing it to .to_linear_spec_list — the same
  # path build_contract_bytes takes.  This exercises upload_data detection, registry
  # update, and spec serialization end-to-end.
  private_env <- m$.__enclos_env__$private
  if (!private_env$applied) private_env$apply()
  reg          <- private_env$registry
  pending_data <- private_env$pending_data

  payload <- mcpower:::.to_linear_spec_list(
    reg,
    scenario_names     = "optimistic",
    alpha              = m$alpha,
    correction         = NULL,
    target_test        = NULL,
    heteroskedasticity = list(driver_var_index = NULL),
    residual_name      = "normal",
    max_failed         = m$max_failed_simulations,
    test_formula       = NULL,
    pending_data       = pending_data
  )

  # Upload block must be present with the correct shape.
  expect_false(is.null(payload$upload))
  expect_equal(payload$upload$mode, "partial")
  expect_equal(payload$upload$n_rows, 50L)
  # All three predictors (hp, wt, am) must appear in the upload columns.
  col_names <- vapply(payload$upload$columns, `[[`, character(1), "name")
  expect_true("hp" %in% col_names)
  expect_true("wt" %in% col_names)
  expect_true("am" %in% col_names)
  # am should be detected as binary (2 distinct values).
  am_entry <- payload$upload$columns[[which(col_names == "am")]]
  expect_equal(am_entry$col_type, "binary")
  # hp and wt should be continuous.
  hp_entry <- payload$upload$columns[[which(col_names == "hp")]]
  expect_equal(hp_entry$col_type, "continuous")
})

test_that("upload_data detects factor columns and updates registry", {
  # 60 rows, 3 distinct values → ratio 20 >= 15 → factor detection
  cyl_vals <- rep(c(4, 6, 8), 20)  # 60 rows, ratio = 20
  df <- data.frame(cyl = cyl_vals,
                   hp  = rnorm(60),
                   stringsAsFactors = FALSE)
  m <- MCPower$new("y ~ cyl + hp")
  m$upload_data(df, mode = "partial", verbose = FALSE)
  # After upload_data, cyl should be a factor in the registry.
  reg <- m$.__enclos_env__$private$registry
  expect_equal(reg$factor_names, "cyl")
  lbls <- reg$`_factors`[["cyl"]]$level_labels
  expect_equal(sort(unlist(lbls)), c("4", "6", "8"))
})
