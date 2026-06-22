# R mirror of the Python APIC chained-surface property tests
# (mcpower/ports/py/tests/test_apic_surface.py). Each block asserts the SAME
# behavior (shape / labels / order / error-kind) as the Python canonical
# reference, adapted to R idiom — never a pinned numeric power value.
#
# The R port deliberately implements a thinner surface than Python: it has no
# standalone `_validate_*` functions (validation is inline in the constructor /
# setters) and no upload_data / to_simulation_spec / snap-to-clusters /
# _check_failure_threshold / _warn_logit_effect_scale. Where R reaches a
# behavior through a different (but equivalent) path, the mirror targets that
# path. Behaviors with no R equivalent are recorded as mirror-gaps in the
# ledger fragment, not faked here.

# --- Constructor validation (APIC-02 / APIC-03 / APIC-27 / APIC-28) ---------

test_that("constructor rejects non-string family (APIC-02)", {
  expect_error(MCPower$new("y ~ x1", family = 5))
  expect_error(MCPower$new("y ~ x1", family = c("ols", "logit")))
})

test_that("constructor accepts NULL estimator and rejects non-canonical (APIC-03)", {
  # NULL estimator derives the family default; canonical strings accepted.
  expect_equal(MCPower$new("y ~ x1", estimator = NULL)$estimator, "ols")
  expect_equal(MCPower$new("y ~ x1", estimator = "glm")$estimator, "glm")
  # case-insensitive
  expect_equal(MCPower$new("y ~ x1", estimator = "GLM")$estimator, "glm")
  expect_error(MCPower$new("y ~ x1", estimator = "zzz"))
  expect_error(MCPower$new("y ~ x1", estimator = 7))
})

# --- set_seed validation (APIC-29) -----------------------------------------

test_that("set_seed rejects negative and non-numeric; accepts NULL (APIC-29)", {
  expect_error(MCPower$new("y ~ x1")$set_seed(-1))
  expect_error(MCPower$new("y ~ x1")$set_seed("abc"))
  m <- MCPower$new("y ~ x1")$set_seed(NULL)
  expect_null(m$seed)
})

# --- set_max_failed_simulations range (APIC-77) ----------------------------

test_that("set_max_failed_simulations rejects values outside [0,1] (APIC-77)", {
  expect_error(MCPower$new("y ~ x1")$set_max_failed_simulations(2))
  expect_error(MCPower$new("y ~ x1")$set_max_failed_simulations(-0.1))
  m <- MCPower$new("y ~ x1")$set_max_failed_simulations(0.0)
  expect_equal(m$max_failed_simulations, 0.0)
})

# --- set_heteroskedasticity_driver (APIC-74) ------------------------------------

test_that("set_heteroskedasticity_driver: default (no args) -> null driver; bad var raises (APIC-74)", {
  m <- MCPower$new("y ~ x1 + x2")
  m$set_heteroskedasticity_driver()  # off -> null driver (λ is scenario-only)
  hs <- m$.__enclos_env__$private$heteroskedasticity
  expect_null(hs$driver_var_index)
  # ratio key is gone (scenario-only)
  expect_null(hs[["ratio"]])
  expect_error(MCPower$new("y ~ x1")$set_heteroskedasticity_driver("zzz"))
})

# --- set_scenario_configs non-dict (APIC-44) -------------------------------

test_that("set_scenario_configs rejects a non-list argument (APIC-44)", {
  expect_error(MCPower$new("y ~ x1")$set_scenario_configs("notalist"))
  expect_error(MCPower$new("y ~ x1")$set_scenario_configs(list(custom = "notalist")))
})

# --- set_scenario_configs key validation ------------------------------------

test_that("set_scenario_configs rejects unknown keys with the key named", {
  m <- MCPower$new("y ~ x1")
  # A typo'd knob must error, naming the key — silently no-oping would
  # produce unperturbed "scenarios".
  expect_error(
    m$set_scenario_configs(list(realistic = list(heterogenity = 0.2))),
    regexp = "unknown config key.*heterogenity"
  )
  # `heteroskedasticity` was the formerly documented name of
  # `heteroskedasticity_ratio`; the old key must error, not silently no-op.
  expect_error(
    m$set_scenario_configs(list(realistic = list(heteroskedasticity = 2.0))),
    regexp = "unknown config key.*heteroskedasticity"
  )
})

test_that("set_scenario_configs accepts mixed-model knobs (Pass B wiring)", {
  # The LME scenario knobs are now wired; user overrides are accepted, no
  # longer gated with "not yet supported" (mirrors Python set_scenario_configs).
  m <- MCPower$new("y ~ x1")
  for (key in c("icc_noise_sd", "random_effect_dist", "random_effect_df")) {
    cfg <- list(realistic = stats::setNames(list(0.1), key))
    m$set_scenario_configs(cfg)
    stored <- m$.__enclos_env__$private$scenario_configs[["realistic"]]
    expect_identical(stored[[key]], 0.1, info = key)
  }
})

test_that("set_scenario_configs accepts every live key", {
  lme_keys <- c("icc_noise_sd", "random_effect_dist", "random_effect_df")
  live_cfg <- mcpower:::.scenario_defaults()[["doomer"]]
  live_cfg <- live_cfg[setdiff(names(live_cfg), lme_keys)]
  expect_gt(length(live_cfg), 0L)
  m <- MCPower$new("y ~ x1")
  m$set_scenario_configs(list(custom = live_cfg))
  stored <- m$.__enclos_env__$private$scenario_configs[["custom"]]
  for (k in names(live_cfg)) {
    expect_identical(stored[[k]], live_cfg[[k]], info = k)
  }
})

# --- set_residual_distribution whitelist (APIC-73) -------------------------

test_that("set_residual_distribution stores the name and aliases are rejected (APIC-73)", {
  # State check for right_skewed (state checks for the other four canonical
  # names live in test-api-surface.R:443 — duplicates removed).
  m <- MCPower$new("y ~ x1")$set_residual_distribution("right_skewed")
  expect_equal(m$summary()$residual_distribution, "right_skewed")
  # Old aliases are now rejected.
  expect_error(MCPower$new("y ~ x1")$set_residual_distribution("heavy_tailed"))
  expect_error(MCPower$new("y ~ x1")$set_residual_distribution("skewed"))
  expect_error(MCPower$new("y ~ x1")$set_residual_distribution("t"))
})

# --- _resolve_scenarios_arg empty list (APIC-43) ---------------------------

test_that("scenarios = empty character vector raises (APIC-43)", {
  m <- MCPower$new("y ~ x1")$set_effects("x1=0.5")
  expect_error(m$find_power(80, scenarios = character(0), n_sims = 50, progress_callback = FALSE))
})

# --- _parse_equation triple + RE tagging + = synonym (APIC-18 / APIC-20) ---

test_that(".parse_equation returns (dependent, fixed, random_effects) with RE tagging (APIC-18)", {
  pe <- mcpower:::.parse_equation("y ~ x1 + (1|g)")
  expect_equal(pe$dependent, "y")
  expect_equal(pe$fixed_formula, "x1")
  expect_equal(length(pe$random_effects), 1L)
  expect_equal(pe$random_effects[[1]]$type, "random_intercept")
  expect_equal(pe$random_effects[[1]]$grouping_var, "g")
})

test_that(".parse_equation accepts '=' as a synonym for '~' (APIC-20)", {
  pe <- mcpower:::.parse_equation("y = x1 + x2")
  expect_equal(pe$dependent, "y")
  expect_equal(pe$fixed_formula, "x1+x2")
  expect_equal(length(pe$random_effects), 0L)
})

# --- _parse_independent_variables shape (APIC-19) --------------------------

test_that(".parse_independent_variables yields main/interaction effect shapes (APIC-19)", {
  piv <- mcpower:::.parse_independent_variables("x1 + x2 + x1:x2")
  expect_equal(unlist(piv$variables), c("x1", "x2"))
  # main effects carry column_index
  expect_equal(piv$effects[[1]]$type, "main")
  expect_equal(piv$effects[[1]]$column_index, 0L)
  # interaction effects carry var_names + column_indices
  expect_equal(piv$effects[[3]]$type, "interaction")
  expect_equal(unlist(piv$effects[[3]]$var_names), c("x1", "x2"))
  expect_equal(piv$effects[[3]]$column_indices, c(0L, 1L))
})

# --- split_assignments top-level comma + unbalanced (APIC-21) --------------

test_that("split_assignments splits top-level commas only; unbalanced ')' raises (APIC-21)", {
  parts <- mcpower:::split_assignments("x1=(binary,0.3), x2=0.5")
  expect_equal(parts, c("x1=(binary,0.3)", "x2=0.5"))
  expect_error(mcpower:::split_assignments("a=)b"))
})

# --- _normalise_correlation_input (APIC-22) --------------------------------

test_that(".normalise_correlation_input rewrites bare (a,b)=v; passes through prefixed (APIC-22)", {
  expect_equal(mcpower:::.normalise_correlation_input("(x1,x2)=0.4"), "corr(x1,x2)=0.4")
  expect_equal(mcpower:::.normalise_correlation_input("corr(x1,x2)=0.4"), "corr(x1,x2)=0.4")
})

# --- assignment-to-legacy key shapes (APIC-23) -----------------------------

test_that(".parse_assignment_kind maps correlation pairs to sorted pair; effects to name keys (APIC-23)", {
  corr <- mcpower:::.parse_assignment_kind("corr(x2,x1)=0.3", "correlation", c("x1", "x2"))
  # pair is sorted regardless of input order
  expect_equal(corr$parsed[[1]]$pair, c("x1", "x2"))
  expect_equal(corr$parsed[[1]]$value, 0.3)
  eff <- mcpower:::.parse_assignment_kind("x1=0.5", "effect", c("x1", "x2"))
  expect_equal(eff$parsed[["x1"]], 0.5)
})

# --- variable_type bare defaults (APIC-24) ---------------------------------

test_that("variable_type bare 'binary'/'factor' defaults via engine (APIC-24)", {
  res <- mcpower:::.parse_assignment_kind("x=binary, y=factor", "variable_type", c("x", "y"))
  expect_equal(length(res$errors), 0L)
  expect_equal(res$parsed$x$type, "binary")
  expect_equal(res$parsed$x$proportion, 0.5)
  expect_equal(res$parsed$y$type, "factor")
  expect_equal(as.numeric(res$parsed$y$n_levels), 3)
  expect_equal(as.numeric(unlist(res$parsed$y$proportions)), rep(1 / 3, 3))
})

# --- factor proportion normalisation (APIC-25) -----------------------------

test_that("variable_type factor proportions normalise to 1; <=0 rejected (APIC-25)", {
  fn <- mcpower:::.parse_assignment_kind("x=(factor,2,2)", "variable_type", "x")
  expect_equal(length(fn$errors), 0L)
  props <- as.numeric(unlist(fn$parsed$x$proportions))
  expect_equal(sum(props), 1.0)
  expect_equal(props, c(0.5, 0.5))
  fneg <- mcpower:::.parse_assignment_kind("x=(factor,1,-1)", "variable_type", "x")
  expect_true(length(fneg$errors) > 0L)
})

# --- variable-type error paths (APIC-26) -----------------------------------

test_that("variable_type rejects unknown type / bad arity / bad level counts (APIC-26)", {
  errs <- function(spec) mcpower:::.parse_assignment_kind(spec, "variable_type", "x")$errors
  expect_true(length(errs("x=zzz")) > 0L)               # unknown type
  expect_true(length(errs("x=(binary,0.3,0.4)")) > 0L)  # bad arity
  expect_true(length(errs("x=(factor,1)")) > 0L)        # n_levels < 2
  expect_true(length(errs("x=(factor,21)")) > 0L)       # > max levels
})

# --- _apply collects all unknown effect errors (APIC-33) -------------------

test_that("set_effects collects all unknown-effect errors (not fail-fast) (APIC-33)", {
  reg <- mcpower:::RVariableRegistry$new("y ~ x1")
  err <- tryCatch(reg$set_effects("zzz=0.1, qqq=0.2"), error = function(e) conditionMessage(e))
  expect_true(grepl("zzz", err, fixed = TRUE))
  expect_true(grepl("qqq", err, fixed = TRUE))
})

# --- string correlation needs >=2 non-factor vars (APIC-34) ----------------

test_that("set_correlation_spec raises with fewer than 2 non-factor variables (APIC-34)", {
  reg <- mcpower:::RVariableRegistry$new("y ~ x1")
  expect_error(reg$set_correlation_spec("(x1,x1)=0.3"))
})

# --- _build_cluster_spec_dict tau transform (APIC-52) ----------------------

test_that(".encode_outcome_and_clusters computes tau = ICC/(1-ICC); ICC=0 -> 0 (APIC-52)", {
  enc <- mcpower:::.encode_outcome_and_clusters("lme", "mle", 0.0,
            list(g = list(icc = 0.2, n_clusters = 10)))
  rp <- jsonlite::fromJSON(enc$clusters_json, simplifyVector = FALSE)
  # transform identity (ratio), not an oracle value: 0.2/(1-0.2)
  expect_equal(rp[[1]]$tau_squared, 0.2 / 0.8)
  enc0 <- mcpower:::.encode_outcome_and_clusters("lme", "mle", 0.0,
            list(g = list(icc = 0.0, n_clusters = 10)))
  rp0 <- jsonlite::fromJSON(enc0$clusters_json, simplifyVector = FALSE)
  expect_equal(rp0[[1]]$tau_squared, 0.0)
})

# --- registry set_correlation symmetry (APIC-58) ---------------------------

test_that("RVariableRegistry$set_correlation is symmetric (APIC-58)", {
  reg <- mcpower:::RVariableRegistry$new("y ~ x1 + x2")
  reg$set_correlation("x1", "x2", 0.5)
  m <- reg$get_correlation_matrix()
  expect_equal(m[1, 2], m[2, 1])
  expect_equal(m[1, 2], 0.5)
})

# --- registry set_variable_type(factor) stores factor_info (APIC-59) -------

test_that("RVariableRegistry$set_variable_type('factor') stores factor_info (APIC-59)", {
  reg <- mcpower:::RVariableRegistry$new("y ~ g")
  reg$set_variable_type("g", "factor", n_levels = 4L)
  fi <- reg$`_factors`[["g"]]
  expect_equal(fi$n_levels, 4L)
  expect_equal(fi$reference_level, 1L)
  expect_true("g" %in% reg$factor_names)
})

# --- expand_factors reference coding shape (APIC-57) -----------------------

test_that("expand_factors produces n_levels-1 dummies named factor[level] (APIC-57)", {
  reg <- mcpower:::RVariableRegistry$new("y ~ x1 + g")
  reg$set_variable_types("g=(factor,3)")
  reg$expand_factors()
  expect_equal(reg$dummy_names, c("g[2]", "g[3]"))
  expect_equal(length(reg$dummy_names), 3L - 1L)
})

# --- _to_linear_spec_dict correlations non-zero only (APIC-61) -------------

test_that(".to_linear_spec_list emits non-zero correlations only; empty when none (APIC-61)", {
  reg <- mcpower:::RVariableRegistry$new("y ~ x1 + x2")
  reg$set_effects("x1=0.5, x2=0.3")
  none <- mcpower:::.to_linear_spec_list(reg, "optimistic", alpha = 0.05,
            correction = NULL, target_test = NULL,
            heteroskedasticity = list(driver_var_index = NULL), residual_name = "normal",
            max_failed = 1.0, test_formula = NULL)
  expect_equal(length(none$correlations), 0L)

  reg$set_correlation_spec("(x1,x2)=0.4")
  withc <- mcpower:::.to_linear_spec_list(reg, "optimistic", alpha = 0.05,
            correction = NULL, target_test = NULL,
            heteroskedasticity = list(driver_var_index = NULL), residual_name = "normal",
            max_failed = 1.0, test_formula = NULL)
  expect_equal(length(withc$correlations), 1L)
  expect_equal(withc$correlations[[1]]$value, 0.4)
})

# --- _to_linear_spec_dict correction alias mapping (APIC-63) ---------------

test_that(".to_linear_spec_list maps bh/fdr -> benjamini_hochberg, NULL -> none (APIC-63)", {
  reg <- mcpower:::RVariableRegistry$new("y ~ x1")
  reg$set_effects("x1=0.5")
  args <- function(corr) mcpower:::.to_linear_spec_list(reg, "optimistic",
            alpha = 0.05, correction = corr, target_test = NULL,
            heteroskedasticity = list(driver_var_index = NULL), residual_name = "normal",
            max_failed = 1.0, test_formula = NULL)
  expect_equal(args("bh")$correction, "benjamini_hochberg")
  expect_equal(args("fdr")$correction, "benjamini_hochberg")
  expect_equal(args(NULL)$correction, "none")
})

# --- _to_linear_spec_dict strips (1|group) (APIC-62) -----------------------

test_that(".to_linear_spec_list strips random-effect terms from the builder formula (APIC-62)", {
  reg <- mcpower:::RVariableRegistry$new("y ~ x1 + (1|g)")
  reg$set_effects("x1=0.5")
  payload <- mcpower:::.to_linear_spec_list(reg, "optimistic", alpha = 0.05,
            correction = NULL, target_test = NULL,
            heteroskedasticity = list(driver_var_index = NULL), residual_name = "normal",
            max_failed = 1.0, test_formula = NULL)
  expect_false(grepl("(1|g)", payload$formula, fixed = TRUE))
  expect_equal(payload$formula, "y ~ x1")
})

# --- _encode_outcome_and_clusters clusters_json present for clustered OLS (APIC-64) ---

test_that(".encode_outcome_and_clusters emits non-empty clusters_json for clustered OLS (APIC-64)", {
  enc <- mcpower:::.encode_outcome_and_clusters("lme", "ols", 0.0,
            list(g = list(icc = 0.1, n_clusters = 20)))
  expect_equal(enc$estimator, "ols")
  expect_false(identical(enc$clusters_json, "[]"))
  rp <- jsonlite::fromJSON(enc$clusters_json, simplifyVector = FALSE)
  expect_equal(rp[[1]]$sizing$FixedClusters$n_clusters, 20L)
})

# --- unwrap single / multi scenario (APIC-65 / APIC-66) --------------------

test_that(".unwrap_scenario_result returns the inner dict for a single scenario (APIC-65)", {
  raw <- list(scenarios = list(optimistic = list(convergence_rate = 1.0)))
  u <- mcpower:::.unwrap_scenario_result(raw, "optimistic")
  expect_false("scenarios" %in% names(u))
  expect_true("convergence_rate" %in% names(u))
})

test_that(".unwrap_scenario_result keeps the full envelope for multiple scenarios (APIC-66)", {
  raw <- list(scenarios = list(optimistic = list(convergence_rate = 1.0),
                               realistic = list(convergence_rate = 0.9)))
  u <- mcpower:::.unwrap_scenario_result(raw, c("optimistic", "realistic"))
  expect_true("scenarios" %in% names(u))
  expect_equal(length(u$scenarios), 2L)
})

# --- _scenario_dict rejects unknown distribution (APIC-80) -----------------

test_that(".scenario_dict / encoders reject unknown distribution names (APIC-80)", {
  expect_error(mcpower:::.encode_dist_list(c("zzz"), "optimistic"))
  expect_error(mcpower:::.encode_residual_list(c("zzz"), "optimistic"))
})

# --- boundary-hit-rate injection shape/range (APIC-67) ---------------------

test_that(".add_boundary_hit_rates injects rates in [0,1]; zeros when no boundary_hit (APIC-67)", {
  r <- list(boundary_hit = c(0L, 1L, 1L, 2L, 0L), convergence_rate = c(1.0))
  out <- mcpower:::.add_boundary_hit_rates(r)
  expect_true(out$boundary_hit_rate_tau_zero >= 0 && out$boundary_hit_rate_tau_zero <= 1)
  expect_true(out$boundary_hit_rate_high_tau >= 0 && out$boundary_hit_rate_high_tau <= 1)
  # rates are fractions of the n_sims vector (shape: 2 of 5 -> 0.4; 1 of 5 -> 0.2)
  expect_equal(out$boundary_hit_rate_tau_zero, 2 / 5)
  expect_equal(out$boundary_hit_rate_high_tau, 1 / 5)
  # no boundary_hit -> per-sample-size zero vector matching convergence_rate length
  r2 <- list(convergence_rate = c(1.0, 1.0))
  out2 <- mcpower:::.add_boundary_hit_rates(r2)
  expect_equal(out2$boundary_hit_rate_tau_zero, rep(0.0, 2L))
  expect_equal(out2$boundary_hit_rate_high_tau, rep(0.0, 2L))
})

# --- G-C Tier-1-B: LME boundary-hit rates live end-to-end -------------------
# Mirrors test_boundary_hit_rates.py cases 1-3: live engine confirms rates are
# returned in [0,1] and have the correct length.

test_that("G-C T1-B: LME find_power returns boundary_hit_rate_* in [0,1]", {
  m <- MCPower$new("y ~ x1 + (1|g)", family = "lme")$
    set_effects("x1=0.5")$
    set_cluster("g", ICC = 0.1, n_clusters = 20)$
    set_simulations(100)
  res <- m$find_power(200, seed = 2137, progress_callback = FALSE)
  tz <- res$boundary_hit_rate_tau_zero
  ht <- res$boundary_hit_rate_high_tau
  expect_false(is.null(tz), info = "boundary_hit_rate_tau_zero missing from LME result")
  expect_false(is.null(ht), info = "boundary_hit_rate_high_tau missing from LME result")
  expect_true(all(tz >= 0 & tz <= 1), info = "tau_zero rate out of [0,1]")
  expect_true(all(ht >= 0 & ht <= 1), info = "high_tau rate out of [0,1]")
})

test_that("G-C T1-B: OLS find_power returns boundary_hit_rate_* = 0.0", {
  m <- MCPower$new("y ~ x1")$set_effects("x1=0.5")$set_simulations(50)
  res <- m$find_power(100, seed = 2137, progress_callback = FALSE)
  expect_equal(res$boundary_hit_rate_tau_zero, 0.0)
  expect_equal(res$boundary_hit_rate_high_tau, 0.0)
})

test_that("G-C T1-B: LME find_sample_size returns boundary_hit_rate_* per-N lists", {
  m <- MCPower$new("y ~ x1 + (1|g)", family = "lme")$
    set_effects("x1=0.5")$
    set_cluster("g", ICC = 0.1, n_clusters = 20)$
    set_simulations(100)
  res <- m$find_sample_size(from_size = 100, to_size = 200, by = 100,
                             seed = 2137, progress_callback = FALSE)
  tz <- res$boundary_hit_rate_tau_zero
  ht <- res$boundary_hit_rate_high_tau
  expect_false(is.null(tz), info = "boundary_hit_rate_tau_zero missing from LME find_sample_size result")
  expect_true(length(tz) >= 1, info = "tau_zero rate should be a per-N vector")
  expect_true(all(tz >= 0 & tz <= 1), info = "tau_zero rates out of [0,1]")
  expect_true(all(ht >= 0 & ht <= 1), info = "high_tau rates out of [0,1]")
})

# --- test_formula LHS strip / unknown-var rejection (APIC-17 / APIC-79) ----
# R does not validate test_formula R-side: the string is passed to the Rust
# builder, which strips the LHS and rejects unknown predictors. The mirror is
# therefore the Rust-builder behavior reachable through find_power.

test_that("find_power accepts a test_formula whose LHS names the dependent var (APIC-79)", {
  m <- MCPower$new("y ~ x1 + x2")$set_effects("x1=0.5, x2=0.3")
  # 'y ~ x1' must not raise a false unknown-variable error on the LHS 'y'.
  # APIC-79: reaching this line proves no error was raised; assert the call returns
  # the unwrapped single-scenario power result (no "scenarios" envelope) rather than
  # the looser OR that also accepted a wrongly-wrapped or convergence-only result.
  res <- m$find_power(80, test_formula = "y ~ x1", n_sims = 50, progress_callback = FALSE)
  expect_false("scenarios" %in% names(res))
  expect_true("power_uncorrected" %in% names(res))
})

test_that("find_power rejects a test_formula naming an unknown predictor (APIC-17)", {
  m <- MCPower$new("y ~ x1 + x2")$set_effects("x1=0.5, x2=0.3")
  expect_error(m$find_power(80, test_formula = "y ~ zzz", n_sims = 50, progress_callback = FALSE))
})

# --- test_formula refits the reduced model (marginal coef) -------------------
# Mirrors the Python test_test_formula_reduced_fit_recovers_marginal_coefficient
# (examples-parity: identical spec, power-band gate). DGP exam ~ 0.1*study +
# 0.8*ability with corr 0.7: the full fit recovers study's *partial* coef (0.1,
# weak power); dropping ability recovers study's *marginal* coef
# (0.1 + 0.7*0.8 = 0.66, ~1.0 power at N=120). An engine that only narrowed the
# reported targets (the historical bug) would return identical powers; the
# reduced-fit power must jump instead.
test_that("test_formula refits the reduced model, recovering the marginal coef", {
  study_power <- function(test_formula) {
    m <- MCPower$new("exam ~ study + ability")$
      set_effects("study=0.1, ability=0.8")$
      set_correlation("corr(study, ability)=0.7")
    res <- m$find_power(120, n_sims = 1000, seed = 2137,
                        test_formula = test_formula, progress_callback = FALSE)
    # study is the first target in both designs; one sample size => one row.
    res$power_uncorrected[[1]][[1]]
  }
  p_full <- study_power(NULL)
  p_reduced <- study_power("exam ~ study")
  expect_lt(p_full, 0.30)
  expect_gt(p_reduced, 0.90)
  expect_gt(p_reduced - p_full, 0.50)
})

# --- baseline-probability logit transform identity (APIC-41) ---------------

test_that("set_baseline_probability stores intercept = logit(p) (APIC-41)", {
  m <- MCPower$new("y ~ x1", family = "logit")$set_baseline_probability(0.3)
  m$summary()  # forces apply
  # transform identity log(p/(1-p)), not an oracle value
  expect_equal(m$intercept, log(0.3 / 0.7))
  m$set_baseline_probability(0.6)
  m$summary()
  expect_equal(m$intercept, log(0.6 / 0.4))
})
