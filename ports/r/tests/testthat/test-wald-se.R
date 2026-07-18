library(mcpower)

# wald_se = "hessian" | "rx" per-call arg surface test.
# Mirrors ports/py/tests/test_wald_se.py.
# Only affects the GLMM estimator; rx (the 1.1.0 fastmode default) is the Schur
# speed knob; hessian is the advanced opt-in per-fit FD-Hessian SE (lme4
# use.hessian = TRUE), at least as conservative.

test_that("wald_se default is rx; hessian accepted and more conservative", {
  m <- MCPower$new("y ~ x1 + (1|grp)", family = "logit")$
    set_effects("x1=0.5")$
    set_baseline_probability(0.3)$
    set_cluster("grp", ICC = 0.3, n_clusters = 20L)
  suppressWarnings({
    pd <- m$find_power(sample_size = 200L, n_sims = 400L,
                       seed = 2137L, progress_callback = FALSE, verbose = FALSE)
    ph <- m$find_power(sample_size = 200L, n_sims = 400L, wald_se = "hessian",
                       seed = 2137L, progress_callback = FALSE, verbose = FALSE)
    prx <- m$find_power(sample_size = 200L, n_sims = 400L, wald_se = "rx",
                        seed = 2137L, progress_callback = FALSE, verbose = FALSE)
  })
  # Default is rx: same seed/data => identical power.
  expect_equal(pd$power_uncorrected[[1]], prx$power_uncorrected[[1]])
  # Hessian is at least as conservative as rx (equal or lower power);
  # allow 0.05 tolerance for Monte Carlo noise at n_sims=400.
  expect_true(ph$power_uncorrected[[1]] <= prx$power_uncorrected[[1]] + 0.05)
  expect_error(
    m$find_power(sample_size = 200L, n_sims = 100L, wald_se = "asymp",
                 progress_callback = FALSE, verbose = FALSE))
})

# M10: R's .wald_se_for_rust(NULL) reads config.json's estimation.wald_se —
# no hardcoded per-port default. Python's build_linear_spec was fixed to
# match this (see test_build_linear_spec_wald_se_default_reads_config in
# ports/py/tests/spec/test_spec_build.py); this pins R's side of the parity.
test_that("wald_se: NULL resolves to the config estimation.wald_se default", {
  expect_identical(mcpower:::.wald_se_for_rust(NULL), mcpower:::.config()$estimation$wald_se)
})
