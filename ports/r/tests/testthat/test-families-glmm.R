library(mcpower)

# family=/outcome_kind//link mapping, baseline-rate intercept, tau_squared vs
# ICC gating for family='poisson' clustering, and agq validation/fallback.
# Mirrors the equivalent Python family + AGQ tests. Small n_sims for speed.

test_that("family='poisson' maps to outcome_kind='count'", {
  m <- MCPower$new("y ~ x1", family = "poisson")
  expect_equal(m$outcome_kind, "count")
})

test_that("family='probit' maps to outcome_kind='binary' and link='probit'", {
  m <- MCPower$new("y ~ x1", family = "probit")
  expect_equal(m$outcome_kind, "binary")
  expect_equal(m$link, "probit")
})

test_that("family='logit' uses the canonical link", {
  m <- MCPower$new("y ~ x1", family = "logit")
  expect_equal(m$outcome_kind, "binary")
  expect_equal(m$link, "canonical")
})

test_that("set_baseline_rate computes intercept = log(rate) for family='poisson'", {
  m <- MCPower$new("y ~ x1", family = "poisson")$set_baseline_rate(2.0)
  m$summary()  # forces apply
  expect_equal(m$intercept, log(2.0))

  expect_error(MCPower$new("y ~ x1", family = "poisson")$set_baseline_rate(0))
})

test_that("set_cluster(tau_squared=) works for family='poisson'; ICC= errors", {
  m <- MCPower$new("y ~ x1 + (1|grp)", family = "poisson")$
    set_effects("x1=0.3")$
    set_baseline_rate(2.0)$
    set_cluster("grp", tau_squared = 0.5, n_clusters = 20)
  suppressWarnings({
    res <- m$find_power(sample_size = 200L, n_sims = 50L,
                        seed = 2137L, progress_callback = FALSE, verbose = FALSE)
  })
  expect_true(is.list(res))
  expect_true(!is.null(res$power_uncorrected))

  m_icc <- MCPower$new("y ~ x1 + (1|grp)", family = "poisson")$set_effects("x1=0.3")
  expect_error(
    m_icc$set_cluster("grp", ICC = 0.3, n_clusters = 20),
    "tau_squared"
  )
})

test_that("tau_squared= is rejected for a non-poisson family", {
  m <- MCPower$new("y ~ x1 + (1|grp)", family = "logit")$set_effects("x1=0.3")
  expect_error(
    m$set_cluster("grp", tau_squared = 0.5, n_clusters = 20),
    "tau_squared= is only for family='poisson'"
  )
})

test_that("agq must be an odd integer in 1..=25", {
  m <- MCPower$new("y ~ x1")$set_effects("x1=0.3")$set_simulations(50)
  expect_error(
    m$find_power(sample_size = 100L, n_sims = 50L, agq = 4L,
                progress_callback = FALSE, verbose = FALSE),
    "agq must be an odd integer"
  )
})

test_that("agq > 1 on an ineligible (unclustered) design warns and falls back to Laplace", {
  m <- MCPower$new("y ~ x1")$set_effects("x1=0.3")$set_simulations(50)
  res <- expect_warning(
    m$find_power(sample_size = 100L, n_sims = 50L, agq = 5L,
                progress_callback = FALSE, verbose = FALSE),
    "agq=5 is not available for this design"
  )
  expect_true(is.list(res))
  expect_true(!is.null(res$power_uncorrected))
})

test_that("agq=5 on an eligible clustered logit GLMM survives eligibility, reaches the contract as nagq, and changes the result vs Laplace", {
  # Eligible shape: single grouping factor, one intercept-only RE — mirrors
  # the AGQ-eligible fixture used elsewhere for this family/cluster combo.
  m <- MCPower$new("y ~ x1 + (1|grp)", family = "logit")$
    set_effects("x1=0.5")$
    set_baseline_probability(0.3)$
    set_cluster("grp", ICC = 0.3, n_clusters = 20L)

  # No warning at agq=5 here — an eligible design must not fall back silently.
  res_laplace <- m$find_power(sample_size = 200L, n_sims = 400L, seed = 2137L,
                              agq = 1L, progress_callback = FALSE, verbose = FALSE)
  res_agq5 <- expect_no_warning(
    m$find_power(sample_size = 200L, n_sims = 400L, seed = 2137L,
                agq = 5L, progress_callback = FALSE, verbose = FALSE)
  )

  # Laplace (agq=1) and 5-point AGQ are different quadrature schemes on the
  # same data draw (same seed) — a real fit at each must diverge. power_uncorrected
  # ties exactly between the two here, so tau_squared_hat_mean (the per-fit
  # RE-variance estimate, which AGQ genuinely moves) is the discriminating
  # metric — otherwise nagq never reached the contract (or was silently
  # stripped back to 1).
  expect_false(isTRUE(all.equal(
    res_laplace$estimator_extras$tau_squared_hat_mean,
    res_agq5$estimator_extras$tau_squared_hat_mean
  )))
})
