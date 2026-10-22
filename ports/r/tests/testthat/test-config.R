test_that("constructor defaults come from config", {
  m <- MCPower$new("y ~ x")
  sim <- mcpower:::.sim_defaults()
  expect_equal(m$seed, as.integer(sim$seed))
  expect_equal(m$alpha, sim$alpha)
  expect_equal(m$power, sim$target_power * 100)
  expect_null(m$n_simulations)  # resolved per design at run time
  expect_equal(m$.__enclos_env__$private$resolved_n_simulations(),
               as.integer(sim$n_sims$ols))
  expect_equal(m$max_failed_simulations, sim$max_failed_fraction)  # 0.1
})

test_that("n_sims default follows the design (ols / mixed / glmm)", {
  # Resolved at run time: unclustered -> `ols`; clustered continuous -> `mixed`;
  # clustered binary/count (GLMM) -> `glmm`. set_simulations overrides all.
  sim <- mcpower:::.sim_defaults()
  resolved <- function(m) m$.__enclos_env__$private$resolved_n_simulations()
  expect_equal(resolved(MCPower$new("y ~ x")), as.integer(sim$n_sims$ols))
  expect_equal(resolved(MCPower$new("y ~ x", family = "logit")),
               as.integer(sim$n_sims$ols))

  lme <- MCPower$new("y = x + (1|g)", family = "lme")$
    set_cluster("g", ICC = 0.2, n_clusters = 20L)
  expect_equal(resolved(lme), as.integer(sim$n_sims$mixed))
  expect_equal(resolved(lme), 800L)

  logit <- MCPower$new("y = x + (1|g)", family = "logit")$
    set_cluster("g", ICC = 0.2, n_clusters = 20L)
  expect_equal(resolved(logit), as.integer(sim$n_sims$glmm))
  expect_equal(resolved(logit), 400L)
  pois <- MCPower$new("y = x + (1|g)", family = "poisson")$
    set_cluster("g", tau_squared = 0.3, n_clusters = 20L)
  expect_equal(resolved(pois), 400L)

  lme$set_simulations(123)
  expect_equal(resolved(lme), 123L)
})
