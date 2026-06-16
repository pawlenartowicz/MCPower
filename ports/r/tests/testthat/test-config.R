test_that("constructor defaults come from config", {
  m <- MCPower$new("y ~ x")
  sim <- mcpower:::.sim_defaults()
  expect_equal(m$seed, as.integer(sim$seed))
  expect_equal(m$alpha, sim$alpha)
  expect_equal(m$power, sim$target_power * 100)
  expect_equal(m$n_simulations, as.integer(sim$n_sims$ols))
  expect_equal(m$max_failed_simulations, sim$max_failed_fraction)  # 0.1
})

test_that("n_sims default is family-aware", {
  # lme fits are more expensive, so they default to the lighter `mixed` budget;
  # OLS and logit (GLM) use the `ols` budget.
  sim <- mcpower:::.sim_defaults()
  expect_equal(MCPower$new("y ~ x")$n_simulations, as.integer(sim$n_sims$ols))
  expect_equal(MCPower$new("y ~ x", family = "logit")$n_simulations,
               as.integer(sim$n_sims$ols))
  expect_equal(MCPower$new("y = x + (1|g)", family = "lme")$n_simulations,
               as.integer(sim$n_sims$mixed))
})
