test_that(".add_boundary_hit_rates computes per-N rates for multi-N results", {
  # 2 sims x 3 N-points, row-major flat: sim0 = (1, 0, 2), sim1 = (1, 0, 0)
  res <- list(
    n_sample_sizes = 3L,
    convergence_rate = c(1, 1, 1),
    boundary_hit = c(1L, 0L, 2L, 1L, 0L, 0L)
  )
  out <- mcpower:::.add_boundary_hit_rates(res)
  expect_equal(out$boundary_hit_rate_tau_zero, c(1.0, 0.0, 0.0))
  expect_equal(out$boundary_hit_rate_high_tau, c(0.0, 0.0, 0.5))
})

test_that(".add_boundary_hit_rates keeps single-N behaviour", {
  res <- list(
    n_sample_sizes = 1L,
    convergence_rate = 1,
    boundary_hit = c(0L, 1L, 2L, 1L)
  )
  out <- mcpower:::.add_boundary_hit_rates(res)
  expect_equal(out$boundary_hit_rate_tau_zero, 0.5)
  expect_equal(out$boundary_hit_rate_high_tau, 0.25)
})

test_that(".add_boundary_hit_rates zero-fills when boundary_hit is absent", {
  res <- list(n_sample_sizes = 2L, convergence_rate = c(1, 1))
  out <- mcpower:::.add_boundary_hit_rates(res)
  expect_equal(out$boundary_hit_rate_tau_zero, c(0.0, 0.0))
  expect_equal(out$boundary_hit_rate_high_tau, c(0.0, 0.0))
})
