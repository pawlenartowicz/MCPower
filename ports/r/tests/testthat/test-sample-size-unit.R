library(mcpower)

# L2 tests for find_sample_size's `unit` argument (mirrors
# ports/py/tests/integration/test_sample_size_unit.py). The engine resolves
# the unit -> divisor and converts bounds/results (see
# engine_orchestrator::unit); these tests check the R binding wires it
# through correctly: both cluster-unit regimes, the two validation error
# paths, that the default is unchanged, and that the sample-size plot spec
# picks up the divisor.

test_that("find_sample_size unit='clusters' on a FixedSize design converts bounds/results", {
  m <- MCPower$new("y ~ x1 + (1|school)", family = "lme")$
    set_effects("x1=0.5")$
    set_cluster("school", ICC = 0.2, cluster_size = 8L)
  res <- m$find_sample_size(from_size = 10L, to_size = 50L, by = 5L,
                             unit = "clusters", n_sims = 20L, verbose = FALSE)
  expect_equal(as.integer(res$sample_sizes),
               c(80L, 120L, 160L, 200L, 240L, 280L, 320L, 360L, 400L))
  expect_equal(as.integer(res$sample_sizes_unit),
               c(10L, 15L, 20L, 25L, 30L, 35L, 40L, 45L, 50L))
  expect_equal(as.integer(res$unit_divisor), 8L)
})

test_that("find_sample_size unit='cluster_size' on a FixedClusters design converts bounds/results", {
  m <- MCPower$new("y ~ x1 + (1|school)", family = "lme")$
    set_effects("x1=0.5")$
    set_cluster("school", ICC = 0.2, n_clusters = 10L)
  res <- m$find_sample_size(from_size = 6L, to_size = 20L, by = 2L,
                             unit = "cluster_size", n_sims = 20L, verbose = FALSE)
  expect_equal(as.integer(res$sample_sizes),
               c(60L, 80L, 100L, 120L, 140L, 160L, 180L, 200L))
  expect_equal(as.integer(res$sample_sizes_unit),
               c(6L, 8L, 10L, 12L, 14L, 16L, 18L, 20L))
  expect_equal(as.integer(res$unit_divisor), 10L)
})

test_that("find_sample_size unit regime mismatch raises", {
  # unit="cluster_size" on a FixedSize (cluster_size=...) design does not fit
  # -- that regime is fixed by size, not by cluster count.
  m <- MCPower$new("y ~ x1 + (1|school)", family = "lme")$
    set_effects("x1=0.5")$
    set_cluster("school", ICC = 0.2, cluster_size = 8L)
  expect_error(
    m$find_sample_size(from_size = 10L, to_size = 50L, by = 5L,
                       unit = "cluster_size", n_sims = 20L, verbose = FALSE),
    regexp = "does not fit the fixed_size"
  )
})

test_that("find_sample_size unit='clusters' on an unclustered model raises", {
  m <- MCPower$new("y = x1")$set_effects("x1=0.5")
  expect_error(
    m$find_sample_size(from_size = 40L, to_size = 200L, by = 40L,
                       unit = "clusters", n_sims = 20L, verbose = FALSE),
    regexp = "needs a clustered model"
  )
})

test_that("find_sample_size default unit is unchanged (observations, divisor 1)", {
  m <- MCPower$new("y ~ x1 + (1|school)", family = "lme")$
    set_effects("x1=0.5")$
    set_cluster("school", ICC = 0.2, cluster_size = 8L)
  res <- m$find_sample_size(from_size = 80L, to_size = 400L, by = 40L,
                             n_sims = 20L, verbose = FALSE)
  expect_equal(as.integer(res$unit_divisor), 1L)
  expect_equal(as.integer(res$sample_sizes_unit), as.integer(res$sample_sizes))
})

test_that("sample-size curve spec uses n_unit + host-set x title under a cluster unit", {
  m <- MCPower$new("y ~ x1 + (1|school)", family = "lme")$
    set_effects("x1=0.5")$
    set_cluster("school", ICC = 0.2, cluster_size = 8L)
  res <- m$find_sample_size(from_size = 10L, to_size = 50L, by = 5L,
                             unit = "clusters", n_sims = 20L, verbose = FALSE)
  blocks <- mcpower:::.plot_blocks(res, "find_sample_size")
  curve_spec <- blocks[["curve"]]
  x_layers <- Filter(function(l) !is.null(l$encoding) && !is.null(l$encoding$x), curve_spec$layer)
  expect_gte(length(x_layers), 2L)
  for (layer in x_layers) {
    expect_equal(layer$encoding$x$field, "n_unit")
    expect_true(!is.null(layer$encoding$x$title) && nzchar(layer$encoding$x$title))
  }
})

test_that("sample-size curve spec keeps field n + engine title under the default unit", {
  m <- MCPower$new("y ~ x1 + (1|school)", family = "lme")$
    set_effects("x1=0.5")$
    set_cluster("school", ICC = 0.2, cluster_size = 8L)
  res <- m$find_sample_size(from_size = 80L, to_size = 400L, by = 40L,
                             n_sims = 20L, verbose = FALSE)
  blocks <- mcpower:::.plot_blocks(res, "find_sample_size")
  curve_spec <- blocks[["curve"]]
  x_layers <- Filter(function(l) !is.null(l$encoding) && !is.null(l$encoding$x), curve_spec$layer)
  expect_gte(length(x_layers), 2L)
  for (layer in x_layers) {
    expect_equal(layer$encoding$x$field, "n")
    expect_equal(layer$encoding$x$title, "Sample size (N)")
  }
})

test_that("long-form summary shows unit columns in every required-N table", {
  # Mirrors ports/py/tests/integration/test_sample_size_unit.py::
  # test_long_form_summary_shows_unit_columns_in_every_required_n_table.
  m <- MCPower$new("y ~ x1 + x2 + (1|school)", family = "lme")$
    set_effects("x1=0.5, x2=0.4")$
    set_cluster("school", ICC = 0.2, cluster_size = 8L)
  res <- m$find_sample_size(from_size = 10L, to_size = 50L, by = 5L, unit = "clusters",
                             n_sims = 20L, verbose = FALSE, scenarios = TRUE)
  text <- paste(capture.output(summary(res)), collapse = "\n")
  expect_true(grepl("unit: Number of school", text, fixed = TRUE))
  # multi-scenario table: unit column then "<scenario> (N)" column
  expect_true(grepl("optimistic   optimistic (N)", text, fixed = TRUE))
  # CI table and joint table: unit column between the label and Required N
  expect_true(grepl("Number of school   Required N   CI 95%", text, fixed = TRUE))
  expect_true(grepl("Joint target         Number of school   Required N", text, fixed = TRUE))
})
