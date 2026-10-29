# test-input-validation.R — boundary tests for harmonized host-side gates.
# Mirrors the validation-harmonization upload/validation gates.

# ── ICC gates (set_cluster) ───────────────────────────────────────────────

test_that("set_cluster: ICC=0.04 rejects (below stability floor 0.05)", {
  m <- MCPower$new("y ~ x1 + (1|g)", family = "lme")
  expect_error(
    m$set_cluster("g", ICC = 0.04, n_clusters = 10L),
    regexp = "0\\.05"
  )
})

test_that("set_cluster: ICC=0.05 accepted (at stability floor)", {
  m <- MCPower$new("y ~ x1 + (1|g)", family = "lme")
  expect_silent(m$set_cluster("g", ICC = 0.05, n_clusters = 10L))
})

test_that("set_cluster: ICC=0.0 accepted (no clustering; bypasses stability band)", {
  m <- MCPower$new("y ~ x1 + (1|g)", family = "lme")
  expect_silent(m$set_cluster("g", ICC = 0.0, n_clusters = 10L))
})

test_that("set_cluster: ICC=0.96 rejects (above stability ceiling 0.95)", {
  m <- MCPower$new("y ~ x1 + (1|g)", family = "lme")
  expect_error(
    m$set_cluster("g", ICC = 0.96, n_clusters = 10L),
    regexp = "0\\.95"
  )
})

test_that("set_cluster: ICC=-0.1 rejects (below 0)", {
  m <- MCPower$new("y ~ x1 + (1|g)", family = "lme")
  expect_error(
    m$set_cluster("g", ICC = -0.1, n_clusters = 10L),
    regexp = "0 and 1"
  )
})

test_that("set_cluster: ICC=1.0 rejects (not exclusive upper bound)", {
  m <- MCPower$new("y ~ x1 + (1|g)", family = "lme")
  expect_error(
    m$set_cluster("g", ICC = 1.0, n_clusters = 10L),
    regexp = "0 and 1"
  )
})

# ── n_clusters gate ───────────────────────────────────────────────────────

test_that("set_cluster: n_clusters=1 rejects", {
  m <- MCPower$new("y ~ x1 + (1|g)", family = "lme")
  expect_error(
    m$set_cluster("g", ICC = 0.2, n_clusters = 1L),
    regexp = ">= 2"
  )
})

test_that("set_cluster: n_clusters=2 accepted", {
  m <- MCPower$new("y ~ x1 + (1|g)", family = "lme")
  expect_silent(m$set_cluster("g", ICC = 0.2, n_clusters = 2L))
})

# ── cluster_size gate ─────────────────────────────────────────────────────

test_that("set_cluster: cluster_size=4 rejects (below reliable floor 5)", {
  m <- MCPower$new("y ~ x1 + (1|g)", family = "lme")
  expect_error(
    m$set_cluster("g", ICC = 0.2, cluster_size = 4L),
    regexp = ">= 5"
  )
})

test_that("set_cluster: cluster_size=5 accepted (at reliable floor)", {
  m <- MCPower$new("y ~ x1 + (1|g)", family = "lme")
  expect_silent(m$set_cluster("g", ICC = 0.2, cluster_size = 5L))
})

# ── set_baseline_probability gates ────────────────────────────────────────

test_that("set_baseline_probability: p=0.04 warns (outside [0.05, 0.95])", {
  m <- MCPower$new("y ~ x1", family = "logit")
  expect_warning(
    m$set_baseline_probability(0.04),
    regexp = "extreme"
  )
})

test_that("set_baseline_probability: p=0.05 is accepted without warning", {
  m <- MCPower$new("y ~ x1", family = "logit")
  expect_silent(m$set_baseline_probability(0.05))
})

test_that("set_baseline_probability: p=0.5 is accepted without warning", {
  m <- MCPower$new("y ~ x1", family = "logit")
  expect_silent(m$set_baseline_probability(0.5))
})

test_that("set_baseline_probability: p=0.0 rejects (not in open interval (0,1))", {
  m <- MCPower$new("y ~ x1", family = "logit")
  expect_error(
    m$set_baseline_probability(0.0),
    regexp = "open interval"
  )
})

test_that("set_baseline_probability: p=1.0 rejects (not in open interval (0,1))", {
  m <- MCPower$new("y ~ x1", family = "logit")
  expect_error(
    m$set_baseline_probability(1.0),
    regexp = "open interval"
  )
})

test_that("set_baseline_probability: p=-0.1 rejects", {
  m <- MCPower$new("y ~ x1", family = "logit")
  expect_error(
    m$set_baseline_probability(-0.1),
    regexp = "open interval"
  )
})

test_that("set_baseline_probability: p=0.96 warns (outside [0.05, 0.95])", {
  m <- MCPower$new("y ~ x1", family = "logit")
  expect_warning(
    m$set_baseline_probability(0.96),
    regexp = "extreme"
  )
})

# ── set_baseline_rate / set_baseline_probability family gate (B4) ─────────
# Before the gate, calling both setters on the same model let the second
# call's intercept silently win with no error and no warning.

test_that("set_baseline_rate: rejects family='logit'", {
  m <- MCPower$new("y ~ x1", family = "logit")
  expect_error(
    m$set_baseline_rate(2.0),
    regexp = "set_baseline_rate is only for family='poisson'"
  )
})

test_that("set_baseline_rate: rejects family='probit'", {
  m <- MCPower$new("y ~ x1", family = "probit")
  expect_error(
    m$set_baseline_rate(2.0),
    regexp = "set_baseline_rate is only for family='poisson'"
  )
})

test_that("set_baseline_probability: rejects family='poisson'", {
  m <- MCPower$new("y ~ x1", family = "poisson")
  expect_error(
    m$set_baseline_probability(0.3),
    regexp = "set_baseline_probability is only for family='logit'/'probit'"
  )
})

test_that("set_baseline_probability: correct pairings unaffected (logit/probit)", {
  expect_silent(MCPower$new("y ~ x1", family = "logit")$set_baseline_probability(0.3))
  expect_silent(MCPower$new("y ~ x1", family = "probit")$set_baseline_probability(0.3))
})

test_that("set_baseline_rate: correct pairing unaffected (poisson)", {
  expect_silent(MCPower$new("y ~ x1", family = "poisson")$set_baseline_rate(2.0))
})

# ── set_alpha gates ───────────────────────────────────────────────────────

test_that("set_alpha: alpha=0.3 warns (above max_alpha=0.25)", {
  m <- MCPower$new("y ~ x1")
  expect_warning(
    m$set_alpha(0.3),
    regexp = "0\\.25"
  )
})

test_that("set_alpha: alpha=0.001 is accepted without warning", {
  m <- MCPower$new("y ~ x1")
  expect_silent(m$set_alpha(0.001))
})

test_that("set_alpha: alpha=0.25 is accepted without warning (at max_alpha boundary)", {
  m <- MCPower$new("y ~ x1")
  expect_silent(m$set_alpha(0.25))
})

test_that("set_alpha: alpha=0.05 (default) is accepted without warning", {
  m <- MCPower$new("y ~ x1")
  expect_silent(m$set_alpha(0.05))
})

# ── set_cluster / runtime grouping coverage (validators.py
#    _validate_cluster_config + model.py _validate_lme_runtime) ─────────────

test_that("set_cluster: a bare nested child name is not a formula grouping", {
  # (1|school/class) parses to "school" and "school:class"; "class" alone would
  # otherwise be simulated as a crossed grouping.
  m <- MCPower$new("y ~ x + (1|school/class)", family = "lme")
  m$set_cluster("school", ICC = 0.2, n_clusters = 20L)
  expect_error(
    m$set_cluster("class", ICC = 0.1, n_clusters = 60L),
    "Grouping variable 'class' not found in formula random effects. Expected one of: school, school:class",
    fixed = TRUE)
})

test_that("set_cluster: needs n_clusters or cluster_size unless nested", {
  m <- MCPower$new("y ~ x1 + (1|g)", family = "lme")
  expect_error(m$set_cluster("g", ICC = 0.2),
               "Must specify either n_clusters or cluster_size", fixed = TRUE)
  expect_error(m$set_cluster("g", ICC = 0.2, n_clusters = 10L, n_per_parent = 1L),
               "n_per_parent must be an integer >= 2, got 1", fixed = TRUE)
})

test_that("set_cluster: a formula without random effects has no grouping to cluster", {
  # Otherwise an OLS model would silently run a clustered DGP.
  m <- MCPower$new("y ~ x")
  expect_error(m$set_cluster("school", ICC = 0.2, n_clusters = 20L),
               "Grouping variable 'school' not found in formula random effects.", fixed = TRUE)
})

test_that("find_*: family='lme' without set_cluster names the missing grouping", {
  m <- MCPower$new("y ~ x1 + (1|school)", family = "lme")$set_effects("x1=0.3")
  msg <- "family='lme' requires set_cluster(...) for each random effect grouping variable; missing: ['school']"
  expect_error(m$find_power(400, n_sims = 10L, verbose = FALSE, progress_callback = FALSE),
               msg, fixed = TRUE)
  expect_error(m$find_sample_size(from_size = 200, to_size = 400, by = 100, n_sims = 10L,
                                  verbose = FALSE, progress_callback = FALSE),
               msg, fixed = TRUE)
})

test_that("find_power: every formula grouping needs its own set_cluster", {
  m <- MCPower$new("y ~ x + (1|school/class)", family = "lme")$set_effects("x=0.3")
  m$set_cluster("school", ICC = 0.2, n_clusters = 20L)
  expect_error(m$find_power(1200, n_sims = 10L, verbose = FALSE, progress_callback = FALSE),
               "random effect '(1|school:class)' has no set_cluster call", fixed = TRUE)
})

# ── set_simulations (validators.py _validate_simulations) ─────────────────

test_that("set_simulations: below 1 rejects, fractional values round", {
  m <- MCPower$new("y ~ x1")
  expect_error(m$set_simulations(0), "Number of simulations must be >= 1, got 0", fixed = TRUE)
  expect_error(m$set_simulations(-5), "Number of simulations must be >= 1, got -5", fixed = TRUE)
  m$set_simulations(200.6)
  expect_identical(m$n_simulations, 201L)
})
