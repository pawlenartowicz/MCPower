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
