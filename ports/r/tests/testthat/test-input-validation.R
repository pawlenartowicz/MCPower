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

test_that("set_cluster: cluster_size=1 rejects (below reliable floor 5)", {
  m <- MCPower$new("y ~ x1 + (1|g)", family = "lme")
  expect_error(
    m$set_cluster("g", ICC = 0.2, cluster_size = 1L),
    regexp = ">= 2"
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
               "the random-effect term on 'school:class' has no set_cluster call", fixed = TRUE)
})

# ── set_simulations (validators.py _validate_simulations) ─────────────────

test_that("set_simulations: below 1 rejects, fractional values round", {
  m <- MCPower$new("y ~ x1")
  expect_error(m$set_simulations(0), "Number of simulations must be >= 1, got 0", fixed = TRUE)
  expect_error(m$set_simulations(-5), "Number of simulations must be >= 1, got -5", fixed = TRUE)
  m$set_simulations(200.6)
  expect_identical(m$n_simulations, 201L)
})

# ── find_power sample_size (validators.py _validate_sample_size and
#    _validate_sample_size_for_model; limits from config limits.sample_size) ──

test_that("find_power: sample_size must be one whole number in [20, 100000]", {
  m <- MCPower$new("y ~ x1")$set_effects("x1=0.3")
  run <- function(n) m$find_power(n, n_sims = 20L, verbose = FALSE, progress_callback = FALSE)
  refused <- list(
    list(10, "sample_size must be at least 20, got 10"),
    list(150.5, "sample_size must be an integer, got 150.5"),
    list(200000, "sample_size too large (200,000). Maximum recommended: 100,000."),
    list(NA, "sample_size must be an integer, got logical"),
    list(c(100, 200), "sample_size must be an integer, got a vector of length 2")
  )
  for (case in refused) {
    expect_error(run(case[[1]]), case[[2]], fixed = TRUE, info = format(case[[1]]))
  }
  expect_no_error(run(200))
  # Floating arithmetic: seq(0.1, 0.9, 0.2) * 1000 gives 300.00000000000006.
  for (n in seq(0.1, 0.9, 0.2) * 1000) expect_no_error(run(n))
  expect_error(run(-1e10), "sample_size must be at least 20, got -10000000000", fixed = TRUE)
  expect_no_error(run(200L))
})

test_that("find_power: sample_size below 15 plus the design columns is refused", {
  m <- MCPower$new("y ~ x1 + x2 + x3 + x4 + x5 + x6")$
    set_effects("x1=0.3, x2=0.3, x3=0.3, x4=0.3, x5=0.3, x6=0.3")
  expect_error(m$find_power(20, n_sims = 20L, verbose = FALSE, progress_callback = FALSE),
               "sample_size (20) is too small for a model with 6 variables. Minimum required: 21 (15 + 6 variables).",
               fixed = TRUE)
})

# ── set_cluster non-finite numbers (NaN, Inf) name their argument ──────────

test_that("set_cluster: a NaN or Inf number names its argument", {
  slope <- function(...) list(list(predictor = "x1", ...))
  lme <- function() MCPower$new("y ~ x1 + (1 + x1|g)", family = "lme")$set_effects("x1=0.3")
  expect_error(lme()$set_cluster("g", ICC = NaN, n_clusters = 20L),
               "ICC must be a finite number, got NaN", fixed = TRUE)
  expect_error(lme()$set_cluster("g", ICC = 0.2, n_clusters = 20L,
                                 random_slopes = slope(variance = NaN)),
               "random_slopes[[1]]$variance must be a finite positive number, got NaN", fixed = TRUE)
  expect_error(lme()$set_cluster("g", ICC = 0.2, n_clusters = 20L,
                                 random_slopes = slope(variance = Inf)),
               "random_slopes[[1]]$variance must be a finite positive number, got Inf", fixed = TRUE)
  expect_error(lme()$set_cluster("g", ICC = 0.2, n_clusters = 20L,
                                 random_slopes = slope(variance = 0.1, corr_with_intercept = NaN)),
               "random_slopes[[1]]$corr_with_intercept must be in [-1, 1]", fixed = TRUE)
  two <- MCPower$new("y ~ x1 + x2 + (1 + x1 + x2|g)", family = "lme")$set_effects("x1=0.3, x2=0.3")
  expect_error(two$set_cluster("g", ICC = 0.2, n_clusters = 20L, random_slopes = list(
                 list(predictor = "x1", variance = 0.1),
                 list(predictor = "x2", variance = 0.1, corr_with = NaN))),
               "random_slopes[[2]]$corr_with values must be finite and in [-1, 1]", fixed = TRUE)
  pois <- MCPower$new("y ~ x1 + (1|g)", family = "poisson")$set_effects("x1=0.3")
  for (bad in c(NaN, Inf)) {
    got <- format(bad)
    expect_error(lme()$set_cluster("g", ICC = 0.2, n_clusters = bad),
                 paste("n_clusters must be an integer >= 2, got", got), fixed = TRUE)
    expect_error(lme()$set_cluster("g", ICC = 0.2, cluster_size = bad),
                 paste0("cluster_size must be an integer >= 2. Got ", got, "."), fixed = TRUE)
    nested <- MCPower$new("y ~ x1 + (1|A/B)", family = "lme")$set_effects("x1=0.3")
    expect_error(nested$set_cluster("A:B", ICC = 0.1, n_per_parent = bad),
                 paste("n_per_parent must be an integer >= 2, got", got), fixed = TRUE)
  }
  expect_error(lme()$set_cluster("g", ICC = "0.2", n_clusters = 20L), "ICC must be a number", fixed = TRUE)
  expect_error(lme()$set_cluster("g", ICC = c(0.2, 0.3), n_clusters = 20L),
               "ICC must be a finite number, got a vector of length 2", fixed = TRUE)
  expect_error(pois$set_cluster("g", tau_squared = Inf, n_clusters = 20L),
               "tau_squared must be a finite non-negative number, got Inf", fixed = TRUE)
})

# ── engine errors reach R verbatim ──────────────────────────────────────────

test_that("an engine error carrying '%' reaches R verbatim", {
  # extendr reads the error text as a C format string; an unescaped '%d'
  # printed garbage and '%n' aborted the R session.
  expect_error(MCPower$new("y ~ x1 + (1|g%d)"), "'g%d'", fixed = TRUE)
  expect_error(MCPower$new("y ~ x1 + (1|g%n)"), "'g%n'", fixed = TRUE)
})

test_that("set_cluster refuses an argument it does not know", {
  m <- MCPower$new("y ~ x + (1|g)", family = "lme")
  expect_error(m$set_cluster("g", icc = 0.5, n_clusters = 30), "unused argument")
})

test_that("set_cluster refuses n_clusters together with cluster_size", {
  m <- MCPower$new("y ~ x + (1|g)", family = "lme")
  expect_error(m$set_cluster("g", ICC = 0.2, n_clusters = 30, cluster_size = 20),
               "Specify either n_clusters OR cluster_size, not both", fixed = TRUE)
})

test_that("set_cluster takes a computed whole number and refuses one past integer range", {
  m <- MCPower$new("y ~ x + (1|g)", family = "lme")
  computed <- seq(0.1, 0.9, 0.2)[2] * 100
  expect_false(computed == 30)
  m$set_cluster("g", ICC = 0.2, n_clusters = computed)
  expect_identical(m$.__enclos_env__$private$pending_clusters$g$n_clusters, 30)
  expect_error(m$set_cluster("g", ICC = 0.2, n_clusters = 3e9),
               "n_clusters must be an integer >= 2, got 3000000000", fixed = TRUE)
})

test_that("find_sample_size checks its range as Python does", {
  m <- MCPower$new("y ~ x")$set_effects("x=0.3")
  fss <- function(...) m$find_sample_size(..., n_sims = 20L, verbose = FALSE,
                                          progress_callback = FALSE)
  expect_error(fss(from_size = 30, to_size = 200, by = 500),
               "Step size 'by' (500) is larger than range (170)", fixed = TRUE)
  expect_error(fss(from_size = 30.9, to_size = 200, by = 10),
               "from_size must be a positive integer, got 30.9", fixed = TRUE)
  expect_error(fss(from_size = 30, to_size = 200, by = 2.5),
               'by must be a positive integer or "auto", got 2.5', fixed = TRUE)
  expect_error(fss(from_size = 200, to_size = 30, by = 10),
               "from_size (200) must be less than to_size (30)", fixed = TRUE)
})
