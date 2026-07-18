test_that("debug_report returns selected stages as a nested list", {
  cj <- mcpower:::build_contract_from_spec(.ols_spec_json("y ~ x1", x1 = 0.5),
                                            "continuous", "canonical", "ols", 0.0, "[]")$contracts
  rep <- mcpower:::debug_report(cj, scenario_index = 0L, seed = 2137, n = 50L, n_sims = 100L,
            stage_input = TRUE, stage_data = TRUE, stage_dispatch = TRUE,
            stage_stats = TRUE, stage_crit = TRUE)
  # (The redundant `!is.null(rep$data)` was dropped — the next line already fails if
  # rep$data is NULL.) Assert each stage's sub-structure, not mere non-NULL-ness.
  expect_equal(rep$data$design$nrow, 50L)
  expect_equal(rep$dispatch$outcome_kind, "continuous")
  expect_equal(rep$dispatch$estimator, "ols")
  # stats && crit => power stage present: a non-empty named list, not just non-NULL
  # (an empty list or a wrong-typed scalar would have satisfied the old check).
  expect_true(is.list(rep$power) && length(rep$power) > 0L)
})

test_that("MCPowerDebug inherits production API and exposes the four stages", {
  m <- MCPowerDebug$new("y ~ x1")$set_effects("x1=0.5")
  expect_true(is.function(m$find_power))         # inherited unchanged
  d <- m$create_data()
  expect_true(is.matrix(d$design))
  expect_equal(nrow(d$design), m$.debug_n)
  disp <- m$dispatch(); expect_equal(disp$estimator, "ols")
  rs <- m$raw_statistics(); expect_true(length(rs$targets) >= 1)
  cv <- m$critical_value(); expect_true(cv$targets[[1]]$critical_value > 0)
})

test_that("create_data reproduces find_power sim-0 (shared seed/n)", {
  m <- MCPowerDebug$new("y ~ x1")$set_effects("x1=0.5")
  d1 <- m$create_data(); d2 <- m$create_data()
  expect_identical(d1$design, d2$design)         # deterministic, fixed seed/n
})

test_that("MCPowerDebug honours set_seed() live, not a construction-time snapshot", {
  m1 <- MCPowerDebug$new("y ~ x1")$set_effects("x1=0.5"); m1$set_seed(101)
  d1 <- m1$create_data()
  m2 <- MCPowerDebug$new("y ~ x1")$set_effects("x1=0.5"); m2$set_seed(202)
  d2 <- m2$create_data()
  # Different live seeds => different sim-0 draw (was identical when the seed
  # was frozen at construction and set_seed() silently ignored).
  expect_false(isTRUE(all.equal(d1$outcome, d2$outcome)))
  # Seed is read at call time: re-pointing m2 to 101 reproduces d1 exactly.
  m2$set_seed(101)
  expect_identical(m2$create_data()$outcome, d1$outcome)
})

test_that("MCPowerDebug .debug_seed overrides the live parent seed", {
  m <- MCPowerDebug$new("y ~ x1")$set_effects("x1=0.5"); m$set_seed(101)
  m$.debug_seed <- 999                       # explicit override wins over self$seed=101
  d_override <- m$create_data()
  ref <- MCPowerDebug$new("y ~ x1")$set_effects("x1=0.5"); ref$set_seed(999)
  expect_identical(d_override$outcome, ref$create_data()$outcome)
})

test_that("MCPowerDebug default seed stays 2137 (production parity) and is not snapshotted", {
  m <- MCPowerDebug$new("y ~ x1")$set_effects("x1=0.5")
  expect_null(m$.debug_seed)                  # no construction-time snapshot
  ref <- MCPowerDebug$new("y ~ x1")$set_effects("x1=0.5"); ref$.debug_seed <- 2137
  # NULL-follows-live-seed (default 2137) must match an explicit 2137 override.
  expect_identical(m$create_data()$outcome, ref$create_data()$outcome)
})

test_that("raw_statistics power matches a recompute from stats + crit", {
  m <- MCPowerDebug$new("y ~ x1")$set_effects("x1=0.5")
  rs <- m$raw_statistics()
  t1 <- rs$targets[[1]]; cv <- m$critical_value()$targets[[1]]
  two_sided <- cv$two_sided
  stat <- t1$statistic[!is.na(t1$statistic)]
  emp <- if (isTRUE(two_sided)) {
    mean(abs(stat) > cv$critical_value)
  } else {
    mean(stat > cv$critical_value)
  }
  expect_equal(emp, rs$power$scenarios[["optimistic"]]$power_uncorrected[[1]][1], tolerance = 1e-9)
})

test_that("raw_statistics power is keyed by .debug_scenario, not a hardcoded label", {
  m <- MCPowerDebug$new("y ~ x1")$set_effects("x1=0.5")
  m$.debug_scenario <- "doomer"              # a non-default configured scenario
  p <- m$raw_statistics()$power
  expect_identical(names(p$scenarios), "doomer")          # outer key (scenarios_envelope)
  # Inner field too — guards against a half-relabel (right key, stale field).
  expect_identical(p$scenarios[[1]]$scenario, "doomer")
})

test_that("MCPowerDebug$load_data round-trips create_data and recovers betas", {
  m <- MCPowerDebug$new("y ~ x1 + x2")$set_effects("x1=0.5, x2=0.3")
  m$.debug_n <- 400L; m$.debug_n_sims <- 1L; m$.debug_seed <- 2137
  d <- m$create_data()

  res <- m$load_data(d)
  expect_true(res$converged)
  expect_equal(length(res$betas), ncol(d$design))
  expect_equal(res$design_columns, d$columns)
  # Two tested coefficients (x1, x2): finite estimates near the truth.
  expect_true(length(res$targets) >= 2)
  betas <- res$betas
  names(betas) <- res$design_columns
  expect_equal(unname(betas["col_1"]), 0.5, tolerance = 0.15)
  expect_equal(unname(betas["col_2"]), 0.3, tolerance = 0.15)
  # Each target carries a positive se, a statistic, and a positive crit.
  for (tg in res$targets) {
    expect_true(tg$se > 0)
    expect_true(is.finite(tg$statistic))
    expect_true(tg$critical_value > 0)
  }
})

test_that("MCPowerDebug$load_data round-trips a logistic dataset", {
  m <- MCPowerDebug$new("y ~ x1 + x2", family = "logit")$set_effects("x1=0.5, x2=0.3")
  m$set_baseline_probability(0.3)
  m$.debug_n <- 800L; m$.debug_n_sims <- 1L; m$.debug_seed <- 2137
  d <- m$create_data()

  res <- m$load_data(d)
  expect_true(res$converged)
  expect_equal(length(res$betas), ncol(d$design))
  expect_equal(res$design_columns, d$columns)
  for (tg in res$targets) {
    expect_true(tg$se > 0)
    expect_true(is.finite(tg$statistic))
    expect_true(tg$critical_value > 0)        # z_crit ≈ 1.96
    expect_equal(tg$statistic_kind, "z")
  }
})

test_that("MCPowerDebug$load_data round-trips a clustered (LME) dataset", {
  m <- MCPowerDebug$new("y ~ x1 + (1|grp)", family = "lme")$set_effects("x1=0.5")
  m$set_cluster("grp", ICC = 0.1, n_clusters = 20L, cluster_size = 20L)
  m$.debug_n <- 400L; m$.debug_n_sims <- 1L; m$.debug_seed <- 2137
  d <- m$create_data()

  res <- m$load_data(d)
  expect_true(res$converged)
  expect_equal(length(res$betas), ncol(d$design))
  expect_true(length(res$targets) >= 1)
  betas <- res$betas; names(betas) <- res$design_columns
  expect_equal(unname(betas["col_1"]), 0.5, tolerance = 0.15)  # x1 near truth
  for (tg in res$targets) {
    expect_true(tg$se > 0)
    expect_true(is.finite(tg$statistic))
    expect_equal(tg$statistic_kind, "z")
  }
})

test_that("get_effects_from_data round-trips uploaded data and returns set_effects string", {
  # Generate synthetic pilot data with known effects.
  set.seed(2137)
  n <- 200L
  x1 <- rnorm(n); x2 <- rnorm(n)
  y  <- 0.4 * x1 + 0.2 * x2 + rnorm(n, sd = 0.8)
  pilot <- data.frame(y = y, x1 = x1, x2 = x2)

  m <- MCPower$new("y ~ x1 + x2")
  m$upload_data(pilot, mode = "partial", verbose = FALSE)
  # get_effects_from_data returns a "x1=..., x2=..." string
  eff_str <- m$get_effects_from_data("y", verbose = FALSE)
  expect_true(is.character(eff_str) && nzchar(eff_str))
  # The string should be parseable by set_effects.
  expect_no_error(MCPower$new("y ~ x1 + x2")$set_effects(eff_str))
  # Recovered effects should be in the right ballpark (large sample).
  # Parse "x1=0.4141, x2=0.2..." by extracting value after each '='.
  pairs <- strsplit(eff_str, ",\\s*")[[1]]
  parsed_vals <- as.numeric(sub("^[^=]+=", "", pairs))
  expect_equal(length(parsed_vals), 2L)
  expect_equal(parsed_vals[1], 0.4, tolerance = 0.15)
  expect_equal(parsed_vals[2], 0.2, tolerance = 0.15)
})

test_that("get_effects_from_data recovers a GLM log-odds coefficient", {
  # Saturated 2x2 table with odds ratio 4 → logistic slope = ln(4). The GLM fit
  # uses the native 0/1 outcome (NOT a z-scored outcome).
  # x=1: 20 successes / 10 failures (odds 2); x=0: 10 / 20 (odds 0.5); OR = 4.
  x <- c(rep(1, 30), rep(0, 30))
  y <- c(rep(1, 20), rep(0, 10), rep(1, 10), rep(0, 20))
  m <- MCPower$new("y ~ x", family = "logit")
  m$set_baseline_probability(0.5)
  m$upload_data(data.frame(y = y, x = x), mode = "partial", verbose = FALSE)
  eff_str <- m$get_effects_from_data("y", verbose = FALSE)
  val <- as.numeric(sub("^x=", "", eff_str))
  expect_equal(val, log(4), tolerance = 1e-3)
})

test_that("get_effects_from_data recovers an MLE fixed effect", {
  # Clustered data y = beta*x + u_cluster + eps. The fixed effect is recovered
  # on the z-scored-x scale (beta * sd(x)), fitting on the native outcome and
  # threading the uploaded grouping column as cluster IDs.
  set.seed(11)
  n <- 200L; n_clusters <- 8L; csize <- n %/% n_clusters
  x <- rnorm(n)
  group <- rep(seq_len(n_clusters) - 1L, each = csize)  # 0..7
  u <- (group - (n_clusters - 1) / 2) * 0.4             # per-cluster intercept
  eps <- rnorm(n, sd = 0.1)
  beta_true <- 1.5
  y <- beta_true * x + u + eps

  m <- MCPower$new("y ~ x + (1|group)", family = "lme")
  m$set_cluster("group", ICC = 0.2, n_clusters = n_clusters)
  m$upload_data(data.frame(y = y, x = x, group = group), mode = "partial", verbose = FALSE)
  eff_str <- m$get_effects_from_data("y", verbose = FALSE)
  val <- as.numeric(sub("^x=", "", eff_str))
  expected <- beta_true * sqrt(mean((x - mean(x))^2))  # population sd(x)
  expect_equal(val, expected, tolerance = 0.15)
})

.icc_from_msgs <- function(msgs) {
  # Pull the ICC value out of the "Estimated ICC...: <x> (..." note line.
  line <- grep("Estimated ICC", msgs, value = TRUE)[1]
  as.numeric(sub("^.*:\\s*([0-9.]+).*$", "\\1", line))
}

test_that("get_effects_from_data reports an estimated ICC for a linear-mixed upload", {
  # Random-intercept data with true ICC ~ 0.33 (var(u)=0.5, var(eps)=1). The
  # recovered ICC = tau^2/(tau^2+sigma^2) must sit in a discriminating band
  # (the latent pi^2/3 formula would land near 0.13, outside it).
  set.seed(7)
  n_clusters <- 40L; csize <- 25L; n <- n_clusters * csize
  x <- rnorm(n)
  u <- rnorm(n_clusters, sd = sqrt(0.5))
  group <- rep(seq_len(n_clusters) - 1L, each = csize)
  eps <- rnorm(n, sd = 1)
  y <- 0.8 * x + u[group + 1L] + eps

  m <- MCPower$new("y ~ x + (1|group)", family = "lme")
  m$set_cluster("group", ICC = 0.3, n_clusters = n_clusters)
  m$upload_data(data.frame(y = y, x = x, group = group), mode = "partial", verbose = FALSE)

  msgs <- testthat::capture_messages(m$get_effects_from_data("y"))
  joined <- paste(msgs, collapse = "")
  expect_match(joined, "Estimated ICC:")
  expect_false(grepl("logit latent scale", joined))
  expect_match(joined, "set_cluster\\('group', ICC = ")
  icc <- .icc_from_msgs(msgs)
  expect_true(icc > 0.15 && icc < 0.55)
  # Gaussian outcome: betas[1] is a mean offset, not a probability — no baseline.
  expect_false(grepl("Estimated baseline probability", joined))
})

test_that("get_effects_from_data reports a latent-scale ICC for a logistic upload", {
  set.seed(13)
  n_clusters <- 30L; csize <- 40L; n <- n_clusters * csize
  x <- rnorm(n)
  u <- rnorm(n_clusters, sd = 1.2)
  group <- rep(seq_len(n_clusters) - 1L, each = csize)
  eta <- 0.7 * x + u[group + 1L]
  p <- 1 / (1 + exp(-eta))
  y <- as.numeric(runif(n) < p)

  m <- MCPower$new("y ~ x + (1|group)", family = "logit")
  m$set_cluster("group", ICC = 0.2, n_clusters = n_clusters)
  m$upload_data(data.frame(y = y, x = x, group = group), mode = "partial", verbose = FALSE)

  msgs <- testthat::capture_messages(m$get_effects_from_data("y"))
  joined <- paste(msgs, collapse = "")
  expect_match(joined, "Estimated ICC \\(logit latent scale\\):")
  icc <- .icc_from_msgs(msgs)
  expect_true(icc > 0 && icc < 1)
  # A binary outcome also reports the recovered baseline probability ∈ (0,1),
  # independent of clustering.
  bline <- grep("Estimated baseline probability:", msgs, value = TRUE)[1]
  expect_false(is.na(bline))
  p_hat <- as.numeric(sub("^.*:\\s*([0-9.]+).*$", "\\1", bline))
  expect_true(p_hat > 0 && p_hat < 1)
  expect_match(joined, "set_baseline_probability\\(")
})

test_that("get_effects_from_data reports a probit-latent-scale ICC (M9)", {
  # Clustered probit upload: the residual variance for the probit latent
  # scale is fixed at 1 (Phi's scale), not logit's pi^2/3 — mirrors
  # driver.rs's cluster_icc branching for BinaryLink::Probit.
  set.seed(17)
  n_clusters <- 30L; csize <- 40L; n <- n_clusters * csize
  x <- rnorm(n)
  u <- rnorm(n_clusters, sd = 1.0)
  group <- rep(seq_len(n_clusters) - 1L, each = csize)
  eta <- 0.7 * x + u[group + 1L]
  p <- pnorm(eta)
  y <- as.numeric(runif(n) < p)

  m <- MCPower$new("y ~ x + (1|group)", family = "probit")
  m$set_cluster("group", ICC = 0.2, n_clusters = n_clusters)
  m$upload_data(data.frame(y = y, x = x, group = group), mode = "partial", verbose = FALSE)

  msgs <- testthat::capture_messages(m$get_effects_from_data("y"))
  joined <- paste(msgs, collapse = "")
  expect_match(joined, "Estimated ICC \\(probit latent scale\\):")
  expect_false(grepl("logit latent scale", joined))
  icc <- .icc_from_msgs(msgs)
  expect_true(icc > 0 && icc < 1)

  # Discriminate against the (wrong) logit residual pi^2/3: the tau^2
  # implied by the printed ICC under the correct residual=1 formula would
  # print a materially different ICC under the logit formula.
  tau_sq_implied <- icc / (1 - icc)
  icc_if_logit_formula <- tau_sq_implied / (tau_sq_implied + pi^2 / 3)
  expect_true(abs(icc - icc_if_logit_formula) > 0.05)
})

test_that("get_effects_from_data emits no ICC line for a clustered Poisson upload (M9)", {
  # A log-link count model has no latent-scale residual to form an ICC ratio
  # against (raw tau^2, not ICC-derived) — mirrors driver.rs's cluster_icc,
  # which returns None for MixedOutcome::Poisson.
  set.seed(19)
  n_clusters <- 20L; csize <- 20L; n <- n_clusters * csize
  x <- rnorm(n)
  u <- rnorm(n_clusters, sd = sqrt(0.3))
  group <- rep(seq_len(n_clusters) - 1L, each = csize)
  eta <- 0.3 * x + u[group + 1L]
  y <- rpois(n, lambda = exp(eta))

  m <- MCPower$new("y ~ x + (1|group)", family = "poisson")
  m$set_cluster("group", tau_squared = 0.3, n_clusters = n_clusters)
  m$upload_data(data.frame(y = y, x = x, group = group), mode = "partial", verbose = FALSE)

  msgs <- testthat::capture_messages(m$get_effects_from_data("y"))
  joined <- paste(msgs, collapse = "")
  expect_match(joined, "APPROXIMATION")
  expect_false(grepl("Estimated ICC", joined))
})

test_that("get_effects_from_data probit round-trip recovers the true baseline probability", {
  # B1 round-trip: data generated from a probit model with true baseline
  # p=0.30 must recover ~0.30, not the logit-mistaken 0.372 (logistic applied
  # to a probit intercept: qnorm(0.30) ~ -0.524, and 1/(1+exp(0.524)) ~ 0.372).
  set.seed(21)
  n <- 8000L
  x <- rnorm(n)
  beta_true <- 0.5
  b0_true <- qnorm(0.30)
  eta <- b0_true + beta_true * x
  p <- pnorm(eta)
  y <- as.numeric(runif(n) < p)

  m <- MCPower$new("y ~ x", family = "probit")
  m$set_baseline_probability(0.30)
  m$upload_data(data.frame(y = y, x = x), mode = "partial", verbose = FALSE)

  msgs <- testthat::capture_messages(m$get_effects_from_data("y"))
  bline <- grep("Estimated baseline probability:", msgs, value = TRUE)[1]
  expect_false(is.na(bline))
  p_hat <- as.numeric(sub("^.*:\\s*([0-9.]+).*$", "\\1", bline))
  expect_equal(p_hat, 0.30, tolerance = 0.05)
})

test_that("get_effects_from_data logit round-trip recovers the true baseline probability", {
  # Companion to the probit round-trip above so the link branch cannot
  # collapse silently.
  set.seed(23)
  n <- 8000L
  x <- rnorm(n)
  beta_true <- 0.5
  b0_true <- log(0.30 / 0.70)
  eta <- b0_true + beta_true * x
  p <- 1 / (1 + exp(-eta))
  y <- as.numeric(runif(n) < p)

  m <- MCPower$new("y ~ x", family = "logit")
  m$set_baseline_probability(0.30)
  m$upload_data(data.frame(y = y, x = x), mode = "partial", verbose = FALSE)

  msgs <- testthat::capture_messages(m$get_effects_from_data("y"))
  bline <- grep("Estimated baseline probability:", msgs, value = TRUE)[1]
  expect_false(is.na(bline))
  p_hat <- as.numeric(sub("^.*:\\s*([0-9.]+).*$", "\\1", bline))
  expect_equal(p_hat, 0.30, tolerance = 0.05)
})

test_that("get_effects_from_data emits no ICC line for non-clustered OLS", {
  set.seed(3)
  x <- rnorm(100)
  y <- 1.5 * x + rnorm(100, sd = 0.3)
  m <- MCPower$new("y ~ x")
  m$upload_data(data.frame(y = y, x = x), mode = "partial", verbose = FALSE)
  msgs <- testthat::capture_messages(m$get_effects_from_data("y"))
  joined <- paste(msgs, collapse = "")
  expect_match(joined, "APPROXIMATION")
  expect_false(grepl("Estimated ICC", joined))
  # Continuous OLS outcome: no baseline probability either.
  expect_false(grepl("Estimated baseline probability", joined))
})

test_that("get_effects_from_data raises without uploaded data", {
  m <- MCPower$new("y ~ x1")
  expect_error(m$get_effects_from_data("y"), regexp = "upload_data")
})
