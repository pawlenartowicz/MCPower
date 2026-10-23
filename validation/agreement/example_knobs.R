#!/usr/bin/env Rscript
# Example 4 check: the scenario knobs, MCPower against an independent lm() loop
# that hard-codes each knob's engine definition.
#
# Model: y ~ treatment + stress * sleep, OLS, N = 130. treatment a two-level
# factor (proportions 0.5/0.5), effect treatment[2] = 0.50; stress, sleep
# continuous, corr 0.5, effects 0.25; stress:sleep fixed at 0.22. Targets
# treatment[2] and stress:sleep. Seven cases: ex4_baseline
# (the optimistic scenario) and one custom scenario per knob, every other knob
# at its optimistic value. MCPower side: find_power at 10,000 sims per scenario.
# Loop side: 2,000 sims. Seed 2137 + n on both sides.
#
# Run from any directory:  Rscript example_knobs.R   (SMOKE=1 for a 2%-sims call check)
# Output: results/example_knobs_r.json ({meta, records}) beside this file.

HERE <- local({
  fa <- grep("^--file=", commandArgs(FALSE), value = TRUE)
  if (length(fa)) dirname(normalizePath(sub("^--file=", "", fa[1]))) else getwd()
})
source(file.path(HERE, "harness.R"))   # record, build_meta, scaled_sims; no run (guarded)

# Read by scaled_sims() and build_meta(). SMOKE=1 checks the calls, not the numbers.
N_SIMS_SCALE <- if (identical(Sys.getenv("SMOKE"), "1")) 0.02 else 1.0
MC_SIMS <- scaled_sims(10000L)
LOOP_SIMS <- scaled_sims(2000L)
ALPHA <- 0.05

N <- 130L
TARGETS <- c("treatment[2]", "stress:sleep")
# find_power results label a factor dummy by its level; the loop's lm term is
# the 0/1 treatment column.
MC_LABEL <- c(`treatment[2]` = "2", `stress:sleep` = "stress:sleep")
LOOP_TERM <- c(`treatment[2]` = "treatment", `stress:sleep` = "stress:sleep")
# Design columns in x_full order: factor dummy, continuous columns, interaction.
BETA <- c(treatment = 0.50, stress = 0.25, sleep = 0.25, `stress:sleep` = 0.22)
# Treatment levels 1 and 2; the factor is outside the correlation matrix.
P_TREAT <- c(0.5, 0.5)
MAIN <- c("stress", "sleep")
R_SS <- 0.5
CORR <- matrix(c(1, R_SS, R_SS, 1), 2, 2, dimnames = list(MAIN, MAIN))

# A custom scenario starts from these optimistic values — mirrors
# configs/scenarios.json "optimistic" and the custom-scenario merge in
# ports/r/R/mcpower.R set_scenario_configs — change together.
OPTIMISTIC <- list(
  heterogeneity = 0, heteroskedasticity_ratio = 1, correlation_noise_sd = 0,
  sampled_factor_proportions = FALSE, distribution_change_prob = 0,
  new_distributions = c("right_skewed", "left_skewed", "uniform"),
  residual_change_prob = 0, residual_dists = c("high_kurtosis", "right_skewed"),
  residual_df = 10
)

KNOBS <- list(
  heterogeneity      = list(heterogeneity = 0.2),
  correlation_noise  = list(correlation_noise_sd = 0.15),
  distribution_swap  = list(distribution_change_prob = 0.5),
  residual_swap      = list(residual_change_prob = 0.5, residual_df = 8),
  heteroskedasticity = list(heteroskedasticity_ratio = 2.0),
  sampled_groups     = list(sampled_factor_proportions = TRUE)
)

CASES <- c(
  list(list(id = "ex4_baseline", scenario = "optimistic", knob = OPTIMISTIC,
            mirrors = "scenario optimistic")),
  lapply(names(KNOBS), function(k) list(
    id = paste0("ex4_", k), scenario = k, knob = modifyList(OPTIMISTIC, KNOBS[[k]]),
    mirrors = paste("knob", paste(sprintf("%s=%s", names(KNOBS[[k]]),
                                          vapply(KNOBS[[k]], format, character(1))),
                                  collapse = ", "))))
)

## --- loop side: engine knob definitions ------------------------------------------

# Correlation noise — mirrors crates/engine-core/src/scenarios.rs
# perturb_correlation — change together: an iid N(0, sd^2) matrix, symmetrized
# as (M + t(M)) / 2, added to the base, clipped to +-0.8, unit diagonal. The
# matrix covers the continuous columns only (stress, sleep); the factor
# treatment is outside it.
perturb_correlation <- function(R, sd) {
  if (sd == 0) return(R)
  k <- nrow(R)
  M <- matrix(rnorm(k * k, sd = sd), k, k)
  out <- pmin(pmax(R + (M + t(M)) / 2, -0.8), 0.8)
  diag(out) <- 1
  out
}

# Lower Cholesky factor with PSD repair — mirrors
# crates/engine-core/src/correlation.rs psd_repair_and_factor — change
# together: plain Cholesky when it succeeds; otherwise floor the eigenvalues at
# 1e-10, rebuild, rescale to unit diagonal and factor that.
chol_lower <- function(R) {
  L <- tryCatch(t(chol(R)), error = function(e) NULL)
  if (!is.null(L)) return(L)
  ev <- eigen(R, symmetric = TRUE)
  A <- ev$vectors %*% diag(pmax(ev$values, 1e-10)) %*% t(ev$vectors)
  d <- sqrt(diag(A))
  A <- A / outer(d, d)
  diag(A) <- 1
  t(chol(A))
}

# Marginal transforms of a latent N(0, 1) column — mirrors
# crates/engine-core/src/data_gen.rs apply_marginal — change together. The
# skewed shapes are an Exp(1) inverse-CDF censored at EXP_CAP and standardized
# by the censored moments; uniform is U(-sqrt(3), sqrt(3)).
EXP_CAP <- 6.95925599364711
EXP_CENSORED_MEAN <- 0.9990501970288289
EXP_CENSORED_STD <- 0.9933676327697134
marginal <- function(z, dist) switch(dist,
  normal       = z,
  right_skewed = (pmin(-log(pnorm(-z)), EXP_CAP) - EXP_CENSORED_MEAN) / EXP_CENSORED_STD,
  left_skewed  = (EXP_CENSORED_MEAN - pmin(-log(pnorm(z)), EXP_CAP)) / EXP_CENSORED_STD,
  uniform      = -sqrt(3) + 2 * sqrt(3) * pnorm(z),
  stop(sprintf("unknown distribution %s", dist)))

# Distribution swap — mirrors scenarios.rs perturb_var_types — change together:
# each unpinned continuous-normal column (stress, sleep; the factor treatment
# never swaps) switches with probability distribution_change_prob to a uniform
# pick from new_distributions, once per simulation.
pick_dists <- function(k) {
  dists <- c(stress = "normal", sleep = "normal")
  for (v in names(dists)) {
    if (runif(1) < k$distribution_change_prob) {
      dists[[v]] <- k$new_distributions[sample.int(length(k$new_distributions), 1L)]
    }
  }
  dists
}

# Residual swap — mirrors scenarios.rs pick_residual and the residual pass in
# data_gen.rs — change together: once per simulation, with probability
# residual_change_prob, the normal residual becomes a uniform pick from
# residual_dists. high_kurtosis is t(df) scaled to unit variance; right_skewed
# is (chi2_df - df) / sqrt(2 df). df is floored at 3 and the chi-square sums
# round(df) squared normals.
draw_residual <- function(k) {
  dist <- "normal"
  if (runif(1) < k$residual_change_prob) {
    dist <- k$residual_dists[sample.int(length(k$residual_dists), 1L)]
  }
  df <- max(k$residual_df, 3)
  switch(dist,
    normal        = rnorm(N),
    high_kurtosis = rnorm(N) / sqrt(rchisq(N, round(df)) / df) * sqrt((df - 2) / df),
    right_skewed  = (rchisq(N, round(df)) - df) / sqrt(2 * df),
    stop(sprintf("unknown residual %s", dist)))
}

# Heterogeneity — mirrors the per-study beta jitter in data_gen.rs — change
# together: once per simulation each effect (the interaction included) gets
# N(0, (heterogeneity * |beta|)^2) added and is clipped toward 0, never across it.
jitter_beta <- function(h) {
  raw <- BETA + rnorm(length(BETA)) * h * abs(BETA)
  ifelse(BETA > 0, pmax(raw, 0), ifelse(BETA < 0, pmin(raw, 0), raw))
}

# Heteroskedasticity — mirrors the scaler in data_gen.rs and
# crates/engine-core/src/spec.rs compute_het_coeffs — change together:
# Var(e_i) is multiplied by exp(g * z_i) / exp(g^2 / 2), g = ln(ratio) / 4, where
# z is the unjittered linear predictor (interaction included) standardized by
# the population moments of the full design, interaction column and its beta
# included. Columns: treatment dummy (mean p, SD sqrt(p(1 - p)), p = 0.5);
# stress, sleep (mean 0, SD 1, correlation R_SS); stress:sleep, the raw product
# (mean R_SS, variance 1 + R_SS^2 by Isserlis). The treatment dummy is
# independent of the continuous block, and every other covariance is an odd
# moment of the continuous block or cancels (Cov(treatment, stress:sleep) =
# p * R_SS - p * R_SS = 0), so the covariance matrix is block diagonal.
COL_MU <- c(P_TREAT[2], 0, 0, R_SS)
COL_COV <- diag(c(P_TREAT[2] * (1 - P_TREAT[2]), 1, 1, 1 + R_SS^2))
COL_COV[2, 3] <- COL_COV[3, 2] <- R_SS
LP_MEAN <- sum(BETA * COL_MU)
LP_SD <- sqrt(drop(t(BETA) %*% COL_COV %*% BETA))

# Fixed factor allocation — mirrors crates/engine-core/src/data_gen.rs
# fixed_level_next — change together: n steps of a largest-remainder walk; each
# step takes the level with the largest deficit w * (steps + 1) - count, the
# lowest index on exact ties (a single factor needs no tie score). It draws no
# random numbers, so the sequence is the same in every simulation.
fixed_levels <- function(weights, n) {
  counts <- numeric(length(weights))
  out <- integer(n)
  for (t in seq_len(n)) {
    g <- which.max(weights * t - counts)
    counts[g] <- counts[g] + 1
    out[t] <- g
  }
  out
}
TREAT_FIXED <- fixed_levels(P_TREAT, N)

# Factor-level allocation — mirrors data_gen.rs, where sampled_factor_proportions
# switches a factor from the fixed walk to one categorical draw per row with the
# level proportions — change together.
draw_treatment <- function(k) {
  if (k$sampled_factor_proportions) sample.int(length(P_TREAT), N, replace = TRUE, prob = P_TREAT)
  else TREAT_FIXED
}

draw_data <- function(k) {
  L <- chol_lower(perturb_correlation(CORR, k$correlation_noise_sd))
  Z <- matrix(rnorm(N * 2L), N, 2L) %*% t(L)
  dists <- pick_dists(k)
  d <- data.frame(treatment = as.numeric(draw_treatment(k) == 2L),
                  stress = marginal(Z[, 1], dists[["stress"]]),
                  sleep = marginal(Z[, 2], dists[["sleep"]]))
  X <- cbind(as.matrix(d), d$stress * d$sleep)
  b <- if (k$heterogeneity > 0) jitter_beta(k$heterogeneity) else BETA
  e <- draw_residual(k)
  g <- log(max(k$heteroskedasticity_ratio, 1)) / 4
  if (g > 0) {
    z <- (drop(X %*% BETA) - LP_MEAN) / LP_SD
    e <- e * sqrt(exp(g * z) * exp(-g^2 / 2))
  }
  d$y <- drop(X %*% b) + e
  d
}

# Two-sided t test at the residual df. A failed or rank-deficient fit counts as
# not significant.
t_sig <- function(fit, term) {
  if (is.null(fit) || fit$rank < length(coef(fit))) return(FALSE)
  s <- summary(fit)$coefficients
  abs(s[term, "t value"]) > qt(1 - ALPHA / 2, fit$df.residual)
}

loop_power <- function(k, n_sims, seed) {
  set.seed(seed)
  hits <- numeric(length(TARGETS))
  for (i in seq_len(n_sims)) {
    d <- draw_data(k)
    fit <- tryCatch(lm(y ~ treatment + stress * sleep, data = d), error = function(e) NULL)
    hits <- hits + vapply(LOOP_TERM[TARGETS], function(t) t_sig(fit, t), logical(1))
  }
  hits / n_sims
}

## --- MCPower side ---------------------------------------------------------------

mc_model <- function() {
  m <- MCPower$new("y = treatment + stress * sleep")
  m$set_variable_type("treatment=(factor,0.5,0.5)")
  m$set_effects(paste(sprintf("%s=%s", sub("^treatment$", "treatment[2]", names(BETA)), BETA), collapse = ", "))
  m$set_correlations("corr(stress, sleep)=0.5")
  m$set_scenario_configs(KNOBS)
  m
}

# Power per target, in `targets` order, for one scenario of a find_power result.
mc_power <- function(res, targets, scenario) {
  df <- tibble::as_tibble(res)
  vapply(targets, function(t) {
    p <- df$power[df$test == MC_LABEL[[t]] & df$scenario == scenario]
    if (length(p) != 1L) stop(sprintf("test '%s' in '%s': %d rows", t, scenario, length(p)))
    p
  }, numeric(1))
}

## --- run ---------------------------------------------------------------------

records <- list()
add_record <- function(case, method, n_sims, elapsed, power) {
  rec <- record(list(id = case$id, family = "ols", targets = TARGETS),
                method, N, n_sims, elapsed, power)
  rec$targets <- I(TARGETS)
  rec$mirrors <- case$mirrors
  records[[length(records) + 1L]] <<- rec
  cat(sprintf("%-22s %-19s n=%4d sims=%5d %7.2fs  [%s]\n", case$id, method, N, n_sims, elapsed,
              paste(sprintf("%.3f", power), collapse = ", ")))
}

m <- mc_model()
for (case in CASES) {
  el <- system.time(res <- m$find_power(sample_size = N, target_test = "all", n_sims = MC_SIMS,
                                        seed = seed_for(N), scenarios = case$scenario,
                                        verbose = FALSE, progress_callback = FALSE))[["elapsed"]]
  add_record(case, "mcpower_find_power", MC_SIMS, el, mc_power(res, TARGETS, case$scenario))
  el <- system.time(lp <- loop_power(case$knob, LOOP_SIMS, seed_for(N)))[["elapsed"]]
  add_record(case, "loop_best", LOOP_SIMS, el, lp)
}

out <- file.path(HERE, "results", "example_knobs_r.json")
dir.create(dirname(out), showWarnings = FALSE, recursive = TRUE)
jsonlite::write_json(list(meta = build_meta("auto"), records = records), out,
                     auto_unbox = TRUE, digits = NA, pretty = TRUE)
cat(sprintf("wrote %s (%d records)\n", out, length(records)))
