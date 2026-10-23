#!/usr/bin/env Rscript
# Example 3 check: cost of misspecifying the analysis model, MCPower against an
# independent lm() loop.
#
# Continuous X, W standardized and independent; M correlated with X at 0.26;
# medium interaction 0.17; N = 100..1000 (nine steps). Seven cases
# ex3_<equation>_<variant>, one target each:
#   y_correct:  generate M:W = 0.17, X:W = 0; test M:W with Y ~ X + W + M + M:W
#   y_over:     same generation; test M:W with Y ~ X + W + M + X:W + M:W
#   y_under:    same generation; test M with Y ~ X + W + M
#   y_complete: generate X:W = 0.17, M:W = 0; test M:W with Y ~ X + W + M + M:W
#   m_correct:  generate X:W = 0.17; test X:W with M ~ X + W + X:W
#   m_under:    same generation; test X with M ~ X + W
#   m_complete: generate X:W = 0; test X:W with M ~ X + W + X:W
# MCPower side: the generating model plus test_formula, find_power at 10,000
# sims. Loop side: 2,000 sims. Seed 2137 + n on both sides.
#
# Run from any directory:  Rscript example_misspec.R   (SMOKE=1 for a 2%-sims call check)
# Output: results/example_misspec_r.json ({meta, records}) beside this file.

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

NS <- c(100L, 150L, 200L, 250L, 300L, 400L, 500L, 750L, 1000L)
INT <- 0.17
A1 <- 0.26  # corr(X, M) via M = a1*X + e, e scaled so M has unit variance

## --- loop side ----------------------------------------------------------------

# Draw X, W independent N(0,1), and M = a1*X + e with e chosen so M has unit
# variance (the MCPower side sets corr(X, M) = 0.26).
gen_xwm <- function(N) {
  X <- rnorm(N)
  W <- rnorm(N)
  e <- rnorm(N, sd = sqrt(1 - A1^2))
  M <- A1 * X + e
  list(X = X, W = W, M = M)
}

sim_y <- function(N, mw, xw) {
  d <- gen_xwm(N)
  X <- d$X; W <- d$W; M <- d$M
  Y <- 0.10 * X + 0.10 * W + 0.26 * M + mw * (M * W) + xw * (X * W) + rnorm(N)
  data.frame(X = X, W = W, M = M, Y = Y)
}

sim_m <- function(N, xw) {
  X <- rnorm(N)
  W <- rnorm(N)
  M <- A1 * X + 0.10 * W + xw * (X * W) + rnorm(N)
  data.frame(X = X, W = W, M = M)
}

# Two-sided t test at the residual df. A failed or rank-deficient fit counts as
# not significant.
t_sig <- function(fit, term) {
  if (is.null(fit) || fit$rank < length(coef(fit))) return(FALSE)
  s <- summary(fit)$coefficients
  # lm() orders an interaction's variables by their first appearance in the
  # formula (M:W after W becomes W:M), so try both orders.
  parts <- strsplit(term, ":")[[1]]
  hit <- intersect(c(term, paste(rev(parts), collapse = ":")), rownames(s))
  if (length(hit) == 0L) stop(sprintf("target %s not among %s", term, paste(rownames(s), collapse = ", ")))
  abs(s[hit[1], "t value"]) > qt(1 - ALPHA / 2, fit$df.residual)
}

loop_power <- function(cell, n, n_sims, seed) {
  set.seed(seed)
  hits <- 0
  for (i in seq_len(n_sims)) {
    df <- cell$gen(n)
    fit <- tryCatch(lm(as.formula(cell$formula), data = df), error = function(e) NULL)
    hits <- hits + t_sig(fit, cell$target)
  }
  hits / n_sims
}

## --- MCPower side ---------------------------------------------------------------

y_model <- function(mw, xw) {
  m <- MCPower$new("Y = X + W + M + X:W + M:W")
  m$set_effects(sprintf("X=0.10, W=0.10, M=0.26, M:W=%s, X:W=%s", mw, xw))
  m$set_correlations("corr(X, M)=0.26")
  m
}

m_model <- function(xw) {
  m <- MCPower$new("M = X + W + X:W")
  m$set_effects(sprintf("X=0.26, W=0.10, X:W=%s", xw))
  m
}

mc_power <- function(res, target) {
  df <- tibble::as_tibble(res)
  p <- df$power[df$test == target & df$scenario == "optimistic"]
  if (length(p) != 1L) stop(sprintf("test '%s': %d rows", target, length(p)))
  p
}

## --- cases ---------------------------------------------------------------------

# formula: the loop's lm() formula; test_formula: the same analysis model in
# MCPower syntax. Both name the target coefficient identically.
CELLS <- list(
  list(id = "ex3_y_correct",  make = function() y_model(INT, 0), gen = function(N) sim_y(N, INT, 0),
       formula = "Y ~ X + W + M + M:W",       test_formula = "Y = X + W + M + M:W",       target = "M:W"),
  list(id = "ex3_y_over",     make = function() y_model(INT, 0), gen = function(N) sim_y(N, INT, 0),
       formula = "Y ~ X + W + M + X:W + M:W", test_formula = "Y = X + W + M + X:W + M:W", target = "M:W"),
  list(id = "ex3_y_under",    make = function() y_model(INT, 0), gen = function(N) sim_y(N, INT, 0),
       formula = "Y ~ X + W + M",             test_formula = "Y = X + W + M",             target = "M"),
  list(id = "ex3_y_complete", make = function() y_model(0, INT), gen = function(N) sim_y(N, 0, INT),
       formula = "Y ~ X + W + M + M:W",       test_formula = "Y = X + W + M + M:W",       target = "M:W"),
  list(id = "ex3_m_correct",  make = function() m_model(INT),    gen = function(N) sim_m(N, INT),
       formula = "M ~ X + W + X:W",           test_formula = "M = X + W + X:W",           target = "X:W"),
  list(id = "ex3_m_under",    make = function() m_model(INT),    gen = function(N) sim_m(N, INT),
       formula = "M ~ X + W",                 test_formula = "M = X + W",                 target = "X"),
  list(id = "ex3_m_complete", make = function() m_model(0),      gen = function(N) sim_m(N, 0),
       formula = "M ~ X + W + X:W",           test_formula = "M = X + W + X:W",           target = "X:W")
)

## --- run ---------------------------------------------------------------------

records <- list()
add_record <- function(cell, method, n, n_sims, elapsed, power) {
  rec <- record(list(id = cell$id, family = "ols", targets = cell$target),
                method, n, n_sims, elapsed, power)
  rec$targets <- I(cell$target)
  rec$mirrors <- paste("test_formula", cell$test_formula)
  records[[length(records) + 1L]] <<- rec
  cat(sprintf("%-15s %-19s n=%4d sims=%5d %7.2fs  [%.3f]\n", cell$id, method, n, n_sims,
              elapsed, power))
}

for (cell in CELLS) {
  m <- cell$make()
  for (n in NS) {
    el <- system.time(res <- m$find_power(sample_size = n, target_test = cell$target,
                                          test_formula = cell$test_formula, n_sims = MC_SIMS,
                                          seed = seed_for(n), verbose = FALSE,
                                          progress_callback = FALSE))[["elapsed"]]
    add_record(cell, "mcpower_find_power", n, MC_SIMS, el, mc_power(res, cell$target))
    el <- system.time(lp <- loop_power(cell, n, LOOP_SIMS, seed_for(n)))[["elapsed"]]
    add_record(cell, "loop_best", n, LOOP_SIMS, el, lp)
  }
}

out <- file.path(HERE, "results", "example_misspec_r.json")
dir.create(dirname(out), showWarnings = FALSE, recursive = TRUE)
jsonlite::write_json(list(meta = build_meta("auto"), records = records), out,
                     auto_unbox = TRUE, digits = NA, pretty = TRUE)
cat(sprintf("wrote %s (%d records)\n", out, length(records)))
