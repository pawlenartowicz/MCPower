#!/usr/bin/env Rscript
# Example 1 check: power from an uploaded pilot (pilot.csv) in strict mode,
# MCPower against an independent resampling loop.
#
# Model: outcome ~ conscientiousness + neuroticism + age + gender, OLS, effects
# 0.25 / -0.25 / 0.10 / 0.20, N = 100..300 by 50, all four targets.
# MCPower side: upload_data(pilot, mode = "strict"), find_power at 10,000 sims.
# Loop side: lm() on whole pilot rows resampled with replacement, 2,000 sims.
# Seed 2137 + n on both sides.
#
# Run from any directory:  Rscript example_upload.R   (SMOKE=1 for a 2%-sims call check)
# Outputs (results/ beside this file): example_upload_r.json ({meta, records}),
# example_upload_pilot_quantiles.csv, example_upload_pilot_spearman.csv.

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

CASE_ID <- "ex1_strict"
MIRRORS <- "upload strict"
NS <- seq(100L, 300L, by = 50L)
TARGETS <- c("conscientiousness", "neuroticism", "age", "gender")
BETA <- c(conscientiousness = 0.25, neuroticism = -0.25, age = 0.10, gender = 0.20)
EFFECTS <- paste(sprintf("%s=%s", names(BETA), BETA), collapse = ", ")

RESULTS <- file.path(HERE, "results")
dir.create(RESULTS, showWarnings = FALSE, recursive = TRUE)
pilot <- read.csv(file.path(HERE, "pilot.csv"))

## --- pilot quantiles and Spearman correlations ------------------------------

write_csv_msg <- function(df, name) {
  path <- file.path(RESULTS, name)
  write.csv(df, path, row.names = FALSE)
  cat(sprintf("wrote %s (%d rows)\n", path, nrow(df)))
}

qprobs <- c(0.05, 0.25, 0.50, 0.75, 0.95)
qrows <- do.call(rbind, lapply(TARGETS, function(v) {
  x <- pilot[[v]]
  q <- quantile(x, probs = qprobs, na.rm = TRUE, names = FALSE)
  data.frame(predictor = v, q05 = q[1], q25 = q[2], q50 = q[3], q75 = q[4], q95 = q[5],
             sd = sd(x, na.rm = TRUE))
}))
write_csv_msg(qrows, "example_upload_pilot_quantiles.csv")

cont <- c("conscientiousness", "neuroticism", "age")
srows <- list()
for (i in seq_along(cont)) {
  for (j in seq_along(cont)) {
    if (j > i) {
      rho <- cor(pilot[[cont[i]]], pilot[[cont[j]]], method = "spearman")
      srows[[length(srows) + 1]] <- data.frame(var1 = cont[i], var2 = cont[j], rho = rho)
    }
  }
}
write_csv_msg(do.call(rbind, srows), "example_upload_pilot_spearman.csv")

## --- loop side ----------------------------------------------------------------

# Strict upload as the engine defines it — mirrors
# crates/engine-spec-builder/src/upload.rs (standardize_column) and the bootstrap
# arm in crates/engine-core/src/data_gen.rs — change together:
#   - a continuous column is z-scored once over the whole pilot with the
#     population SD (ddof = 0), never per resample;
#   - a binary column is stored as x - p and decoded back to 0/1 by the
#     bootstrap arm, so gender enters the design as its raw 0/1 value;
#   - each design row copies one pilot row drawn uniformly with replacement, so
#     the four columns keep their joint distribution.
pop_z <- function(x) (x - mean(x)) / sqrt(mean((x - mean(x))^2))
FRAME <- data.frame(conscientiousness = pop_z(pilot$conscientiousness),
                    neuroticism = pop_z(pilot$neuroticism),
                    age = pop_z(pilot$age),
                    gender = pilot$gender)

# Two-sided t test at the residual df. A failed or rank-deficient fit counts as
# not significant.
t_sig <- function(fit, term) {
  if (is.null(fit) || fit$rank < length(coef(fit))) return(FALSE)
  s <- summary(fit)$coefficients
  abs(s[term, "t value"]) > qt(1 - ALPHA / 2, fit$df.residual)
}

loop_power <- function(n, n_sims, seed) {
  set.seed(seed)
  hits <- numeric(length(TARGETS))
  for (i in seq_len(n_sims)) {
    d <- FRAME[sample.int(nrow(FRAME), n, replace = TRUE), , drop = FALSE]
    d$y <- as.vector(as.matrix(d[names(BETA)]) %*% BETA) + rnorm(n)
    fit <- tryCatch(lm(y ~ conscientiousness + neuroticism + age + gender, data = d),
                    error = function(e) NULL)
    hits <- hits + vapply(TARGETS, function(t) t_sig(fit, t), logical(1))
  }
  hits / n_sims
}

## --- MCPower side ---------------------------------------------------------------

mc_model <- function() {
  m <- MCPower$new("outcome = conscientiousness + neuroticism + age + gender")
  m$upload_data(pilot, mode = "strict", verbose = FALSE)
  m$set_effects(EFFECTS)
  m
}

# Power per target, in `targets` order, for one scenario of a find_power result.
mc_power <- function(res, targets, scenario = "optimistic") {
  df <- tibble::as_tibble(res)
  vapply(targets, function(t) {
    p <- df$power[df$test == t & df$scenario == scenario]
    if (length(p) != 1L) stop(sprintf("test '%s' in '%s': %d rows", t, scenario, length(p)))
    p
  }, numeric(1))
}

## --- run ---------------------------------------------------------------------

records <- list()
add_record <- function(method, n, n_sims, elapsed, power) {
  rec <- record(list(id = CASE_ID, family = "ols", targets = TARGETS),
                method, n, n_sims, elapsed, power)
  rec$targets <- I(TARGETS)
  rec$mirrors <- MIRRORS
  records[[length(records) + 1L]] <<- rec
  cat(sprintf("%-11s %-19s n=%4d sims=%5d %7.2fs  [%s]\n", CASE_ID, method, n, n_sims, elapsed,
              paste(sprintf("%.3f", power), collapse = ", ")))
}

m <- mc_model()
for (n in NS) {
  el <- system.time(res <- m$find_power(sample_size = n, target_test = "all", n_sims = MC_SIMS,
                                        seed = seed_for(n), verbose = FALSE,
                                        progress_callback = FALSE))[["elapsed"]]
  add_record("mcpower_find_power", n, MC_SIMS, el, mc_power(res, TARGETS))
  el <- system.time(lp <- loop_power(n, LOOP_SIMS, seed_for(n)))[["elapsed"]]
  add_record("loop_best", n, LOOP_SIMS, el, lp)
}

out <- file.path(RESULTS, "example_upload_r.json")
jsonlite::write_json(list(meta = build_meta("auto"), records = records), out,
                     auto_unbox = TRUE, digits = NA, pretty = TRUE)
cat(sprintf("wrote %s (%d records)\n", out, length(records)))
