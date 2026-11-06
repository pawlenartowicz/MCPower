# glmm_slope_glmer.R — is the over-rejection of glmm_slope_type1.py shared by lme4?
#
# On three cells of that grid, generates null logit datasets with MCPower and
# fits each one twice, so every rate below is on the same data:
#   * MCPower through load_data(), Wald z, rx and Hessian standard errors
#   * lme4::glmer with its defaults (Laplace, calc.derivs = TRUE), Wald z, read
#     as summary() reports it (Hessian, or rx on a fit with no usable Hessian)
#     and from vcov(use.hessian = FALSE) (rx)
# A draw whose fit fails or does not converge counts as not rejected for that
# method, as in a power run. Rejection is |z| > qnorm(.975) for every method.
# The design matrix already carries the intercept, so glmer fits through the
# origin on it. Columns: intercept, x, w, x:w; targets x and x:w.
#
# Run from mcpower/validation/:
#   GLMER_CORES=8 pin 8 -- Rscript glmm_slope_glmer.R
# Env knobs: GLMER_K (draws per cell, default 2000), GLMER_CORES.
# Writes data/glmm_slope_glmer.csv, one row per cell x method x target.

suppressPackageStartupMessages({
  library(mcpower)
  library(lme4)
})

K     <- as.integer(Sys.getenv("GLMER_K", "2000"))
CORES <- as.integer(Sys.getenv("GLMER_CORES", "1"))
CHUNK <- 50L
Z_CRIT <- stats::qnorm(.975)
BASE_SEED <- 61000000L
TARGETS <- c("x", "x:w")
TIDX <- c(1L, 3L)                        # 0-based design columns of the targets
METHODS <- c("mcpower_rx", "mcpower_hessian", "glmer_rx", "glmer_hessian")

CELLS <- list(
  list(g = 20L, m = 20L, v = 0.3),
  list(g = 60L, m = 8L, v = 0.3),
  list(g = 60L, m = 20L, v = 0.1))

new_model <- function(cell) {
  mod <- MCPowerDebug$new("y ~ x * w + (1 + x|g)", family = "logit",
                          debug_n = as.integer(cell$g * cell$m), debug_n_sims = 1L)
  mod$set_variable_type("w=binary")
  mod$set_baseline_probability(0.5)
  mod$set_effects("x=0, w=0.3, x:w=0")
  mod$set_cluster("g", ICC = 0.3, n_clusters = cell$g, cluster_level_vars = "w",
                  random_slopes = list(list(predictor = "x", variance = cell$v)))
  mod
}

mcpower_z <- function(mod, d, wald_se) {
  fit <- tryCatch(mod$load_data(d, wald_se = wald_se, df_method = "asymptotic"),
                  error = function(e) NULL)
  if (is.null(fit) || !isTRUE(fit$converged)) return(c(NA_real_, NA_real_))
  t <- fit$targets[match(TIDX, vapply(fit$targets, `[[`, 0L, "target_index"))]
  vapply(t, function(x) x$statistic[[1]], numeric(1))
}

glmer_z <- function(d) {
  X <- d$design
  df <- data.frame(V1 = X[, 1], V2 = X[, 2], V3 = X[, 3], V4 = X[, 4], .y = d$outcome,
                   .g = factor(d$cluster_ids))
  # A fit that warns (lme4's convergence checks) does not converge; a singular
  # fit only messages.
  fit <- tryCatch(suppressMessages(
    lme4::glmer(.y ~ 0 + V1 + V2 + V3 + V4 + (1 + V2 | .g), data = df, family = binomial())),
    warning = function(w) NULL, error = function(e) NULL)
  if (is.null(fit)) return(c(NA, NA, NA, NA, NA))
  b <- lme4::fixef(fit)[c("V2", "V4")]
  # vcov() as summary() calls it: the Hessian where one is usable, else rx. Naming
  # use.hessian = TRUE errors on singular fits, which would hide what glmer reports.
  se_h <- suppressWarnings(sqrt(diag(as.matrix(stats::vcov(fit)))))[c(2, 4)]
  se_r <- sqrt(diag(as.matrix(stats::vcov(fit, use.hessian = FALSE))))[c(2, 4)]
  c(b / se_r, b / se_h, as.numeric(lme4::isSingular(fit)))
}

score_range <- function(cell, seed, from, to) {
  mod <- new_model(cell)
  rows <- vector("list", to - from + 1L)
  for (k in seq.int(from, to)) {
    mod$.debug_seed <- as.numeric(seed + k)
    d <- tryCatch(mod$create_data(), error = function(e) NULL)
    r <- rep(NA_real_, 9)
    if (!is.null(d)) r <- c(mcpower_z(mod, d, "rx"), mcpower_z(mod, d, "hessian"), glmer_z(d))
    rows[[k - from + 1L]] <- r
    if (k %% 50L == 0L) gc(FALSE)
  }
  do.call(rbind, rows)
}

t_all <- Sys.time()
out <- list()
for (i in seq_along(CELLS)) {
  cell <- CELLS[[i]]
  seed <- BASE_SEED + 100000L * i
  label <- sprintf("glmm_slope_g%d_m%d_v%g", cell$g, cell$m, cell$v)
  t0 <- Sys.time()
  starts <- seq.int(1L, K, by = CHUNK)
  parts <- parallel::mclapply(starts, function(a) score_range(cell, seed, a, min(a + CHUNK - 1L, K)),
                              mc.cores = max(1L, CORES), mc.preschedule = FALSE)
  bad <- vapply(parts, function(p) inherits(p, "try-error") || !is.matrix(p), logical(1))
  if (any(bad)) stop(sprintf("worker failed (%s): %s", label, paste(unlist(parts[bad]), collapse = " | ")))
  Z <- do.call(rbind, parts)
  singular <- mean(Z[, 9], na.rm = TRUE)
  for (mi in seq_along(METHODS)) for (ti in 1:2) {
    z <- Z[, 2L * (mi - 1L) + ti]
    p <- sum(abs(z) > Z_CRIT, na.rm = TRUE) / K
    out[[length(out) + 1L]] <- data.frame(
      label = label, g = cell$g, cluster_size = cell$m, slope_var = cell$v, n = cell$g * cell$m,
      method = METHODS[mi], target = TARGETS[ti], rejection_rate = p, se = sqrt(p * (1 - p) / K),
      fits_ok = sum(!is.na(z)), glmer_singular = singular, n_draws = K, seed = seed)
  }
  cat(sprintf("%s: %.0fs\n", label, as.numeric(difftime(Sys.time(), t0, units = "secs"))))
  utils::write.csv(do.call(rbind, out), file.path("data", "glmm_slope_glmer.csv"), row.names = FALSE)
}
cat(sprintf("done in %.0f s\n", as.numeric(difftime(Sys.time(), t_all, units = "secs"))))
