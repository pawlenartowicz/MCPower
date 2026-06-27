# oracle2_power.R — Oracle-2: STATISTICAL truth for the GLMM Wald-SE (Tasks 15+16).
#
# For each harsh cell (oracle_cells.R) and each condition (null β=0, alt β=β*):
#   * Per-draw Monte Carlo (K draws): refit every draw twice on the SAME bytes —
#     wald_se="rx" and wald_se="hessian" — to read the per-fit rx / hessian SE and
#     the (mode-independent) β̂. Skip non-converged / non-finite-SE draws (counted).
#   * sd_true = sd(β̂) over alt draws is the TRUE finite-sample SE — the oracle.
#   * Type-I / power for each per-draw flavour use the engine's NORMAL Wald
#     reference z = qnorm(0.975) = 1.959964 (deliberate: the feature fixes the SE
#     CONVENTION, not the small-sample reference — Type-I > α at few-cluster cells
#     is the normal reference, not an asymp bug; the asymp-vs-hessian delta cancels
#     the reference and is the clean §11.1 decision signal).
#   * asymp + hessian DECISION metrics come from the REAL shipped find_power
#     (n_sims=K, same model, same n & cluster config) — there is no R calibrate
#     entry, so asymp power MUST come from find_power(wald_se="asymp").
#
# SCOPE: pilot K (Sys.getenv("ORACLE2_K","2000")) — NOT the full K=50k (the
# controller green-lights that with the user). Diagnostics, not a CI gate.

suppressPackageStartupMessages(library(mcpower))
source("common.R")       # build_m4_model, %||%
source("formulas.R")     # M4_GLMM_CASES (not used directly; keeps env parity)
source("tolerances.R")   # sourced for parity with the rest of the suite
source("oracle_cells.R") # ORACLE_CELLS, oracle_models, oracle_target

Z_CRIT <- stats::qnorm(0.975)   # 1.959964 — engine's normal Wald reference
K <- as.integer(Sys.getenv("ORACLE2_K", "2000"))
# Per-draw MC is single-core by default; ORACLE2_CORES>1 fans the draw range
# across forked workers. The per-draw path (load_data, one fit) is single-
# threaded — no rayon — so forking is safe; find_power stays in the master.
CORES <- as.integer(Sys.getenv("ORACLE2_CORES", "1"))
cat(sprintf("Oracle-2 — K = %d draws/condition/cell, z = %.6f, cores = %d\n\n",
            K, Z_CRIT, CORES))

# ---- per-draw MC over a draw RANGE [k_from, k_to] of one condition ------------
# Draw k is fully determined by .debug_seed = base_seed + k, so any split of the
# 1:K range across workers — recombined in chunk order — yields identical kept
# vectors, and the downstream aggregates (sd / mean / reject-rate) are order-
# invariant regardless. A draw is skipped when either fit is non-converged or its
# target SE is NaN/<=0.
per_draw_mc_range <- function(model, target_col, base_seed, k_from, k_to) {
  beta_hat <- se_rx <- se_hess <- numeric(0)
  skipped <- 0L
  for (k in seq.int(k_from, k_to)) {
    model$.debug_seed <- as.numeric(base_seed + k)
    d  <- tryCatch(model$create_data(), error = function(e) NULL)
    if (is.null(d)) { skipped <- skipped + 1L; next }
    fr <- tryCatch(model$load_data(d, wald_se = "rx"),      error = function(e) NULL)
    fh <- tryCatch(model$load_data(d, wald_se = "hessian"), error = function(e) NULL)
    if (is.null(fr) || is.null(fh) ||
        !isTRUE(fr$converged) || !isTRUE(fh$converged)) { skipped <- skipped + 1L; next }
    tr <- oracle_target(fr, target_col); th <- oracle_target(fh, target_col)
    sr <- tr$se; sh <- th$se; bh <- th$beta
    if (!is.finite(sr) || !is.finite(sh) || sr <= 0 || sh <= 0 || !is.finite(bh)) {
      skipped <- skipped + 1L; next
    }
    beta_hat <- c(beta_hat, bh); se_rx <- c(se_rx, sr); se_hess <- c(se_hess, sh)
  }
  list(beta_hat = beta_hat, se_rx = se_rx, se_hess = se_hess, skipped = skipped)
}

# Run K draws for one condition ("alt"/"null"), single-core or fanned across
# CORES forked workers. Each worker rebuilds the model (cheap, deterministic) for
# full isolation; copy-on-write fork inherits the already-loaded engine.
run_per_draw <- function(cell, which, base_seed, K) {
  mk <- function() { mm <- oracle_models(cell, 0L); if (which == "alt") mm$alt else mm$null }
  if (CORES <= 1L)
    return(per_draw_mc_range(mk(), cell$target_col, base_seed, 1L, K))
  bnds <- floor(seq(0, K, length.out = CORES + 1L))
  jobs <- Map(function(a, b) c(a + 1L, b), bnds[-length(bnds)], bnds[-1L])
  jobs <- Filter(function(r) r[1L] <= r[2L], jobs)
  parts <- parallel::mclapply(jobs, function(r)
    per_draw_mc_range(mk(), cell$target_col, base_seed, r[1L], r[2L]),
    mc.cores = CORES, mc.preschedule = FALSE)
  bad <- vapply(parts, function(p) inherits(p, "try-error") || !is.list(p), logical(1))
  if (any(bad))
    stop(sprintf("mclapply worker failed in per-draw MC (%s): %s",
                 which, paste(unlist(parts[bad]), collapse = " | ")))
  list(  # recombine in k-chunk order; skips summed
    beta_hat = unlist(lapply(parts, `[[`, "beta_hat"), use.names = FALSE),
    se_rx    = unlist(lapply(parts, `[[`, "se_rx"),    use.names = FALSE),
    se_hess  = unlist(lapply(parts, `[[`, "se_hess"),  use.names = FALSE),
    skipped  = sum(vapply(parts, `[[`, integer(1), "skipped"))
  )
}

# ---- find_power power read (target marginal, NORMAL Wald reference) -----------
fp_power <- function(model, n, n_sims, mode, seed) {
  r <- suppressWarnings(model$find_power(sample_size = n, n_sims = n_sims,
                                         wald_se = mode, seed = seed,
                                         progress_callback = FALSE, verbose = FALSE))
  r$power_uncorrected[[1]]
}

results <- list()
for (nm in names(ORACLE_CELLS)) {
  cell <- ORACLE_CELLS[[nm]]
  cat(sprintf("=== %s (%s) n=%d, %d×%d, ICC=%.2f ===\n", nm, cell$structure,
              cell$n, cell$cluster$n_clusters, cell$cluster$cluster_size, cell$cluster$ICC))

  mm <- oracle_models(cell, 0L)   # kept for the find_power calls below

  # Per-draw MC, alt then null. Distinct seed offsets keep the two conditions
  # from sharing a draw stream. run_per_draw fans across CORES when >1.
  alt  <- run_per_draw(cell, "alt",  cell$seed,            K)
  null <- run_per_draw(cell, "null", cell$seed + 1000000L, K)

  kept_alt  <- length(alt$beta_hat)
  kept_null <- length(null$beta_hat)
  skip_rate <- (alt$skipped + null$skipped) / (2 * K)

  sd_true        <- stats::sd(alt$beta_hat)
  mean_se_rx     <- mean(alt$se_rx)
  mean_se_hess   <- mean(alt$se_hess)
  se_mean_hessian <- mean_se_hess            # the rejected MEAN flavour == mean-of-hessian-SE

  se_bias_pct_rx   <- 100 * (mean_se_rx   / sd_true - 1)
  se_bias_pct_hess <- 100 * (mean_se_hess / sd_true - 1)

  # Type-I (null draws) and power (alt draws), per flavour. mean_hessian uses the
  # constant alt-cell se_mean_hessian denominator.
  typeI_rx           <- mean(abs(null$beta_hat / null$se_rx)   > Z_CRIT)
  typeI_hessian      <- mean(abs(null$beta_hat / null$se_hess) > Z_CRIT)
  typeI_mean_hessian <- mean(abs(null$beta_hat) / se_mean_hessian > Z_CRIT)

  power_true         <- mean(abs(alt$beta_hat / sd_true)      > Z_CRIT)
  power_rx           <- mean(abs(alt$beta_hat / alt$se_rx)    > Z_CRIT)
  power_hessian      <- mean(abs(alt$beta_hat / alt$se_hess)  > Z_CRIT)
  power_mean_hessian <- mean(abs(alt$beta_hat) / se_mean_hessian > Z_CRIT)

  # Real shipped find_power — asymp + hessian, alt (power) and null (Type-I).
  # Paired seed so asymp/hessian see the SAME draws (the §11.1 delta cancels MC).
  power_asymp     <- fp_power(mm$alt,  cell$n, K, "asymp",   cell$seed)
  power_hessian_fp <- fp_power(mm$alt, cell$n, K, "hessian", cell$seed)
  typeI_asymp     <- fp_power(mm$null, cell$n, K, "asymp",   cell$seed + 2000000L)
  typeI_hessian_fp <- fp_power(mm$null, cell$n, K, "hessian", cell$seed + 2000000L)

  decision_delta <- abs(power_asymp - power_hessian_fp)
  consistency    <- abs(power_hessian_fp - power_hessian)

  row <- data.frame(
    cell = nm, n = cell$n, n_clusters = cell$cluster$n_clusters,
    cluster_size = cell$cluster$cluster_size, ICC = cell$cluster$ICC,
    structure = cell$structure,
    kept_alt = kept_alt, kept_null = kept_null, skip_rate = skip_rate,
    sd_true = sd_true, mean_se_rx = mean_se_rx, mean_se_hess = mean_se_hess,
    se_bias_pct_rx = se_bias_pct_rx, se_bias_pct_hess = se_bias_pct_hess,
    typeI_rx = typeI_rx, typeI_hessian = typeI_hessian,
    typeI_mean_hessian = typeI_mean_hessian, typeI_asymp = typeI_asymp,
    typeI_hessian_fp = typeI_hessian_fp,
    power_true = power_true, power_rx = power_rx, power_hessian = power_hessian,
    power_mean_hessian = power_mean_hessian, power_asymp = power_asymp,
    power_hessian_fp = power_hessian_fp,
    decision_delta = decision_delta, consistency_hess = consistency,
    stringsAsFactors = FALSE
  )
  results[[nm]] <- row
  saveRDS(row, file.path("data", sprintf("oracle2_%s.rds", nm)))

  # Sanity asserts — WARN only (harsh cells may legitimately violate small-sample
  # expectations); never hard-stop.
  if (!(mean_se_rx <= mean_se_hess + 1e-9))
    cat(sprintf("  WARN core invariant violated: mean_se_rx %.4f > mean_se_hess %.4f\n",
                mean_se_rx, mean_se_hess))
  if (!(typeI_rx >= typeI_hessian - 1e-9))
    cat(sprintf("  WARN typeI_rx %.4f < typeI_hessian %.4f (rx not the more anticonservative)\n",
                typeI_rx, typeI_hessian))
  if (consistency > 0.05)
    cat(sprintf("  WARN find_power(hessian)=%.3f diverges from per-draw hessian=%.3f (|Δ|=%.3f)\n",
                power_hessian_fp, power_hessian, consistency))

  cat(sprintf("  kept alt=%d null=%d  skip_rate=%.3f\n", kept_alt, kept_null, skip_rate))
  cat(sprintf("  se: sd_true=%.4f  mean_rx=%.4f (bias %+.1f%%)  mean_hess=%.4f (bias %+.1f%%)\n",
              sd_true, mean_se_rx, se_bias_pct_rx, mean_se_hess, se_bias_pct_hess))
  cat(sprintf("  Type-I: rx=%.3f hess=%.3f mean_hess=%.3f | asymp=%.3f hess_fp=%.3f\n",
              typeI_rx, typeI_hessian, typeI_mean_hessian, typeI_asymp, typeI_hessian_fp))
  cat(sprintf("  power : true=%.3f rx=%.3f hess=%.3f mean_hess=%.3f | asymp=%.3f hess_fp=%.3f\n",
              power_true, power_rx, power_hessian, power_mean_hessian, power_asymp, power_hessian_fp))
  cat(sprintf("  >>> §11.1 DECISION DELTA |power_asymp - power_hessian_fp| = %.4f  (consistency |Δhess|=%.3f)\n\n",
              decision_delta, consistency))
}

summ <- do.call(rbind, results)
cat("\n================ COMBINED SUMMARY ================\n")
print(summ[, c("cell", "kept_alt", "skip_rate", "se_bias_pct_rx", "se_bias_pct_hess",
               "typeI_rx", "typeI_hessian", "typeI_asymp", "typeI_hessian_fp",
               "power_asymp", "power_hessian_fp", "decision_delta")], row.names = FALSE)
cat("\n§11.1 decision deltas |power_asymp - power_hessian_fp|:\n")
for (nm in names(results))
  cat(sprintf("  %-14s %.4f\n", nm, results[[nm]]$decision_delta))
saveRDS(summ, file.path("data", "oracle2_summary.rds"))
cat("\nSaved per-cell + summary RDS to validation/data/oracle2_*.rds\n")
