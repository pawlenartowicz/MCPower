#!/usr/bin/env Rscript
# regression.R — fast L3 regression gate. Compares MCPower to FROZEN R-SOTA
# goldens (data/<label>.golden.rds) within the SOLVING_TOL bands. Runs NO live
# R/lme4 fits — only MCPower load_data() + the frozen oracle. Exits nonzero on
# any breach. The .rmd campaign is the sole producer of goldens; this only reads.
# A missing golden is an ERROR (deletion must be visible in git).
suppressMessages(library(mcpower))

script_dir <- local({
  fa <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
  if (length(fa)) dirname(normalizePath(sub("^--file=", "", fa[1]))) else getwd()
})
setwd(script_dir)
source("common.R"); source("tolerances.R"); source("formulas.R")

fails <- character(0)
note  <- function(label, msg) fails <<- c(fails, sprintf("%-22s %s", label, msg))

for (case in CASES) {
  # A<->B: DGP content-hash tripwire (moments-vs-spec is a soft campaign diagnostic, not gated here).
  hc <- tryCatch(hash_check(case), error = function(e) list(ok = FALSE))
  if (!isTRUE(hc$ok)) note(case$label, "DGP content-hash mismatch (regenerate or re-freeze)")
  # B<->C: MCPower vs frozen R-SOTA golden.
  g <- tryCatch(check_golden(case), error = function(e) { note(case$label, conditionMessage(e)); NULL })
  if (!is.null(g) && !g$ok) note(case$label, "B<->C golden breach")
}

# Phase 2: M2/M3/M4 — hash tripwire + frozen extras-golden gate (no live lme4).
# Note: glmm_laplace_bias is excluded from EXTRA_CASES (validator rejects its
# cluster_size=4); that bias cell is gated inline in the campaign rmd only.
for (case in EXTRA_CASES) {
  hc <- tryCatch(hash_check(case), error = function(e) list(ok = FALSE))
  if (!isTRUE(hc$ok)) note(case$label, "DGP content-hash mismatch")
  g <- tryCatch(check_extras_golden(case),
                error = function(e) { note(case$label, conditionMessage(e)); NULL })
  if (!is.null(g) && !g$ok)
    note(case$label, paste("B<->C golden breach:", paste(g$fails, collapse = ",")))
}
# G2/get_effects: deterministic analytic oracle (no golden, no lme4). For each
# GE_CASES case, MCPower's K-draw mean recovered effect must match the analytic
# expected_recovery value (beta/sqrt(beta'Sigma beta + 1) for OLS; beta directly
# for GLM/MLE) within GETEFFECTS_TOL$mean_abs. Reuses recover_once and
# expected_recovery from common.R, GE_CASES from formulas.R.
local({
  GE_N     <- 4000L
  GE_K     <- 20L
  GE_SEED0 <- 2137L
  for (cc_raw in GE_CASES) {
    cc   <- scale_case(cc_raw, GE_N)
    cont <- continuous_names(cc)
    draws <- tryCatch(
      matrix(
        vapply(seq_len(GE_K),
               function(k) recover_once(cc, GE_SEED0 + k - 1L),
               numeric(length(cont))),
        nrow = length(cont)),
      error = function(e) { note(cc$label, paste("get_effects error:", conditionMessage(e))); NULL })
    if (is.null(draws)) next
    mean_rec <- rowMeans(draws)
    exp_v    <- as.numeric(expected_recovery(cc))
    if (any(abs(mean_rec - exp_v) > GETEFFECTS_TOL$mean_abs))
      note(cc$label, sprintf("get_effects oracle breach (max |err|=%.4f, tol=%.4f)",
                             max(abs(mean_rec - exp_v)), GETEFFECTS_TOL$mean_abs))
  }
})

# Phase 4a: crossing — coarse run vs frozen dense golden (CROSS_CASES from formulas.R).
# For each case: load golden (ERROR if missing), run only the cheap coarse grid,
# compare each fitted target's coarse n_achievable to frozen dense within CROSSING_TOL$n_rel.
# The partial target (cross_partial's x2=0.05) is skipped (not-fitted by design).
local({
  FROM_SIZE     <- 20L
  TO_SIZE       <- 300L
  N_SIMS_COARSE <- 1600L
  SEED_COARSE   <- 2138L
  # build_cross_model lives in common.R — shared with validation_crossing.rmd so
  # the coarse gate and the dense golden producer can never diverge on family.

  for (cc in CROSS_CASES) {
    gpath <- file.path("data", paste0(cc$label, ".golden.rds"))
    if (!file.exists(gpath)) {
      note(cc$label, "Missing crossing golden — render validation_crossing.rmd first.")
      next
    }
    golden <- readRDS(gpath)

    m <- tryCatch(build_cross_model(cc),
                  error = function(e) { note(cc$label, paste("build error:", conditionMessage(e))); NULL })
    if (is.null(m)) next

    coarse_res <- tryCatch(
      m$find_sample_size(from_size = FROM_SIZE, to_size = TO_SIZE,
                         n_sims = N_SIMS_COARSE, seed = SEED_COARSE,
                         verbose = FALSE, progress_callback = FALSE),
      error = function(e) { note(cc$label, paste("coarse run error:", conditionMessage(e))); NULL })
    if (is.null(coarse_res)) next

    coarse_fitted <- coarse_res$fitted
    target_labels <- names(golden$dense)

    for (i in seq_along(target_labels)) {
      lbl <- target_labels[i]
      key <- as.character(i - 1L)
      dg  <- golden$dense[[lbl]]

      # Skip partial target (not-fitted by design in the dense run).
      if (dg$status != "fitted") next

      c_fit <- coarse_fitted[[key]]
      if (c_fit$status != "fitted") {
        note(sprintf("%s[%s]", cc$label, lbl), "coarse status != 'fitted' (dense was fitted)")
        next
      }
      rel_diff <- abs(c_fit$n_achievable - dg$n_achievable) / dg$n_achievable
      if (rel_diff > CROSSING_TOL$n_rel)
        note(sprintf("%s[%s]", cc$label, lbl),
             sprintf("coarse vs frozen-dense rel_diff=%.4f > %.4f (coarse=%d dense=%d)",
                     rel_diff, CROSSING_TOL$n_rel, c_fit$n_achievable, dg$n_achievable))
    }
  }
})

# Phase 4b: upload deterministic oracle — recomputes the two strict assertions
# validation_upload.rmd makes, using the shared helpers in common.R:
#   (1) parabola joint-structure check (upload_strict_nonlinear, mode="strict"):
#       mean max |x2_std - f(x1_std)| across K draws <= DGP_TOL$parabola_abs.
#   (2) binary-predictor zero-correlation slot (upload_cont_binary, mode="partial"):
#       |K-draw mean binary×continuous r| <= DGP_TOL$moment_abs (oracle = 0).
# No golden, no lme4. Uses the report's case definitions from UPLOAD_CASES.
local({
  K_ORACLE <- 200L   # enough draws for a reliable mean; full rmd uses 1600

  # (1) Parabola joint-structure assertion — strict bootstrap.
  case_para <- UPLOAD_CASES[["upload_strict_nonlinear"]]
  if (!is.null(case_para)) {
    s <- tryCatch(simulate_upload_case(case_para, k_draws = K_ORACLE),
                  error = function(e) { note("upload_parabola", conditionMessage(e)); NULL })
    if (!is.null(s) && !is.null(s$parabola_max_resid)) {
      mean_resid <- mean(s$parabola_max_resid)
      if (mean_resid > DGP_TOL$parabola_abs)
        note("upload_parabola",
             sprintf("parabola max |x2_std-f(x1_std)| mean=%.2e > tol=%.2e",
                     mean_resid, DGP_TOL$parabola_abs))
    }
  }

  # (2) Binary-predictor zero-correlation slot — partial mode.
  case_bin <- UPLOAD_CASES[["upload_cont_binary"]]
  if (!is.null(case_bin)) {
    s <- tryCatch(simulate_upload_case(case_bin, k_draws = K_ORACLE),
                  error = function(e) { note("upload_binary_zero", conditionMessage(e)); NULL })
    if (!is.null(s) && !is.null(s$corr_bc) && !is.null(s$bc_pairs)) {
      for (r in seq_len(nrow(s$bc_pairs))) {
        bi <- s$bc_pairs$b[r]; ci <- s$bc_pairs$c[r]
        bin_nm  <- case_bin$binary_cols[bi]
        cont_nm <- case_bin$cont_cols[ci]
        k_avg_r <- mean(s$corr_bc[[r]])
        if (abs(k_avg_r) > DGP_TOL$moment_abs)
          note(sprintf("upload_binary_zero[%s x %s]", bin_nm, cont_nm),
               sprintf("binary×cont r mean=%.4f > tol=%.4f (oracle=0)",
                       abs(k_avg_r), DGP_TOL$moment_abs))
      }
    }
  }
})

if (length(fails)) {
  cat("L3 REGRESSION GATE: FAIL\n", paste(fails, collapse = "\n"), "\n", sep = "")
  quit(status = 1L)
}
cat("L3 REGRESSION GATE: OK — all frozen-golden + in-process-oracle checks pass.\n")  # don't hard-code a count: Phases 2/4 add EXTRA_CASES + crossing + oracle checks
