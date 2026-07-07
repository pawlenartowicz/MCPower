# common.R — shared helpers for the L3 validation suite.
#
# Sourced by data_generation.r (formula -> data) and the validation .rmd reports
# (A<->B, B<->C). Holds only pure helpers: model construction, the content-hash
# DGP tripwire, spec parsing (effects/correlations/factor proportions -> design
# columns), R-SOTA refits, and the solver-free variance decomposition. No
# top-level side effects (no file writes, no case loop) so it is safe to source.
#
# Refit + content-hash logic for the L3 campaign and the regression gate
# (validation/regression.R).

suppressPackageStartupMessages({
  library(mcpower)
  library(digest)
})

`%||%` <- function(a, b) if (is.null(a)) b else a

# ---- model construction ------------------------------------------------------

# Build a fresh MCPowerDebug from a case's DGP block. The caller sets the
# draw-specific .debug_n / .debug_n_sims / .debug_seed afterwards (the validation suite
# loops seeds across K draws; data_generation.r uses the committed seed). Factor
# types are declared before effects so the g[k] dummy names resolve. Mirrors the
# setter order the engine applies.
build_model <- function(case) {
  m <- MCPowerDebug$new(case$formula, family = case$family)
  # variable_types may be a character vector (one entry per factor); the loop
  # registers each via a separate set_variable_type call. The setters accumulate
  # (last-wins per key), so every factor survives — overwrite kept only the last.
  if (!is.null(case$variable_types)) for (vt in case$variable_types) m$set_variable_type(vt)
  m$set_effects(case$effects)
  if (!is.null(case$correlations)) m$set_correlations(case$correlations)
  if (!is.null(case$residual)) {
    m$set_residual_distribution(case$residual$name)
  }
  if (!is.null(case$baseline_probability)) {
    m$set_baseline_probability(case$baseline_probability)
  }
  if (!is.null(case$cluster)) {
    cl <- case$cluster
    m$set_cluster(cl$var, ICC = cl$ICC,
                  n_clusters = cl$n_clusters %||% NULL,
                  cluster_size = cl$cluster_size %||% NULL)
  }
  m
}

# ---- M2/M3/M4 model builders (lifted from validation_MLE_solving.rmd) -------
# These builders are shared by the campaign (.rmd) and data_generation.r so
# both drive M2/M3/M4 cases through the same MCPowerDebug setup path.
# The lme4-side reference helpers (m2_frame, m3_frame, m4_frame, lme4_re_summary,
# etc.) remain in the rmd — they are R-SOTA-reference-side only.

build_m2_model <- function(case, seed_off = 0L) {
  m <- MCPowerDebug$new(case$formula, family = "lme",
                        debug_n = case$n, debug_n_sims = 200L)
  m$.debug_seed <- case$seed + seed_off
  m$set_effects(case$effects)
  m$set_cluster(case$cluster$var, ICC = case$cluster$ICC,
                n_clusters = case$cluster$n_clusters)
  m$set_extra_groupings_debug(case$extra_groupings)
  m
}

build_m3_model <- function(case, seed_off = 0L) {
  m <- MCPowerDebug$new(case$formula, family = "lme",
                        debug_n = case$n, debug_n_sims = 200L)
  m$.debug_seed <- case$seed + seed_off
  m$set_effects(case$effects)
  m$set_cluster(case$cluster$var, ICC = case$cluster$ICC,
                n_clusters = case$cluster$n_clusters)
  m$set_slopes_debug(case$slopes)
  # composition: translate the M3 human-readable extra format (var/kind/n_clusters)
  # to the contract shape (relation/tau_squared[/slopes]) that set_extra_groupings_debug expects.
  if (!is.null(case$extra)) {
    contract_extras <- lapply(case$extra, function(e) {
      rel <- if (e$kind == "crossed") {
        list(Crossed = list(n_clusters = e$n_clusters))
      } else {
        list(NestedWithin = list(n_per_parent = e$n_clusters))
      }
      out <- list(relation = rel, tau_squared = e$tau_squared)
      if (!is.null(e$slopes)) out$slopes <- e$slopes   # forward extra-grouping slopes
      out
    })
    m$set_extra_groupings_debug(contract_extras)
  }
  m
}

build_m4_model <- function(case, seed_off = 0L) {
  m <- MCPowerDebug$new(case$formula, family = "logit",
                        debug_n = case$n, debug_n_sims = 200L)
  m$.debug_seed <- case$seed + seed_off
  m$set_effects(case$effects)
  m$set_baseline_probability(case$baseline_probability)
  m$set_cluster(case$cluster$var, ICC = case$cluster$ICC,
                n_clusters = case$cluster$n_clusters)
  if (!is.null(case$slopes)) m$set_slopes_debug(case$slopes)
  # Composition: translate the M4 human-readable extra format (var/kind/n_clusters)
  # to the contract shape (relation/tau_squared[/slopes]) that set_extra_groupings_debug expects — mirrors M3.
  if (!is.null(case$extra)) {
    contract_extras <- lapply(case$extra, function(e) {
      rel <- if (e$kind == "crossed") list(Crossed = list(n_clusters = e$n_clusters))
             else list(NestedWithin = list(n_per_parent = e$n_clusters))
      out <- list(relation = rel, tau_squared = e$tau_squared)
      if (!is.null(e$slopes)) out$slopes <- e$slopes   # forward extra-grouping slopes
      out
    })
    m$set_extra_groupings_debug(contract_extras)
  }
  m
}

# Dispatch by case shape: M4 logit GLMM -> m4; random slopes -> m3; else m2.
build_extras_model <- function(case, seed_off = 0L) {
  if (case$family == "logit") return(build_m4_model(case, seed_off))
  if (!is.null(case$slopes))  return(build_m3_model(case, seed_off))
  build_m2_model(case, seed_off)
}

# ---- content hash (the DGP-drift tripwire) -----------------------------------

# Stable SHA-256 of one dataset's (design, outcome[, cluster]). A saved hash that
# no longer matches its regenerated dataset means the DGP drifted (build/platform
# change) — a stat comparison against frozen values would then be meaningless.
content_hash <- function(design, outcome, cluster_ids = NULL) {
  digest::digest(
    list(
      design  = as.numeric(design),                     # column-major flatten
      outcome = as.numeric(outcome),
      cluster = if (is.null(cluster_ids)) NULL else as.integer(cluster_ids)
    ),
    algo = "sha256"
  )
}

# ---- spec parsing ------------------------------------------------------------
# Maps a case's human-written spec strings onto MCPower's positional design
# columns (intercept, col_1.., factor_dummy_0.., interaction_0..). The column
# order is the engine's contract order: intercept, continuous predictors, factor
# dummies, interactions (see engine-core contract_adapter interaction_base).

# "x1=0.25, g[2]=0.50, x1:x2=-0.20" -> named numeric c(x1=0.25, "g[2]"=0.50, "x1:x2"=-0.20)
parse_effects <- function(s) {
  parts <- trimws(strsplit(s, ",")[[1]])
  kv    <- regmatches(parts, regexpr("=", parts), invert = TRUE)
  nm    <- trimws(vapply(kv, `[`, character(1), 1))
  val   <- as.numeric(trimws(vapply(kv, `[`, character(1), 2)))
  stats::setNames(val, nm)
}

# Names of factor variables declared in the case (across one or more
# variable_types entries).
factor_names <- function(case) {
  if (is.null(case$variable_types)) return(character(0))
  decls <- unlist(strsplit(case$variable_types, ";"))
  decls <- trimws(decls[grepl("factor", decls)])
  trimws(sub("=.*", "", decls))
}

# True coefficient per design column, aligned to `columns`. Positional: the
# design is [intercept, main predictors..., interactions...] (engine contract
# order), so the case's effects must be listed main-effects-first (in column
# order: continuous, then factor dummies) then interactions. A 2-level factor is
# materialised as one binary `col_` column, so col_-vs-factor_dummy_ naming is
# NOT a reliable main/dummy distinction -- map by position instead. Intercept is
# 0 (OLS/LME) or qlogis(baseline_probability) (logit).
true_beta_vector <- function(case, columns) {
  e         <- parse_effects(case$effects)
  is_inter  <- grepl(":", names(e), fixed = TRUE)
  main_eff  <- e[!is_inter]                          # listed in column order
  inter_eff <- e[is_inter]

  tb <- stats::setNames(numeric(length(columns)), columns)
  mi <- 0L; ii <- 0L
  for (j in seq_along(columns)) {
    col <- columns[j]
    if (col == "intercept") {
      tb[j] <- if (!is.null(case$baseline_probability)) stats::qlogis(case$baseline_probability) else 0
    } else if (grepl("^interaction_", col)) {
      ii <- ii + 1L; tb[j] <- inter_eff[[ii]]
    } else {                                          # col_ or factor_dummy_ main effect
      mi <- mi + 1L; tb[j] <- main_eff[[mi]]
    }
  }
  tb
}

# Human term name per design column, aligned to `columns` (same positional logic
# as true_beta_vector): "intercept", then the main-effect names in column order,
# then the interaction names. Lets reports label rows by meaning (x1, g[2],
# x1:x2) instead of the engine's positional names (col_1, factor_dummy_0, ...).
term_labels <- function(case, columns) {
  e        <- parse_effects(case$effects)
  is_inter <- grepl(":", names(e), fixed = TRUE)
  main_nm  <- names(e)[!is_inter]
  inter_nm <- names(e)[is_inter]
  lab <- character(length(columns)); mi <- 0L; ii <- 0L
  for (j in seq_along(columns)) {
    if (columns[j] == "intercept") {
      lab[j] <- "intercept"
    } else if (grepl("^interaction_", columns[j])) {
      ii <- ii + 1L; lab[j] <- inter_nm[ii]
    } else {
      mi <- mi + 1L; lab[j] <- main_nm[mi]
    }
  }
  lab
}

# Continuous-predictor effect names, in column order (col_1, col_2, ...). Used to
# resolve correlation-pair names to design column indices.
continuous_names <- function(case) {
  e    <- parse_effects(case$effects)
  fac  <- factor_names(case)
  base <- sub("\\[.*", "", names(e))
  names(e)[!grepl(":", names(e), fixed = TRUE) & !(base %in% fac)]
}

# "corr(x1,x2)=0.5" -> list(cols = c(2,3), rho = 0.5) where cols index `columns`
# (1-based, intercept = 1). NULL when the case sets no correlation.
parse_correlation <- function(case) {
  if (is.null(case$correlations)) return(NULL)
  s    <- case$correlations
  vars <- regmatches(s, regexpr("\\(([^)]*)\\)", s))
  vars <- trimws(strsplit(gsub("[()]", "", vars), ",")[[1]])
  rho  <- as.numeric(sub(".*=", "", s))
  cont <- continuous_names(case)
  # design column index = 1 (intercept) + position among continuous predictors
  cols <- 1L + match(vars, cont)
  list(cols = cols, rho = rho, vars = vars)
}

# "g=(factor, 0.5, 0.3, 0.2)" -> c(0.5, 0.3, 0.2) (base level first). NULL unless
# the case declares exactly one factor (the proportion probe maps every
# factor_dummy_ column to one factor; with two factors the columns are mixed).
parse_factor_props <- function(case) {
  if (length(factor_names(case)) != 1L) return(NULL)
  vt <- paste(case$variable_types, collapse = " ")
  m  <- regmatches(vt, regexpr("factor,([^)]*)", vt))
  if (length(m) == 0L) return(NULL)
  nums <- as.numeric(trimws(strsplit(sub("factor,", "", m), ",")[[1]]))
  nums[!is.na(nums)]
}

# ---- R-SOTA refits (the B side) ----------------------------------------------
# Fit the canonical R solver on a (design, outcome[, cluster]) dataset and return
# betas + SEs aligned to the design columns. design already carries the intercept
# as its first all-ones column, so all fits are through the origin on the matrix
# (matching MCPower's own test definition: OLS marginal t, GLM Wald z, LME REML
# Wald z via lmer — NOT lmerTest Satterthwaite).
fit_sota <- function(family, design, outcome, cluster_ids = NULL) {
  if (family == "lme") {
    df <- as.data.frame(design)
    vn <- paste0("V", seq_len(ncol(design)))
    names(df) <- vn
    df$.y <- outcome
    df$.g <- factor(cluster_ids)
    fml <- stats::as.formula(
      paste(".y ~ 0 +", paste(vn, collapse = " + "), "+ (1 | .g)")
    )
    fit  <- lme4::lmer(fml, data = df, REML = TRUE,
                       control = lme4::lmerControl(calc.derivs = FALSE))
    beta <- as.numeric(lme4::fixef(fit))
    se   <- sqrt(diag(as.matrix(stats::vcov(fit))))
  } else if (family == "logit") {
    fit <- stats::glm(outcome ~ design + 0, family = stats::binomial())
    sm  <- summary(fit)$coefficients
    beta <- sm[, "Estimate"]
    se   <- sm[, "Std. Error"]
  } else {                                              # ols
    fit <- stats::lm(outcome ~ design + 0)
    sm  <- summary(fit)$coefficients
    beta <- sm[, "Estimate"]
    se   <- sm[, "Std. Error"]
  }
  list(beta = as.numeric(beta), se = as.numeric(se))
}

# ---- MCPower's own fit (the C side) ------------------------------------------
# Fit a saved/loaded dataset through MCPower's data -> results path (load_data),
# using the case's configured spec (test definition) and the committed seed.
# Returns the raw load_data list (betas, design_columns, converged, targets).
fit_mcpower <- function(case, d) {
  m <- build_model(case)
  m$.debug_n      <- as.integer(case$n)
  m$.debug_n_sims <- 1L
  m$.debug_seed   <- as.numeric(case$seed)
  m$load_data(d)
}

# ---- provenance tripwire (shared by every B<->C report) ----------------------
# Regenerating the saved dataset from its committed seed must reproduce its
# content fingerprint; a mismatch means the DGP drifted and any comparison
# against the saved bytes is meaningless. Dispatches to build_extras_model for
# M2/M3/M4 cases (which carry slopes/extra_groupings), build_model for the rest.
hash_check <- function(case) {
  saved <- readRDS(file.path("data", paste0(case$label, ".rds")))
  needs_extras <- !is.null(case$slopes) || !is.null(case$extra) ||
                  !is.null(case$extra_groupings)
  m <- if (needs_extras) build_extras_model(case) else build_model(case)
  m$.debug_n <- as.integer(case$n); m$.debug_n_sims <- 1L
  m$.debug_seed <- as.numeric(case$seed)
  d <- m$create_data()
  list(ok = identical(content_hash(d$design, d$outcome, d$cluster_ids), saved$hash))
}

# ---- B<->C (+ C<->A) comparison for one case ---------------------------------
# Fits the SAME saved bytes in R-SOTA (B) and MCPower (C), aligned by design
# column, and returns a per-term data frame with the gate verdicts. Covers all
# three estimators: the R fit, the R reference critical value, and the estimate
# band all switch on case$family -- fit_sota(case$family, ...) (OLS lm / GLM
# binomial glm / LME lmer), a qt critical value for OLS-t vs qnorm for GLM/MLE-z,
# and the estimate band `estimate_rel_ols` (closed-form OLS) vs `estimate_rel_iter`
# (the iterative IRLS / REML-Brent fits). The crit band splits the same way:
# `crit_abs` (OLS t vs qt) vs `crit_abs_iter` (GLM/MLE z, where the engine's own
# normal/chi2 quantile is coarser than qnorm). Bands stay central in SOLVING_TOL
# (tolerances.R); the report sources it, so SOLVING_TOL is in scope when this is
# called. C<->A (vs the true formula) rides along as a readable sanity column.

# B side — R-SOTA on the saved bytes. The frozen golden is exactly this list
# (an INDEPENDENT oracle: lm/glm/lmer fit, not MCPower's own output).
sota_side <- function(case, saved) {
  alpha <- 0.05
  df    <- saved$n - saved$n_predictors
  bf    <- fit_sota(case$family, saved$design, saved$outcome, saved$cluster_ids)
  list(
    beta = as.numeric(bf$beta),
    se   = as.numeric(bf$se),
    stat = as.numeric(bf$beta / bf$se),                                   # marginal t / Wald z
    crit = if (case$family == "ols") stats::qt(1 - alpha/2, df) else stats::qnorm(1 - alpha/2)
  )
}

# C side — MCPower load_data() on the saved bytes. Targets map by 0-based kernel idx.
mcpower_side <- function(case, saved) {
  cols <- saved$columns
  cf <- fit_mcpower(case, list(design = saved$design, columns = cols,
                               outcome = saved$outcome, cluster_ids = saved$cluster_ids))
  c_se <- c_stat <- c_crit <- rep(NA_real_, length(cols))
  for (tg in cf$targets) {
    j <- tg$target_index + 1L
    c_se[j]   <- tg$se
    c_stat[j] <- tg$statistic
    c_crit[j] <- tg$critical_value
  }
  list(beta = cf$betas, se = c_se, stat = c_stat, crit = c_crit,
       converged = isTRUE(cf$converged))
}

# Shared B<->C comparator — identical logic live (sota = sota_side) and frozen
# (sota = golden). `sota$crit` is a scalar reference critical value.
compare_sides <- function(case, sota, mc, saved) {
  cols  <- saved$columns
  tb    <- true_beta_vector(case, cols)
  terms <- term_labels(case, cols)
  est_tol  <- if (case$family == "ols") SOLVING_TOL$estimate_rel_ols else SOLVING_TOL$estimate_rel_iter
  crit_tol <- if (case$family == "ols") SOLVING_TOL$crit_abs        else SOLVING_TOL$crit_abs_iter
  rel <- function(a, b) if (b == 0) abs(a - b) else abs(a - b) / abs(b)
  rows <- lapply(seq_along(cols), function(j) {
    is_tgt  <- !is.na(mc$stat[j])
    beta_ok <- rel(mc$beta[j], sota$beta[j]) <= est_tol
    stat_ok <- if (is_tgt) rel(mc$stat[j], abs(sota$stat[j])) <= est_tol else TRUE
    crit_ok <- if (is_tgt) abs(mc$crit[j] - sota$crit) <= crit_tol else TRUE
    data.frame(
      Term            = terms[j],
      `True (formula)`= tb[j],
      `R beta`        = sota$beta[j],
      `MCPower beta`  = mc$beta[j],
      `R t`           = if (is_tgt) abs(sota$stat[j]) else NA_real_,
      `MCPower stat`  = if (is_tgt) mc$stat[j]   else NA_real_,
      `R crit`        = if (is_tgt) sota$crit     else NA_real_,
      `MCPower crit`  = if (is_tgt) mc$crit[j]   else NA_real_,
      Verdict         = if (beta_ok && stat_ok && crit_ok) "PASS" else "FAIL",
      check.names = FALSE, row.names = NULL)
  })
  list(table = do.call(rbind, rows), converged = mc$converged)
}

# Live B<->C (unchanged behaviour — same table + converged as before the refactor).
solving_rows <- function(case, saved) {
  compare_sides(case, sota_side(case, saved), mcpower_side(case, saved), saved)
}

# ---- solver-report presentation (shared by OLS/GLM/MLE solving reports) -------

# log10 floor for an exact (bit-identical) B<->C agreement — log10(0) is -Inf, so
# clamp it just under machine eps (~2.2e-16) where exact matches pile up.
SOLVER_AGREEMENT_FLOOR <- -16.5

# Relative B<->C discrepancy from a compare_sides() row, guarding the b==0 case
# (then it degrades to the absolute difference, as compare_sides itself does).
.rel_bc <- function(a, b) if (isTRUE(b == 0)) abs(a - b) else abs(a - b) / abs(b)

# One-figure agreement summary: the realized relative discrepancy of every fitted
# quantity (β for each coefficient, t/z for each target) on a log10 axis against
# the gate. This is the claim the per-formula "R beta | MCPower beta" tables can't
# show — at six printed digits the two columns just look duplicated, while the only
# number that matters (how far inside tolerance the fit lands) stays invisible.
# `results` is the rmd's RESULTS list (each element carries $table from compare_sides).
plot_solver_agreement <- function(results, est_tol, gate_label) {
  beta_e <- numeric(0); stat_e <- numeric(0)
  for (r in results) {
    tb <- r$table
    for (i in seq_len(nrow(tb))) {
      beta_e <- c(beta_e, .rel_bc(tb[["MCPower beta"]][i], tb[["R beta"]][i]))
      if (!is.na(tb[["R t"]][i]))
        stat_e <- c(stat_e, .rel_bc(tb[["MCPower stat"]][i], tb[["R t"]][i]))
    }
  }
  lg <- function(v) pmax(ifelse(v <= 0, SOLVER_AGREEMENT_FLOOR, log10(v)), SOLVER_AGREEMENT_FLOOR)
  groups <- list(`coefficient β` = lg(beta_e), `statistic t / z` = lg(stat_e))
  gate <- log10(est_tol)
  xlim <- range(c(SOLVER_AGREEMENT_FLOOR, gate + 0.5, unlist(groups)))
  op <- graphics::par(mar = c(4.2, 7.5, 2.4, 1)); on.exit(graphics::par(op))
  graphics::stripchart(groups, method = "jitter", jitter = 0.18, pch = 16, las = 1,
                       col = grDevices::rgb(0.20, 0.40, 0.70, 0.5), xlim = xlim,
                       xlab = "log10 relative discrepancy (MCPower vs R) — left is better",
                       main = sprintf("B↔C agreement vs gate (%s)", gate_label))
  graphics::abline(v = gate, col = "red", lwd = 2)
  graphics::text(gate, 1.5, labels = "gate", col = "red", pos = 2, cex = 0.9)
}

# Collapse the twin "R x | MCPower x" columns into one value column plus the
# realized margin: the duplicated 6-digit pairs hid how close the fit actually is,
# so show each quantity once and the worst β/stat relative discrepancy + the
# critical-value absolute discrepancy explicitly. Numeric value columns render via
# kable(digits=); the two discrepancy columns are pre-formatted in scientific
# notation so a sub-picoscale margin reads as "1.2e-12", not "0".
compact_solver_table <- function(tb) {
  n <- nrow(tb)
  rel_est <- vapply(seq_len(n), function(i) {
    rb <- .rel_bc(tb[["MCPower beta"]][i], tb[["R beta"]][i])
    rs <- if (is.na(tb[["R t"]][i])) 0 else .rel_bc(tb[["MCPower stat"]][i], tb[["R t"]][i])
    max(rb, rs)
  }, numeric(1))
  crit_abs <- abs(tb[["MCPower crit"]] - tb[["R crit"]])
  data.frame(
    Term                  = tb$Term,
    `True (formula)`      = tb[["True (formula)"]],
    `β (R = MCPower)`= tb[["R beta"]],
    `t / z`               = tb[["R t"]],
    `crit`                = tb[["R crit"]],
    `rel. diff (β, stat)` = formatC(rel_est, format = "e", digits = 1),
    `crit abs. diff`      = ifelse(is.na(crit_abs), "—", formatC(crit_abs, format = "e", digits = 1)),
    Verdict               = tb$Verdict,
    check.names = FALSE, row.names = NULL)
}

# ---- solver-free variance decomposition (ICC / within) -----------------------

# One-way random-effects ANOVA decomposition of a value vector over the cluster
# grouping — a solver-free ICC and between/within variance components (method of
# moments, unbalanced-corrected). Mirrors the DGP, not lmer's REML estimate. The
# caller supplies the vector to decompose: TRUE-beta residuals (outcome - Xbeta)
# for the conditional ICC tau^2/(tau^2+sigma^2), the raw outcome for the marginal
# ICC, or Xbeta itself for the fixed part's between/within split. Decomposing a
# value vector directly — NOT a cluster-naive lm.fit's residuals — is deliberate:
# an OLS fit absorbs between-cluster signal into the fixed-effect estimates and
# deflates the between component, badly on factor designs whose dummies have
# high-variance cluster means, so the conditional check removes the fixed part
# with the KNOWN true betas instead of estimating it.
var_components <- function(values, cluster_ids) {
  g  <- factor(cluster_ids)
  k  <- nlevels(g)
  N  <- length(values)
  nj <- as.numeric(table(g))
  gm <- tapply(values, g, mean)
  ss_between <- sum(nj * (gm - mean(values))^2)
  ss_within  <- sum((values - gm[as.integer(g)])^2)
  ms_between <- ss_between / (k - 1)
  ms_within  <- ss_within / (N - k)
  n0          <- (N - sum(nj^2) / N) / (k - 1)          # unbalanced correction
  var_between <- max((ms_between - ms_within) / n0, 0)
  var_within  <- ms_within
  list(icc = var_between / (var_between + var_within),
       between = var_between, within = var_within)
}

# ---- golden writer + regression gate -----------------------------------------

# Freeze R-SOTA's B-side outputs for a case to data/<label>.golden.rds. Producer:
# the campaign render only (after its live-R gate passes). Stamped with versions.
write_golden <- function(case, saved, sota) {
  saveRDS(
    list(label = case$label, family = case$family,
         beta = sota$beta, se = sota$se, stat = sota$stat, crit = sota$crit,
         mcpower      = saved$mcpower,
         r_version    = as.character(getRversion()),
         lme4_version = if (case$family == "lme") as.character(utils::packageVersion("lme4")) else NA_character_,
         frozen_on    = as.character(Sys.Date())),
    file = file.path("data", paste0(case$label, ".golden.rds"))
  )
}

# Regression gate for one solving case: load frozen bytes + frozen R-SOTA golden,
# run MCPower fresh, compare via the SAME comparator. NO live R. Missing golden errors.
check_golden <- function(case) {
  gpath <- file.path("data", paste0(case$label, ".golden.rds"))
  if (!file.exists(gpath))
    stop(sprintf("Missing golden for '%s' — run the campaign to freeze it (deletion must be visible in git).", case$label))
  saved  <- readRDS(file.path("data", paste0(case$label, ".rds")))
  golden <- readRDS(gpath)
  sota   <- list(beta = golden$beta, se = golden$se, stat = golden$stat, crit = golden$crit)
  res    <- compare_sides(case, sota, mcpower_side(case, saved), saved)
  list(label = case$label,
       ok = all(res$table$Verdict == "PASS") && res$converged,
       table = res$table, converged = res$converged)
}

# Regression gate for M2/M3/M4 cases: load frozen bytes + frozen extras golden,
# run MCPower load_data() fresh on the saved bytes, compare beta/z/vc/corr/sigma2.
# NO live lme4 — the golden carries the frozen R-SOTA values. MLMM_TOL + SOLVING_TOL
# must be in scope (sourced from tolerances.R before calling this). M4 (logit) is a
# documented XFAIL: ALL its quantities (beta/z/vc/corr) use MLMM_TOL$xfail_backstop,
# mirroring the rmd's M4 B<->C stopifnot — the gate must not be tighter than the campaign.
check_extras_golden <- function(case) {
  gpath <- file.path("data", paste0(case$label, ".golden.rds"))
  if (!file.exists(gpath))
    stop(sprintf("Missing golden for '%s' — run the campaign to freeze it.", case$label))
  saved  <- readRDS(file.path("data", paste0(case$label, ".rds")))
  golden <- readRDS(gpath)
  m <- build_extras_model(case)
  m$.debug_n      <- as.integer(saved$n)
  m$.debug_n_sims <- 1L
  m$.debug_seed   <- as.numeric(case$seed)
  fit <- m$load_data(list(design = saved$design, columns = saved$columns,
                          outcome = saved$outcome, cluster_ids = saved$cluster_ids,
                          extra_grouping_ids = saved$extra_grouping_ids))
  fails <- character(0)
  xfail <- case$family == "logit"  # M4 GLMM: documented XFAIL, use generous backstop
  beta_ok <- function(a, b)
    if (xfail) abs(a - b) <= MLMM_TOL$xfail_backstop$beta_abs
    else abs(a - b) <= MLMM_TOL$beta_abs_floor ||
         abs(a - b) <= SOLVING_TOL$estimate_rel_iter * max(abs(a), abs(b))
  stat_ok <- function(a, b)
    if (xfail) abs(abs(a) - abs(b)) <= MLMM_TOL$xfail_backstop$z_abs
    else abs(abs(a) - abs(b)) <= MLMM_TOL$stat_abs_floor ||
         abs(abs(a) - abs(b)) <= SOLVING_TOL$estimate_rel_iter * max(abs(a), abs(b))
  for (j in seq_along(fit$betas)) {
    a <- golden$beta[j]; b <- fit$betas[j]
    if (!beta_ok(a, b)) fails <- c(fails, sprintf("beta[%d] %.8f vs %.8f", j, a, b))
  }
  for (t in fit$targets) {
    # z comparison: golden$z is signed (R lmer "t value" / glmer "z value"); align signs.
    a <- golden$z[t$target_index + 1L]; b <- t$statistic * sign(t$beta)
    if (!stat_ok(a, b)) fails <- c(fails, sprintf("z[%d] %.6f vs %.6f", t$target_index + 1L, a, b))
  }
  # Variance components: logit (GLMM) is documented XFAIL — use the backstop vc_abs (0.1);
  # for LMM use the tight solver-pair relative band (slope_var_rel) + absolute floor.
  for (g in seq_along(fit$variance_components)) {
    a <- golden$vars[g]; b <- fit$variance_components[g]
    ok_vc <- if (xfail) abs(a - b) <= MLMM_TOL$xfail_backstop$vc_abs
             else abs(a - b) <= SOLVING_TOL$vc_abs ||
                  abs(a - b) <= SOLVING_TOL$slope_var_rel * max(abs(a), abs(b))
    if (!ok_vc) fails <- c(fails, sprintf("vc[%d] %.8f vs %.8f", g, a, b))
  }
  # RE correlation: logit XFAIL uses backstop corr_abs (0.1); LMM uses slope_corr_abs (2e-3).
  crab <- if (xfail) MLMM_TOL$xfail_backstop$corr_abs else SOLVING_TOL$slope_corr_abs
  for (g in seq_along(fit$re_corr))
    if (g <= length(golden$corr) && !is.na(golden$corr[g]) &&
        abs(golden$corr[g] - fit$re_corr[g]) > crab)
      fails <- c(fails, sprintf("corr[%d] %.6f vs %.6f", g, golden$corr[g], fit$re_corr[g]))
  # Residual variance (LMM only; logit golden$sigma2 is NA). vc_rel (not slope_var_rel)
  # because sigma2 is the error variance, not a random-slope variance.
  if (!is.null(golden$sigma2) && !is.na(golden$sigma2)) {
    a <- golden$sigma2; b <- fit$sigma_sq_hat
    if (!(abs(a - b) <= SOLVING_TOL$vc_abs || abs(a - b) <= SOLVING_TOL$vc_rel * max(abs(a), abs(b))))
      fails <- c(fails, sprintf("sigma2 %.8f vs %.8f", a, b))
  }
  list(label = case$label, ok = length(fails) == 0, fails = fails)
}

# ==============================================================================
# Upload validation helpers (validation_upload.rmd + regression.R)
# ==============================================================================
# Helpers for the NORTA-over-frame path. Shared so regression.R can reuse the
# same two strict assertions validation_upload.rmd already makes (parabola
# joint-structure check and binary-predictor zero-correlation-slot check).
#
# N_REF and K (oracle draw size / K draws per case) are campaign parameters;
# callers that need them for their own draws define them locally. Only the two
# strict-assertion primitives are called from regression.R (no K-draw loop).

# Absolute path to the upload CSV for a case (path stored relative to validation/).
upload_csv_abs <- function(case) {
  file.path(getwd(), case$upload_csv)
}

# Load the frame for a case and extract the upload columns. A case names its
# source either as a built-in R dataset (`upload_builtin`, e.g. "mtcars" — no
# fixture file needed) or a CSV path (`upload_csv`, used by the synthetic
# parabola case).
load_upload_frame <- function(case) {
  df <- if (!is.null(case$upload_builtin)) {
    get(case$upload_builtin, envir = asNamespace("datasets"))
  } else {
    read.csv(upload_csv_abs(case), row.names = 1L, stringsAsFactors = FALSE)
  }
  df[, case$upload_cols, drop = FALSE]
}

# Compute raw empirical moments from the upload frame.
frame_raw_moments <- function(frame, case) {
  cont_mean <- vapply(case$cont_cols, function(n) mean(frame[[n]]),     numeric(1))
  cont_sd   <- vapply(case$cont_cols, function(n) stats::sd(frame[[n]]), numeric(1))
  bin_prop  <- vapply(case$binary_cols, function(n) mean(as.numeric(frame[[n]])), numeric(1))
  cont_corr <- if (length(case$cont_cols) >= 2L) {
    pairs <- combn(case$cont_cols, 2, simplify = FALSE)
    vapply(pairs, function(p) stats::cor(frame[[p[1]]], frame[[p[2]]]), numeric(1))
  } else numeric(0)
  list(
    cont_mean = stats::setNames(cont_mean, case$cont_cols),
    cont_sd   = stats::setNames(cont_sd,   case$cont_cols),
    bin_prop  = stats::setNames(bin_prop,  case$binary_cols),
    cont_corr = cont_corr
  )
}

# Build an MCPowerDebug with upload_data() applied.
build_upload_model <- function(case, frame, mode = NULL) {
  if (is.null(mode)) mode <- case$upload_mode %||% "partial"
  m <- MCPowerDebug$new(case$formula, family = case$family)
  m$set_effects(case$effects)
  suppressMessages(m$upload_data(frame, mode = mode, verbose = FALSE))
  m$.debug_n_sims <- 1L
  m
}

# Measure the engine's own population moments via one large-n reference draw.
# Returns oracle_mean, oracle_sd, oracle_prop, oracle_corr_cc, cont_idx, bin_idx,
# and frame_raw for the nonlinear parabola case.
engine_oracle <- function(case, frame, n_ref = 200000L) {
  m <- build_upload_model(case, frame)
  m$.debug_n    <- n_ref
  m$.debug_seed <- as.numeric(case$seed + 99999L)
  d <- m$create_data()
  des <- d$design

  n_cont <- length(case$cont_cols)
  n_bin  <- length(case$binary_cols)
  cont_idx <- seq_len(n_cont) + 1L
  bin_idx  <- seq_len(n_bin)  + 1L + n_cont

  oracle_mean <- vapply(cont_idx, function(j) mean(des[, j]), numeric(1))
  oracle_sd   <- vapply(cont_idx, function(j) stats::sd(des[, j]), numeric(1))
  oracle_prop <- if (n_bin > 0L) vapply(bin_idx, function(j) mean(des[, j]), numeric(1)) else numeric(0)
  oracle_corr_cc <- if (n_cont >= 2L) {
    stats::cor(des[, cont_idx[1]], des[, cont_idx[2]])
  } else NA_real_

  frame_raw <- list()
  if (isTRUE(case$nonlinear) && length(case$cont_cols) >= 2L) {
    for (nm in case$cont_cols) {
      x <- frame[[nm]]
      frame_raw[[nm]] <- list(mean = mean(x), sd = sqrt(mean((x - mean(x))^2)))
    }
  }

  list(oracle_mean    = stats::setNames(oracle_mean, case$cont_cols),
       oracle_sd      = stats::setNames(oracle_sd,   case$cont_cols),
       oracle_prop    = stats::setNames(oracle_prop, case$binary_cols),
       oracle_corr_cc = oracle_corr_cc,
       cont_idx       = cont_idx, bin_idx = bin_idx,
       frame_raw      = frame_raw)
}

# Run K simulated draws for one upload case. Returns everything upload_assertion_rows needs.
simulate_upload_case <- function(case, k_draws = 1600L) {
  frame  <- load_upload_frame(case)
  raw_fm <- frame_raw_moments(frame, case)
  oracle <- engine_oracle(case, frame)
  n_cont <- length(case$cont_cols)
  n_bin  <- length(case$binary_cols)

  m <- build_upload_model(case, frame)
  m$.debug_n <- as.integer(case$n)

  cmean   <- matrix(NA_real_, k_draws, n_cont)
  csd     <- matrix(NA_real_, k_draws, n_cont)
  bprop   <- if (n_bin > 0L) matrix(NA_real_, k_draws, n_bin) else NULL
  corr_cc <- if (n_cont >= 2L) numeric(k_draws) else NULL

  bc_pairs <- if (n_bin > 0L && n_cont > 0L) {
    expand.grid(b = seq_len(n_bin), c = seq_len(n_cont), stringsAsFactors = FALSE)
  } else NULL
  corr_bc <- if (!is.null(bc_pairs)) lapply(seq_len(nrow(bc_pairs)), function(...) numeric(k_draws)) else NULL

  parabola_max_resid <- if (isTRUE(case$nonlinear) && n_cont >= 2L) numeric(k_draws) else NULL
  para_coef <- if (!is.null(parabola_max_resid)) {
    fr <- oracle$frame_raw
    m1 <- fr[[case$cont_cols[1]]]$mean;  s1 <- fr[[case$cont_cols[1]]]$sd
    m2 <- fr[[case$cont_cols[2]]]$mean;  s2 <- fr[[case$cont_cols[2]]]$sd
    list(a = s1^2 / s2, b = 2 * m1 * s1 / s2, c0 = (m1^2 - m2) / s2)
  } else NULL

  for (k in seq_len(k_draws)) {
    m$.debug_seed <- as.numeric(case$seed + k)
    d <- m$create_data()
    des <- d$design
    for (i in seq_len(n_cont)) {
      x <- des[, oracle$cont_idx[i]]; cmean[k, i] <- mean(x); csd[k, i] <- stats::sd(x)
    }
    if (!is.null(bprop)) {
      for (i in seq_len(n_bin)) bprop[k, i] <- mean(des[, oracle$bin_idx[i]])
    }
    if (!is.null(corr_cc)) {
      corr_cc[k] <- stats::cor(des[, oracle$cont_idx[1]], des[, oracle$cont_idx[2]])
    }
    if (!is.null(corr_bc)) {
      for (r in seq_len(nrow(bc_pairs))) {
        ci <- oracle$cont_idx[bc_pairs$c[r]]
        bi <- oracle$bin_idx[bc_pairs$b[r]]
        corr_bc[[r]][k] <- stats::cor(des[, ci], des[, bi])
      }
    }
    if (!is.null(parabola_max_resid)) {
      x1s <- des[, oracle$cont_idx[1]]
      x2s <- des[, oracle$cont_idx[2]]
      x2s_pred <- para_coef$a * x1s^2 + para_coef$b * x1s + para_coef$c0
      parabola_max_resid[k] <- max(abs(x2s - x2s_pred))
    }
  }

  list(case = case, raw_fm = raw_fm, oracle = oracle,
       cmean = cmean, csd = csd, bprop = bprop,
       corr_cc = corr_cc, corr_bc = corr_bc, bc_pairs = bc_pairs,
       parabola_max_resid = parabola_max_resid, para_coef = para_coef)
}

# Build the assertion table for one upload case result.
upload_assertion_rows <- function(s, bands = NULL) {
  # bands defaults to the DGP_TOL-based BANDS the rmd uses; callers that source
  # tolerances.R can pass their own list(mean=, sd=, prop=, corr_cc=) if needed.
  if (is.null(bands)) bands <- list(mean    = DGP_TOL$moment_abs,
                                    sd      = DGP_TOL$moment_abs,
                                    prop    = DGP_TOL$moment_abs,
                                    corr_cc = DGP_TOL$moment_abs)
  rows <- list()
  add <- function(what, oracle_val, k_avg, band) {
    off <- abs(k_avg - oracle_val)
    ok  <- off <= band
    rows[[length(rows) + 1L]] <<- data.frame(
      Assertion          = what,
      `Oracle (engine)`  = oracle_val,
      `K-draw average`   = k_avg,
      `Difference`       = off,
      `Tolerance`        = band,
      Verdict            = if (ok) "PASS" else "FAIL",
      check.names = FALSE, stringsAsFactors = FALSE)
  }
  for (i in seq_len(length(s$case$cont_cols))) {
    nm <- s$case$cont_cols[i]
    add(sprintf("mean of %s", nm),
        s$oracle$oracle_mean[nm], mean(s$cmean[, i]), bands$mean)
    add(sprintf("sd of %s", nm),
        s$oracle$oracle_sd[nm], mean(s$csd[, i]), bands$sd)
  }
  if (!is.null(s$bprop)) {
    for (i in seq_len(length(s$case$binary_cols))) {
      nm <- s$case$binary_cols[i]
      add(sprintf("proportion(1) of %s", nm),
          s$oracle$oracle_prop[nm], mean(s$bprop[, i]), bands$prop)
    }
  }
  if (!is.null(s$corr_cc)) {
    pair_nm  <- paste(s$case$cont_cols[1], "×", s$case$cont_cols[2])
    mode_lbl <- s$case$upload_mode %||% "partial"
    add(sprintf("correlation %s (mode=%s)", pair_nm, mode_lbl),
        s$oracle$oracle_corr_cc, mean(s$corr_cc), bands$corr_cc)
  }
  if (!is.null(s$corr_bc) && !is.null(s$bc_pairs)) {
    for (r in seq_len(nrow(s$bc_pairs))) {
      bi <- s$bc_pairs$b[r]; ci <- s$bc_pairs$c[r]
      bin_nm  <- s$case$binary_cols[bi]
      cont_nm <- s$case$cont_cols[ci]
      add(sprintf("correlation %s × %s ≈ 0 (binary independent)", bin_nm, cont_nm),
          oracle_val = 0.0, k_avg = mean(s$corr_bc[[r]]), band = bands$corr_cc)
    }
  }
  if (!is.null(s$parabola_max_resid)) {
    add("parabola: max |x2_std - f(x1_std)| per draw (strict preserves joint; NORTA would fail)",
        oracle_val = 0.0, k_avg = mean(s$parabola_max_resid), band = DGP_TOL$parabola_abs)
  }
  do.call(rbind, rows)
}

# ==============================================================================
# G2 — get_effects_from_data round-trip helpers (validation_get_effects.rmd + regression.R)
# ==============================================================================
# Shared by the campaign rmd and the regression gate. The rmd keeps its own GE_N/GE_K/GE_SEED0
# campaign parameters; regression.R uses the same defaults (4000 / 20 / 2137). GE_CASES is
# defined in formulas.R (requires CASES, so cannot live here).

# Scale a case to a round-trip sample size. Flat (ols/glm) sets n directly; LME
# keeps cluster_size and scales n_clusters so n_clusters*cluster_size ~ n_big.
scale_case <- function(case, n_big) {
  cc <- case
  if (!is.null(case$cluster)) {
    cs  <- case$cluster$cluster_size
    ncl <- max(2L, as.integer(round(n_big / cs)))
    cc$cluster$n_clusters <- ncl; cc$cluster$cluster_size <- cs; cc$n <- ncl * cs
  } else cc$n <- as.integer(n_big)
  cc
}

# One simulate -> recover cycle: generate a dataset from the case's DGP, rebuild a
# raw predictor frame (+ outcome, + grouping for LME), upload it, and recover the
# standardized effects via get_effects_from_data.
recover_once <- function(cc, seed) {
  m <- build_model(cc)
  m$.debug_n <- as.integer(cc$n); m$.debug_n_sims <- 1L; m$.debug_seed <- as.numeric(seed)
  d    <- m$create_data()
  cont <- continuous_names(cc)
  frame <- as.data.frame(d$design[, seq_along(cont) + 1L, drop = FALSE])
  names(frame) <- cont
  frame$y <- d$outcome
  if (!is.null(cc$cluster)) frame[[cc$cluster$var]] <- d$cluster_ids
  m2 <- build_model(cc)
  m2$upload_data(frame, mode = "partial", verbose = FALSE)
  as.numeric(parse_effects(m2$get_effects_from_data("y", verbose = FALSE))[cont])
}

# Convention-predicted recovery per estimator. OLS z-scores the outcome, so it
# recovers the standardized coefficient beta / sqrt(Var(X beta) + 1), where
# Var(X beta) = beta' Sigma beta (reduces to sum(beta^2) when predictors are
# independent). With correlation the cross terms inflate Var(X beta), producing
# stronger shrinkage. GLM (logit) and MLE (mixed) fit the native outcome and
# recover beta directly.
expected_recovery <- function(cc) {
  cont      <- continuous_names(cc)
  beta_cont <- parse_effects(cc$effects)[cont]
  if (cc$family != "ols") return(beta_cont)
  k <- length(beta_cont)
  Sigma <- diag(k)
  if (!is.null(cc$correlations)) {
    pc <- parse_correlation(cc)            # list(cols, rho, vars); rho is a scalar
    a  <- match(pc$vars[1], cont)
    b  <- match(pc$vars[2], cont)
    if (!is.na(a) && !is.na(b)) { Sigma[a, b] <- pc$rho; Sigma[b, a] <- pc$rho }
  }
  var_xb <- as.numeric(t(beta_cont) %*% Sigma %*% beta_cont)
  beta_cont / sqrt(var_xb + 1)
}

# ==============================================================================
# L5 — scenario-perturbation probes (validation_scenarios.rmd)
# ==============================================================================
# Measurement helpers for the L5 layer: each scenario knob's realised magnitude
# vs its documented law. The oracle is the perturbation law (a distribution),
# not a point value, so these probes recover the knob's magnitude from the
# generated data. Per-draw replication: with .debug_n_sims = 1 every
# create_data() call draws scenario block 0, so seed + k (k = 1..K) gives K
# independent block-level perturbation draws; the same seed under two
# scenarios shares both RNG streams (paired draws).

# Engine marginal-transform constants (data_gen.rs apply_marginal).
# Skewed marginal = censored Exp(1) at E <= EXP_CAP, standardized by the
# CENSORED moments (mean_c = 1 - e^-c; E[min(E,c)^2] = 2 - (2c+2)e^-c) so
# variance is exactly 1 and support is [-1.006, +6] SD. The literals mirror
# data_gen.rs — change together.
SQRT3             <- sqrt(3)
EXP_CAP           <- 6.95925599364711
EXP_CENSORED_MEAN <- 0.9990501970288289
EXP_CENSORED_STD  <- 0.9933676327697134

# ---- scenario model construction + draws -------------------------------------

# build_model + the case's heteroskedasticity driver pin and custom scenario
# configs. Caller sets .debug_seed / .debug_scenario per draw.
scenario_model <- function(case) {
  m <- build_model(case)
  if (!is.null(case$hsk_var)) m$set_heteroskedasticity_driver(case$hsk_var)
  if (!is.null(case$scenario_configs)) m$set_scenario_configs(case$scenario_configs)
  m$.debug_n      <- as.integer(case$n)
  m$.debug_n_sims <- 1L
  m
}

# One scenario draw: scenario block 0 + data stream, both keyed on `s`.
scen_draw <- function(m, scen, s) {
  m$.debug_scenario <- scen
  m$.debug_seed     <- as.numeric(s)
  m$create_data()
}

# ---- gate primitives (the beta_rows pattern) ----------------------------------

# The L5 gate: K per-draw statistics vs the documented law.
# z = (mean - law) / (sd/sqrt(K)); PASS iff |z| <= SCENARIO_TOL$z_c.
zgate <- function(stats_k, law) {
  m  <- mean(stats_k)
  se <- stats::sd(stats_k) / sqrt(length(stats_k))
  z  <- (m - law) / se
  list(est = m, law = law, se = se, z = z, ok = abs(z) <= SCENARIO_TOL$z_c)
}

# Same gate with a caller-supplied (analytic) SE — for binomial frequencies
# where the exact SE beats the empirical one (pick-uniformity counts).
zgate_se <- function(est, law, se) {
  z <- (est - law) / se
  list(est = est, law = law, se = se, z = z, ok = abs(z) <= SCENARIO_TOL$z_c)
}

# Gate on the across-block SD of a statistic (the correlation-noise law).
# SE(SD) = SD/sqrt(2(K-1)) — normal approximation, adequate because the
# per-block r is itself near-normal here.
sdgate <- function(stats_k, law_sd) {
  s  <- stats::sd(stats_k)
  se <- s / sqrt(2 * (length(stats_k) - 1))
  z  <- (s - law_sd) / se
  list(est = s, law = law_sd, se = se, z = z, ok = abs(z) <= SCENARIO_TOL$z_c)
}

# ---- sample-shape statistics + classifiers ------------------------------------

samp_skew   <- function(x) { m <- mean(x); mean((x - m)^3) / stats::sd(x)^3 }
samp_exkurt <- function(x) { m <- mean(x); mean((x - m)^4) / stats::sd(x)^4 - 3 }

# Which engine marginal generated this design column? The candidate shapes are
# extreme enough that at the case n (>= 2000) every cut sits many SDs from both
# sides: uniform is EXACTLY bounded at ±sqrt(3) while every other marginal
# exceeds the bound a.s. by n = 2000; the censored-Exp(1) skewed marginal's
# sample skew runs ~1.9 (population ±1.90) vs 0 for normal/t3 (skew SE ~
# sqrt(6/n) ≈ 0.05); censored t3's excess kurtosis is ~6.4 vs 0 for normal
# (SE ~ 0.08).
classify_marginal <- function(x) {
  if (max(abs(x)) <= SQRT3 + 1e-9) return("uniform")
  sk <- samp_skew(x)
  if (sk >  1.0) return("right_skewed")
  if (sk < -1.0) return("left_skewed")
  if (samp_exkurt(x) > 1.0) return("high_kurtosis")
  "normal"
}

# Which residual family generated these (true-beta) residuals? Pool identities:
# skewed = (chi2(df)-df)/sqrt(2df), skew = sqrt(8/df) (0.89 at df 10); heavy =
# t(df)*sqrt((df-2)/df), excess kurtosis 6/(df-4) (1.0 at df 10); normal = 0/0.
# Cuts at the midpoints; at n = 4000 each true class sits >= ~3-10 SDs from a
# cut, so misclassification is rare (and only nudges the pick-split fractions,
# never the swap-frequency gate, which counts any non-normal verdict).
classify_residual <- function(e) {
  if (abs(samp_skew(e)) > 0.45) return("skewed")
  if (samp_exkurt(e) > 0.5) return("heavy_tailed")
  "normal"
}

# ---- He: jitter-variance probe -------------------------------------------------

# Per-predictor slope of squared true-beta residuals on the squared predictor.
# Under heterogeneity h the jitter delta_j ~ N(0, (h b_j)^2) is drawn ONCE per
# study (constant across the draw's rows — data_gen.rs per-study beta-jitter), so
# within a draw e = sum_j delta_j x_j + eps and the slope on x_j^2 recovers the
# REALISED delta_j^2; averaged over K draws E[delta_j^2] = (h b_j)^2 (the gate's
# law). The per-study structure makes the per-draw slope chi-square_1-shaped (so a
# sizeable fraction of draws have a negative slope) — the zgate uses the empirical
# across-draw SD, which correctly reflects that spread. Clean at lambda = 1; under
# a lambda driver the cosh(gamma z) even component contaminates the driver column's
# slope — B2 pins the driver to the OTHER column for exactly this.
jitter_slopes <- function(d, tb) {
  e2   <- (d$outcome - as.numeric(d$design %*% tb))^2
  pred <- which(d$columns != "intercept")
  X2   <- cbind(1, d$design[, pred, drop = FALSE]^2)
  stats::setNames(stats::lm.fit(X2, e2)$coefficients[-1], d$columns[pred])
}

# ---- Hs: log-e² slope + binned ratio -------------------------------------------

# gamma-hat = slope of log(e^2) on the standardized driver. The kernel scales
# residuals by exp(gamma z/2 - gamma^2/4), so log e^2 = log eps^2 + gamma z + c
# EXACTLY — slope = gamma = ln(lambda)/4 for any residual shape (B3 relies on
# this shape-blindness). The realised lambda is exp(4 gamma-hat).
hsk_gamma <- function(d, tb, driver) {
  e2 <- (d$outcome - as.numeric(d$design %*% tb))^2
  stats::lm.fit(cbind(1, driver), log(e2))$coefficients[2]
}

# ±2σ binned variance ratio — the readable companion to the slope gate (finite
# bins make its law only approximately lambda, so it is reported, not gated).
hsk_bin_ratio <- function(d, tb, driver, half_width = 0.25) {
  e  <- d$outcome - as.numeric(d$design %*% tb)
  hi <- e[abs(driver - 2) <= half_width]
  lo <- e[abs(driver + 2) <= half_width]
  if (length(hi) < 20 || length(lo) < 20) return(NA_real_)
  stats::var(hi) / stats::var(lo)
}

# ---- Co: clamp-truncation law ---------------------------------------------------

# Mean/SD of clamp(X, lo, hi) for X ~ N(mu, tau^2) — the ±0.8-clamp truncation
# law for a perturbed correlation entry (clamping censors: probability mass
# piles AT the bounds). Standard censored-normal moments.
clamp_normal_moments <- function(mu, tau, lo = -0.8, hi = 0.8) {
  a <- (lo - mu) / tau; b <- (hi - mu) / tau
  p_lo <- stats::pnorm(a); p_hi <- 1 - stats::pnorm(b)
  pm   <- stats::pnorm(b) - stats::pnorm(a)
  dm   <- stats::dnorm(b) - stats::dnorm(a)
  m1   <- lo * p_lo + hi * p_hi + mu * pm - tau * dm
  m2   <- lo^2 * p_lo + hi^2 * p_hi +
    (mu^2 + tau^2) * pm - tau^2 * (b * stats::dnorm(b) - a * stats::dnorm(a)) -
    2 * mu * tau * dm
  list(mean = m1, sd = sqrt(max(m2 - m1^2, 0)))
}

# ---- engine marginal transforms (NORTA oracle + moment laws) -------------------

# Exact mean/var of the engine's high_kurtosis marginal — mirrors
# T3PpfTable::build (2048 knots on u in [0.00121, 0.99879], LINEAR interpolation
# between knots, percentile clamped outside, then divided by the table's own
# SD so var = 1 by construction — change together with t3.rs). Piecewise-
# linear in u, so each segment integrates in closed form; the clamped tails
# contribute point mass at the first/last knot value. sd_raw is the
# pre-normalization SD of the censored table (≈ 1.596, vs √3 for full t(3)).
t3_table_moments <- function(resolution = 2048L, p_min = 0.00121, p_max = 0.99879) {
  u  <- p_min + (p_max - p_min) * (0:(resolution - 1)) / (resolution - 1)
  v  <- qt(u, df = 3)
  du <- diff(u)
  v0 <- utils::head(v, -1); v1 <- utils::tail(v, -1)
  m1 <- p_min * v[1] + (1 - p_max) * v[resolution] + sum(du * (v0 + v1) / 2)
  m2 <- p_min * v[1]^2 + (1 - p_max) * v[resolution]^2 +
    sum(du * (v0^2 + v0 * v1 + v1^2) / 3)
  sd_raw <- sqrt(m2 - m1^2)
  list(mean = m1 / sd_raw, var = (m2 - m1^2) / sd_raw^2, sd_raw = sd_raw)
}

# Pre-normalization SD of the censored t3 table — the divisor the engine
# applies at table build. Computed once; marg_transform shares it.
T3_SD <- t3_table_moments()$sd_raw

# R mirrors of data_gen.rs apply_marginal for the swappable kinds. The skewed
# pair is the censored standardized Exp(1): E = -log(pnorm(-z)) (the pnorm(-z)
# form avoids cancellation as z -> +Inf), capped at EXP_CAP. high_kurtosis
# uses the exact qt with the engine's [0.00121, 0.99879] percentile clamp and
# the table-SD normalization; the engine's 2048-knot linear interpolation is
# mirrored separately by t3_table_moments (the table IS the marginal identity,
# so moment laws come from the table construction, not the ideal t3).
marg_transform <- function(name) {
  switch(name,
    normal        = function(z) z,
    right_skewed  = function(z)
      (pmin(-log(stats::pnorm(-z)), EXP_CAP) - EXP_CENSORED_MEAN) / EXP_CENSORED_STD,
    left_skewed   = function(z)
      (EXP_CENSORED_MEAN - pmin(-log(stats::pnorm(z)), EXP_CAP)) / EXP_CENSORED_STD,
    uniform       = function(z) -SQRT3 + 2 * SQRT3 * stats::pnorm(z),
    high_kurtosis = function(z) qt(pmin(pmax(stats::pnorm(z), 0.00121), 0.99879), df = 3) / T3_SD,
    stop("unknown marginal ", name)
  )
}

# ---- NORTA oracle ----------------------------------------------------------------

# Gauss–Hermite nodes/weights for E[f(Z)], Z ~ N(0,1) (Golub–Welsch on the
# Hermite Jacobi matrix; probabilists' scaling). Base-R, no extra packages.
gauss_hermite <- function(n) {
  i <- seq_len(n - 1)
  J <- matrix(0, n, n)
  J[cbind(i, i + 1)] <- sqrt(i / 2)
  J[cbind(i + 1, i)] <- sqrt(i / 2)
  e <- eigen(J, symmetric = TRUE)
  list(nodes = e$values * sqrt(2), weights = e$vectors[1, ]^2)
}

# Realized Pearson r between two transformed marginals under latent correlation
# rho: r = Cov(T1(Z1), T2(Z2)) / (sd1 sd2) with (Z1, Z2) standard bivariate
# normal — the NORTA prediction for a swapped-marginal pair (B4). Numerical:
# nested Gauss–Hermite (Z2 = rho Z1 + sqrt(1-rho^2) Z2'). Closed-form anchor
# for the report: uniform pair (6/pi) asin(rho/2); the censored-Exp(1) skewed
# pair has no closed form (Gauss–Hermite only).
norta_r <- function(name1, name2, rho, n_gh = 80) {
  T1 <- marg_transform(name1); T2 <- marg_transform(name2)
  gh <- gauss_hermite(n_gh)
  z  <- gh$nodes; w <- gh$weights
  inner <- vapply(z, function(a) sum(w * T2(rho * a + sqrt(1 - rho^2) * z)), numeric(1))
  m1  <- sum(w * T1(z)); m2 <- sum(w * T2(z))
  e12 <- sum(w * T1(z) * inner)
  v1  <- sum(w * T1(z)^2) - m1^2
  v2  <- sum(w * T2(z)^2) - m2^2
  (e12 - m1 * m2) / sqrt(v1 * v2)
}

# ---- Hs driver anchor (B5) --------------------------------------------------------

# sigma0 = the engine's het_coeffs lp_pop_std for an all-continuous-normal
# design: sqrt(beta' Sigma beta) over the non-intercept block (compute_het_coeffs
# with unit-variance columns; the intercept column has variance 0). Only valid
# for the continuous-only cases L5 points it at.
lp_sigma0 <- function(tb, rho_mat) {
  b <- tb[names(tb) != "intercept" & !grepl("^factor_dummy_", names(tb))]
  sqrt(as.numeric(t(b) %*% rho_mat %*% b))
}

# ---- Fa: allocation counts ---------------------------------------------------------

# Per-level counts of the single declared factor (3+-level: reference level =
# rows with all dummies 0). sampled_factor_proportions laws: FALSE -> counts are
# a pure function of (n, proportions) — identical across draws, each within ~1 of
# n*p (largest-remainder walk); TRUE -> count_g ~ Binomial(n, p_g).
factor_counts <- function(d) {
  dm <- d$design[, grepl("^factor_dummy_", d$columns), drop = FALSE]
  c(sum(rowSums(dm) == 0), colSums(dm))
}

# ---- GLM flip-rate predictor --------------------------------------------------------

# Predicted h-toggle flip rate for a shared-seed logit pair. X and the
# Bernoulli uniforms are drawn BEFORE the jitter normals, so the pair shares
# them bit-identically and P(flip | row i) = E_delta |sigmoid(lp_i + delta_i) -
# sigmoid(lp_i)| with delta_i ~ N(0, h^2 (1 + sum_j x_ij^2 b_j^2)) — the 1 is
# the Binary intercept jitter (s0 = h, NOT scaled by |b0|; data_gen.rs). The
# expectation is a 1-D Gauss–Hermite integral per row; returns the n-row mean.
glm_fliprate_pred <- function(design, tb, h, n_gh = 40) {
  lp <- as.numeric(design %*% tb)
  s  <- h * sqrt(1 + as.numeric(design[, -1, drop = FALSE]^2 %*% (tb[-1]^2)))
  gh <- gauss_hermite(n_gh)
  PH <- stats::plogis(sweep(outer(s, gh$nodes), 1, lp, `+`))
  mean(abs(PH - stats::plogis(lp)) %*% gh$weights)
}

# Pseudo-true (population-averaged) logit betas under heterogeneity h. The
# Per-study pseudo-true coefficient under the heterogeneity β-jitter (logit only).
# The engine draws ONE jitter δ per simulation (data_gen.rs, per-study β-jitter),
# constant across that study's rows — so each study's data is a clean logit with its
# OWN β_eff, the MLE recovers β_eff, and averaged over K studies the fitted coefficient
# → E[β_eff]. There is NO per-observation population-averaging (that would attenuate);
# the per-study structure leaves the slopes essentially at the true β. Slopes are clipped
# toward zero (s_j = h·|β_j|): E[max(0, β_j+δ)] = β_j·(Φ(1/h)+h·φ(1/h)), a tiny nudge AWAY
# from zero (×1.0008 at h = 0.4). The binary intercept jitter (s_0 = h) is symmetric and
# unclipped, so its mean stays β_0. OLS needs no nudge (linear averaging keeps it exact).
glm_perstudy_beta <- function(tb, h) {
  b <- as.numeric(tb)
  if (h <= 0) return(b)
  nudge  <- stats::pnorm(1 / h) + h * stats::dnorm(1 / h)  # E[max(0, β+δ)]/β, δ ~ N(0, (h|β|)²)
  out    <- b * nudge                                       # slopes: clip nudges |coef| up
  out[1] <- b[1]                                            # intercept: symmetric unclipped jitter, mean unchanged
  out
}
