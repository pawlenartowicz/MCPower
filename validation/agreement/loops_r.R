#!/usr/bin/env Rscript
# R DIY-loop baselines (naive off-the-shelf + best hand-rolled), keyed by family.
# These loops produce the reference powers the agreement check compares
# MCPower against, and are also timed as the DIY-loop comparators.
#
# Kernel signature: fn(case, n, n_sims, seed) -> list(power, joint, converged)
# where power is a per-coef rejection-rate vector (length P) aligned to the
# fixed column convention: [continuous mains, factor dummies, interactions].
# Rates are over all n_sims: a failed fit counts as not significant, as in
# MCPower; `converged` is the number of fits that were scored.
#
# Decision rules follow MCPower's defaults: OLS t at the residual df; logit
# Wald z; LME and GLMM Wald statistic against a t at the between-within df
# (mixed_df(), per coefficient); a fit with every variance component zero uses
# the single-level reference instead (single_level_fit()).
# Parser constraints: every formula variable not listed in variable_types is
# continuous; only pairwise interactions are parsed (a*b*c loses the 3-way term).

suppressPackageStartupMessages({
  library(lme4)
  library(lmerTest)
})

ALPHA <- 0.05

`%||%` <- function(a, b) if (is.null(a)) b else a

# ---------------------------------------------------------------------------
# Design parser
# Column order: [continuous mains | factor dummy blocks in variable_types order
#                | interaction terms in formula order]
# Interactions expand: cc -> 1 col, cf -> one col per dummy of f,
#                      ff -> dummies of f1 x dummies of f2, f1-major.
# ---------------------------------------------------------------------------

loop_design <- function(case) {
  rhs <- sub("^[^~=]*[~=]", "", case$formula)
  rhs <- gsub("\\([^)]*\\|[^)]*\\)", "", rhs)            # drop (1|g)

  fac_names <- names(case$variable_types)
  proportions <- lapply(case$variable_types, function(spec)
    if (identical(spec, "binary")) c(0.5, 0.5)
    else as.numeric(regmatches(spec, gregexpr("[0-9]*\\.?[0-9]+", spec))[[1]]))
  n_dummies_of <- vapply(proportions, function(p) length(p) - 1L, integer(1))

  vars <- unique(regmatches(rhs, gregexpr("[A-Za-z_][A-Za-z0-9_.]*", rhs))[[1]])
  cont <- setdiff(vars, fac_names)
  n_cont <- length(cont)
  n_dummies <- sum(n_dummies_of)

  # interaction terms in formula order: kind "cc" | "cf" | "ff", 1-based indices
  pair_strs <- regmatches(rhs, gregexpr("\\w+\\s*[:*]\\s*\\w+", rhs))[[1]]
  interactions <- lapply(pair_strs, function(s) {
    v <- regmatches(s, gregexpr("\\w+", s))[[1]]
    a <- v[1]; b <- v[2]
    if (a %in% cont && b %in% cont)
      list(kind = "cc", i = match(a, cont), j = match(b, cont))
    else if (a %in% cont && b %in% fac_names)
      list(kind = "cf", i = match(a, cont), j = match(b, fac_names))
    else if (a %in% fac_names && b %in% cont)
      list(kind = "cf", i = match(b, cont), j = match(a, fac_names))
    else if (a %in% fac_names && b %in% fac_names)
      list(kind = "ff", i = match(a, fac_names), j = match(b, fac_names))
    else stop(sprintf("unparseable interaction %s in %s", s, case$id))
  })
  term_width <- vapply(interactions, function(t) switch(t$kind,
    cc = 1L,
    cf = n_dummies_of[[t$j]],
    ff = n_dummies_of[[t$i]] * n_dummies_of[[t$j]]), integer(1))
  # 0-based "columns before this term", used 1-based at lookup
  inter_offset <- n_cont + n_dummies + c(0L, cumsum(term_width))[seq_along(interactions)]
  P <- n_cont + n_dummies + sum(term_width)
  fac_offset <- n_cont + c(0L, cumsum(n_dummies_of))[seq_along(fac_names)]

  atom <- function(tok) {   # list(kind = "c"|"f", idx, lvl)
    tok <- trimws(tok)
    if (grepl("\\[", tok)) {
      fac <- sub("\\[.*", "", tok)
      lvl <- as.integer(sub(".*\\[([0-9]+)\\].*", "\\1", tok))   # digits INSIDE [] — factor names may contain digits (f2[3] -> 3)
      return(list(kind = "f", idx = match(fac, fac_names), lvl = lvl))
    }
    if (tok %in% cont) return(list(kind = "c", idx = match(tok, cont), lvl = NA_integer_))
    if (!is.null(fac_names) && tok %in% fac_names &&
        length(proportions[[match(tok, fac_names)]]) == 2L)
      return(list(kind = "f", idx = match(tok, fac_names), lvl = 2L))   # bare 2-level factor
    stop(sprintf("cannot resolve effect token '%s' in %s", tok, case$id))
  }

  term_index <- function(kind, i, j) {
    hit <- which(vapply(interactions, function(t) t$kind == kind && t$i == i && t$j == j, logical(1)))
    if (!length(hit)) stop(sprintf("no interaction term %s(%d,%d) in %s", kind, i, j, case$id))
    hit[1]
  }

  col_of <- function(name) {
    name <- trimws(name)
    if (grepl(":", name)) {
      sides <- strsplit(name, ":")[[1]]
      a <- atom(sides[1]); b <- atom(sides[2])
      if (a$kind == "c" && b$kind == "c")
        return(as.integer(inter_offset[term_index("cc", a$idx, b$idx)] + 1L))
      if (a$kind == "f" && b$kind == "c") { tmp <- a; a <- b; b <- tmp }   # f:c names the same column as c:f
      if (a$kind == "c" && b$kind == "f")
        return(as.integer(inter_offset[term_index("cf", a$idx, b$idx)] + (b$lvl - 2L) + 1L))
      if (a$kind == "f" && b$kind == "f") {
        ti <- term_index("ff", a$idx, b$idx)
        return(as.integer(inter_offset[ti] + (a$lvl - 2L) * n_dummies_of[[b$idx]] + (b$lvl - 2L) + 1L))
      }
      stop(sprintf("unsupported interaction order in '%s' (%s)", name, case$id))
    }
    a <- atom(name)
    if (a$kind == "c") return(a$idx)
    as.integer(fac_offset[a$idx] + (a$lvl - 2L) + 1L)
  }

  beta <- numeric(P)
  for (part in strsplit(case$effects, ",")[[1]]) {
    kv <- strsplit(trimws(part), "=")[[1]]
    beta[col_of(kv[1])] <- as.numeric(kv[2])
  }
  targets <- vapply(case$targets, col_of, integer(1))

  # Random-slope design columns (1-based), via the same name->column map as the
  # fixed effects. Slopes are on continuous predictors, which
  # come first in the column order (x1->1, x2->2).
  # Empty unless the case declares cluster$random_slopes.
  slope_cols <- integer(0)
  if (!is.null(case$cluster) && length(case$cluster$random_slopes) > 0L)
    slope_cols <- vapply(case$cluster$random_slopes, col_of, integer(1))

  # Columns constant within clusters: dummies of cluster_level factors, and
  # interactions whose factors are all cluster-level (a continuous component varies
  # within clusters). Mirrors crates/engine-core/src/critvals.rs
  # column_is_cluster_level — change together.
  cl_facs <- match(unlist(case$cluster$cluster_level), fac_names)
  if (anyNA(cl_facs)) stop(sprintf("cluster_level names a non-factor variable in %s", case$id))
  inter_cl <- vapply(interactions, function(t)
    switch(t$kind, cc = FALSE, cf = FALSE, ff = t$i %in% cl_facs && t$j %in% cl_facs), logical(1))
  cluster_level_cols <- c(rep(FALSE, n_cont),
                          rep(seq_along(fac_names) %in% cl_facs, times = n_dummies_of),
                          rep(inter_cl, times = term_width))

  chol_mat <- NULL
  if (!is.null(case$correlations)) {
    C <- diag(n_cont)
    terms <- regmatches(case$correlations,
                        gregexpr("corr\\([^)]*\\)\\s*=\\s*-?[0-9.]+", case$correlations))[[1]]
    for (tm in terms) {
      vars <- regmatches(tm, gregexpr("[A-Za-z_]\\w*", sub("=.*", "", tm)))[[1]]  # "corr","x1","x2"
      r <- as.numeric(sub(".*=", "", tm))
      i <- match(vars[2], cont); j <- match(vars[3], cont)
      C[i, j] <- r; C[j, i] <- r
    }
    chol_mat <- chol(C)     # upper-triangular U, C = U'U; draw Z %*% U
  }

  pair_terms <- gsub("[[:space:]]", "", gsub("*", ":", pair_strs, fixed = TRUE))   # "a:b" in formula order

  list(n_cont = n_cont, cont_names = cont, pair_terms = pair_terms, fac_names = fac_names,
       proportions = proportions, n_dummies_of = n_dummies_of,
       interactions = interactions, beta = beta, targets = targets, chol = chol_mat,
       slope_cols = slope_cols, cluster_level_facs = cl_facs, cluster_level_cols = cluster_level_cols)
}

# ---------------------------------------------------------------------------
# Design matrix draw — mirrors loops_py _draw_X()
# cluster_ids (0-based, one per row) is needed only when the spec has
# cluster_level factors: those are drawn once per cluster and repeated over
# the cluster's rows.
# ---------------------------------------------------------------------------

draw_X <- function(n, spec, cluster_ids = NULL) {
  Xc <- matrix(rnorm(n * spec$n_cont), n, spec$n_cont)
  if (!is.null(spec$chol)) Xc <- Xc %*% spec$chol
  dummies <- list()
  for (fi in seq_along(spec$proportions)) {
    props <- spec$proportions[[fi]]
    L <- length(props)
    if (fi %in% spec$cluster_level_facs) {
      if (is.null(cluster_ids)) stop("draw_X: cluster_ids required for cluster_level variables")
      lvl_c <- sample(0L:(L - 1L), max(cluster_ids) + 1L, replace = TRUE, prob = props / sum(props))
      lvl <- lvl_c[cluster_ids + 1L]
    } else {
      lvl <- sample(0L:(L - 1L), n, replace = TRUE, prob = props / sum(props))
    }
    dmat <- sapply(1L:(L - 1L), function(k) as.numeric(lvl == k))
    if (!is.matrix(dmat)) dmat <- matrix(dmat, ncol = 1L)
    dummies[[fi]] <- dmat
  }
  blocks <- c(list(Xc), dummies)
  for (t in spec$interactions) {
    if (t$kind == "cc") {
      blocks[[length(blocks) + 1L]] <- Xc[, t$i] * Xc[, t$j]
    } else if (t$kind == "cf") {
      blocks[[length(blocks) + 1L]] <- Xc[, t$i] * dummies[[t$j]]   # recycles per column
    } else {                                                          # ff, f1-major
      d1 <- dummies[[t$i]]; d2 <- dummies[[t$j]]
      cols <- do.call(cbind, lapply(seq_len(ncol(d1)), function(k1)
        sapply(seq_len(ncol(d2)), function(k2) d1[, k1] * d2[, k2])))
      if (!is.matrix(cols)) cols <- matrix(cols, ncol = 1L)
      blocks[[length(blocks) + 1L]] <- cols
    }
  }
  do.call(cbind, blocks)
}

# ---------------------------------------------------------------------------
# Cluster assignment — mirrors loops_py _assign_clusters()
# base/extra split: first `extra` clusters get base+1 obs, rest get base.
# ---------------------------------------------------------------------------

assign_clusters <- function(n, k) {
  base  <- n %/% k
  extra <- n %%  k
  sizes <- rep(base, k)
  if (extra > 0L) sizes[seq_len(extra)] <- sizes[seq_len(extra)] + 1L
  rep(seq_len(k) - 1L, sizes)   # 0-based cluster ids (R: use +1 to index b)
}

# Between-within denominator df (SAS DDFM=BETWITHIN), MCPower's default mixed
# reference, one value per coefficient (design columns, intercept excluded). A
# coefficient constant within clusters takes the between df
# k - (cluster-level fixed effects, intercept included); every other one takes
# the residual df n - (P + 1) minus that between df. Random slopes do not change
# either. Mirrors crates/engine-core/src/critvals.rs (df_between, df_within) —
# change together.
mixed_df <- function(n, spec, k) {
  P <- length(spec$beta)
  between <- k - (1L + sum(spec$cluster_level_cols))
  within <- n - (P + 1L) - between
  ifelse(spec$cluster_level_cols, between, within)
}

# A converged fit with every variance component pinned at zero is the plain
# single-level model, so it is tested against the single-level reference instead
# of the between-within one: LMM t at n - (P + 1), GLMM Wald z. Decided per fit.
# A component counts as pinned when its diagonal theta is <= 1e-4; off-diagonal
# Cholesky entries (lower bound -Inf) are ignored.
# Mirrors crates/engine-core/src/critvals.rs BetweenWithinDf::collapsed and glmm's
# PIN_THETA (1e-4) — change together.
single_level_fit <- function(fit) {
  th <- lme4::getME(fit, "theta")
  all(th[lme4::getME(fit, "lower") == 0] <= 1e-4)
}

# ---------------------------------------------------------------------------
# Parallel pooling for the *_best kernels: chunked sims across mclapply, fixed
# per-chunk seed stride. Statistically equivalent to serial, not bit-equal —
# the same stance the engine takes on RNG across worker counts.
# mclapply serializes to 1 core on Windows.
# ---------------------------------------------------------------------------

loop_best_pool <- function(chunk_fn, case, n, n_sims, seed) {
  n_workers <- min(parallel::detectCores(), n_sims)
  if (.Platform$OS.type == "windows") n_workers <- 1L
  base  <- n_sims %/% n_workers
  extra <- n_sims %%  n_workers
  counts <- base + as.integer(seq_len(n_workers) <= extra)
  counts <- counts[counts > 0L]
  parts <- parallel::mclapply(seq_along(counts), function(i)
    chunk_fn(case, n, counts[[i]], seed + (i - 1L) * 100003L),
    mc.cores = length(counts))
  bad <- vapply(parts, function(p) inherits(p, "try-error") || is.null(p$t_rej), logical(1))
  if (any(bad)) stop(sprintf("loop_best chunk failed in %s", case$id))
  t_rej  <- Reduce(`+`, lapply(parts, `[[`, "t_rej"))
  f_rej  <- Reduce(`+`, lapply(parts, `[[`, "f_rej"))
  usable <- Reduce(`+`, lapply(parts, `[[`, "usable"))
  list(power = t_rej / n_sims, joint = f_rej / n_sims, converged = usable)
}

# ---------------------------------------------------------------------------
# OLS kernels
# ---------------------------------------------------------------------------

ols_best_chunk <- function(case, n, n_sims, seed) {
  set.seed(seed)
  spec    <- loop_design(case)
  BETA    <- spec$beta
  TARGETS <- spec$targets
  P       <- length(BETA)
  df      <- n - (P + 1L)
  tcrit   <- qt(1.0 - ALPHA / 2.0, df)
  fcrit   <- qf(1.0 - ALPHA, length(TARGETS), df)

  t_rej <- integer(P)
  f_rej <- 0L
  usable <- 0L

  for (i in seq_len(n_sims)) {
    X   <- draw_X(n, spec)
    y   <- X %*% BETA + rnorm(n)
    Xd  <- cbind(1, X)
    # A draw can be rank-deficient (e.g. a factor level absent at small n); it scores as not significant.
    xtxi <- tryCatch(solve(crossprod(Xd)), error = function(e) NULL)
    if (is.null(xtxi)) next
    usable <- usable + 1L
    coef  <- xtxi %*% crossprod(Xd, y)
    resid <- y - Xd %*% coef
    s2    <- sum(resid^2) / df
    se    <- sqrt(s2 * diag(xtxi))
    tstat <- coef[-1] / se[-1]
    t_rej <- t_rej + as.integer(abs(tstat) > tcrit)

    idx <- TARGETS + 1L
    b   <- coef[idx]
    V   <- s2 * xtxi[idx, idx, drop = FALSE]
    Vf  <- tryCatch(solve(V, b), error = function(e) NULL)
    if (!is.null(Vf)) {
      fstat <- as.numeric(t(b) %*% Vf) / length(TARGETS)
      if (fstat > fcrit) f_rej <- f_rej + 1L
    }
  }

  list(t_rej = t_rej, f_rej = f_rej, usable = usable)
}

ols_best <- function(case, n, n_sims, seed) loop_best_pool(ols_best_chunk, case, n, n_sims, seed)

ols_naive <- function(case, n, n_sims, seed) {
  set.seed(seed)
  spec    <- loop_design(case)
  BETA    <- spec$beta
  TARGETS <- spec$targets
  P       <- length(BETA)

  t_rej <- integer(P)
  f_rej <- 0L
  usable <- 0L

  for (i in seq_len(n_sims)) {
    X   <- draw_X(n, spec)
    y   <- X %*% BETA + rnorm(n)
    fit <- lm(y ~ X)
    # A rank-deficient draw scores as not significant: lm() drops aliased coefs, so summary() would be short.
    if (fit$rank < (P + 1L)) next
    usable <- usable + 1L
    pv  <- summary(fit)$coefficients[-1L, 4L]
    t_rej <- t_rej + as.integer(pv < ALPHA)

    # Joint: reduced vs full via anova()
    p_joint <- tryCatch({
      if (length(TARGETS) == P) {
        red <- lm(y ~ 1)
      } else {
        red <- lm(y ~ X[, -TARGETS, drop = FALSE])
      }
      anova(red, fit)[2L, "Pr(>F)"]
    }, error = function(e) NA_real_)
    if (!is.na(p_joint) && p_joint < ALPHA) f_rej <- f_rej + 1L
  }

  list(power = t_rej / n_sims, joint = f_rej / n_sims, converged = usable)
}

# ---------------------------------------------------------------------------
# GLM logit kernels
# ---------------------------------------------------------------------------

glm_best_chunk <- function(case, n, n_sims, seed) {
  set.seed(seed)
  spec    <- loop_design(case)
  BETA    <- spec$beta
  TARGETS <- spec$targets
  P       <- length(BETA)
  INTERCEPT <- qlogis(case$baseline_p)
  zcrit   <- qnorm(1.0 - ALPHA / 2.0)
  chi2crit <- qchisq(1.0 - ALPHA, length(TARGETS))

  t_rej     <- integer(P)
  f_rej     <- 0L
  converged <- 0L

  for (i in seq_len(n_sims)) {
    X   <- draw_X(n, spec)
    eta <- INTERCEPT + X %*% BETA
    y   <- rbinom(n, 1L, plogis(eta))
    Xd  <- cbind(1, X)

    fit <- tryCatch(
      glm.fit(Xd, y, family = binomial()),
      error = function(e) NULL
    )
    if (is.null(fit) || !isTRUE(fit$converged)) next
    converged <- converged + 1L

    coef_fit <- fit$coefficients
    # Covariance = (X'WX)^-1 via chol on the cross-product. NOT qr.R(qr(.)):
    # R's default qr pivots near-degenerate columns and returns R in pivoted
    # order, silently misaligning the SEs. chol errors on non-PD -> not significant.
    V <- tryCatch({
      mu  <- fit$fitted.values
      wts <- pmax(mu * (1 - mu), 1e-8)
      chol2inv(chol(crossprod(Xd * sqrt(wts))))
    }, error = function(e) NULL)
    if (is.null(V)) next

    se    <- sqrt(diag(V))
    zstat <- coef_fit[-1L] / se[-1L]
    t_rej <- t_rej + as.integer(abs(zstat) > zcrit)

    idx   <- TARGETS + 1L
    b     <- coef_fit[idx]
    Vt    <- V[idx, idx, drop = FALSE]
    Vs    <- tryCatch(solve(Vt, b), error = function(e) NULL)
    if (!is.null(Vs)) {
      w_stat <- as.numeric(t(b) %*% Vs)
      if (w_stat > chi2crit) f_rej <- f_rej + 1L
    }
  }

  list(t_rej = t_rej, f_rej = f_rej, usable = converged)
}

glm_best <- function(case, n, n_sims, seed) loop_best_pool(glm_best_chunk, case, n, n_sims, seed)

glm_naive <- function(case, n, n_sims, seed) {
  set.seed(seed)
  spec    <- loop_design(case)
  BETA    <- spec$beta
  TARGETS <- spec$targets
  P       <- length(BETA)
  INTERCEPT <- qlogis(case$baseline_p)

  t_rej     <- integer(P)
  f_rej     <- 0L
  converged <- 0L

  for (i in seq_len(n_sims)) {
    X   <- draw_X(n, spec)
    eta <- INTERCEPT + X %*% BETA
    y   <- rbinom(n, 1L, plogis(eta))

    fit <- tryCatch(glm(y ~ X, family = binomial()), error = function(e) NULL)
    if (is.null(fit) || !isTRUE(fit$converged)) next
    converged <- converged + 1L

    pv    <- summary(fit)$coefficients[-1L, 4L]
    t_rej <- t_rej + as.integer(pv < ALPHA)

    # Joint via drop1 LRT or anova reduced-vs-full
    p_joint <- tryCatch({
      if (length(TARGETS) == P) {
        red <- glm(y ~ 1, family = binomial())
      } else {
        red <- glm(y ~ X[, -TARGETS, drop = FALSE], family = binomial())
      }
      anova(red, fit, test = "Chisq")[2L, "Pr(>Chi)"]
    }, error = function(e) NA_real_)
    if (!is.na(p_joint) && p_joint < ALPHA) f_rej <- f_rej + 1L
  }

  list(power = t_rej / n_sims, joint = f_rej / n_sims, converged = converged)
}

# ---------------------------------------------------------------------------
# LME kernels
# ---------------------------------------------------------------------------

lme_best_chunk <- function(case, n, n_sims, seed) {
  set.seed(seed)
  spec       <- loop_design(case)
  BETA       <- spec$beta
  TARGETS    <- spec$targets
  P          <- length(BETA)
  N_CLUSTERS <- case$cluster$n_clusters
  ICC        <- case$cluster$ICC
  TAU        <- sqrt(ICC / (1.0 - ICC))
  df         <- mixed_df(n, spec, N_CLUSTERS)
  tcrit      <- qt(1.0 - ALPHA / 2.0, df)
  fcrit      <- qf(1.0 - ALPHA, length(TARGETS), min(df[TARGETS]))   # joint test: smallest target df
  df1        <- n - (P + 1L)
  tcrit1     <- qt(1.0 - ALPHA / 2.0, df1)
  fcrit1     <- qf(1.0 - ALPHA, length(TARGETS), df1)

  t_rej      <- integer(P)
  f_rej      <- 0L
  converged  <- 0L

  cluster_ids <- assign_clusters(n, N_CLUSTERS)   # 0-based

  # Random slopes: one independent slope RE per declared column, uncorrelated
  # with the intercept and each other (the fit still estimates the correlations).
  SLOPE_COLS <- spec$slope_cols
  SLOPE_SD   <- if (length(SLOPE_COLS)) sqrt(case$cluster$slope_variance) else numeric(0)

  # Build lmer formula once (.y ~ V1 .. VP + (1 + Vk..|.g)); intercept included,
  # mirroring loops_py's sm.add_constant (the DGP intercept is 0).
  vn  <- paste0("V", seq_len(P))
  re  <- if (length(SLOPE_COLS)) paste(c("1", vn[SLOPE_COLS]), collapse = " + ") else "1"
  fml <- as.formula(
    paste(".y ~", paste(vn, collapse = " + "), "+ (", re, "| .g)")
  )

  for (i in seq_len(n_sims)) {
    X   <- draw_X(n, spec, cluster_ids)
    b   <- rnorm(N_CLUSTERS) * TAU
    mu  <- X %*% BETA + b[cluster_ids + 1L]
    for (sc in SLOPE_COLS) mu <- mu + (rnorm(N_CLUSTERS) * SLOPE_SD)[cluster_ids + 1L] * X[, sc]
    y   <- mu + rnorm(n)

    df_fit           <- as.data.frame(X)
    names(df_fit)    <- vn
    df_fit$.y        <- as.numeric(y)
    df_fit$.g        <- factor(cluster_ids)

    # Muffle warnings inline (single fit); a real error -> NULL (not significant).
    # (Do NOT re-run lmer in a warning handler — that double-fits and inflates timing.)
    fit <- tryCatch(
      withCallingHandlers(
        lme4::lmer(fml, data = df_fit, REML = TRUE,
                   control = lme4::lmerControl(calc.derivs = FALSE)),
        warning = function(w) invokeRestart("muffleWarning")
      ),
      error = function(e) NULL
    )
    if (is.null(fit)) next
    # Any returned fit is scored, warnings included (mirrors loops_py).
    converged <- converged + 1L

    coef_fix <- as.numeric(lme4::fixef(fit))   # [intercept, V1..VP]
    V_cov    <- as.matrix(stats::vcov(fit))
    se       <- sqrt(diag(V_cov))
    wstat    <- coef_fix[-1L] / se[-1L]
    one      <- single_level_fit(fit)
    t_rej    <- t_rej + as.integer(abs(wstat) > (if (one) tcrit1 else tcrit))

    idx  <- TARGETS + 1L
    b_t  <- coef_fix[idx]
    Vt   <- V_cov[idx, idx, drop = FALSE]
    Vs   <- tryCatch(solve(Vt, b_t), error = function(e) NULL)
    if (!is.null(Vs)) {
      w_stat <- as.numeric(t(b_t) %*% Vs) / length(TARGETS)
      if (w_stat > (if (one) fcrit1 else fcrit)) f_rej <- f_rej + 1L
    }
  }

  list(t_rej = t_rej, f_rej = f_rej, usable = converged)
}

lme_best <- function(case, n, n_sims, seed) loop_best_pool(lme_best_chunk, case, n, n_sims, seed)

lme_naive <- function(case, n, n_sims, seed) {
  set.seed(seed)
  spec       <- loop_design(case)
  BETA       <- spec$beta
  TARGETS    <- spec$targets
  P          <- length(BETA)
  N_CLUSTERS <- case$cluster$n_clusters
  ICC        <- case$cluster$ICC
  TAU        <- sqrt(ICC / (1.0 - ICC))
  tcrit      <- qt(1.0 - ALPHA / 2.0, mixed_df(n, spec, N_CLUSTERS))
  tcrit1     <- qt(1.0 - ALPHA / 2.0, n - (P + 1L))

  t_rej     <- integer(P)
  f_rej     <- 0L
  converged <- 0L

  cluster_ids <- assign_clusters(n, N_CLUSTERS)   # 0-based

  # Random slopes — mirrors lme_best_chunk (see the comment there).
  SLOPE_COLS <- spec$slope_cols
  SLOPE_SD   <- if (length(SLOPE_COLS)) sqrt(case$cluster$slope_variance) else numeric(0)

  vn  <- paste0("V", seq_len(P))
  re  <- if (length(SLOPE_COLS)) paste(c("1", vn[SLOPE_COLS]), collapse = " + ") else "1"
  fml <- as.formula(   # intercept included — mirrors loops_py's sm.add_constant
    paste(".y ~", paste(vn, collapse = " + "), "+ (", re, "| .g)")
  )

  for (i in seq_len(n_sims)) {
    X   <- draw_X(n, spec, cluster_ids)
    b   <- rnorm(N_CLUSTERS) * TAU
    mu  <- X %*% BETA + b[cluster_ids + 1L]
    for (sc in SLOPE_COLS) mu <- mu + (rnorm(N_CLUSTERS) * SLOPE_SD)[cluster_ids + 1L] * X[, sc]
    y   <- mu + rnorm(n)

    df_fit        <- as.data.frame(X)
    names(df_fit) <- vn
    df_fit$.y     <- as.numeric(y)
    df_fit$.g     <- factor(cluster_ids)

    # Muffle warnings inline (single fit); a real error -> NULL (not significant).
    fit <- tryCatch(
      withCallingHandlers(
        lmerTest::lmer(fml, data = df_fit, REML = TRUE),
        warning = function(w) invokeRestart("muffleWarning")
      ),
      error = function(e) NULL
    )
    if (is.null(fit)) next
    converged <- converged + 1L

    # lmerTest's summary (the off-the-shelf call, Satterthwaite df included);
    # the decision reads its t value against the between-within df like the
    # other tiers, not its Satterthwaite p-value.
    sm <- tryCatch(summary(fit)$coefficients, error = function(e) NULL)
    if (is.null(sm)) next

    tv    <- sm[-1L, "t value"]                # drop intercept row
    t_rej <- t_rej + as.integer(abs(tv) > (if (single_level_fit(fit)) tcrit1 else tcrit))

    # Joint: skip (no straightforward joint test for lmerTest in naive mode)
    # Use Wald chi2 on targets via car::linearHypothesis or just skip — keep simple
    f_rej <- f_rej + 0L
  }

  list(power = t_rej / n_sims, joint = f_rej / n_sims, converged = converged)
}

# ---------------------------------------------------------------------------
# GLMM kernels (clustered logistic) -- glmer Laplace, Wald statistic against
# the between-within t (mixed_df()), RX-based SEs (MCPower's wald_se = "rx"). Mirrors
# lme_best_chunk's clustered DGP but with a binary outcome (like glm_best) and a
# glmer fit. The random intercept lives on the LATENT logit scale, where the
# residual variance is pi^2/3 (not 1), so TAU = sqrt(ICC/(1-ICC) * pi^2/3) --
# matching mcpower's latent-scale logit-ICC convention. The naive tier keeps
# glmer's default fit (Hessian computed) but reads the RX-based vcov.
# No Python counterpart (statsmodels lacks a frequentist Laplace GLMM); the
# Python harness skips the loop tier for this family.
# ---------------------------------------------------------------------------

glmm_best_chunk <- function(case, n, n_sims, seed) {
  set.seed(seed)
  spec       <- loop_design(case)
  BETA       <- spec$beta
  TARGETS    <- spec$targets
  P          <- length(BETA)
  N_CLUSTERS <- case$cluster$n_clusters
  ICC        <- case$cluster$ICC
  TAU        <- sqrt(ICC / (1.0 - ICC) * pi^2 / 3.0)   # latent-scale logit ICC
  INTERCEPT  <- qlogis(case$baseline_p)
  df         <- mixed_df(n, spec, N_CLUSTERS)
  tcrit      <- qt(1.0 - ALPHA / 2.0, df)
  fcrit      <- qf(1.0 - ALPHA, length(TARGETS), min(df[TARGETS]))   # joint test: smallest target df
  zcrit      <- qnorm(1.0 - ALPHA / 2.0)
  chicrit    <- qchisq(1.0 - ALPHA, length(TARGETS)) / length(TARGETS)   # w_stat is the Wald chi-square / k

  t_rej      <- integer(P)
  f_rej      <- 0L
  converged  <- 0L

  cluster_ids <- assign_clusters(n, N_CLUSTERS)   # 0-based

  # Random slopes (GLMM slope cases): one independent latent-scale slope RE per
  # declared column, uncorrelated with the intercept and each other (mirrors the
  # engine's benchmark slope config). SLOPE_SD lives on the latent logit scale
  # like TAU. Empty slope_cols -> intercept-only, byte-identical to the pre-slope
  # path (no rnorm consumed). RE term grows to (1 + Vk... | .g). Mirrors glmm_naive.
  SLOPE_COLS <- spec$slope_cols
  SLOPE_SD   <- if (length(SLOPE_COLS)) sqrt(case$cluster$slope_variance) else numeric(0)

  vn  <- paste0("V", seq_len(P))
  re  <- if (length(SLOPE_COLS)) paste(c("1", vn[SLOPE_COLS]), collapse = " + ") else "1"
  fml <- as.formula(
    paste(".y ~", paste(vn, collapse = " + "), "+ (", re, "| .g)")
  )

  for (i in seq_len(n_sims)) {
    X   <- draw_X(n, spec)
    b   <- rnorm(N_CLUSTERS) * TAU
    eta <- INTERCEPT + X %*% BETA + b[cluster_ids + 1L]
    for (sc in SLOPE_COLS) eta <- eta + (rnorm(N_CLUSTERS) * SLOPE_SD)[cluster_ids + 1L] * X[, sc]
    y   <- rbinom(n, 1L, plogis(eta))

    df_fit        <- as.data.frame(X)
    names(df_fit) <- vn
    df_fit$.y     <- as.numeric(y)
    df_fit$.g     <- factor(cluster_ids)

    fit <- tryCatch(
      withCallingHandlers(
        lme4::glmer(fml, data = df_fit, family = binomial(),
                    control = lme4::glmerControl(calc.derivs = FALSE)),
        warning = function(w) invokeRestart("muffleWarning")
      ),
      error = function(e) NULL
    )
    if (is.null(fit)) next
    converged <- converged + 1L

    coef_fix <- as.numeric(lme4::fixef(fit))   # [intercept, V1..VP]
    V_cov    <- as.matrix(stats::vcov(fit))
    se       <- sqrt(diag(V_cov))
    wstat    <- coef_fix[-1L] / se[-1L]
    one      <- single_level_fit(fit)
    t_rej    <- t_rej + as.integer(abs(wstat) > (if (one) zcrit else tcrit))

    idx  <- TARGETS + 1L
    b_t  <- coef_fix[idx]
    Vt   <- V_cov[idx, idx, drop = FALSE]
    Vs   <- tryCatch(solve(Vt, b_t), error = function(e) NULL)
    if (!is.null(Vs)) {
      w_stat <- as.numeric(t(b_t) %*% Vs) / length(TARGETS)
      if (w_stat > (if (one) chicrit else fcrit)) f_rej <- f_rej + 1L
    }
  }

  list(t_rej = t_rej, f_rej = f_rej, usable = converged)
}

glmm_best <- function(case, n, n_sims, seed) loop_best_pool(glmm_best_chunk, case, n, n_sims, seed)

glmm_naive <- function(case, n, n_sims, seed) {
  set.seed(seed)
  spec       <- loop_design(case)
  BETA       <- spec$beta
  TARGETS    <- spec$targets
  P          <- length(BETA)
  N_CLUSTERS <- case$cluster$n_clusters
  ICC        <- case$cluster$ICC
  TAU        <- sqrt(ICC / (1.0 - ICC) * pi^2 / 3.0)
  INTERCEPT  <- qlogis(case$baseline_p)
  tcrit      <- qt(1.0 - ALPHA / 2.0, mixed_df(n, spec, N_CLUSTERS))
  zcrit      <- qnorm(1.0 - ALPHA / 2.0)

  t_rej     <- integer(P)
  f_rej     <- 0L
  converged <- 0L

  cluster_ids <- assign_clusters(n, N_CLUSTERS)

  # Random slopes — mirrors glmm_best_chunk (see the comment there). Empty
  # slope_cols -> intercept-only, byte-identical to the pre-slope path.
  SLOPE_COLS <- spec$slope_cols
  SLOPE_SD   <- if (length(SLOPE_COLS)) sqrt(case$cluster$slope_variance) else numeric(0)

  vn  <- paste0("V", seq_len(P))
  re  <- if (length(SLOPE_COLS)) paste(c("1", vn[SLOPE_COLS]), collapse = " + ") else "1"
  fml <- as.formula(
    paste(".y ~", paste(vn, collapse = " + "), "+ (", re, "| .g)")
  )

  for (i in seq_len(n_sims)) {
    X   <- draw_X(n, spec)
    b   <- rnorm(N_CLUSTERS) * TAU
    eta <- INTERCEPT + X %*% BETA + b[cluster_ids + 1L]
    for (sc in SLOPE_COLS) eta <- eta + (rnorm(N_CLUSTERS) * SLOPE_SD)[cluster_ids + 1L] * X[, sc]
    y   <- rbinom(n, 1L, plogis(eta))

    df_fit        <- as.data.frame(X)
    names(df_fit) <- vn
    df_fit$.y     <- as.numeric(y)
    df_fit$.g     <- factor(cluster_ids)

    fit <- tryCatch(
      withCallingHandlers(
        lme4::glmer(fml, data = df_fit, family = binomial()),
        warning = function(w) invokeRestart("muffleWarning")
      ),
      error = function(e) NULL
    )
    if (is.null(fit)) next
    converged <- converged + 1L

    # vcov warns when the Hessian and RX covariances differ; RX is the choice here.
    se <- tryCatch(suppressWarnings(
            sqrt(diag(as.matrix(stats::vcov(fit, use.hessian = FALSE))))),
          error = function(e) NULL)
    if (is.null(se)) next
    wstat <- as.numeric(lme4::fixef(fit))[-1L] / se[-1L]   # drop intercept
    t_rej <- t_rej + as.integer(abs(wstat) > (if (single_level_fit(fit)) zcrit else tcrit))
  }

  list(power = t_rej / n_sims, joint = f_rej / n_sims, converged = converged)
}

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

LOOPS <- list(
  ols   = list(best = ols_best,  naive = ols_naive),
  logit = list(best = glm_best,  naive = glm_naive),
  lme   = list(best = lme_best,  naive = lme_naive),
  glmm  = list(best = glmm_best, naive = glmm_naive)
)
