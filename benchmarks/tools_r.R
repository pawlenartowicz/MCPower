#!/usr/bin/env Rscript
# Dedicated-tool adapters (R only): simr, Superpower, simglm -- each used as its
# docs show, returning the tool's power estimate(s) for the sanity column.
# Contract: tool_<name>(case, n, nsim, seed) -> list(power = numeric vector).
# The harness times the call; adapters use lazy :: so package load cost lands
# in the harness's micro warm-up, not the timed run.
# Requires loop_design() from loops_r.R (harness.R sources that first).

TOOLS <- list(
  simr       = function(case, n, nsim, seed) tool_simr(case, n, nsim, seed),
  superpower = function(case, n, nsim, seed) tool_superpower(case, n, nsim, seed),
  simglm     = function(case, n, nsim, seed) tool_simglm(case, n, nsim, seed)
)

# ---------------------------------------------------------------------------
# simr -- the lme and glmm cases. True-parameter workflow: generate one DGP
# draw with
# tau^2 = ICC/(1-ICC), sigma = 1 (matching the loop kernels' TAU
# parameterisation: TAU = sqrt(ICC/(1-ICC)), b ~ N(0,TAU^2)); fit lme4::lmer
# on it; then PIN the generating parameters into the fit (slot assignment:
# @beta, @theta, devcomp sigma) so simr::powerSim simulates at the TRUE
# effect sizes, not at single-draw estimates. simulate.merMod reads exactly
# these slots (getME "beta"/"theta", sigma()), so this replicates what
# simr::makeLmer's fixef<-/VarCorr<-/sigma<- setters would produce.
#
# makeLmer itself is avoided: its VarCorr<-/sigma<- setters call
# lme4::getReCovs(), removed in lme4 >= 1.1.35 (hard error; verified, no
# upstream fix as of simr 1.0.9-1). simrTag marks the fit as a constructed
# true-parameter object, suppressing powerSim's "observed power" warning.
#
# nsim and progress are simrOptions, passed through powerSim's ... argument
# (see body(simr::powerSim): opts <- simrOptions(...)).
# ---------------------------------------------------------------------------

tool_simr <- function(case, n, nsim, seed) {
  if (identical(case$family, "glmm"))
    return(tool_simr_glmm(case, n, nsim, seed))
  set.seed(seed)
  spec <- loop_design(case)

  N_CLUSTERS <- case$cluster$n_clusters
  ICC        <- case$cluster$ICC
  TAU        <- sqrt(ICC / (1.0 - ICC))

  # Build pilot data from the DGP (matches loop kernel data generation)
  d <- as.data.frame(matrix(rnorm(n * spec$n_cont), n, spec$n_cont))
  if (spec$n_cont > 0L) names(d) <- spec$cont_names
  for (fi in seq_along(spec$fac_names)) {
    L <- length(spec$proportions[[fi]])
    d[[spec$fac_names[fi]]] <- factor(rep(seq_len(L), length.out = n))
  }
  cluster_ids <- factor(rep(seq_len(N_CLUSTERS), length.out = n))
  d[[case$cluster$var]] <- cluster_ids

  # Generate response: eta from design matrix + random intercepts + noise
  rhs_fixed <- sub("\\s*\\+\\s*\\([^)]*\\|[^)]*\\).*", "",
                   sub("^[^~]*~\\s*", "", case$formula))
  fml_fixed <- stats::as.formula(paste("~", rhs_fixed))
  X_full <- stats::model.matrix(fml_fixed, data = d)[, -1L, drop = FALSE]
  b_cluster <- rnorm(N_CLUSTERS) * TAU
  y_vals <- as.numeric(X_full %*% spec$beta) +
            b_cluster[as.integer(cluster_ids)] + rnorm(n)
  d[[".y_simr"]] <- y_vals

  # Fit lmer replacing the original LHS with .y_simr
  fml_fit <- stats::as.formula(
    paste(".y_simr ~", sub("^[^~]*~", "", case$formula))
  )
  fit <- suppressWarnings(
    lme4::lmer(fml_fit, data = d, REML = TRUE,
               control = lme4::lmerControl(calc.derivs = FALSE))
  )

  # Pin the generating parameters: powerSim must simulate at the true values,
  # not at this single draw's estimates (intercept 0, sigma 1, theta = TAU/1).
  fit@beta  <- c(0, spec$beta)
  fit@theta <- TAU
  fit@devcomp$cmp[["sigmaML"]]   <- 1.0
  fit@devcomp$cmp[["sigmaREML"]] <- 1.0
  attr(fit, "simrTag") <- TRUE

  ps <- simr::powerSim(fit, test = simr::fixed(case$targets[1], "z"),
                       nsim = nsim, progress = FALSE)
  s <- summary(ps)
  list(power = as.numeric(s$mean))
}

# The glmm cases: same true-parameter workflow on the logit scale. The random
# intercept lives on the LATENT logit scale where the residual variance is
# pi^2/3, so TAU = sqrt(ICC/(1-ICC) * pi^2/3), and slope REs are independent
# with sd sqrt(slope_variance) -- both matching glmm_best_chunk's DGP in
# loops_r.R. For a binomial glmer theta IS the lower Cholesky of the RE
# covariance (no sigma scaling, no devcomp pin); independent REs make it the
# diagonal c(TAU, SLOPE_SD, ...) laid out column-major over the lower triangle.
tool_simr_glmm <- function(case, n, nsim, seed) {
  set.seed(seed)
  spec <- loop_design(case)

  N_CLUSTERS <- case$cluster$n_clusters
  ICC        <- case$cluster$ICC
  TAU        <- sqrt(ICC / (1.0 - ICC) * pi^2 / 3.0)   # latent-scale logit ICC
  INTERCEPT  <- qlogis(case$baseline_p)
  SLOPE_COLS <- spec$slope_cols
  SLOPE_SD   <- if (length(SLOPE_COLS)) sqrt(case$cluster$slope_variance) else numeric(0)

  # Pilot data from the DGP (matches glmm_best_chunk's data generation)
  d <- as.data.frame(matrix(rnorm(n * spec$n_cont), n, spec$n_cont))
  if (spec$n_cont > 0L) names(d) <- spec$cont_names
  for (fi in seq_along(spec$fac_names)) {
    L <- length(spec$proportions[[fi]])
    d[[spec$fac_names[fi]]] <- factor(rep(seq_len(L), length.out = n))
  }
  cluster_ids <- factor(rep(seq_len(N_CLUSTERS), length.out = n))
  d[[case$cluster$var]] <- cluster_ids

  rhs_fixed <- sub("\\s*\\+\\s*\\([^)]*\\|[^)]*\\).*", "",
                   sub("^[^~]*~\\s*", "", case$formula))
  fml_fixed <- stats::as.formula(paste("~", rhs_fixed))
  X_full <- stats::model.matrix(fml_fixed, data = d)[, -1L, drop = FALSE]
  b_cluster <- rnorm(N_CLUSTERS) * TAU
  eta <- INTERCEPT + as.numeric(X_full %*% spec$beta) +
         b_cluster[as.integer(cluster_ids)]
  for (sc in SLOPE_COLS)
    eta <- eta + (rnorm(N_CLUSTERS) * SLOPE_SD)[as.integer(cluster_ids)] * X_full[, sc]
  d[[".y_simr"]] <- rbinom(n, 1L, plogis(eta))

  fml_fit <- stats::as.formula(
    paste(".y_simr ~", sub("^[^~]*~", "", case$formula))
  )
  fit <- suppressWarnings(suppressMessages(
    lme4::glmer(fml_fit, data = d, family = binomial(),
                control = lme4::glmerControl(calc.derivs = FALSE))
  ))

  # Pin the generating parameters (true intercept + betas, diagonal Cholesky).
  k <- 1L + length(SLOPE_COLS)
  L <- diag(c(TAU, rep(SLOPE_SD, length(SLOPE_COLS))), k, k)
  fit@beta  <- c(INTERCEPT, spec$beta)
  fit@theta <- L[lower.tri(L, diag = TRUE)]
  attr(fit, "simrTag") <- TRUE

  ps <- simr::powerSim(fit, test = simr::fixed(case$targets[1], "z"),
                       nsim = nsim, progress = FALSE)
  s <- summary(ps)
  list(power = as.numeric(s$mean))
}

# ---------------------------------------------------------------------------
# Superpower -- the 3 between-subject ANOVA cases. Dummy betas -> cell means
# (sd = 1, n per cell = n/cells). Cell order is row-major over factor levels
# (a1_b1, a1_b2, ..., a2_b1, ...), matching ANOVA_design's labeling.
# ---------------------------------------------------------------------------

superpower_mu <- function(spec) {
  stopifnot(spec$n_cont == 0L)
  beta <- spec$beta
  Ls <- vapply(spec$proportions, length, integer(1))
  if (length(Ls) == 1L) return(c(0, beta[seq_len(Ls - 1L)]))
  L1 <- Ls[1]; L2 <- Ls[2]
  b_f1  <- beta[seq_len(L1 - 1L)]
  b_f2  <- beta[L1 - 1L + seq_len(L2 - 1L)]
  b_int <- beta[(L1 - 1L) + (L2 - 1L) + seq_len((L1 - 1L) * (L2 - 1L))]
  mu <- matrix(0, L1, L2)
  for (i in seq_len(L1)) for (j in seq_len(L2)) {
    m <- 0
    if (i > 1L) m <- m + b_f1[i - 1L]
    if (j > 1L) m <- m + b_f2[j - 1L]
    if (i > 1L && j > 1L) m <- m + b_int[(i - 2L) * (L2 - 1L) + (j - 1L)]
    mu[i, j] <- m
  }
  as.vector(t(mu))   # row-major
}

superpower_design_string <- function(spec) {
  paste0(vapply(spec$proportions, length, integer(1)), "b", collapse = "*")
}

tool_superpower <- function(case, n, nsim, seed) {
  set.seed(seed)
  spec <- loop_design(case)
  cells <- prod(vapply(spec$proportions, length, integer(1)))
  if (n %% cells != 0L)   # n %/% cells below would silently shrink the study
    stop(sprintf("n=%d not divisible by %d cells", n, cells))
  des <- Superpower::ANOVA_design(design = superpower_design_string(spec),
                                  n = n %/% cells, mu = superpower_mu(spec),
                                  sd = 1, plot = FALSE)
  res <- Superpower::ANOVA_power(des, alpha_level = 0.05, nsims = nsim,
                                 verbose = FALSE)
  list(power = as.numeric(res$main_results$power) / 100)   # Superpower reports percent
}

# ---------------------------------------------------------------------------
# simglm -- the ols_*/glm_* cases. sim_args built per case: continuous
# covariates N(0,1), factors via categorical var_type, binomial outcome with
# qlogis(baseline_p) intercept for logit, reg_weights = c(intercept, betas).
# Call shapes pinned against the simglm 0.8.9 vignettes:
#   - "Tidy Simulation with simglm" (tidy_simulation.Rmd): fixed/var_type,
#     factor levels+prob, outcome_type='binary', reg_weights, the
#     replicate_simulation %>% compute_statistics power pipeline.
#   - "Simulation Argument Details for simglm" (simulation_arguments.Rmd):
#     error$variance, model_fit$model_function/family, the power list
#     (dist/alpha). simglm()'s driver only fits a model when sim_args$model_fit
#     is non-null, so model_fit is always supplied. compute_statistics returns
#     one row per model term; term column = the lm/glm coef name (continuous
#     'x1'; factor level k named 'lev<k>' -> term 'f' + 'lev<k>'; interactions
#     joined with ':'). Factor levels carry case proportions via prob=.
# Column order matches loop_design's [cont mains | factor dummies |
# interactions] convention, so reg_weights = c(intercept, spec$beta) directly.
# alpha 0.05, two-tailed z-test (dist='qnorm').
# ---------------------------------------------------------------------------

# Factor level labels: 'lev1', 'lev2', ... so target 'f[k]' -> term 'flev<k>'
# and the reference level is 'lev1' (first), matching the dummy-coded betas.
simglm_levels <- function(L) paste0("lev", seq_len(L))

# Map a case target token (e.g. "x1", "f[2]", "x1:f[2]", bare 2-level "f1") to
# the simglm/lm coefficient name. Mirrors loop_design's atom/col_of resolution.
simglm_term_of <- function(tok, spec) {
  tok <- trimws(tok)
  if (grepl(":", tok)) {
    sides <- strsplit(tok, ":")[[1]]
    return(paste(vapply(sides, simglm_term_of, character(1), spec = spec),
                 collapse = ":"))
  }
  if (grepl("\\[", tok)) {                                   # f[k]
    fac <- sub("\\[.*", "", tok)
    lvl <- as.integer(sub(".*\\[([0-9]+)\\].*", "\\1", tok))
    return(paste0(fac, "lev", lvl))
  }
  if (tok %in% spec$cont_names) return(tok)                  # continuous main
  if (tok %in% spec$fac_names) return(paste0(tok, "lev2"))   # bare 2-level factor
  stop(sprintf("simglm: cannot map target token '%s'", tok))
}

# Build the formula RHS in loop_design column order:
# [cont mains] + [factor names] + [interaction terms in formula order].
simglm_rhs <- function(spec) {
  parts <- c(spec$cont_names, spec$fac_names)
  for (t in spec$interactions) {
    a <- switch(t$kind, cc = spec$cont_names[t$i], cf = spec$cont_names[t$i],
                ff = spec$fac_names[t$i])
    b <- switch(t$kind, cc = spec$cont_names[t$j], cf = spec$fac_names[t$j],
                ff = spec$fac_names[t$j])
    parts <- c(parts, paste0(a, ":", b))
  }
  parts
}

tool_simglm <- function(case, n, nsim, seed) {
  set.seed(seed)
  old_plan <- future::plan(future::sequential)   # plan() returns the previous plan
  on.exit(future::plan(old_plan), add = TRUE)
  spec <- loop_design(case)

  fixed <- list()
  for (cn in spec$cont_names)
    fixed[[cn]] <- list(var_type = "continuous", mean = 0, sd = 1)
  for (fi in seq_along(spec$fac_names)) {
    L <- length(spec$proportions[[fi]])
    fixed[[spec$fac_names[fi]]] <- list(
      var_type = "factor", levels = simglm_levels(L),
      prob = spec$proportions[[fi]] / sum(spec$proportions[[fi]]))
  }

  rhs <- paste(c("1", simglm_rhs(spec)), collapse = " + ")
  fml <- stats::as.formula(paste("y ~", rhs))

  logit   <- identical(case$family, "logit")
  intcpt  <- if (logit) qlogis(case$baseline_p) else 0
  weights <- c(intcpt, spec$beta)

  sim_args <- list(
    formula      = fml,
    fixed        = fixed,
    sample_size  = n,
    reg_weights  = weights,
    replications = nsim,
    power        = list(dist = "qnorm", alpha = 0.05),
    extract_coefficients = TRUE
  )
  if (logit) {
    sim_args$outcome_type <- "binary"
    sim_args$model_fit    <- list(model_function = "glm", family = binomial)
  } else {
    sim_args$error     <- list(variance = 1)
    sim_args$model_fit <- list(model_function = "lm")
  }

  res <- simglm::compute_statistics(
    simglm::replicate_simulation(sim_args), sim_args)
  res <- as.data.frame(res)

  terms <- vapply(case$targets, simglm_term_of, character(1), spec = spec)
  hit <- match(terms, res$term)
  if (anyNA(hit))   # a simglm term-naming change would otherwise record NA power
    stop(sprintf("simglm: term(s) not in compute_statistics output: %s",
                 paste(terms[is.na(hit)], collapse = ", ")))
  list(power = as.numeric(res$power[hit]))
}
