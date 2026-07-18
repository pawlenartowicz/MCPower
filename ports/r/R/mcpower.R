# mcpower.R — R6 MCPower class: constructor + set_* setters + find_power / find_sample_size.
#
# Two independent axes:
#   * family=     — the data-generating process (outcome kind + clustering).
#   * estimator=  — the model fitted to each simulated dataset.

#' MCPower — Monte Carlo power analysis (R6 class)
#'
#' Mirrors the Python `MCPower` API: a formula plus chained `set_*` setters,
#' then `find_power()` / `find_sample_size()`.
#'
#' `find_sample_size()`'s power-vs-N curve uses common random numbers across
#' sample sizes, so it is smooth and the reported N is stable from run to run.
#'
#' @export
MCPower <- R6::R6Class(
  "MCPower",
  public = list(
    # Public Python-parity fields (model.py __init__).
    family = NULL,           # "ols" | "logit" | "probit" | "poisson" | "lme"
    outcome_kind = NULL,     # "continuous" | "binary" | "count"
    link = NULL,             # "canonical" | "probit"
    estimator = NULL,        # "ols" | "glm" | "mle"
    seed = NULL,             # base RNG seed (default 2137; NULL = seed 0, still deterministic)
    power = NULL,            # target power for find_sample_size (default 80)
    alpha = NULL,            # type-I error rate (default 0.05)
    n_simulations = NULL,    # per-call MC count (default 1600)
    max_failed_simulations = NULL,  # failed-sim tolerance (default 1.0)
    intercept = NULL,        # logit(p) once set_baseline_probability is called

    #' @param formula An R `formula` or a string (e.g. `"y ~ x1 + x2"`).
    #' @param family  "ols" (default), "logit", "probit", "poisson", or "lme".
    #' @param estimator Override the analysis estimator ("ols"/"glm"/"mle"); NULL derives from family.
    #' @param solve_as Synonym for estimator; estimator wins when both are given.
    initialize = function(formula, family = "ols", estimator = NULL, solve_as = NULL) {
      # formula accepts an R formula or string; deparse to a canonical string.
      # No R-side parsing — the registry calls Rust parse_formula.
      formula_string <- if (inherits(formula, "formula")) {
        paste(deparse(formula), collapse = " ")
      } else if (is.character(formula)) {
        formula
      } else {
        stop("formula must be an R formula or a string")
      }

      if (!is.character(family) || length(family) != 1L) {
        stop("family must be a single string")
      }
      family_norm <- tolower(family)
      if (!family_norm %in% c("ols", "logit", "probit", "poisson", "lme")) {
        stop(sprintf(
          "unsupported family %s; expected 'ols', 'logit', 'probit', 'poisson', or 'lme'",
          sQuote(family)))
      }
      self$family <- family_norm

      # estimator= / solve_as= override; estimator wins when both supplied.
      estimator_raw <- if (!is.null(estimator)) estimator else solve_as
      if (!is.null(estimator_raw)) {
        if (!is.character(estimator_raw) || length(estimator_raw) != 1L) {
          stop("estimator/solve_as must be a single string or NULL")
        }
        est_norm <- tolower(estimator_raw)
        if (!est_norm %in% c("ols", "glm", "mle")) {
          stop(sprintf("unsupported estimator %s; expected 'ols', 'glm', or 'mle'",
                       sQuote(estimator_raw)))
        }
      }

      # outcome_kind + link from family (mirrors Python model.py __init__):
      #   logit/probit -> binary  (probit sets the non-canonical link)
      #   poisson      -> count
      #   ols/lme      -> continuous
      self$outcome_kind <- if (family_norm %in% c("logit", "probit")) {
        "binary"
      } else if (family_norm == "poisson") {
        "count"
      } else {
        "continuous"
      }
      # Non-canonical link override sent on the wire ("probit"), else canonical.
      self$link <- if (family_norm == "probit") "probit" else "canonical"

      # Default estimator coupling: logit/probit/poisson -> glm (GLMM when
      # clustered), lme -> mle, else ols.
      default_estimator <- if (family_norm %in% c("logit", "probit", "poisson")) {
        "glm"
      } else if (family_norm == "lme") {
        "mle"
      } else {
        "ols"
      }
      self$estimator <- if (!is.null(estimator_raw)) tolower(estimator_raw) else default_estimator

      # Core configuration — defaults from configs/config.json.
      .sim <- .sim_defaults()
      self$seed <- as.integer(.sim$seed)
      self$power <- .sim$target_power * 100
      self$alpha <- .sim$alpha
      # Family-aware: lme fits are more expensive, so they default to the
      # lighter `mixed` budget; OLS and logit (GLM) use the `ols` budget.
      self$n_simulations <- as.integer(.sim$n_sims[[if (family_norm == "lme") "mixed" else "ols"]])
      self$max_failed_simulations <- .sim$max_failed_fraction
      self$intercept <- 0.0

      # Variable registry, parsed from the formula.
      private$registry <- RVariableRegistry$new(formula_string)

      # Scenario configs start at the defaults (deep copy via modifyList on a fresh list).
      private$scenario_configs <- .scenario_defaults()

      # Residual distribution (name), pinned flag, encoded at build time.
      private$residual_dist_name <- "normal"
      private$residual_pinned <- FALSE

      # Heteroskedasticity input for the Rust spec-builder.
      # Always a named list with driver_var_index so jsonlite emits a JSON
      # object {} rather than [] (a bare list() serializes to [] and fails Rust
      # serde). null = "null" in toJSON means NULL entries serialize as JSON null,
      # which Rust Option<T> with serde(default) deserializes correctly.
      private$heteroskedasticity <- list(driver_var_index = NULL)

      # Pending settings — applied lazily before find_* / summary.
      private$pending_variable_types <- NULL
      private$pending_effects <- NULL
      private$pending_correlations <- NULL
      private$applied <- FALSE
      private$effects_set <- FALSE

      # Binary/count-specific pending state. Both stay set after applying (v1
      # semantics): the canonical record that the user supplied a baseline.
      # probability → logit/probit intercept; rate → Poisson log-link intercept.
      private$pending_baseline_probability <- NULL
      private$pending_baseline_rate <- NULL

      # LME state: pending_clusters keyed by grouping var, each entry holds
      # icc / n_clusters / cluster_size. effective_n_clusters is populated at
      # runtime from sample_size for cluster_size-only specs.
      private$pending_clusters <- list()
      private$effective_n_clusters <- NULL

      # Upload data state — populated by upload_data().
      # pending_data holds: columns_typed (matched predictors), raw_columns
      # (all columns), mode, uploaded_n.  NULL = no data uploaded.
      private$pending_data <- NULL

      invisible(self)
    },

    # ------------------------------------------------------------------
    # Configuration setters (model.py). Each returns invisible(self).
    # ------------------------------------------------------------------

    set_seed = function(seed) {
      if (!is.null(seed)) {
        if (length(seed) != 1L || !is.numeric(seed)) {
          stop("seed must be a single integer or NULL")
        }
        if (seed < 0) stop("seed must be non-negative")
      }
      self$seed <- seed
      invisible(self)
    },

    set_power = function(power) {
      if (length(power) != 1L || !is.numeric(power)) stop("power must be numeric")
      self$power <- as.numeric(power)
      invisible(self)
    },

    set_alpha = function(alpha) {
      if (length(alpha) != 1L || !is.numeric(alpha)) stop("alpha must be numeric")
      alpha <- as.numeric(alpha)
      max_alpha <- .config()$limits$max_alpha
      if (alpha > max_alpha) {
        warning(sprintf(
          "Alpha = %g is above the usual maximum of %g; power at such a high significance level is rarely meaningful.",
          alpha, max_alpha
        ), call. = FALSE)
      }
      self$alpha <- alpha
      invisible(self)
    },

    set_simulations = function(n_simulations) {
      if (length(n_simulations) != 1L || !is.numeric(n_simulations)) {
        stop("n_simulations must be numeric")
      }
      self$n_simulations <- as.integer(n_simulations)
      invisible(self)
    },

    set_max_failed_simulations = function(fraction) {
      if (length(fraction) != 1L || !is.numeric(fraction) ||
          fraction < 0 || fraction > 1) {
        stop("fraction must be between 0 and 1")
      }
      self$max_failed_simulations <- as.numeric(fraction)
      invisible(self)
    },

    set_effects = function(spec) {
      if (!is.character(spec) || length(spec) != 1L) {
        stop("set_effects expects a string")
      }
      if (!nzchar(trimws(spec))) stop("effects spec cannot be empty")
      # Accumulate: each call appends a fragment, replayed in order at apply
      # time (last-wins per effect in the registry). Overwriting would silently
      # drop earlier declarations. Mirrors model.py set_effects.
      private$pending_effects <- c(private$pending_effects, spec)
      private$effects_set <- TRUE
      private$applied <- FALSE
      invisible(self)
    },

    set_correlation = function(spec) {
      # Mirror Python set_correlations (model.py): a string assignment spec
      # or a full correlation matrix. (Python also takes a dict; R has no
      # equivalent idiom, so string-or-matrix is the parity surface.)
      #
      # Accumulation with matrix-resets semantics: pending_correlations is a
      # list whose elements are either string fragments or a full matrix. A
      # string APPENDS (partial pairwise spec); a matrix RESETS the accumulator
      # (a full matrix is a complete spec), so a string can only follow a matrix
      # and apply() layers its pairwise entries on top of it.
      if (is.matrix(spec)) {
        private$pending_correlations <- list(spec)
      } else if (is.character(spec) && length(spec) == 1L) {
        private$pending_correlations <- c(private$pending_correlations, list(spec))
      } else {
        stop("set_correlation expects a string or matrix")
      }
      private$applied <- FALSE
      invisible(self)
    },

    # Compatibility alias for v1's pluralized name (model.py).
    set_correlations = function(spec) self$set_correlation(spec),

    set_variable_type = function(spec) {
      if (!is.character(spec) || length(spec) != 1L) {
        stop("set_variable_type expects a string")
      }
      # Accumulate (last-wins per predictor in the registry). Chained/separate
      # calls (one predictor at a time) must each survive; overwriting would
      # silently demote every earlier factor to a continuous column. (A combined
      # string also works — the parser is paren-aware.) Mirrors model.py.
      private$pending_variable_types <- c(private$pending_variable_types, spec)
      private$applied <- FALSE
      invisible(self)
    },

    set_residual_distribution = function(name) {
      # Validate via .residual_code (all five canonical names accepted:
      # normal, right_skewed, left_skewed, high_kurtosis, uniform).
      .residual_code(name)
      private$residual_dist_name <- name
      # Any explicit call (including "normal") pins the residual — scenario
      # distribution swaps will leave it alone.
      private$residual_pinned <- TRUE
      invisible(self)
    },

    set_heteroskedasticity_driver = function(var = NULL) {
      # driver_var_index: NULL means use the linear predictor Xβ as the driver.
      # Else validate and convert to 0-based non-factor predictor index.
      driver_var_index <- NULL
      if (!is.null(var)) {
        non_factor <- private$registry$non_factor_names
        if (!(var %in% non_factor)) {
          stop(sprintf(paste0(
            "heteroskedasticity variable %s must be a non-factor predictor; ",
            "available: %s"), sQuote(var), paste(non_factor, collapse = ", ")))
        }
        driver_var_index <- as.integer(match(var, non_factor) - 1L)  # 0-based
      }

      # The variance ratio λ is scenario-only — no model pin here.
      # Always store driver_var_index so jsonlite emits a JSON object {} (not []).
      # null = "null" in toJSON serializes NULL as JSON null; Rust Option<u32>
      # with serde(default) deserializes JSON null -> None correctly.
      private$heteroskedasticity <- list(driver_var_index = driver_var_index)
      invisible(self)
    },

    #' Add a random-intercept grouping factor to the model.
    #'
    #' @param grouping_var Name of the grouping variable (must appear in the
    #'   formula as a random-effect term, e.g. \code{(1|school)}).
    #' @param ICC Conditional (residual) ICC: within-cluster correlation after
    #'   predictors are accounted for. \code{tau^2 = ICC / (1 - ICC)}.
    #' @param n_clusters Number of distinct clusters; or \code{NULL} for
    #'   cluster-size-only specs.
    #' @param cluster_size Observations per cluster.
    #' @param random_slopes Optional list of random-slope specs, one per slope.
    #'   Each element: \code{list(predictor = "x1", variance = 0.05,
    #'   corr_with_intercept = 0.3, corr_with = c(...))}. \code{corr_with} is
    #'   the vector of correlations with each earlier slope in the same call
    #'   (\code{numeric(0)} for the first slope); defaults to \code{numeric(0)}.
    #' @param cluster_level_vars Character vector of predictor names whose values
    #'   are constant within a cluster (cluster-level covariates). Must all be
    #'   modeled predictors. Default \code{NULL} (no cluster-level covariates).
    #' @param n_per_parent For nested random effects \code{(1|A/B)}: the number of
    #'   B-units per A-unit. Use with the child grouping var (e.g.
    #'   \code{set_cluster("A:B", n_per_parent = 5)}).
    #' @param tau_squared Raw random-intercept variance, for
    #'   \code{family = "poisson"} only (Poisson has no residual-variance term
    #'   for \code{ICC} to divide against). Mutually exclusive with \code{ICC}.
    #' @return The \code{MCPower} object invisibly (for chaining).
    set_cluster = function(grouping_var, ICC = NULL, n_clusters = NULL,
                           cluster_size = NULL,
                           random_slopes = NULL,
                           cluster_level_vars = NULL, n_per_parent = NULL,
                           tau_squared = NULL, ...) {
      # ICC is the CONDITIONAL (residual) ICC tau^2/(tau^2+sigma^2): the
      # within-cluster correlation after the predictors are accounted for. The
      # random intercept is sized tau^2 = ICC/(1-ICC). The raw outcome's marginal
      # ICC is lower whenever the fixed effects explain variance (Var(X*beta) adds
      # to the denominator) -- expected, not a bug.

      # Validate random_slopes: each entry must have predictor (character),
      # variance (positive numeric), corr_with_intercept (numeric in [-1,1]).
      if (!is.null(random_slopes) && length(random_slopes) > 0L) {
        for (i in seq_along(random_slopes)) {
          sl <- random_slopes[[i]]
          if (!is.character(sl$predictor) || length(sl$predictor) != 1L) {
            stop(sprintf("random_slopes[[%d]]$predictor must be a single string", i),
                 call. = FALSE)
          }
          if (!sl$predictor %in% private$registry$predictor_names) {
            stop(sprintf(
              "random_slopes[[%d]]$predictor %s is not a modeled predictor",
              i, sQuote(sl$predictor)
            ), call. = FALSE)
          }
          if (!is.numeric(sl$variance) || sl$variance <= 0) {
            stop(sprintf("random_slopes[[%d]]$variance must be a positive number", i),
                 call. = FALSE)
          }
          corr <- sl$corr_with_intercept %||% 0.0
          if (!is.numeric(corr) || corr < -1 || corr > 1) {
            stop(sprintf(
              "random_slopes[[%d]]$corr_with_intercept must be in [-1, 1]", i
            ), call. = FALSE)
          }
        }
      }

      # Validate cluster_level_vars (a–d mirror Python validators.py).
      if (!is.null(cluster_level_vars)) {
        # (a) must be character
        if (!is.character(cluster_level_vars)) {
          stop("cluster_level_vars must be a character vector of predictor names",
               call. = FALSE)
        }
        # (b) must be non-empty if supplied
        if (length(cluster_level_vars) == 0L) {
          stop("cluster_level_vars must be non-empty when supplied; pass NULL to omit",
               call. = FALSE)
        }
        # (c) grouping_var itself may not also be a cluster-level covariate.
        # Checked before the predictor-membership test so the grouping variable
        # (which is not a fixed predictor) yields this clearer message rather
        # than "not in the formula" (mirrors Python set_cluster ordering).
        if (grouping_var %in% cluster_level_vars) {
          stop(sprintf(
            "cluster_level_vars may not include the grouping variable itself (%s)",
            grouping_var
          ), call. = FALSE)
        }
        # (d) each remaining name must be a modeled predictor
        bad <- setdiff(cluster_level_vars, private$registry$predictor_names)
        if (length(bad) > 0L) {
          stop(sprintf(
            "cluster_level_vars contains names not in the formula: %s",
            paste(bad, collapse = ", ")
          ), call. = FALSE)
        }
      }

      # Poisson (count) mixed models size the random effect by a RAW τ² — no
      # standard latent-scale ICC exists for a log-link count model (Decision 8).
      # Every other family uses ICC. Gate the two so they can't be mixed up.
      if (identical(self$family, "poisson")) {
        if (!is.null(ICC)) {
          stop(paste0(
            "family='poisson' sizes the random effect by tau_squared, not ICC; ",
            "pass tau_squared= (raw τ²) instead of ICC="), call. = FALSE)
        }
        if (is.null(tau_squared)) tau_squared <- 0.0
        if (!is.numeric(tau_squared) || length(tau_squared) != 1L ||
            is.na(tau_squared) || tau_squared < 0) {
          stop(sprintf("tau_squared must be a non-negative number, got %s",
                       format(tau_squared)), call. = FALSE)
        }
      } else if (!is.null(tau_squared)) {
        stop(sprintf(paste0(
          "tau_squared= is only for family='poisson'; family='%s' sizes the ",
          "random effect by ICC="), self$family), call. = FALSE)
      }

      if (is.null(ICC)) ICC <- 0.0
      icc_val <- as.numeric(ICC)
      # Hard range: ICC must be in [0, 1).
      if (icc_val < 0 || icc_val >= 1) {
        stop(sprintf(
          "ICC must be between 0 and 1 (exclusive on upper end), got %g",
          icc_val
        ), call. = FALSE)
      }
      # Stability band: nonzero ICC must be in [icc_lo, icc_hi] for numerical stability.
      lims <- .config()$limits
      icc_lo <- lims$icc_stability[[1]]
      icc_hi <- lims$icc_stability[[2]]
      if (icc_val != 0 && (icc_val < icc_lo || icc_val > icc_hi)) {
        stop(sprintf(
          "ICC must be 0 (no clustering) or between %g and %g for numerical stability. Got %g. Extreme ICC values (< %g or > %g) cause convergence issues in mixed models.",
          icc_lo, icc_hi, icc_val, icc_lo, icc_hi
        ), call. = FALSE)
      }
      # n_clusters must be at least 2 when provided.
      if (!is.null(n_clusters) && as.integer(n_clusters) < 2L) {
        stop(sprintf(
          "n_clusters must be an integer >= 2, got %d",
          as.integer(n_clusters)
        ), call. = FALSE)
      }
      # cluster_size must be at least reliable_rows_per_cluster when provided.
      if (!is.null(cluster_size)) {
        reliable <- as.integer(lims$reliable_rows_per_cluster)
        if (as.integer(cluster_size) < reliable) {
          stop(sprintf(
            "cluster_size must be an integer >= %d for reliable mixed model estimation. Got %d.",
            reliable, as.integer(cluster_size)
          ), call. = FALSE)
        }
      }
      private$pending_clusters[[grouping_var]] <- list(
        n_clusters   = n_clusters,
        cluster_size = cluster_size,
        icc          = as.numeric(ICC),
        # Poisson raw τ² (Decision 8); NULL for ICC-sized families.
        tau_squared  = if (!is.null(tau_squared)) as.numeric(tau_squared) else NULL,
        n_per_parent = n_per_parent,   # NULL for crossed; integer for nested
        raw_slopes   = if (!is.null(random_slopes) && length(random_slopes) > 0L)
                         random_slopes else NULL,
        # Cluster-level covariate names for THIS grouping var. Stored per group
        # (not in a flat field) so a second set_cluster for a different grouping
        # var can't drop the first's vars; unioned across all groups at
        # payload-build into the engine's flat GenerationSpec.cluster_level_columns
        # (mirrors Python set_cluster).
        cluster_level_vars = if (!is.null(cluster_level_vars)) cluster_level_vars else character(0)
      )
      private$applied <- FALSE
      invisible(self)
    },

    set_baseline_probability = function(p) {
      # family gate: a probability set on a Poisson model would silently
      # overwrite whichever baseline was set last (rate or probability) with
      # no warning; mirrors the set_cluster ICC/tau_squared gate.
      if (!self$family %in% c("logit", "probit")) {
        stop(sprintf(paste0(
          "set_baseline_probability is only for family='logit'/'probit'; ",
          "family='%s' sizes the intercept by set_baseline_rate="), self$family),
          call. = FALSE)
      }
      # Store pending p; intercept = log(p/(1-p)) computed at apply time.
      if (!is.numeric(p) || length(p) != 1L) {
        stop("baseline probability must be a number", call. = FALSE)
      }
      pv <- as.numeric(p)
      # Hard reject: p must be in the open interval (0, 1).
      if (pv <= 0 || pv >= 1) {
        stop(sprintf(
          "baseline probability must be in the open interval (0, 1), got %g",
          pv
        ), call. = FALSE)
      }
      # Soft warn: extreme baselines lead to near-separation and unstable power estimates.
      lims <- .config()$limits
      p_lo <- lims$baseline_p_warn[[1]]
      p_hi <- lims$baseline_p_warn[[2]]
      if (pv < p_lo || pv > p_hi) {
        warning(sprintf(
          "baseline probability %g is extreme (outside [%g, %g]); expect near-separation and unstable power estimates",
          pv, p_lo, p_hi
        ), call. = FALSE)
      }
      private$pending_baseline_probability <- pv
      private$applied <- FALSE
      invisible(self)
    },

    # model.py set_baseline_rate — baseline event rate λ₀ for family='poisson'.
    # Stored pending; intercept = log(λ₀) computed at apply time (log link).
    set_baseline_rate = function(rate) {
      # family gate: a rate set on a logit/probit model would silently
      # overwrite whichever baseline was set last (rate or probability) with
      # no warning; mirrors the set_cluster ICC/tau_squared gate.
      if (!identical(self$family, "poisson")) {
        stop(sprintf(paste0(
          "set_baseline_rate is only for family='poisson'; ",
          "family='%s' sizes the intercept by set_baseline_probability="), self$family),
          call. = FALSE)
      }
      if (!is.numeric(rate) || length(rate) != 1L) {
        stop("baseline rate must be a number", call. = FALSE)
      }
      rv <- as.numeric(rate)
      if (rv <= 0) {
        stop(sprintf("baseline rate must be > 0, got %g", rv), call. = FALSE)
      }
      private$pending_baseline_rate <- rv
      private$applied <- FALSE
      invisible(self)
    },

    # model.py upload_data — ingest empirical predictor data.
    # mode controls faithfulness: "none" (marginals only), "partial"
    # (marginals + empirical correlation matrix), "strict" (bootstrap whole rows).
    upload_data = function(data, columns = NULL, mode = "partial", verbose = TRUE) {
      if (!mode %in% c("none", "partial", "strict")) {
        stop("mode must be one of 'none', 'partial', 'strict'", call. = FALSE)
      }

      # Normalize input → (matrix, names).
      norm <- .normalize_upload_input(data, columns)
      mat  <- norm$matrix
      cols <- norm$names

      n_rows <- nrow(mat)
      n_cols <- length(cols)

      if (n_rows == 0L || n_cols == 0L) {
        stop("uploaded data must have at least one row and one column", call. = FALSE)
      }

      # Detect types using config params.
      upload_cfg <- .config()$upload
      min_rows <- as.integer(upload_cfg$min_rows)
      max_rows <- as.integer(upload_cfg$max_rows)
      if (n_rows < min_rows) {
        stop(sprintf(
          "Need at least %d samples for reliable quantile matching, got %d",
          min_rows, n_rows
        ), call. = FALSE)
      }
      if (n_rows > max_rows) {
        stop(sprintf(
          "Uploaded data has too many rows (%s); the maximum is %s.",
          format(n_rows, big.mark = ","), format(max_rows, big.mark = ",")
        ), call. = FALSE)
      }
      max_k     <- as.integer(upload_cfg$max_factor_k_soft)
      max_ratio <- as.numeric(upload_cfg$max_factor_ratio)
      detected  <- .detect_column_types(mat, cols, max_k, max_ratio)
      types_vec  <- detected$types
      labels_list <- detected$labels

      # Store raw columns for ALL uploaded columns.
      raw_columns <- lapply(seq_len(n_cols), function(j) as.list(mat[, j]))
      names(raw_columns) <- cols

      # Identify columns that match modeled predictors (before _apply expansion).
      modeled_names  <- private$registry$predictor_names
      columns_typed  <- list()  # list of list(name, col_type, raw_vals, col_labels)

      for (j in seq_len(n_cols)) {
        col_name   <- cols[j]
        col_type   <- types_vec[j]
        col_labels <- labels_list[[j]]
        raw_vals   <- as.list(mat[, j])

        if (!col_name %in% modeled_names) next

        columns_typed[[length(columns_typed) + 1L]] <- list(
          name      = col_name,
          col_type  = col_type,
          raw_vals  = raw_vals,
          col_labels = col_labels
        )

        # Update registry with detected type for matched predictors.
        if (col_type == "factor") {
          n_lvl <- length(col_labels)
          private$registry$set_variable_type(
            col_name, "factor",
            n_levels = n_lvl,
            labels   = col_labels,
            reference = if (length(col_labels) > 0L) col_labels[[1]] else NULL
          )
        } else if (col_type == "binary") {
          private$registry$set_variable_type(col_name, "binary")
        }
        # continuous: leave as default "normal" — engine handles via upload path
      }

      if (isTRUE(verbose)) {
        cat(sprintf("Uploaded %d rows, %d columns.\n", n_rows, n_cols))
        for (j in seq_len(n_cols)) {
          status <- if (cols[j] %in% modeled_names) "matched" else "extra"
          cat(sprintf("  %s: %s (%s)\n", cols[j], types_vec[j], status))
        }
      }

      private$pending_data <- list(
        columns_typed = columns_typed,
        raw_columns   = raw_columns,
        mode          = mode,
        uploaded_n    = n_rows
      )
      private$applied <- FALSE
      invisible(self)
    },

    # model.py get_effects_from_data — estimate standardized effects from
    # uploaded data. Estimator follows the family: OLS (linear) z-scores the
    # outcome; GLM (logit) and MLE (mixed) fit the native outcome (clustered
    # recovery is fixed-effects-only and needs the grouping column uploaded).
    # Returns a set_effects-style string "x=0.13, y=0.41"; does NOT auto-apply
    # the effects. For a clustered model the verbose note also reports the
    # estimated ICC (latent log-odds scale for logistic) with a copy-paste
    # set_cluster snippet for single-grouping models — also not auto-applied.
    get_effects_from_data = function(y, verbose = TRUE) {
      pending <- private$pending_data
      if (is.null(pending)) {
        stop("no uploaded data; call upload_data(...) before get_effects_from_data()",
             call. = FALSE)
      }

      # raw_columns holds ALL uploaded columns (needed for outcome lookup below);
      # columns_typed$raw_vals holds only matched predictors with their col_type/labels.
      # Both are intentionally kept: outcome 'y' lives in raw_columns but not in
      # columns_typed (predictors only), so they cannot be merged without adding y
      # to columns_typed.
      raw_columns <- pending$raw_columns
      if (!y %in% names(raw_columns)) {
        stop(sprintf("outcome column %s not found in uploaded data; available columns: %s",
                     sQuote(y), paste(sort(names(raw_columns)), collapse = ", ")),
             call. = FALSE)
      }

      # Expand factors so dummy_names / effect_names match the canonical order.
      if (!private$applied) private$apply()

      reg    <- private$registry
      n_rows <- as.integer(pending$uploaded_n)

      # Validate that every modeled main-effect predictor is present in the
      # upload. columns_typed holds list(name, col_type, raw_vals, col_labels) —
      # predictors in both upload and formula. The design matrix itself is
      # assembled engine-side (build_recovery_design), single-sourced with py/Tauri.
      present <- vapply(pending$columns_typed, function(e) e$name, character(1))
      for (name in reg$non_factor_names) {
        if (!name %in% present) {
          stop(paste0(
            "predictor ", sQuote(name), " is in the model but missing from the uploaded data; ",
            "get_effects_from_data needs every modeled main-effect predictor as an upload column"),
            call. = FALSE)
        }
      }
      for (factor_name in reg$factor_names) {
        if (!factor_name %in% present) {
          stop(paste0(
            "factor ", sQuote(factor_name), " is in the model but missing from the uploaded data; ",
            "get_effects_from_data needs every modeled main-effect predictor as an upload column"),
            call. = FALSE)
        }
      }

      # Build the LinearSpec JSON (carries the coded upload columns) and let the
      # engine assemble the recovery design [Intercept, non-factors, dummies,
      # interactions], column-major, with a parallel semantic-name list.
      payload <- .to_linear_spec_list(
        reg, "optimistic",
        alpha = self$alpha, correction = NULL, target_test = NULL,
        heteroskedasticity = private$heteroskedasticity,
        residual_name = private$residual_dist_name,
        residual_pinned = private$residual_pinned,
        max_failed = self$max_failed_simulations, test_formula = NULL,
        scenario_configs = private$scenario_configs, pending_data = pending,
        estimator = self$estimator)
      json <- jsonlite::toJSON(payload, auto_unbox = TRUE, null = "null", digits = NA)
      rd <- build_recovery_design(json)
      design_flat    <- rd$design_flat
      semantic_names <- rd$semantic_names
      ncol_design    <- rd$ncol

      # Outcome scaling depends on the estimator: OLS z-scores it (recovers the
      # standardized beta) via the shared engine helper; GLM (logit) and MLE
      # (mixed) fit it on its native scale (raw 0/1 for logit, raw response).
      y_raw <- as.numeric(unlist(raw_columns[[y]]))
      outcome <- if (identical(self$estimator, "ols")) {
        standardize_continuous(y_raw)
      } else {
        y_raw
      }

      # Clustered recovery (linear mixed `mle` OR logistic GLMM `glm`) is
      # fixed-effects-only: thread the uploaded grouping column as contiguous
      # 0-based cluster IDs and pin the contract's cluster count to the data's
      # distinct-group count so the fitter's per-cluster buffers line up. Gated
      # on the model being clustered (not the estimator) so glm clusters reach
      # the engine's GLMM branch — otherwise the engine never sees the groups.
      cluster_ids <- integer(0)
      if (length(private$pending_clusters) > 0L) {
        grouping_var <- names(private$pending_clusters)[1]
        if (is.null(grouping_var) || !grouping_var %in% names(raw_columns)) {
          stop(paste0(
            "clustered get_effects_from_data needs the grouping column ",
            sQuote(grouping_var), " present in the uploaded data"), call. = FALSE)
        }
        group_raw <- unlist(raw_columns[[grouping_var]])
        cluster_ids <- as.integer(match(group_raw, unique(group_raw)) - 1L)
        # Pin n_clusters to the data's distinct-group count for the contract
        # build, restoring the model state on exit.
        saved_nc <- private$pending_clusters[[grouping_var]]$n_clusters
        private$pending_clusters[[grouping_var]]$n_clusters <- length(unique(group_raw))
        on.exit(
          private$pending_clusters[[grouping_var]]$n_clusters <- saved_nc,
          add = TRUE)
      }

      # Build contract via existing spec-build path (placeholder zero effects).
      contracts <- private$build_contract_bytes("optimistic")

      base_seed <- private$resolve_base_seed(NULL)

      fit <- debug_load_data(
        contracts,
        scenario_index = 0L,
        seed           = as.numeric(base_seed),
        design         = design_flat,
        nrow           = as.integer(n_rows),
        ncol           = as.integer(ncol_design),
        outcome        = outcome,
        cluster_ids    = cluster_ids,
        wald_se        = ""               # use the contract's configured SE mode
      )

      betas <- fit$betas
      if (length(betas) != length(semantic_names)) {
        stop(sprintf(
          "fit returned %d betas but %d design columns were built; canonical column order mismatch",
          length(betas), length(semantic_names)), call. = FALSE)
      }

      # Drop intercept (index 1), format remaining betas.
      parts <- mapply(function(nm, beta) {
        sprintf("%s=%s", nm, round(as.numeric(beta), 4))
      }, semantic_names[-1], betas[-1])
      effects_str <- paste(parts, collapse = ", ")

      if (isTRUE(verbose)) {
        message(paste0(
          "Note: these effect sizes are an APPROXIMATION only - ",
          "standardization, the random-X assumption, and sampling error ",
          "all bias them away from the population values. They are NOT ",
          "auto-applied; call set_effects(...) to use them."))

        # Clustered fits also recover the random-intercept variance: report the
        # estimated ICC so the user need not guess it for set_cluster.
        # variance_components[1] is the primary grouping's intercept variance
        # tau^2 — exactly the quantity set_cluster(ICC=...) reconstructs. A
        # degenerate (non-converged) fit yields an empty vector; note that the
        # ICC is unavailable rather than erroring on the missing component.
        # Poisson has no residual-variance scale to form an ICC ratio against
        # (raw tau^2, not ICC-derived) — no meaningful ICC to report; mirrors
        # driver.rs's cluster_icc, which returns None for it.
        vc <- fit$variance_components
        if (length(private$pending_clusters) > 0L && !identical(self$family, "poisson") &&
            length(vc) == 0L) {
          message("Estimated ICC: unavailable (the random-intercept fit did ",
                  "not converge to a variance estimate).")
        } else if (length(private$pending_clusters) > 0L && !identical(self$family, "poisson")) {
          tau_sq <- as.numeric(vc[1])
          if (identical(self$family, "probit")) {
            # Probit's latent-variable residual variance is fixed at 1 (Phi's
            # scale), unlike logit's pi^2/3 — mirrors driver.rs.
            icc <- tau_sq / (tau_sq + 1.0)
            scale_note <- " (probit latent scale)"
          } else if (identical(self$outcome_kind, "binary")) {
            # Latent (log-odds) scale: residual variance is pi^2/3, not sigma^2
            # (sigma_sq_hat is a 1.0 placeholder for the binomial fit). Inverse
            # of the set_cluster latent conversion. Mirrors spec-builder.R.
            icc <- tau_sq / (tau_sq + pi^2 / 3)
            scale_note <- " (logit latent scale)"
          } else {
            icc <- tau_sq / (tau_sq + as.numeric(fit$sigma_sq_hat))
            scale_note <- ""
          }
          if (length(private$pending_clusters) > 1L) {
            # variance_components[1] is the primary grouping only; a single
            # set_cluster snippet would misrepresent the others.
            message(sprintf(
              "Estimated ICC%s for the primary grouping: %.4f (APPROXIMATION; not auto-applied).",
              scale_note, icc))
          } else {
            grouping_var <- names(private$pending_clusters)[1]
            # Straight quotes (not sQuote): this is a copy-paste R snippet, and
            # fancy curly quotes would not parse if pasted back.
            message(sprintf(
              paste0("Estimated ICC%s: %.4f (APPROXIMATION; not auto-applied). ",
                     "To use it: set_cluster('%s', ICC = %s)."),
              scale_note, icc, grouping_var, round(icc, 4)))
          }
        }

        # Binary fits also recover the baseline event probability: the
        # inverse link of the fitted intercept betas[1], undoing the forward
        # link (logit(p) or probit's qnorm(p)) the generator applies.
        # Reported for any binary outcome (clustered or not), unlike the ICC,
        # which is clustered-only.
        if (identical(self$outcome_kind, "binary")) {
          p_hat <- if (identical(self$family, "probit")) {
            stats::pnorm(as.numeric(betas[1]))
          } else {
            1 / (1 + exp(-as.numeric(betas[1])))
          }
          # Straight quotes (copy-paste R snippet): see the set_cluster note above.
          message(sprintf(
            paste0("Estimated baseline probability: %.4f (APPROXIMATION; not auto-applied). ",
                   "To use it: set_baseline_probability(%s)."),
            p_hat, round(p_hat, 4)))
        }
      }

      effects_str
    },

    # model.py set_parallel tombstone — parallelism is automatic.
    set_parallel = function(...) {
      stop(paste0(
        "mcpower has no set_parallel — parallelism is automatic and ",
        "controlled by mcpower::set_n_threads(n)."),
        call. = FALSE)
    },

    set_scenario_configs = function(configs) {
      if (!is.list(configs)) stop("configs must be a named list")
      # All scenario knob keys are now live; unknown keys still error (typo guard).
      live_keys <- names(.scenario_defaults()[["optimistic"]])
      merged <- .scenario_defaults()
      for (name in names(configs)) {
        user_cfg <- configs[[name]]
        if (!is.list(user_cfg)) {
          stop(sprintf("scenario config for %s must be a list", sQuote(name)))
        }
        for (key in names(user_cfg)) {
          if (!key %in% live_keys) {
            stop(sprintf(
              "scenario %s: unknown config key %s; valid keys: %s",
              sQuote(name), sQuote(key), paste(sort(live_keys), collapse = ", ")
            ))
          }
        }
        if (name %in% names(merged)) {
          merged[[name]] <- modifyList(merged[[name]], user_cfg)
        } else {
          # truth_start is a scenario ASSUMPTION ("estimation is well-behaved"),
          # not a generic knob — a brand-new custom scenario stays cold-start
          # (FALSE) unless the user sets it explicitly, even though it inherits
          # every other key from optimistic.
          custom_cfg <- modifyList(.scenario_defaults()[["optimistic"]], user_cfg)
          if (is.null(user_cfg[["truth_start"]])) {
            custom_cfg$truth_start <- FALSE
          }
          merged[[name]] <- custom_cfg
        }
      }
      private$scenario_configs <- merged
      invisible(self)
    },

    # ------------------------------------------------------------------
    # Analysis entry points (model.py).
    # ------------------------------------------------------------------

    #' @param agq Adaptive Gauss-Hermite quadrature node count for a clustered
    #'   binary/count GLMM fit; \code{NULL} or \code{1} uses Laplace (the
    #'   default). An odd value in \code{3..=25} opts into AGQ, but only when
    #'   the design is eligible (a Binary or Count GLMM with a single grouping
    #'   factor and at most 3 random effects per group) -- an ineligible or
    #'   even/out-of-range value warns and runs at Laplace instead. No-op for
    #'   OLS/LMM.
    find_power = function(sample_size, target_test = NULL, correction = NULL,
                          wald_se = NULL, agq = NULL, test_formula = NULL,
                          n_sims = NULL,
                          seed = NULL, scenarios = FALSE, progress_callback = TRUE,
                          verbose = TRUE) {
      if (!private$applied) private$apply()

      # Validate correction (raises on unknown names).
      private$validate_correction_arg(correction)
      # Resolve wald_se / agq against config; warn-and-strip ineligible agq.
      est <- private$resolve_estimation(wald_se, agq)

      names <- private$resolve_scenarios_arg(scenarios)

      # Logit pre-flight (model.py _validate_logit_runtime).
      private$validate_logit_runtime(names)

      n <- if (!is.null(n_sims)) as.integer(n_sims) else as.integer(self$n_simulations)
      base_seed <- private$resolve_base_seed(seed)

      # Resolve tests to extract posthoc_factors for report_meta.
      wire_posthoc_factors <- character(0)
      wire_targets_for_d4  <- character(0)
      if (!is.null(target_test)) {
        tests_resolved <- .resolve_tests(
          private$registry, target_test,
          overall_available = .overall_test_available(self$estimator, private$registry))
        wire_posthoc_factors <- tests_resolved$posthoc_factors %||% character(0)
        wire_targets_for_d4  <- tests_resolved$targets %||% character(0)
      }

      # D4 rule: Tukey HSD + explicit named beta targets → warn and proceed.
      correction_wire <- .correction_for_rust(correction)
      has_named_beta_targets <- length(wire_targets_for_d4) > 0L &&
        !identical(wire_targets_for_d4, "overall")
      if (identical(correction_wire, "tukey_hsd") && has_named_beta_targets) {
        warning(paste0(
          "Tukey HSD applies only to post-hoc contrast families; the marginal ",
          "coefficient test(s) you requested are reported uncorrected."),
          call. = FALSE)
      }

      contracts <- private$build_contract_bytes(
        names, target_test = target_test, correction = correction,
        wald_se = est$wald_se, nagq = est$nagq, test_formula = test_formula)

      progress <- private$resolve_progress(progress_callback)
      raw <- find_power(contracts, as.integer(sample_size), n, base_seed,
                        progress)
      raw <- .unwrap_scenario_result(raw, names)
      # Post-batch: check that the failure rate (1 - convergence_rate) does not
      # exceed the configured threshold. Mirrors Python model.py find_power.
      .check_result_failure_threshold(raw, self$max_failed_simulations)
      # min_cluster_size: explicit cluster_size, else sample_size %/% n_clusters
      # (primary grouping; smallest atomic cluster at the requested N). NULL unless
      # a clustered binary run (GLMM). Feeds both the transient Laplace warn and the
      # persistent report line — derived identically to Python model.py find_power.
      min_cs <- NULL
      if (length(private$pending_clusters) > 0L && identical(self$outcome_kind, "binary")) {
        primary <- private$pending_clusters[[1L]]
        if (!is.null(primary$cluster_size)) {
          min_cs <- as.integer(primary$cluster_size)
        } else {
          nc <- private$effective_n_clusters %||% primary$n_clusters %||% 1L
          min_cs <- as.integer(sample_size) %/% as.integer(nc)
        }
      }
      baseline_req <- if (identical(self$outcome_kind, "binary"))
        private$pending_baseline_probability else NULL
      meta <- private$.build_report_meta(correction, posthoc_factors = wire_posthoc_factors,
                                         baseline_prob_requested = baseline_req,
                                         min_cluster_size = min_cs)
      result <- .make_result(raw, meta, "find_power")
      # Surface engine preflight warnings exactly once per distinct message.
      .surface_grid_warnings(raw)
      # Laplace-bias warning: fires once per GLMM call with large tau^2 + small
      # clusters. Same derived min_cs as the report line. Mirrors the Python port.
      .check_glmm_laplace_bias_warning(raw, cluster_size = min_cs)
      if (isTRUE(verbose)) print(result)
      # Strict-bootstrap reuse diagnostics (mirrors model.py find_power).
      if (!is.null(private$pending_data) &&
          identical(private$pending_data$mode, "strict") &&
          private$pending_data$uploaded_n > 0L) {
        U     <- as.integer(private$pending_data$uploaded_n)
        N     <- as.integer(sample_size)
        ratio <- .config()$upload$strict_warning_ratio
        frac  <- .reuse_fraction(U, N)
        if (isTRUE(verbose)) {
          cat(sprintf(
            "[strict bootstrap] N=%d, uploaded rows U=%d: ~%.0f%% of rows reused per simulated dataset.\n",
            N, U, frac))
        }
        w <- .strict_reuse_warning(U, N, ratio)
        if (!is.null(w)) warning(w, call. = FALSE)
      }
      result
    },

    #' @param agq Adaptive Gauss-Hermite quadrature node count for a clustered
    #'   binary/count GLMM fit; \code{NULL} or \code{1} uses Laplace (the
    #'   default). An odd value in \code{3..=25} opts into AGQ, but only when
    #'   the design is eligible (a Binary or Count GLMM with a single grouping
    #'   factor and at most 3 random effects per group) -- an ineligible or
    #'   even/out-of-range value warns and runs at Laplace instead. No-op for
    #'   OLS/LMM.
    find_sample_size = function(target_test = NULL, correction = NULL,
                                wald_se = NULL, agq = NULL, test_formula = NULL,
                                target_power = NULL,
                                from_size = NULL, to_size = NULL, by = NULL,
                                mode = "linear",
                                n_sims = NULL, seed = NULL, scenarios = FALSE,
                                progress_callback = TRUE, verbose = TRUE) {
      if (!private$applied) private$apply()
      .ssb <- .sim_defaults()$sample_size_bounds
      if (is.null(from_size)) from_size <- .ssb$from
      if (is.null(to_size))   to_size   <- .ssb$to
      if (is.null(by))        by        <- .ssb$by   # "auto"

      # Validate correction (raises on unknown names).
      private$validate_correction_arg(correction)
      # Resolve wald_se / agq against config; warn-and-strip ineligible agq.
      est <- private$resolve_estimation(wald_se, agq)

      # LME cluster_size-only guard (model.py _validate_lme_find_sample_size_cluster_size).
      private$validate_lme_find_sample_size_cluster_size()

      names <- private$resolve_scenarios_arg(scenarios)
      n <- if (!is.null(n_sims)) as.integer(n_sims) else as.integer(self$n_simulations)
      base_seed <- private$resolve_base_seed(seed)

      # Resolve tests to extract posthoc_factors for report_meta.
      wire_posthoc_factors <- character(0)
      wire_targets_for_d4  <- character(0)
      if (!is.null(target_test)) {
        tests_resolved <- .resolve_tests(
          private$registry, target_test,
          overall_available = .overall_test_available(self$estimator, private$registry))
        wire_posthoc_factors <- tests_resolved$posthoc_factors %||% character(0)
        wire_targets_for_d4  <- tests_resolved$targets %||% character(0)
      }

      # D4 rule: Tukey HSD + explicit named beta targets → warn and proceed.
      correction_wire <- .correction_for_rust(correction)
      has_named_beta_targets <- length(wire_targets_for_d4) > 0L &&
        !identical(wire_targets_for_d4, "overall")
      if (identical(correction_wire, "tukey_hsd") && has_named_beta_targets) {
        warning(paste0(
          "Tukey HSD applies only to post-hoc contrast families; the marginal ",
          "coefficient test(s) you requested are reported uncorrected."),
          call. = FALSE)
      }

      contracts <- private$build_contract_bytes(
        names, target_test = target_test, correction = correction,
        wald_se = est$wald_se, nagq = est$nagq, test_formula = test_formula)

      tp <- if (!is.null(target_power)) as.numeric(target_power) else as.numeric(self$power)
      progress <- private$resolve_progress(progress_callback)

      if (identical(by, "auto")) {
        by_value <- as.integer(.sim_defaults()$cluster_auto_count)
        by_kind  <- "auto"
      } else {
        by_value <- as.integer(by)
        by_kind  <- "fixed"
      }

      raw <- find_sample_size(
        contracts, tp, as.integer(from_size), as.integer(to_size), n, base_seed,
        "grid",
        by_value,
        by_kind,
        as.character(mode),
        NULL,
        progress)
      raw <- .unwrap_scenario_result(raw, names)
      # Post-batch: check failure threshold (mirrors Python model.py find_sample_size).
      .check_result_failure_threshold(raw, self$max_failed_simulations)
      # min_cluster_size: explicit cluster_size, else from_size %/% n_clusters
      # (lower bound of the search range). NULL unless a clustered binary run
      # (GLMM). Feeds both the transient Laplace warn and the persistent report
      # line — derived identically to Python model.py find_sample_size.
      min_cs <- NULL
      if (length(private$pending_clusters) > 0L && identical(self$outcome_kind, "binary")) {
        primary <- private$pending_clusters[[1L]]
        if (!is.null(primary$cluster_size)) {
          min_cs <- as.integer(primary$cluster_size)
        } else {
          nc <- private$effective_n_clusters %||% primary$n_clusters %||% 1L
          min_cs <- as.integer(from_size) %/% as.integer(nc)
        }
      }
      baseline_req <- if (identical(self$outcome_kind, "binary"))
        private$pending_baseline_probability else NULL
      meta <- private$.build_report_meta(correction, posthoc_factors = wire_posthoc_factors,
                                         baseline_prob_requested = baseline_req,
                                         min_cluster_size = min_cs)
      meta$target_power <- tp
      result <- .make_result(raw, meta, "find_sample_size")
      # Surface engine preflight warnings exactly once per distinct message.
      .surface_grid_warnings(raw)
      # Laplace-bias warning: fires once per GLMM call with large tau^2 + small
      # clusters. Same derived min_cs as the report line. Mirrors the Python port.
      .check_glmm_laplace_bias_warning(raw, cluster_size = min_cs)
      if (isTRUE(verbose)) print(result)
      # Strict-bootstrap reuse diagnostics per achieved-N (mirrors model.py find_sample_size).
      if (!is.null(private$pending_data) &&
          identical(private$pending_data$mode, "strict") &&
          private$pending_data$uploaded_n > 0L) {
        U     <- as.integer(private$pending_data$uploaded_n)
        ratio <- .config()$upload$strict_warning_ratio
        # Collect inner scenario lists (single or multi-scenario envelope).
        inner_list <- if (!is.null(result$scenarios)) {
          result$scenarios
        } else {
          list(result)
        }
        for (inner in inner_list) {
          fa <- inner$first_achieved
          if (is.null(fa)) next
          for (pos in seq_along(fa)) {
            achieved_n <- fa[[pos]]
            if (is.null(achieved_n) || (length(achieved_n) == 1L && is.na(achieved_n))) next
            achieved_n <- as.integer(achieved_n)
            frac <- .reuse_fraction(U, achieved_n)
            if (isTRUE(verbose)) {
              cat(sprintf(
                "[strict bootstrap] target %d: first N=%d, uploaded rows U=%d: ~%.0f%% of rows reused per simulated dataset.\n",
                pos - 1L, achieved_n, U, frac))
            }
            w <- .strict_reuse_warning(U, achieved_n, ratio)
            if (!is.null(w)) warning(w, call. = FALSE)
          }
        }
      }
      result
    },

    summary = function() {
      if (!private$applied) private$apply()
      effect_names <- private$registry$effect_names
      effect_sizes <- as.numeric(private$registry$get_effect_sizes())
      effects <- as.list(effect_sizes)
      names(effects) <- effect_names
      list(
        formula = self$equation,
        family = self$family,
        outcome_kind = self$outcome_kind,
        estimator = self$estimator,
        effects = effects,
        n_simulations = self$n_simulations,
        alpha = self$alpha,
        power_target = self$power,
        residual_distribution = private$residual_dist_name,
        residual_pinned = private$residual_pinned,
        scenarios = sort(names(private$scenario_configs))
      )
    }
  ),

  active = list(
    # model.py equation property.
    equation = function() private$registry$equation
  ),

  private = list(
    registry = NULL,
    scenario_configs = NULL,
    residual_dist_name = NULL,
    residual_pinned = FALSE,
    heteroskedasticity = NULL,
    # Accumulating fragment stores (NULL = empty). variable_types / effects are
    # character vectors grown with c(); correlations is a list() of string-or-
    # matrix fragments. apply() replays each in call order.
    pending_variable_types = NULL,
    pending_effects = NULL,
    pending_correlations = NULL,
    pending_baseline_probability = NULL,
    pending_baseline_rate = NULL,
    pending_clusters = NULL,
    effective_n_clusters = NULL,
    applied = FALSE,
    effects_set = FALSE,
    pending_data = NULL,
    # Captured after each build_contract_from_spec call; NULL until the first
    # find_power / find_sample_size call.  .build_report_meta reads this.
    skeleton_json = NULL,

    # model.py _apply — apply pending settings to the registry.
    apply = function() {
      reg <- private$registry

      # 1. Variable types. Fragments replay in call order; a predictor
      #    re-declared in a later fragment last-wins in the registry.
      for (frag in private$pending_variable_types) {
        reg$set_variable_types(frag)
      }

      # 1b. Upload type-lock: for each matched uploaded column, verify the
      # modeled class (factor/binary/continuous) agrees with the detected class.
      # The upload is authoritative; a conflicting user declaration is an error.
      # Matching continuous columns: re-apply is a no-op (leave declared dist).
      if (!is.null(private$pending_data)) {
        for (entry in private$pending_data$columns_typed) {
          col_name      <- entry$name
          detected_type <- entry$col_type   # "continuous" | "binary" | "factor"
          col_labels    <- entry$col_labels
          raw_vals      <- entry$raw_vals

          modeled_class <- private$upload_type_class(reg, col_name)

          if (modeled_class != detected_type) {
            stop(sprintf(
              "Column '%s' was detected as %s from your uploaded data; it can't be modeled as %s. Uploaded columns take their type from the data.",
              col_name, detected_type, modeled_class),
              call. = FALSE)
          }

          # Data wins: re-apply detected type/levels for factor and binary.
          if (detected_type == "factor") {
            n_lvl <- length(col_labels)
            reg$set_variable_type(
              col_name, "factor",
              n_levels  = n_lvl,
              labels    = col_labels,
              reference = if (n_lvl > 0L) col_labels[[1]] else NULL
            )
          } else if (detected_type == "binary") {
            reg$set_variable_type(col_name, "binary")
          }
          # continuous: leave the declared distribution untouched
        }
      }

      # 2. Expand factors.
      if (length(reg$factor_names) > 0L) {
        reg$expand_factors()
      }

      # 3. Effects. Fragments replay in call order; an effect re-declared in a
      #    later fragment last-wins (the registry overwrites its size).
      for (frag in private$pending_effects) {
        reg$set_effects(frag)
      }

      # 4. Baseline probability -> intercept (logit only; no-op otherwise).
      private$apply_baseline_probability()

      # 5. Correlations — the accumulator holds string fragments and/or a full
      #    matrix; set_correlation resets it on a matrix call, so any matrix is
      #    the first element and later string fragments layer pairwise entries on
      #    top of it (set_correlation_spec edits the matrix, not replaces it).
      for (frag in private$pending_correlations) {
        if (is.matrix(frag)) {
          private$apply_correlation_matrix(frag)
        } else {
          reg$set_correlation_spec(frag)
        }
      }

      private$applied <- TRUE
      invisible(NULL)
    },

    # model.py — full-matrix correlation branch of _apply().
    # Validates the structural properties the wire format (upper triangle only)
    # cannot preserve; range (|r| <= 1) and positive semi-definiteness are left
    # to the engine, exactly as the Python port delegates them.
    apply_correlation_matrix = function(matrix) {
      reg <- private$registry
      n <- length(reg$non_factor_names)
      if (nrow(matrix) != n || ncol(matrix) != n) {
        stop(sprintf(
          "Matrix shape (%s) doesn't match %d non-factor variables",
          paste(dim(matrix), collapse = ", "), n),
          call. = FALSE)
      }
      # Structural guards mirroring validators.py _validate_correlation_matrix.
      errors <- character(0)
      if (!isTRUE(all.equal(diag(matrix), rep(1.0, n)))) {
        errors <- c(errors, "Diagonal elements of correlation matrix must be 1")
      }
      if (!isTRUE(all.equal(matrix, t(matrix), check.attributes = FALSE))) {
        errors <- c(errors, "Correlation matrix must be symmetric")
      }
      if (length(errors) > 0L) {
        stop(paste0("Validation failed:\n",
                    paste(sprintf("• %s", errors), collapse = "\n")),
             call. = FALSE)
      }
      reg$set_correlation_matrix(matrix)
      invisible(NULL)
    },

    # model.py _apply_baseline_probability.
    # Binary: log(p/(1-p)) (logit) or qnorm(p) (probit). Count (Poisson):
    # log(λ₀) (log link). Pending values are NOT cleared (canonical record).
    apply_baseline_probability = function() {
      if (!is.null(private$pending_baseline_probability)) {
        p <- private$pending_baseline_probability
        self$intercept <- if (identical(self$family, "probit")) {
          stats::qnorm(p)
        } else {
          log(p / (1.0 - p))
        }
      }
      if (!is.null(private$pending_baseline_rate)) {
        self$intercept <- log(private$pending_baseline_rate)
      }
      invisible(NULL)
    },

    # Mirror of Python _class_of: registry var_type → type class string.
    # "factor" → "factor", "binary" → "binary", everything else → "continuous".
    upload_type_class = function(reg, col_name) {
      pred <- reg$get_predictor(col_name)
      if (is.null(pred)) return("continuous")
      vt <- pred$var_type
      if (isTRUE(pred$is_factor) || identical(vt, "factor")) return("factor")
      if (identical(vt, "binary")) return("binary")
      "continuous"
    },

    # model.py _validate_correction_arg — validate correction name.
    # normalize: lower-case + replace [-space] with _, check against known aliases.
    validate_correction_arg = function(correction) {
      # Validates via .correction_for_rust (which raises on truly unknown names).
      .correction_for_rust(correction)
      invisible(NULL)
    },

    # model.py _validate_logit_runtime — pre-flight for logit/probit/poisson.
    # `scenario_names` is the resolved character vector from resolve_scenarios_arg.
    validate_logit_runtime = function(scenario_names) {
      if (!self$family %in% c("logit", "probit", "poisson")) return(invisible(NULL))

      # Missing baseline. Binary families need a probability; Poisson a rate.
      if (self$family %in% c("logit", "probit")) {
        if (is.null(private$pending_baseline_probability)) {
          stop(sprintf(paste0(
            "baseline probability required for family='%s'; call ",
            "set_baseline_probability(p) before find_power"), self$family),
            call. = FALSE)
        }
      } else {  # poisson
        if (is.null(private$pending_baseline_rate)) {
          stop(paste0(
            "baseline rate required for family='poisson'; call ",
            "set_baseline_rate(rate) before find_power"),
            call. = FALSE)
        }
      }

      # Intercept-only model.
      if (length(private$registry$effect_names) == 0L) {
        stop(sprintf(paste0(
          "family='%s' requires at least one predictor; ",
          "intercept-only models have no testable effect"), self$family),
          call. = FALSE)
      }

      invisible(NULL)
    },

    # model.py _agq_eligible — whether the design admits AGQ (nagq > 1).
    # Binary/count GLMM, single grouping factor, ≤3 REs per group (intercept +
    # slopes). Mirrors contract invariant 25 + glmm assert_model_shape.
    agq_eligible = function() {
      if (!self$outcome_kind %in% c("binary", "count")) return(FALSE)
      if (length(private$pending_clusters) != 1L) return(FALSE)
      cfg <- private$pending_clusters[[1L]]
      n_re <- 1L + length(cfg$raw_slopes %||% list())  # intercept + slopes
      n_re <= 3L
    },

    # model.py _resolve_estimation — resolve wald_se / agq against the config
    # `estimation` defaults, validate, and warn-and-strip an ineligible agq > 1
    # to Laplace (nagq = 1). Returns list(wald_se = <str>, nagq = <int>).
    resolve_estimation = function(wald_se, agq) {
      est <- .config()$estimation
      if (is.null(wald_se)) wald_se <- est$wald_se
      wald_se_wire <- .wald_se_for_rust(wald_se)  # validates
      nagq <- if (is.null(agq)) as.integer(est$nagq) else as.integer(agq)
      if (is.na(nagq) || nagq < 1L || nagq > 25L || nagq %% 2L == 0L) {
        stop(sprintf("agq must be an odd integer in 1..=25, got %s", format(agq)),
             call. = FALSE)
      }
      if (nagq > 1L && !private$agq_eligible()) {
        warning(sprintf(paste0(
          "agq=%d is not available for this design; running at agq=1 (Laplace). ",
          "AGQ requires a clustered binary or count (logit/probit/poisson) model ",
          "with a single grouping factor and at most 3 random effects per group."),
          nagq), call. = FALSE)
        nagq <- 1L
      }
      list(wald_se = wald_se_wire, nagq = nagq)
    },

    # cluster_size-only LME guard for find_sample_size (model.py _validate_lme_find_sample_size_cluster_size).
    validate_lme_find_sample_size_cluster_size = function() {
      if (length(private$pending_clusters) == 0L) return(invisible(NULL))
      cfg <- private$pending_clusters[[1L]]
      if (is.null(cfg$n_clusters)) {
        stop(paste0(
          "find_sample_size with cluster_size-only LME specs is ",
          "not yet supported; pass n_clusters to set_cluster"),
          call. = FALSE)
      }
      invisible(NULL)
    },

    # model.py _resolve_scenarios_arg.
    resolve_scenarios_arg = function(scenarios) {
      configs <- private$scenario_configs
      if (isFALSE(scenarios)) return("optimistic")
      if (isTRUE(scenarios)) {
        names <- names(configs)
        if ("optimistic" %in% names) {
          names <- c("optimistic", setdiff(names, "optimistic"))
        }
        return(names)
      }
      if (!is.character(scenarios)) {
        stop("scenarios must be TRUE, FALSE, or a character vector")
      }
      if (length(scenarios) == 0L) stop("scenarios list cannot be empty")
      invalid <- setdiff(scenarios, names(configs))
      if (length(invalid) > 0L) {
        stop(sprintf("Unknown scenario(s): %s; configured: %s",
                     paste(invalid, collapse = ", "),
                     paste(sort(names(configs)), collapse = ", ")))
      }
      scenarios
    },

    # base_seed: explicit seed= wins; else self$seed; else 0.
    resolve_base_seed = function(seed) {
      if (!is.null(seed)) return(as.numeric(seed))
      if (!is.null(self$seed)) return(as.numeric(self$seed))
      0
    },

    # progress_callback wiring (mirrors model.py resolve_progress_callback).
    # The engine now delivers events to the callback on the R main thread
    # (engine-r run_with_progress), so TRUE maps to a default console reporter,
    # a function is used as-is, and FALSE/NULL pass NULL (silent).
    resolve_progress = function(progress_callback) {
      if (is.function(progress_callback)) return(progress_callback)
      if (isTRUE(progress_callback)) return(.default_progress_reporter())
      NULL
    },

    # Build the metadata list that .make_result attaches to the raw result.
    # Reads the registry's effect_names and factor reference levels.
    # posthoc_factors: character vector of factor names (in request order) for
    # which posthoc_requests were wired; used to build posthoc_factors in meta.
    # effect_skeleton: parsed from private$skeleton_json (set by build_contract_bytes),
    # β-column aligned (intercept at 0); .build_rows renders names from it + factor levels.
    .build_report_meta = function(correction, posthoc_factors = character(0),
                                  baseline_prob_requested = NULL, min_cluster_size = NULL) {
      if (!private$applied) private$apply()
      reg <- private$registry
      factors <- list()
      for (fname in names(reg$`_factors`)) {
        factors[[fname]] <- list(
          baseline = reg$`_factors`[[fname]]$reference_level,
          # Full ordered label list the skeleton's `level` index resolves against
          # (reference included) — mirrors Python _report_meta.
          levels = reg$factor_levels(fname)
        )
      }
      # Parse the skeleton JSON captured by the most-recent build_contract_bytes call.
      # simplifyVector=FALSE is REQUIRED so nested objects stay as named lists, not data.frames.
      effect_skeleton <- if (!is.null(private$skeleton_json)) {
        jsonlite::fromJSON(private$skeleton_json, simplifyVector = FALSE)
      } else {
        NULL
      }
      meta <- list(
        effect_names = reg$effect_names,
        effect_skeleton = effect_skeleton,
        effect_sizes = reg$get_effect_sizes(),
        factors = factors,
        estimator = self$estimator,
        # "binary" for logit-link outcomes (logistic regression and binary GLMM,
        # whose estimator is "mle" not "glm"); gates the OR = exp(β) readout.
        # Mirrors Python _report_meta.
        outcome_kind = self$outcome_kind,
        alpha = self$alpha,
        correction = correction %||% "none",
        target_power = self$power,
        formula = self$equation,
        residual = private$residual_dist_name,
        # Meta-level diagnostics inputs (one per run): requested GLM event
        # probability (drives the live baseline-drift gate) and smallest cluster
        # size at the evaluated N (drives the persistent Laplace line). NULL for
        # OLS / non-binary / non-clustered. Mirrors Python _report_meta.
        baseline_prob_requested = baseline_prob_requested,
        min_cluster_size = min_cluster_size
      )
      # Build posthoc_factors: list of list(name=, levels=) in request order.
      if (length(posthoc_factors) > 0L) {
        ph_meta <- vector("list", length(posthoc_factors))
        for (i in seq_along(posthoc_factors)) {
          fname <- posthoc_factors[[i]]
          ph_meta[[i]] <- list(name = fname, levels = reg$factor_levels(fname))
        }
        meta$posthoc_factors <- ph_meta
      } else {
        meta$posthoc_factors <- list()
      }
      meta
    },

    # Shared contract-bytes builder used by find_power, find_sample_size, and tests.
    # Builds the registry-derived payload, serializes it, and calls build_contract_from_spec.
    # Captures $skeleton from the engine response into private$skeleton_json so
    # .build_report_meta can include the effect_skeleton without a separate call.
    build_contract_bytes = function(scenario_names, target_test = NULL,
                                    correction = NULL, wald_se = NULL,
                                    nagq = 1L, test_formula = NULL) {
      if (!private$applied) private$apply()
      reg <- private$registry

      # Collect cluster_level_vars across all pending cluster specs (mirrors
      # Python build path); the engine surface is a single flat column list.
      clv <- unlist(lapply(private$pending_clusters, `[[`, "cluster_level_vars"),
                    use.names = FALSE)

      payload <- .to_linear_spec_list(
        reg, scenario_names,
        alpha = self$alpha,
        correction = correction,
        wald_se = wald_se,
        nagq = nagq,
        target_test = target_test,
        heteroskedasticity = private$heteroskedasticity,
        residual_name = private$residual_dist_name,
        residual_pinned = private$residual_pinned,
        max_failed = self$max_failed_simulations,
        test_formula = test_formula,
        scenario_configs = private$scenario_configs,
        pending_data = private$pending_data,
        cluster_level_vars = if (length(clv) > 0L) clv else NULL,
        estimator = self$estimator)

      # CRITICAL: null = "null" so nullable fields serialize as JSON null (Rust
      # Option<T> deserializes correctly); digits = NA avoids rounding floats.
      json <- jsonlite::toJSON(payload, auto_unbox = TRUE, null = "null", digits = NA)

      # Resolve raw_slopes predictor names to 0-based generation column
      # indices (mirrors Python's slope-resolution path). Column index for a
      # simple main effect = eff$column_index (singular scalar; column_indices,
      # plural, is interaction-only and integer(0) for a main effect). corr_with
      # defaults to numeric(0) for the first slope; each subsequent slope's
      # corr_with has length = position-1.
      resolved_clusters <- private$pending_clusters
      for (gv in names(resolved_clusters)) {
        raw_slp <- resolved_clusters[[gv]]$raw_slopes
        if (!is.null(raw_slp)) {
          resolved <- lapply(seq_along(raw_slp), function(i) {
            sl  <- raw_slp[[i]]
            eff <- reg$get_effect(sl$predictor)
            if (is.null(eff)) stop(sprintf(
              "random slope predictor %s not found in registry", sQuote(sl$predictor)),
              call. = FALSE)
            col_idx <- eff$column_index  # 0-based generation column (main effect: column_index, singular; column_indices is interaction-only / integer(0) for main effects)
            # [["corr_with"]] (exact match) — `$corr_with` would PARTIAL-match
            # the sibling `corr_with_intercept` key when corr_with is absent,
            # wrongly giving the first slope a length-1 corr_with.
            corr_w  <- sl[["corr_with"]] %||% numeric(0)
            list(
              column              = as.integer(col_idx),
              variance            = as.numeric(sl$variance),
              corr_with_intercept = as.numeric(sl$corr_with_intercept %||% 0.0),
              corr_with           = I(as.numeric(corr_w))
            )
          })
          resolved_clusters[[gv]]$slopes <- resolved
        }
      }

      enc <- .encode_outcome_and_clusters(
        self$family, self$link, self$estimator, self$intercept, resolved_clusters)

      out <- build_contract_from_spec(
        json, enc$outcome_kind, enc$link, enc$estimator, enc$intercept,
        enc$clusters_json)
      # Capture the skeleton so .build_report_meta can use it (avoids a second call).
      private$skeleton_json <- out$skeleton
      out$contracts
    }
  )
)

# Build the canonical Laplace-approximation bias message for one GLMM scenario's
# estimator_extras, or NULL when it does not apply. Reads both thresholds from
# .config() — not hardcoded. Fires when the GLM τ̂² exceeds glmm_tau_sq_warn AND
# min_cluster_size < recommended_rows_per_cluster (the validator already rejects
# configs below `reliable`, so the reachable band is [reliable, recommended):
# allowed but Laplace-risky). Mirrors the Python port's _glmm_laplace_bias_warning
# (model.py) — message text + firing rule change all ports together. Shared by the
# transient fit-time warn (.check_glmm_laplace_bias_warning) and the persistent
# report line (.diagnostic_warnings) so the two never diverge.
.glmm_laplace_bias_warning <- function(extras, min_cluster_size) {
  if (is.null(min_cluster_size) || is.na(min_cluster_size)) return(NULL)
  if (is.null(extras) || !identical(extras$estimator, "glm")) return(NULL)
  tau_hat <- extras$tau_squared_hat_mean
  if (is.null(tau_hat) || is.na(tau_hat)) return(NULL)
  cfg      <- .config()
  thr      <- cfg$report$thresholds$glmm_tau_sq_warn
  min_size <- cfg$limits$recommended_rows_per_cluster
  if (tau_hat > thr && min_cluster_size < min_size) {
    return(sprintf(
      paste0(
        "Laplace-approximation bias likely: estimated random-intercept variance ",
        "τ̂² = %.2f exceeds %.2f with small clusters (min cluster size %d < %d). ",
        "GLMM power may be optimistic — interpret with caution or increase cluster size."
      ),
      tau_hat, thr, as.integer(min_cluster_size), as.integer(min_size)
    ))
  }
  NULL
}

# Fire the Laplace-approximation bias warning for GLMM (binary + cluster) results.
# Recurses into multi-scenario envelopes and warns once if ANY scenario breaches,
# delegating the message + firing rule to .glmm_laplace_bias_warning.
#
# cluster_size: smallest cluster size at the evaluated N (host-derived:
#   explicit cluster_size, or sample_size %/% n_clusters). NULL/NA suppresses the
#   warning (size unknown).
.check_glmm_laplace_bias_warning <- function(result, cluster_size) {
  if (is.null(cluster_size) || is.na(cluster_size)) return(invisible(NULL))
  scenarios <- if (!is.null(result$scenarios)) result$scenarios else list(result)
  for (sc in scenarios) {
    msg <- .glmm_laplace_bias_warning(sc$estimator_extras, cluster_size)
    if (!is.null(msg)) {
      warning(msg, call. = FALSE)
      break  # one warning per call
    }
  }
  invisible(NULL)
}

# Surface engine grid_warnings exactly once per distinct message (mirrors Python
# _surface_warnings in model.py). Handles both the single-scenario result and
# the multi-scenario envelope; dedupes preserving first-seen order.
# Called after both find_power and find_sample_size result construction.
.surface_grid_warnings <- function(result) {
  seen <- character(0)
  emit <- function(inner) {
    for (w in (inner$grid_warnings %||% character(0))) {
      if (!w %in% seen) {
        seen <<- c(seen, w)
        warning(w, call. = FALSE)
      }
    }
  }
  if (!is.null(result$scenarios)) {
    for (sc in result$scenarios) emit(sc)
  } else {
    emit(result)
  }
  invisible(NULL)
}

# Default console progress reporter (the TRUE branch of resolve_progress).
# Mirrors the Python tqdm default: stays silent for the first ~2 s so fast runs
# draw no bar, then opens a utils::txtProgressBar on stderr and tracks the
# engine's (current, total) reports. The engine calls this on the R main thread
# (engine-r run_with_progress), so direct console I/O here is safe.
.PROGRESS_DELAY_SECONDS <- 2.0

.default_progress_reporter <- function() {
  bar   <- NULL
  total <- NULL
  start <- Sys.time()
  function(current, total_in) {
    # Suppress the bar until the run is slow enough to be worth one; a fast run
    # finishes inside the window and never opens a bar.
    if (is.null(bar)) {
      if (as.numeric(Sys.time() - start, units = "secs") < .PROGRESS_DELAY_SECONDS)
        return(invisible(TRUE))
      total <<- total_in
      bar   <<- utils::txtProgressBar(min = 0, max = total_in, initial = current,
                                      style = 3, file = stderr())
    }
    # The engine may report a different total per scenario; rebuild on change.
    if (!identical(total, total_in)) {
      total <<- total_in
      bar   <<- utils::txtProgressBar(min = 0, max = total_in, initial = current,
                                      style = 3, file = stderr())
    }
    utils::setTxtProgressBar(bar, current)
    if (current >= total_in) {
      close(bar)
      bar <<- NULL
    }
    invisible(TRUE)
  }
}

