# mcpower-debug.R — R6 MCPowerDebug subclass for engine introspection.
#
# Inherits MCPower unchanged (no re-implemented statistics).  Adds four
# stage-access methods — create_data(), dispatch(), raw_statistics(),
# critical_value() — that all share .debug_seed / .debug_n / .debug_n_sims
# so the four stages describe the *same* run. The seed is resolved live per
# call (resolve_base_seed), not snapshotted, so set_seed() is honoured.
#
# Each method builds a single-scenario contract from the parent's state
# (via private$build_contract_bytes, after applying pending setters exactly
# as find_power does) then calls debug_report with the appropriate stage
# mask and decodes the relevant portion. The scenario is .debug_scenario
# ("optimistic" by default), so scenario-perturbed DGPs are debuggable too.

#' MCPowerDebug — engine-introspection subclass of MCPower
#'
#' Inherits the full \code{MCPower} API and adds four stage-access methods
#' (\code{create_data}, \code{dispatch}, \code{raw_statistics},
#' \code{critical_value}) backed by a fixed seed/n/n_sims so every call
#' describes the same simulated run.
#'
#' @export
MCPowerDebug <- R6::R6Class(
  "MCPowerDebug",
  inherit = MCPower,

  public = list(
    # Public debug configuration fields — visible so tests can read them.
    # .debug_seed is an OPTIONAL override: NULL (default) follows the live
    # parent seed via resolve_base_seed (so set_seed() takes effect); a
    # non-NULL value pins the debug draw (the validation harness sets it per case).
    .debug_seed  = NULL,
    .debug_n     = NULL,
    .debug_n_sims = NULL,
    # Scenario whose perturbations the debug run applies. A single name —
    # build_contract_bytes wraps it in a one-element blob, so scenario_index
    # stays 0L. Validated against the configured scenarios at contract build.
    .debug_scenario = "optimistic",

    #' @param formula An R formula or string (forwarded to MCPower).
    #' @param family  "ols" (default), "logit", or "lme".
    #' @param estimator Override the analysis estimator; NULL derives from family.
    #' @param solve_as Synonym for estimator.
    #' @param debug_n   Sample size used for all debug calls (default 50).
    #' @param debug_n_sims Number of simulations for all debug calls (default 200).
    initialize = function(formula, family = "ols", estimator = NULL,
                          solve_as = NULL, debug_n = 50L, debug_n_sims = 200L) {
      super$initialize(formula, family = family, estimator = estimator,
                       solve_as = solve_as)
      # .debug_seed stays NULL (follow the live parent seed); only the
      # debug-only n / n_sims are genuine config to fix here.
      self$.debug_n      <- as.integer(debug_n)
      self$.debug_n_sims <- as.integer(debug_n_sims)
      invisible(self)
    },

    # ------------------------------------------------------------------
    # Stage-access public methods
    # ------------------------------------------------------------------

    #' Generate the design matrix + outcome for sim-0.
    #'
    #' Returns a list with:
    #' \describe{
    #'   \item{design}{numeric matrix (nrow = .debug_n, ncol = n_predictors)}
    #'   \item{columns}{character vector of column names}
    #'   \item{outcome}{numeric vector of length .debug_n}
    #'   \item{cluster_ids}{integer vector or NULL}
    #' }
    create_data = function() {
      rep <- private$.debug(data = TRUE)
      dsgn <- rep$data$design
      mat <- matrix(dsgn$data, nrow = dsgn$nrow, ncol = dsgn$ncol)
      list(
        design             = mat,
        columns            = rep$data$design_columns,
        outcome            = rep$data$outcome,
        cluster_ids        = rep$data$cluster_ids,
        extra_grouping_ids = rep$data$extra_grouping_ids
      )
    },

    #' Fit a provided dataset with the configured solver (the data -> results
    #' path). Inverse of create_data(): pass the list create_data() returns (or
    #' any list with the same shape) and get MCPower's own betas/se/statistic/
    #' critical_value back.
    #'
    #' @param d A list with $design (numeric matrix, nrow x ncol), $outcome
    #'   (numeric, length nrow), and optionally $cluster_ids (integer or NULL).
    #'   $columns is ignored (the engine re-labels positionally).
    #' Returns a list with:
    #' \describe{
    #'   \item{betas}{numeric, length ncol (aligned to design_columns)}
    #'   \item{design_columns}{character, length ncol}
    #'   \item{converged}{logical}
    #'   \item{targets}{list of per-target lists (beta, se, statistic,
    #'     critical_value, target_index, target_label, df, two_sided, ...)}
    #' }
    #' @param wald_se Validation-only override of the Wald-SE kernel mode for this
    #'   single fit. NULL (default) uses the contract's configured mode; accepts
    #'   "rx" (Schur SE) or "hessian" (FD-Hessian SE). Used by the GLMM Oracle
    #'   harness to read per-fit rx vs hessian SE on one dataset.
    load_data = function(d, wald_se = NULL, nagq = 1L) {
      if (!private$applied) private$apply()
      # nagq is baked into the contract (build_contract_bytes -> LinearSpec.nagq);
      # the engine's fit_provided_data honours it (AGQ when > 1). Used by the
      # GLMM Oracle harness to read a per-fit AGQ solve on one dataset.
      blob <- private$build_contract_bytes(self$.debug_scenario, nagq = as.integer(nagq))
      design <- d$design
      stopifnot(is.matrix(design), is.numeric(d$outcome),
                nrow(design) == length(d$outcome))
      cl <- d$cluster_ids
      debug_load_data(
        blob,
        scenario_index = 0L,
        seed           = private$resolve_base_seed(self$.debug_seed),
        design         = as.numeric(design),          # column-major (R matrix order)
        nrow           = nrow(design),
        ncol           = ncol(design),
        outcome        = as.numeric(d$outcome),
        cluster_ids    = if (is.null(cl)) integer(0) else as.integer(cl),
        wald_se        = if (is.null(wald_se)) "" else wald_se
      )
    },

    #' Debug-only: attach extra grouping factors (crossed / nested random
    #' intercepts) to the pending cluster spec, in the contract's serde JSON
    #' shape, e.g.
    #'   list(list(relation = list(Crossed = list(n_clusters = 12L)),
    #'             tau_squared = 0.15))
    #' Call AFTER set_cluster(). Validation tooling for the engine surface;
    #' the user-facing API is not yet exposed.
    #'
    #' @param groupings A list of grouping specs in the contract's serde JSON
    #'   shape (each with $relation and $tau_squared).
    set_extra_groupings_debug = function(groupings) {
      stopifnot(length(private$pending_clusters) > 0L)
      nm <- names(private$pending_clusters)[1]
      private$pending_clusters[[nm]]$extra_groupings <- groupings
      invisible(self)
    },

    #' Debug-only: set the primary random slopes. `slopes` is a list of one or
    #' more list(column=<0-based generation col>, variance=, corr_with_intercept=,
    #' corr_with=<numeric vector of correlations with EARLIER slopes; numeric(0)
    #' for the first slope>).
    #' Call AFTER set_cluster(). Validation tooling for the engine surface;
    #' the user-facing API is not yet exposed.
    #'
    #' @param slopes A list of slope specs in the contract's serde JSON shape.
    set_slopes_debug = function(slopes) {
      stopifnot(length(private$pending_clusters) > 0L)
      nm <- names(private$pending_clusters)[1]
      private$pending_clusters[[nm]]$slopes <- slopes
      invisible(self)
    },

    #' Return dispatch metadata for the debug run.
    #'
    #' Returns the raw \code{$dispatch} sub-list from \code{debug_report}.
    dispatch = function() {
      private$.debug(dispatch = TRUE)$dispatch
    },

    #' Return raw per-simulation statistics, convergence flags, and power.
    #'
    #' Returns a list with \code{targets}, \code{converged}, and \code{power}
    #' (so the empirical reject-rate can be verified against engine-reported
    #' power without a separate call).
    raw_statistics = function() {
      rep <- private$.debug(stats = TRUE, crit = TRUE)
      list(
        targets   = rep$stats$targets,
        converged = rep$stats$converged,
        power     = rep$power
      )
    },

    #' Return critical-value metadata for each target test.
    #'
    #' Returns the raw \code{$crit} sub-list from \code{debug_report}.
    critical_value = function() {
      private$.debug(crit = TRUE)$crit
    }
  ),

  private = list(
    # Build the optimistic-scenario contract bytes (after applying pending
    # setters) and call debug_report with the requested stage mask.
    # Unmapped stage flags default to FALSE.
    .debug = function(input = FALSE, data = FALSE, dispatch = FALSE,
                      stats = FALSE, crit = FALSE) {
      # Apply pending setters (mirrors how find_power triggers private$apply).
      if (!private$applied) private$apply()

      blob <- private$build_contract_bytes(self$.debug_scenario)

      rep <- debug_report(
        blob,
        scenario_index  = 0L,
        seed            = private$resolve_base_seed(self$.debug_seed),
        n               = self$.debug_n,
        n_sims          = self$.debug_n_sims,
        stage_input     = isTRUE(input),
        stage_data      = isTRUE(data),
        stage_dispatch  = isTRUE(dispatch),
        stage_stats     = isTRUE(stats),
        stage_crit      = isTRUE(crit)
      )

      # The bridge hardcodes the power scenario label to "optimistic" (it gets
      # only scenario_index, not the label), so relabel here where
      # .debug_scenario is known. The label lives in two coupled places, both
      # seeded from that hardcode — the outer key (scenarios_envelope,
      # orchestrator_bridge.rs) and the inner $scenario field
      # (power_result_to_host, result_host.rs) — change them together or the
      # result self-contradicts (right key, wrong inner field).
      if (!is.null(rep$power) && !is.null(rep$power$scenarios)) {
        names(rep$power$scenarios)        <- self$.debug_scenario
        rep$power$scenarios[[1]]$scenario <- self$.debug_scenario
      }
      rep
    }
  )
)
