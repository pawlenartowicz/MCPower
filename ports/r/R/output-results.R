# output-results.R — thin pass-through helpers for engine result lists.
#
# The Rust engine emits an already-aggregated result envelope.  This file
# mirrors the shape of engine-py's results.py: boundary-hit rate injection
# and single-scenario unwrapping.  Kept minimal for v1 (no numpy; R operates
# on plain numeric vectors).

#' Inject boundary_hit_rate_tau_zero and boundary_hit_rate_high_tau into a
#' raw engine result list (or into each scenario sub-list inside a
#' multi-scenario envelope).
#'
#' boundary_hit is a flat row-major integer vector of shape
#' (n_sims x n_sample_sizes) emitted by the engine. Values: 0 = no boundary,
#' 1 = tau=0, 2 = high tau.
#'
#' @param result A result list from find_power or find_sample_size.
#' @return The same list with two additional fields: boundary_hit_rate_tau_zero
#'   and boundary_hit_rate_high_tau, each a numeric vector of length
#'   n_sample_sizes (length 1 for find_power — numerically identical to the
#'   former scalar).
#' @keywords internal
.add_boundary_hit_rates <- function(result) {
  if ("scenarios" %in% names(result)) {
    result$scenarios <- lapply(result$scenarios, .add_boundary_hit_rates)
    return(result)
  }
  bh <- result[["boundary_hit"]]
  n_ss <- result[["n_sample_sizes"]]
  if (is.null(n_ss) || n_ss < 1L) n_ss <- max(1L, length(result[["convergence_rate"]]))
  if (is.null(bh) || length(bh) == 0L) {
    result[["boundary_hit_rate_tau_zero"]] <- rep(0.0, n_ss)
    result[["boundary_hit_rate_high_tau"]] <- rep(0.0, n_ss)
    return(result)
  }
  # Engine emits boundary_hit row-major (n_sims x n_sample_sizes); per-N rates
  # mirror Python's add_boundary_hit_rates (one rate per sample-size point).
  n_sims <- length(bh) %/% n_ss
  m <- matrix(bh, nrow = n_sims, ncol = n_ss, byrow = TRUE)
  result[["boundary_hit_rate_tau_zero"]] <- colSums(m == 1L) / n_sims
  result[["boundary_hit_rate_high_tau"]] <- colSums(m == 2L) / n_sims
  result
}

#' Unwrap the multi-scenario envelope when a single scenario was requested.
#'
#' The orchestrator always emits the multi-scenario envelope; callers that
#' asked for a single scenario get the inner list directly.  Boundary-hit
#' rates are injected either way.
#'
#' @param raw      Named list returned by find_power / find_sample_size.
#' @param names    Character vector of scenario names requested.
#' @return Either the inner scenario list (single) or the full envelope (multi).
#' @keywords internal
.unwrap_scenario_result <- function(raw, names) {
  if (length(names) == 1L) {
    inner <- raw$scenarios[[names[[1L]]]]
    return(.add_boundary_hit_rates(inner))
  }
  .add_boundary_hit_rates(raw)
}

#' Check that the worst-case failure rate across all sample-size points does not
#' exceed the configured threshold.
#'
#' Mirrors Python \code{_check_failure_threshold} in \file{output/results.py}.
#' Raises if \code{max(1 - convergence_rate) > threshold} (strict \code{>}).
#' A threshold of 1.0 never raises.
#'
#' @param convergence_rate   Numeric vector; per-N convergence rates in [0,1].
#' @param boundary_hit_rate_tau_zero  Numeric vector; per-N tau=0 boundary rates.
#' @param boundary_hit_rate_high_tau  Numeric vector; per-N high-tau boundary rates.
#' @param threshold  Maximum acceptable failure rate in [0,1].
#' @return Invisibly NULL.
#' @keywords internal
.check_failure_threshold <- function(convergence_rate,
                                     boundary_hit_rate_tau_zero,
                                     boundary_hit_rate_high_tau,
                                     threshold) {
  failure_rates <- 1.0 - convergence_rate
  worst_idx     <- which.max(failure_rates)
  worst_rate    <- failure_rates[[worst_idx]]
  if (worst_rate > threshold) {
    tz <- boundary_hit_rate_tau_zero[[worst_idx]]
    ht <- boundary_hit_rate_high_tau[[worst_idx]]
    stop(sprintf(
      paste0(
        "LME convergence failure rate %.1f%% exceeds the configured threshold %.1f%% ",
        "(sample-size index %d). ",
        "Boundary-hit breakdown at that N: ",
        "tau_zero=%.1f%% (τ̂=0, common for small ICC), ",
        "high_tau=%.1f%% (τ̂ implausibly large, potential red flag). ",
        "Raise the threshold via set_max_failed_simulations() or increase ",
        "n_clusters / sample size."
      ),
      worst_rate * 100, threshold * 100, worst_idx - 1L,
      tz * 100, ht * 100
    ), call. = FALSE)
  }
  invisible(NULL)
}

#' Check result list (or multi-scenario envelope) against the failure threshold.
#'
#' Mirrors Python \code{_check_result_failure_threshold} in \file{model.py}.
#' Recurses into scenario sub-lists when a multi-scenario envelope is present.
#'
#' @param result     Named list returned by \code{.unwrap_scenario_result}.
#' @param threshold  Maximum acceptable failure rate in [0,1].
#' @return Invisibly NULL.
#' @keywords internal
.check_result_failure_threshold <- function(result, threshold) {
  if ("scenarios" %in% names(result)) {
    for (scenario_dict in result$scenarios) {
      .check_result_failure_threshold(scenario_dict, threshold)
    }
    return(invisible(NULL))
  }
  cr <- result[["convergence_rate"]]
  if (is.null(cr)) {
    # Non-LME result or empty run: no failure tracking.
    return(invisible(NULL))
  }
  .check_failure_threshold(
    convergence_rate           = as.numeric(cr),
    boundary_hit_rate_tau_zero = as.numeric(result[["boundary_hit_rate_tau_zero"]] %||%
                                              rep(0.0, length(cr))),
    boundary_hit_rate_high_tau = as.numeric(result[["boundary_hit_rate_high_tau"]] %||%
                                              rep(0.0, length(cr))),
    threshold = threshold
  )
}

# Wrap an unwrapped engine result in its S3 class, attaching the port-owned
# label/header metadata the report renderer needs (mirrors Python make_*_result).
.make_result <- function(raw, meta, kind) {
  cls <- if (kind == "find_sample_size") "mcpower_sample_size_result" else "mcpower_result"
  attr(raw, "mcpower_meta") <- meta
  attr(raw, "mcpower_kind") <- kind
  class(raw) <- c(cls, class(raw))
  raw
}
