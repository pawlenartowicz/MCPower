# zzz.R — engine-sourced lookup tables (distribution/correction/residual) and the .onLoad hook.

# Synthetic- and residual-distribution name -> integer code tables are
# single-sourced in the engine (engine-spec-builder) and read via the
# `dist_codes` / `residual_codes` bridges. Cached lazily (like `.config()`)
# because the native routines are unavailable when this file runs at install
# time. `as.list()` keeps list `[[ ]]` semantics (NULL for a missing name).
.dist_codes <- function() {
  if (is.null(.mcpower_cache$dist_codes)) {
    .mcpower_cache$dist_codes <- as.list(jsonlite::fromJSON(dist_codes(), simplifyVector = TRUE))
  }
  .mcpower_cache$dist_codes
}

.residual_codes <- function() {
  if (is.null(.mcpower_cache$residual_codes)) {
    .mcpower_cache$residual_codes <- as.list(jsonlite::fromJSON(residual_codes(), simplifyVector = TRUE))
  }
  .mcpower_cache$residual_codes
}

.residual_code <- function(name) {
  v <- .residual_codes()[[name]]
  if (is.null(v)) stop(sprintf(
    "unknown residual distribution %s; valid: %s",
    sQuote(name), paste(sort(names(.residual_codes())), collapse = ", ")
  ))
  v
}

# Random-effect distribution name → integer code table (normal/heavy_tailed/right_skewed).
# Single-sourced from the engine via the re_dist_codes bridge (engine-r crate).
# Mirrors the .residual_codes() caching pattern.
.re_dist_codes <- function() {
  if (is.null(.mcpower_cache$re_dist_codes)) {
    .mcpower_cache$re_dist_codes <- as.list(jsonlite::fromJSON(re_dist_codes(), simplifyVector = TRUE))
  }
  .mcpower_cache$re_dist_codes
}

# Canonical `Correction` enum variant names (engine-spec-builder input.rs); the
# engine deserializes these directly, so they need no alias. Input aliases
# ("bh"/"fdr" -> benjamini_hochberg, "tukey" -> tukey_hsd) are single-sourced in
# configs/config.json `correction_aliases`. Python None maps to NULL in R.
.correction_for_rust <- function(correction) {
  if (is.null(correction)) return("none")
  key <- gsub("[- ]", "_", tolower(correction))
  aliases <- .config()$correction_aliases
  if (key %in% names(aliases)) return(unname(aliases[[key]]))
  if (key %in% c("none", "bonferroni", "holm", "benjamini_hochberg", "tukey_hsd")) return(key)
  stop(sprintf("unknown correction %s", sQuote(correction)))
}

.mcpower_cache <- new.env(parent = emptyenv())

.config <- function() {
  if (is.null(.mcpower_cache$config)) {
    .mcpower_cache$config <- jsonlite::fromJSON(config(), simplifyVector = TRUE)
  }
  .mcpower_cache$config
}

.scenario_defaults <- function() {
  if (is.null(.mcpower_cache$scenarios)) {
    .mcpower_cache$scenarios <- jsonlite::fromJSON(
      scenarios(), simplifyVector = TRUE, simplifyDataFrame = FALSE
    )
  }
  .mcpower_cache$scenarios
}

.report_config <- function() .config()$report
.sim_defaults  <- function() .config()$simulation
.limits        <- function() .config()$limits

.onLoad <- function(libname, pkgname) invisible(NULL)

# ── Strict-bootstrap reuse diagnostics ───────────────────────────────────────
# Mirrors mcpower/ports/py/mcpower/model.py:_reuse_fraction / _strict_reuse_warning.

#' Expected % of uploaded rows reused within one strict-bootstrap dataset.
#'
#' A dataset of size N is drawn with replacement from U uploaded rows.
#' Closed form: g = 100 * [1 - (1-1/U)^N - (N/U)*(1-1/U)^(N-1)].
#' Guard: U <= 0 -> 0.0; U == 1 -> 100.0.
#' @keywords internal
.reuse_fraction <- function(U, N) {
  if (U <= 0L) return(0.0)
  if (U == 1L) return(100.0)
  p <- 1.0 - 1.0 / U
  100.0 * (1.0 - p^N - (N / U) * p^(N - 1L))
}

#' Return a warning string when N > ratio*U (strict >) suggesting a lighter mode, else NULL.
#' @keywords internal
.strict_reuse_warning <- function(U, N, ratio) {
  if (N > ratio * U) {
    return(sprintf(
      "N=%d is more than %.4gx the uploaded rows (%d). Each strict-bootstrap dataset will reuse many rows; consider mode='partial' or mode='none' for a faster and more generalizable simulation.",
      as.integer(N), ratio, as.integer(U)
    ))
  }
  NULL
}

#' Set the number of rayon worker threads used by the engine
#'
#' Configure the thread pool before the first call to \code{find_power()} or
#' \code{find_sample_size()}.  The pool is initialised lazily on first use
#' (defaulting to the number of logical CPU cores); once initialised it cannot
#' be changed — a second call raises an error.  Mirrors
#' \code{mcpower._engine.set_n_threads(n)} in the Python port.
#'
#' @param n Integer \eqn{\ge 1}: number of rayon worker threads.
#' @return Invisibly \code{NULL}.
#' @export
set_n_threads <- function(n) {
  if (length(n) != 1L || !is.numeric(n) || is.na(n)) {
    stop("n must be a single integer", call. = FALSE)
  }
  .Call(wrap__set_n_threads, as.integer(n))
  invisible(NULL)
}
