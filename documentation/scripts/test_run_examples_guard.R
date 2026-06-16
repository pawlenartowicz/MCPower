# test_run_examples_guard.R — Standalone sanity test for the nzchar(list())
# re-run guard in run_examples.R.
#
# Run with: Rscript test_run_examples_guard.R
# (Not part of the R package testthat suite — this is documentation tooling.)
#
# The guard expression:
#   if (is.character(plot_name) && length(plot_name) == 1L && nzchar(plot_name))
# guards against bare nzchar(entry$plot) crashing when entry$plot is list()
# (the JSON null round-trip via jsonlite).

guard <- function(plot_name) {
  is.character(plot_name) && length(plot_name) == 1L && nzchar(plot_name)
}

# Simulate JSON null round-trip: jsonlite reads JSON null as list() in
# simplifyVector=FALSE mode.
stopifnot(!guard(list()))           # empty list -> FALSE (no crash)
stopifnot(!guard(NULL))             # NULL -> FALSE
stopifnot(!guard(""))               # empty string -> FALSE
stopifnot( guard("chart.png"))      # real filename -> TRUE
stopifnot(!guard(c("a", "b")))      # vector length > 1 -> FALSE

cat("run_examples guard: OK\n")
