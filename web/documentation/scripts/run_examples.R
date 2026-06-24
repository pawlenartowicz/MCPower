#!/usr/bin/env Rscript
# Run the R documentation examples and cache their real output.
#
# Mirror of run_examples.py for the R port. Reads examples-r.json, evaluates
# each entry's `code` capturing console output into `output`, and stamps
# `captured_at`.
#
# If an entry sets `plot`, the code must leave the result object to be plotted
# in a variable named `result`. The R port ships no built-in PNG renderer, so
# rather than render here this writes the themed Vega-Lite spec to
# ../assets/examples/<plot>.vl.json; `run_examples.py` then converts every such
# spec to PNG with vl_convert (one renderer for both ports). So: run this, then
# run run_examples.py to materialise the R plots.
#
# This fills the cache that inject_examples.py pastes into the tutorial pages:
# author `code`, run this to fill `output`/`captured_at` (and emit plot specs),
# then run inject_examples.py to update the pages. Not wired into the leyline
# build.
#
# Usage:
#   Rscript run_examples.R            # refresh every entry
#   Rscript run_examples.R id1 id2    # refresh only the named entries

suppressMessages({
  library(jsonlite)
  library(mcpower)
})

here   <- dirname(normalizePath(sub("--file=", "", grep("--file=", commandArgs(FALSE), value = TRUE))))
cache  <- file.path(here, "examples-r.json")
assets <- file.path(dirname(here), "assets", "examples")

# Write the primary Vega-Lite spec block for a result as a .vl.json file.
# Uses .plot_blocks() (the current internal API) to obtain the named block list,
# then picks the representative block (power for find_power, curve for
# find_sample_size). run_examples.py converts every .vl.json to PNG via
# vl_convert — so the R port ships no PNG renderer of its own.
# Fails gracefully with a warning when vegawidget/V8 are absent (they are not
# needed here — the spec is light-print-themed JSON written straight to disk).
write_plot_spec <- function(result, plot_name) {
  dir.create(assets, recursive = TRUE, showWarnings = FALSE)
  kind <- attr(result, "mcpower_kind")
  if (is.null(kind)) kind <- "find_power"
  blocks <- tryCatch(
    mcpower:::.plot_blocks(result, kind),
    error = function(e) {
      warning(sprintf("write_plot_spec: .plot_blocks() failed: %s", conditionMessage(e)))
      NULL
    }
  )
  if (is.null(blocks)) {
    return("(skipped — .plot_blocks() failed; see warning above)")
  }
  block_key <- if (kind == "find_sample_size") "curve" else "power"
  spec <- if (!is.null(blocks[[block_key]])) blocks[[block_key]] else blocks[[1L]]
  spec_json <- mcpower:::.apply_theme(
    toJSON(spec, auto_unbox = TRUE, null = "null"),
    "light-print"
  )
  spec_path <- file.path(assets, sub("\\.png$", ".vl.json", plot_name))
  writeLines(spec_json, spec_path)
  sprintf("(wrote light-print-themed spec %s; run_examples.py renders it to PNG)", basename(spec_path))
}

run_entry <- function(entry) {
  env <- new.env()
  out <- capture.output(eval(parse(text = entry$code), envir = env))
  entry$output <- paste(out, collapse = "\n")
  entry$captured_at <- format(Sys.time(), "%Y-%m-%dT%H:%M:%S%z")
  # plot is a single filename for chart entries; no-plot entries carry `null`,
  # which round-trips through jsonlite to `{}` (empty list) on re-run — guard
  # for a non-empty scalar string so both shapes are treated as "no plot".
  plot_name <- entry$plot
  if (is.character(plot_name) && length(plot_name) == 1L && nzchar(plot_name)) {
    result <- get0("result", envir = env)
    if (is.null(result)) stop(sprintf("entry %s sets 'plot' but left no 'result'", entry$id))
    message("  plot: ", write_plot_spec(result, plot_name))
  }
  entry
}

main <- function() {
  args    <- commandArgs(trailingOnly = TRUE)
  entries <- fromJSON(cache, simplifyVector = FALSE)
  for (i in seq_along(entries)) {
    if (length(args) && !(entries[[i]]$id %in% args)) next
    message("running ", entries[[i]]$id, " ...")
    entries[[i]] <- run_entry(entries[[i]])
  }
  writeLines(toJSON(entries, pretty = TRUE, auto_unbox = TRUE), cache)
  message("wrote ", basename(cache))
}

main()
