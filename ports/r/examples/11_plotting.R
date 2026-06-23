# Plotting MCPower Results
# ========================
#
# Visualise a result: plot() writes a print-themed stacked HTML and opens it
# (no extra packages — the runtime loads from a CDN); save_plot() renders to
# html/svg/png/pdf with an optional theme override. Tours the print default,
# theme override, per-block file sets, stacked HTML, and list_plot_themes().
#
# Run with:  Rscript 11_plotting.R

suppressMessages(library(mcpower))

# Example: Clinical trial — does the new therapy speed recovery?
# Compute a result once, then tour every way to look at it.

# 1. Define the model and effects (treatment binary; see 01_basic_power.R).
model <- MCPower$new("recovery ~ treatment + dose + age")
model$set_effects("treatment=0.5, dose=0.3, age=0.2")
model$set_variable_type("treatment=binary")

# 2. A find_power result. target_test = "all" so the chart carries every effect
#    plus the omnibus row — the same view summary() prints as a table.
result <- model$find_power(sample_size = 120, target_test = "all", verbose = FALSE)

# 3. plot() with no file argument writes a print-themed stacked HTML
#    (find_power.html) and opens it. The HTML needs a CDN connection to load
#    the Vega-Lite runtime but requires NO extra R packages. Headless? It
#    writes the file and prints the path.
cat(">>> plot(result)  # power-at-N: print-themed HTML, opened in browser\n")
plot(result)

# 4. list_plot_themes() shows the available theme names. The default is
#    'print': white background, black axes, light-grey grid, colourblind-safe
#    palette. Pass theme = NULL for a theme-naked spec.
cat(sprintf("\n>>> list_plot_themes()  ->  %s\n",
            paste(list_plot_themes(), collapse = ", ")))

# 5. save_plot() saves to a file; format follows the extension (html/svg/png/pdf).
#    Default theme is 'print'. HTML is the lightest format — no extra packages.
cat("\n>>> save_plot(result, 'power_report.html')  # stacked HTML, all blocks\n")
save_plot(result, "power_report.html")
cat("  saved power_report.html\n")

# 6. Default print theme, and theme = to override it. SVG/PNG/PDF need optional
#    renderers: vegawidget + V8 for SVG; rsvg + system librsvg for PNG/PDF.
tryCatch({
  cat("\n>>> save_plot(result, 'power_default.png')  # print theme by default\n")
  save_plot(result, "power_default.png")
  cat("  saved power_default.png\n")
  cat(">>> save_plot(result, 'power_dark.svg', theme = 'dark')\n")
  save_plot(result, "power_dark.svg", theme = "dark")
  cat("  saved power_dark.svg\n")
}, error = function(e) message(conditionMessage(e)))

# 7. Power curves. A find_sample_size result plots as power-vs-N — one line per
#    effect, with the target-power reference line — the headline planning view.
#    Same plot() / save_plot() API; same print default.
curve <- model$find_sample_size(
  target_test = "all", from_size = 40, to_size = 300, by = 20, verbose = FALSE
)
cat("\n>>> plot(curve)  # power curves: print-themed HTML, opened in browser\n")
plot(curve)

# 8. Multi-target sample-size: save_plot writes ONE FILE PER BLOCK with derived
#    names. For a single-scenario result with 3 targets the blocks are:
#      curve              -> power_curve.png           (the main per-effect curves)
#      at_least_k         -> power_curve_at_least_k.png
#      exactly_k          -> power_curve_exactly_k.png
#    The 'exactly_k' block shows P(exactly k effects significant) vs N, including
#    k=0 — new in this release.
tryCatch({
  cat("\n>>> save_plot(curve, 'power_curve.png')  # per-block files (curve + joint)\n")
  save_plot(curve, "power_curve.png")
  cat("  saved power_curve.png, power_curve_at_least_k.png, power_curve_exactly_k.png\n")
}, error = function(e) message(conditionMessage(e)))

# HTML is one stacked file even for multi-block results.
cat("\n>>> save_plot(curve, 'power_curve.html')  # stacked HTML, all blocks\n")
save_plot(curve, "power_curve.html")
cat("  saved power_curve.html\n")

# 9. Scenarios. scenarios = TRUE runs optimistic / realistic / doomer.
#    On find_power: grouped bars (one group per scenario).
scen <- model$find_power(
  sample_size = 120, target_test = "all", scenarios = TRUE, verbose = FALSE
)
cat("\n>>> plot(scen)  # scenario bars — opens in the browser\n")
plot(scen)

# On find_sample_size: per-scenario blocks + overlay grid.
# save_plot writes one file per block:
#   power_scen_<label>.png  for each scenario
#   power_scen_overlay.png  for the 3-column overlay grid
scen_curve <- model$find_sample_size(
  target_test = "all", from_size = 40, to_size = 300, by = 20,
  scenarios = TRUE, verbose = FALSE
)
cat(">>> plot(scen_curve)  # scenario curves — opens in the browser\n")
plot(scen_curve)
cat(">>> save_plot(scen_curve, 'power_scen.html')  # stacked HTML, all scenario blocks\n")
save_plot(scen_curve, "power_scen.html")
cat("  saved power_scen.html\n")

cat(
  "\nplot() and save_plot() write self-contained HTML charts that open in any browser —",
  "no extra packages needed. For PNG/SVG/PDF export instead, install:",
  "install.packages(c('vegawidget', 'V8', 'rsvg')).\n"
)
