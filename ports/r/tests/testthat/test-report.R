test_that("report config loads format constants from the engine", {
  cfg <- mcpower:::.report_config()
  expect_equal(cfg$format$power_decimals_short, 1)
  expect_equal(cfg$format$power_decimals_long, 1)
  expect_equal(cfg$thresholds$convergence_min, 0.95)
  expect_equal(cfg$overall_label_by_estimator$ols, "Overall F")
})

.r_ols_model <- function() {
  MCPower$new("y ~ x1 + condition")$set_effects("x1=0.5, condition=0.4")$set_simulations(200)
}

test_that("find_power returns a classed result that still carries raw fields", {
  res <- .r_ols_model()$find_power(sample_size = 120, verbose = FALSE, progress_callback = FALSE)
  expect_s3_class(res, "mcpower_result")
  expect_true("power_uncorrected" %in% names(res))
})

test_that("print.mcpower_result short form: headline, no CI, no glyph", {
  res <- .r_ols_model()$find_power(sample_size = 120, verbose = FALSE, progress_callback = FALSE)
  out <- paste(capture.output(print(res)), collapse = "\n")
  expect_match(out, "Power Analysis — OLS")
  expect_match(out, "N=120"); expect_match(out, "target=")
  expect_match(out, "formula:")
  expect_match(out, "Target")
  expect_false(grepl("CI 95%", out))
  expect_false(grepl("✓|✗", out))   # no ✓ / ✗
})

test_that("print.mcpower_result short form scenarios: no Delta column", {
  res <- MCPower$new("y ~ x1")$set_effects("x1=0.5")$set_simulations(200)$
    find_power(sample_size = 100, scenarios = TRUE, verbose = FALSE, progress_callback = FALSE)
  out <- paste(capture.output(print(res)), collapse = "\n")
  expect_match(out, "optimistic"); expect_match(out, "doomer")
  expect_false(grepl("pp", out)); expect_false(grepl("Δ", out))  # no Δ / pp
})

test_that("print.mcpower_result keeps the Overall (omnibus) row across axis states", {
  # Parity with Python main_power_tables: the omnibus row is part of the main
  # table in every state (correction / scenarios / both), not only the plain
  # single-scenario case. A multi-predictor model emits an omnibus test.
  corr <- paste(capture.output(print(
    .r_ols_model()$find_power(sample_size = 120, correction = "holm",
                              verbose = FALSE, progress_callback = FALSE))), collapse = "\n")
  expect_match(corr, "Overall F")
  expect_match(corr, "\\(same\\)")  # omnibus has no multiplicity correction
  scen <- paste(capture.output(print(
    .r_ols_model()$find_power(sample_size = 120, scenarios = TRUE,
                              verbose = FALSE, progress_callback = FALSE))), collapse = "\n")
  expect_match(scen, "Overall F")
})

test_that("verbose = TRUE auto-prints", {
  expect_output(
    .r_ols_model()$find_power(sample_size = 120, verbose = TRUE, progress_callback = FALSE),
    "Power Analysis"
  )
})

test_that("summary() returns mcpower_report and prints the long-form sections", {
  res <- .r_ols_model()$find_power(sample_size = 120, verbose = FALSE, progress_callback = FALSE)
  rep <- summary(res)
  expect_s3_class(rep, "mcpower_report")
  out <- paste(capture.output(print(rep)), collapse = "\n")
  expect_match(out, "Per-test power")
  expect_match(out, "Joint significance distribution")
  # Diagnostics is gated (mirrors Python): a healthy OLS run shows no such section.
  expect_false(grepl("Diagnostics", out))
  # New footer (charter §6 — plot() method, not save_plot)
  expect_match(out, "plot\\(result\\)")
  # No old "needs 'vegawidget'" footer
  expect_false(grepl("needs 'vegawidget'", out))
})

test_that("summary renders when test_formula drops the leading term", {
  # Regression for the test_formula target-index bug (mirrors the Python test
  # test_summary_renders_when_leading_term_dropped): dropping the LEADING
  # generation term (study) and reporting a later one (caffeine) sent
  # .build_rows' skeleton[[idx + 1]] lookup out of range and aborted the render.
  # The kept term's generation-kernel column (2) differed from its reduced
  # test-design term position (1); the engine now reports the reduced position.
  res <- MCPower$new("score ~ study + caffeine")$
    set_effects("study=0.3, caffeine=0")$
    set_correlations("corr(study, caffeine)=0.6")$
    find_power(sample_size = 100, n_sims = 200, seed = 2137,
               target_test = "caffeine", test_formula = "score ~ caffeine",
               verbose = FALSE, progress_callback = FALSE)
  expect_equal(res$n_targets, 1L)
  # Short and long forms must both render without aborting and name the kept term.
  short <- paste(capture.output(print(res)), collapse = "\n")
  long  <- paste(capture.output(print(summary(res))), collapse = "\n")
  expect_match(short, "caffeine")
  expect_match(long, "caffeine")
  expect_false(grepl("out of range", long))
})

test_that("long form: Diagnostics section is gated — shows only on a threshold trip", {
  # Mirrors Python Report._diagnostics: empty on a healthy run; "⚠ Diagnostics"
  # + "! <bare warning>" once a configured threshold trips. Flip convergence
  # below the 0.95 floor on an otherwise-clean single-scenario find_power result.
  make_inner <- function(extra) {
    base <- list(
      scenario = "default", target_indices = 1L, contrast_pairs = list(),
      sample_sizes = 100L, n = 100L, n_sims = 200L,
      power_uncorrected = list(0.9), power_corrected = list(0.9),
      ci_uncorrected = list(list(c(0.85, 0.95))), ci_corrected = list(list(c(0.85, 0.95))),
      convergence_rate = 1.0,
      boundary_hit_rate_tau_zero = 0.0, boundary_hit_rate_high_tau = 0.0,
      estimator_extras = list(estimator = "ols"),
      overall_significant_rate = NULL, overall_significant_ci = NULL,
      success_count_histogram_uncorrected = c(20L, 180L))
    modifyList(base, extra)
  }
  meta <- list(
    effect_names = "x1", effect_sizes = list(0.5), factors = list(),
    estimator = "ols", alpha = 0.05, correction = "none",
    target_power = 80, formula = "y ~ x1",
    effect_skeleton = list(list(kind = "intercept"),
                           list(kind = "continuous", predictor = "x1")))
  out_ok <- paste(capture.output(print(summary(
    mcpower:::.make_result(make_inner(list()), meta, "find_power")))), collapse = "\n")
  expect_false(grepl("Diagnostics", out_ok))
  out_bad <- paste(capture.output(print(summary(
    mcpower:::.make_result(make_inner(list(convergence_rate = 0.5)), meta, "find_power")))), collapse = "\n")
  expect_match(out_bad, "⚠ Diagnostics")
  expect_match(out_bad, "! convergence")
})

# ──long-form boxed header, effects line, CI section, footer ─────────

test_that("long form: boxed header, effects line, CI section, footer", {
  res <- .r_ols_model()$find_power(sample_size = 120, verbose = FALSE, progress_callback = FALSE)
  out <- paste(capture.output(print(summary(res))), collapse = "\n")
  expect_match(out, "=====")
  expect_match(out, "MCPower")
  expect_match(out, "effects:")
  expect_match(out, "Per-test power")
  expect_match(out, "Power & 95% CI")
  expect_match(out, "Monte-Carlo \\(Wilson\\)")
  expect_match(out, "plot\\(result\\)")
})

# T2-B: CI footnote must report the actual n_sims value used in the run.
# Mirrors Python test_report_long.py::test_long_form_ci_section_single_scenario
# ("n_sims=200" appears in the CI footnote when set_simulations(200)).
test_that("long form CI footnote contains n_sims matching the run", {
  # .r_ols_model() calls set_simulations(200), so the footnote must say n_sims=200.
  res <- .r_ols_model()$find_power(sample_size = 120, verbose = FALSE, progress_callback = FALSE)
  out <- paste(capture.output(print(summary(res))), collapse = "\n")
  # Footnote lives after "Monte-Carlo (Wilson)"
  expect_match(out, "Monte-Carlo \\(Wilson\\), n_sims=200")
})

# T2-C: find_sample_size long-form summary must show joint required-N table.
# Mirrors Python test_joint_detection.py::test_long_summary_has_joint_required_n_table.
test_that("sample-size long-form summary shows joint detection lines with correct count", {
  res <- MCPower$new("y ~ a + b + c")$
    set_effects("a=0.5, b=0.3, c=0.1")$
    find_sample_size(from_size = 20, to_size = 400, by = 20,
                     target_power = 80, n_sims = 500,
                     verbose = FALSE, progress_callback = FALSE)
  txt <- paste(utils::capture.output(print(summary(res))), collapse = "\n")
  # "Joint detection" header must appear
  expect_match(txt, "Joint detection")
  # Must reference "3" (the number of tests)
  expect_match(txt, "of 3 tests")
  # The "≥ k of n tests" wording must appear (at least one joint row)
  expect_match(txt, "≥ [0-9]+ of 3 tests")
})

test_that("as_tibble returns a long-format frame", {
  skip_if_not_installed("tibble")
  res <- .r_ols_model()$find_power(sample_size = 120, verbose = FALSE, progress_callback = FALSE)
  tb <- tibble::as_tibble(res)
  expect_true(all(c("test", "scenario", "power", "ci_lo", "ci_hi") %in% names(tb)))
})

# ──CDN-HTML plot() ─────────────────────────────────────────────────

test_that("plot(result) writes a self-contained CDN HTML and returns the path", {
  res <- .r_ols_model()$find_power(sample_size = 120, verbose = FALSE, progress_callback = FALSE)
  tmp <- tempfile(); dir.create(tmp); old <- setwd(tmp); on.exit(setwd(old))
  p <- plot(res)
  expect_true(file.exists(p))
  html <- paste(readLines(p), collapse = "\n")
  expect_match(html, "vega-embed")
  expect_match(html, "vegaEmbed")
})

test_that("plot(result, file) delegates to save_plot suffix handling", {
  res <- .r_ols_model()$find_power(sample_size = 120, verbose = FALSE, progress_callback = FALSE)
  expect_error(plot(res, tempfile(fileext = ".gif")), "svg|png|pdf|html")
})

# ── Multi-scenario short-form print ──────────────────────────────────────────

# Build a synthetic 2-scenario mcpower_result matching the shape .scenarios()
# expects: result$scenarios = named list of inner per-scenario results.
# Each inner mirrors a real single-scenario unwrap (target_indices,
# power_uncorrected per-N list, ci_uncorrected per-N list, sample_sizes, etc.).
.make_synthetic_multi <- function() {
  make_inner <- function(power_val, label) {
    list(
      scenario              = label,
      target_indices        = 1L,
      sample_sizes          = 100L,
      n_sims                = 400L,
      power_uncorrected     = list(power_val),          # [[N_idx]][[pos]]
      power_corrected       = list(power_val),
      ci_uncorrected        = list(list(c(power_val - 0.05, min(power_val + 0.05, 1)))),
      ci_corrected          = list(list(c(power_val - 0.05, min(power_val + 0.05, 1)))),
      convergence_rate      = list(1.0),
      boundary_hit          = integer(0L),
      boundary_hit_rate_tau_zero = 0.0,
      boundary_hit_rate_high_tau = 0.0,
      estimator_extras      = list(estimator = "ols"),
      overall_significant_rate  = NULL,
      overall_significant_ci    = NULL,
      success_count_histogram_uncorrected = integer(0L),
      success_count_histogram_corrected   = integer(0L)
    )
  }

  raw <- list(
    scenarios  = list(optimistic = make_inner(0.90, "optimistic"),
                      pessimistic = make_inner(0.70, "pessimistic")),
    comparison = list()
  )
  meta <- list(
    effect_names = "x1",
    factors      = list(),
    estimator    = "ols",
    alpha        = 0.05,
    correction   = "none",
    target_power = 80,   # stored as 0-100
    formula      = "y ~ x1"
  )
  mcpower:::.make_result(raw, meta, "find_power")
}

test_that("print.mcpower_result renders multi-scenario columns, no Delta (synthetic)", {
  res_multi <- .make_synthetic_multi()
  out <- paste(capture.output(print(res_multi)), collapse = "\n")

  expect_match(out, "Power Analysis")
  # Both scenario labels appear in the header line
  expect_match(out, "optimistic")
  expect_match(out, "pessimistic")
  # No Δ / pp column (moved to long-form Robustness)
  expect_false(grepl("pp", out))
  expect_false(grepl("Δ", out))
  # The Target column appears
  expect_match(out, "Target")
  # The x1 predictor row appears
  expect_match(out, "x1")
})

test_that("print.mcpower_result renders multi-scenario from a real find_power call", {
  res <- MCPower$new("y ~ x1")$
    set_effects("x1=0.5")$
    set_simulations(200)$
    find_power(sample_size = 100, scenarios = TRUE,
               verbose = FALSE, progress_callback = FALSE)
  out <- paste(capture.output(print(res)), collapse = "\n")
  expect_match(out, "Power Analysis")
  # All three default scenario names appear
  expect_match(out, "optimistic")
  expect_match(out, "realistic")
  expect_match(out, "doomer")
  # No Δ / pp column (moved to long-form Robustness)
  expect_false(grepl("pp", out))
})

# ──sample-size short form print ────────────────────────────────────

test_that("sample-size short form: required N, config labels, no At-target col", {
  res <- MCPower$new("y ~ x1 + x2")$set_effects("x1=0.5, x2=0.3")$
    find_sample_size(from_size = 40, to_size = 200, by = 40, verbose = FALSE, progress_callback = FALSE)
  out <- paste(capture.output(print(res)), collapse = "\n")
  expect_match(out, "Power Analysis \\(sample size\\) — OLS")
  expect_match(out, "Required N")
  expect_match(out, "First N achieving all targets")
})

# ── find_sample_size long-form summary (report) ───────────────────────────────

test_that("sample-size long-form summary shows joint required-N table and drops stale message", {
  res <- MCPower$new("y ~ a + b + c")$
    set_effects("a=0.5, b=0.3, c=0.1")$
    find_sample_size(from_size = 20, to_size = 400, by = 20,
                     target_power = 80, n_sims = 500,
                     verbose = FALSE, progress_callback = FALSE)
  txt <- paste(utils::capture.output(print(summary(res))), collapse = "\n")
  expect_true(grepl("Joint detection", txt))
  expect_true(grepl("of 3 tests", txt))
  expect_false(grepl("analytical fast-path", txt))
})

# Mirrors Python test_report_long.py::test_sample_size_overall_required_n_row_renders.
test_that("sample-size long-form summary shows the overall (omnibus) required-N row", {
  res <- MCPower$new("y ~ a + b")$
    set_effects("a=0.5, b=0.3")$
    find_sample_size(from_size = 20, to_size = 200, by = 20,
                     target_power = 80, n_sims = 200,
                     verbose = FALSE, progress_callback = FALSE)
  txt <- paste(utils::capture.output(print(summary(res))), collapse = "\n")
  # OLS ⇒ the omnibus required-N row is labelled "Overall F".
  expect_match(txt, "Overall F")
})

# Mirrors Python test_report_short.py::test_sample_size_short_form_shows_overall_row.
test_that("sample-size short form (print) shows the overall (omnibus) required-N row", {
  res <- MCPower$new("y ~ a + b")$
    set_effects("a=0.5, b=0.3")$
    find_sample_size(from_size = 20, to_size = 200, by = 20,
                     target_power = 80, n_sims = 200,
                     verbose = FALSE, progress_callback = FALSE)
  out <- paste(utils::capture.output(print(res)), collapse = "\n")
  expect_match(out, "Overall F")
})

test_that("sample-size header does not report the lowest grid N", {
  res <- MCPower$new("y ~ a + b + c")$
    set_effects("a=0.5, b=0.3, c=0.1")$
    find_sample_size(from_size = 20, to_size = 400, by = 20,
                     target_power = 80, n_sims = 500,
                     verbose = FALSE, progress_callback = FALSE)
  txt <- paste(utils::capture.output(print(summary(res))), collapse = "\n")
  expect_false(grepl("N=20 ", txt))
})

# ── Plot-set block architecture tests ─────────────────────────────────────────

# Helper: get all mark types from a spec recursively
.collect_marks <- function(node) {
  if (!is.list(node)) return(character(0))
  marks <- character(0)
  mark <- node$mark
  mark_type <- if (is.list(mark)) mark$type %||% "" else if (is.character(mark)) mark else ""
  if (nzchar(mark_type)) marks <- c(marks, mark_type)
  for (child in node) marks <- c(marks, .collect_marks(child))
  marks
}

test_that("list_plot_themes returns the four embedded names", {
  themes <- list_plot_themes()
  expect_true(is.character(themes))
  expect_setequal(themes, c("light", "dark", "print", "wild"))
})

test_that(".plot_blocks find_power returns exactly one 'power' block with CI marks", {
  res <- .r_ols_model()$find_power(sample_size = 120, verbose = FALSE, progress_callback = FALSE)
  blocks <- mcpower:::.plot_blocks(res, "find_power")
  expect_equal(names(blocks), "power")
  # The power block must contain errorbar marks (CI styling applied)
  marks <- .collect_marks(blocks[["power"]])
  expect_true("errorbar" %in% marks)
})

test_that(".plot_blocks find_power carries the dashed target-power rule", {
  res <- .r_ols_model()$find_power(sample_size = 120, verbose = FALSE, progress_callback = FALSE)
  blocks <- mcpower:::.plot_blocks(res, "find_power")
  spec_json <- jsonlite::toJSON(blocks[["power"]], auto_unbox = TRUE, null = "null")
  # Target-power line: engine emits a 'rule' mark in the power block
  marks <- .collect_marks(blocks[["power"]])
  expect_true("rule" %in% marks)
})

test_that(".plot_blocks find_sample_size S=1 m=1 returns 'curve' block", {
  res <- .r_ols_model()$find_sample_size(
    from_size = 40, to_size = 200, by = 40, progress_callback = FALSE
  )
  blocks <- mcpower:::.plot_blocks(res, "find_sample_size")
  expect_true("curve" %in% names(blocks))
})

test_that(".plot_blocks find_sample_size S=1 m>=2 returns curve + at_least_k + exactly_k", {
  res <- MCPower$new("y ~ a + b")$
    set_effects("a=0.5, b=0.3")$
    find_sample_size(from_size = 20, to_size = 200, by = 20,
                     target_power = 80, n_sims = 300,
                     verbose = FALSE, progress_callback = FALSE)
  blocks <- mcpower:::.plot_blocks(res, "find_sample_size")
  expect_true("curve" %in% names(blocks))
  expect_true("at_least_k" %in% names(blocks))
  expect_true("exactly_k" %in% names(blocks))
})

test_that("save_plot HTML: correct file set for S=1 m=1 (one file, no suffix)", {
  res <- .r_ols_model()$find_power(sample_size = 120, verbose = FALSE, progress_callback = FALSE)
  f <- tempfile(fileext = ".html")
  save_plot(res, f)
  expect_true(file.exists(f))
  # No extra files like _power.html
  expect_equal(length(list.files(dirname(f), basename(f))), 1L)
})

test_that("save_plot HTML: S=1 m=2 writes ONE stacked file with 3 vegaEmbed calls", {
  # HTML format always writes a single stacked file with all blocks.
  # For S=1 m=2 find_sample_size: curve + at_least_k + exactly_k all embedded in one HTML.
  res <- MCPower$new("y ~ a + b")$
    set_effects("a=0.5, b=0.3")$
    find_sample_size(from_size = 20, to_size = 200, by = 20,
                     target_power = 80, n_sims = 300,
                     verbose = FALSE, progress_callback = FALSE)
  f <- tempfile(fileext = ".html")
  save_plot(res, f)
  expect_true(file.exists(f))
  # Single stacked file: no per-block suffixes created
  stem <- tools::file_path_sans_ext(f)
  expect_false(file.exists(paste0(stem, "_at_least_k.html")))
  expect_false(file.exists(paste0(stem, "_exactly_k.html")))
  # Verify 3 blocks are in the HTML: the template loops with forEach,
  # so the JSON array must have 3 elements (curve + at_least_k + exactly_k)
  html <- paste(readLines(f), collapse = "\n")
  # Extract the JSON array from the script
  blocks <- mcpower:::.plot_blocks(res, "find_sample_size")
  expect_equal(length(blocks), 3L)
})

test_that("save_plot HTML: multi-scenario writes per-scenario + _overlay, no bare base", {
  res <- MCPower$new("y ~ x1")$
    set_effects("x1=0.5")$
    set_simulations(200)$
    find_power(sample_size = 100, scenarios = TRUE,
               verbose = FALSE, progress_callback = FALSE)
  stem <- tempfile()
  f_base <- paste0(stem, ".html")
  save_plot(res, f_base)
  # Multi-scenario find_power: 'power' block -> base path; no 'scenario:' blocks for find_power
  # (find_power always returns one 'power' block regardless of scenarios)
  expect_true(file.exists(f_base))
})

test_that("save_plot HTML default theme: spec config has background=#ffffff and legend", {
  res <- .r_ols_model()$find_power(sample_size = 120, verbose = FALSE, progress_callback = FALSE)
  f <- tempfile(fileext = ".html")
  save_plot(res, f)
  html <- paste(readLines(f), collapse = "\n")
  # Print theme has background #ffffff
  expect_match(html, "#ffffff")
  # Print theme has a legend block
  expect_match(html, '"legend"')
})

test_that("save_plot HTML theme='dark': spec has dark background", {
  res <- .r_ols_model()$find_power(sample_size = 120, verbose = FALSE, progress_callback = FALSE)
  f <- tempfile(fileext = ".html")
  save_plot(res, f, theme = "dark")
  html <- paste(readLines(f), collapse = "\n")
  expect_match(html, "#1e1e1e")
})

test_that("save_plot HTML theme=NULL: no config block added (theme-naked)", {
  res <- .r_ols_model()$find_power(sample_size = 120, verbose = FALSE, progress_callback = FALSE)
  f <- tempfile(fileext = ".html")
  save_plot(res, f, theme = NULL)
  html <- paste(readLines(f), collapse = "\n")
  # Theme-naked: no background from print theme
  expect_false(grepl("#ffffff", html))
})

test_that("save_plot HTML correction: Power axis title gains correction note", {
  res <- .r_ols_model()$find_power(sample_size = 120, correction = "holm",
                                    verbose = FALSE, progress_callback = FALSE)
  f <- tempfile(fileext = ".html")
  save_plot(res, f)
  html <- paste(readLines(f), collapse = "\n")
  expect_match(html, "Holm-corrected")
})

test_that("save_plot HTML: contains CDN script tags, vegaEmbed, {{SPECS}} substituted", {
  res <- .r_ols_model()$find_power(sample_size = 120, verbose = FALSE, progress_callback = FALSE)
  f <- tempfile(fileext = ".html")
  save_plot(res, f)
  html <- paste(readLines(f), collapse = "\n")
  expect_match(html, "cdn.jsdelivr.net/npm/vega@5")
  expect_match(html, "vegaEmbed")
  # {{SPECS}} placeholder must be fully substituted
  expect_false(grepl("\\{\\{SPECS\\}\\}", html))
  # The specs JSON must be parseable (basic sanity check)
  expect_match(html, '"\\$schema"')
})

# ── save_plot ─────────────────────────────────────────────────────────────────

test_that("save_plot errors on an unknown suffix", {
  res <- .r_ols_model()$find_power(sample_size = 120, verbose = FALSE, progress_callback = FALSE)
  expect_error(save_plot(res, tempfile(fileext = ".gif")), "svg|png|pdf|html")
})

test_that("save_plot writes an SVG file", {
  skip_if_not_installed("vegawidget")
  skip_if_not_installed("V8")
  res <- .r_ols_model()$find_power(sample_size = 120, verbose = FALSE, progress_callback = FALSE)
  f <- tempfile(fileext = ".svg")
  save_plot(res, f)
  expect_true(file.exists(f) && file.info(f)$size > 0)
})

test_that("save_plot writes a PNG file", {
  skip_if_not_installed("rsvg")
  skip_if_not_installed("vegawidget")
  skip_if_not_installed("V8")
  res <- .r_ols_model()$find_power(sample_size = 120, verbose = FALSE, progress_callback = FALSE)
  f <- tempfile(fileext = ".png")
  save_plot(res, f)
  expect_true(file.exists(f) && file.info(f)$size > 0)
})

test_that("save_plot PNG scale=2 is twice the pixel width of scale=1", {
  skip_if_not_installed("rsvg")
  skip_if_not_installed("vegawidget")
  skip_if_not_installed("V8")
  skip_if_not_installed("png")
  res <- .r_ols_model()$find_power(sample_size = 120, verbose = FALSE, progress_callback = FALSE)
  f1 <- tempfile(fileext = ".png")
  f2 <- tempfile(fileext = ".png")
  save_plot(res, f1, scale = 1)
  save_plot(res, f2, scale = 2)
  img1 <- png::readPNG(f1)
  img2 <- png::readPNG(f2)
  # scale=2 should yield approximately twice the width of scale=1
  w1 <- dim(img1)[2]
  w2 <- dim(img2)[2]
  expect_true(w2 > w1 * 1.5 && w2 < w1 * 2.5,
              info = sprintf("scale=1 width=%d, scale=2 width=%d", w1, w2))
})

# ──.build_report_meta carries effect_sizes and residual ─────────────

test_that(".build_report_meta carries effect_sizes and residual", {
  m <- MCPower$new("y ~ x1 + x2")$set_effects("x1=0.5, x2=0.3")
  meta <- m$.__enclos_env__$private$.build_report_meta(NULL)
  expect_true("effect_sizes" %in% names(meta))
  expect_true("residual" %in% names(meta))
  expect_equal(meta$residual, "normal")
  expect_equal(unname(meta$effect_sizes[[1]]), 0.5)
})

# ── .required_n_headline chain unit tests ────────────────────────────────────
# Five cases: fitted, at_or_below_min, not_reached, non_monotone fallback,
# missing fitted fallback. Mirrors Python tables.py:_required_n_headline.

# Helper: build a minimal inner list with a single-target fitted map.
.make_inner_with_fitted <- function(fitted_entry, first_achieved_val = NULL, sample_sizes = 40:200) {
  list(
    fitted          = list(`0` = fitted_entry),
    first_achieved  = list(`0` = first_achieved_val),
    sample_sizes    = sample_sizes
  )
}

test_that(".required_n_headline: fitted -> n_achievable as display + numeric", {
  inner <- .make_inner_with_fitted(
    list(status = "fitted", n_star = 123.4, n_achievable = 124L, ci_lo = 118.7, ci_hi = 130.2))
  h <- mcpower:::.required_n_headline(inner, pos = 1L)
  expect_equal(h$display, "124")
  expect_equal(h$numeric, 124L)
})

test_that(".required_n_headline: at_or_below_min -> ≤ n_min + numeric", {
  inner <- .make_inner_with_fitted(list(status = "at_or_below_min", n_min = 40L))
  h <- mcpower:::.required_n_headline(inner, pos = 1L)
  expect_equal(h$display, "≤ 40")
  expect_equal(h$numeric, 40L)
})

test_that(".required_n_headline: not_reached -> ≥ ceiling + NULL numeric", {
  inner <- .make_inner_with_fitted(list(status = "not_reached", n_approx = NA_integer_))
  h <- mcpower:::.required_n_headline(inner, pos = 1L)
  expect_equal(h$display, sprintf("≥ %d", max(inner$sample_sizes)))
  expect_null(h$numeric)
})

test_that(".required_n_headline: non_monotone falls back to first_achieved", {
  inner <- .make_inner_with_fitted(
    list(status = "non_monotone", max_violation = 0.05), first_achieved_val = 80L)
  h <- mcpower:::.required_n_headline(inner, pos = 1L)
  expect_equal(h$display, "80")
  expect_equal(h$numeric, 80L)
})

test_that(".required_n_headline: missing fitted falls back to first_achieved", {
  inner <- list(
    fitted         = list(),       # empty — no key "0"
    first_achieved = list(`0` = 95L),
    sample_sizes   = 40:200
  )
  h <- mcpower:::.required_n_headline(inner, pos = 1L)
  expect_equal(h$display, "95")
  expect_equal(h$numeric, 95L)
})

# ── short form: non_monotone warning line appears ─────────────────────────────
# A model in a pathological range should exercise the non_monotone code path
# when it fires. We hand-build a fake result that has non_monotone status to
# unit-test the warning emission without needing a real non-monotone run.

test_that("short form: non_monotone warning line is printed", {
  # Build a minimal mcpower_sample_size_result with one non_monotone fitted entry.
  make_ss_inner <- function(label) {
    list(
      scenario              = label,
      n_sims                = 200L,
      n_sample_sizes        = 2L,
      n_targets             = 1L,
      target_indices        = 1L,
      target_power          = 0.8,
      power_uncorrected     = list(list(0.5), list(0.7)),
      power_corrected       = list(list(0.5), list(0.7)),
      ci_uncorrected        = list(list(c(0.4, 0.6)), list(c(0.6, 0.8))),
      ci_corrected          = list(list(c(0.4, 0.6)), list(c(0.6, 0.8))),
      convergence_rate      = list(1.0, 1.0),
      sample_sizes          = c(40L, 80L),
      first_achieved        = list(`0` = 80L),
      first_joint_achieved  = list(`0` = 80L),
      success_count_histogram_uncorrected = c(100L, 100L),
      success_count_histogram_corrected   = c(100L, 100L),
      boundary_hit          = integer(0L),
      boundary_hit_rate_tau_zero = 0.0,
      boundary_hit_rate_high_tau = 0.0,
      grid_warnings         = character(0L),
      estimator_extras      = list(estimator = "ols"),
      # fitted: non_monotone so the warning line fires
      fitted                = list(`0` = list(status = "non_monotone", max_violation = 0.07)),
      fitted_joint          = list(`0` = list(status = "fitted", n_star = 78.0, n_achievable = 78L,
                                              ci_lo = 70.0, ci_hi = 86.0)),
      cluster_atom          = 1L
    )
  }
  # skeleton: intercept at index 0, x1 at index 1 (engine 0-based convention)
  # target_indices = 1L means "target the x1 predictor (β-column 1)"
  meta <- list(
    effect_names   = "x1",
    effect_sizes   = list(0.5),
    factors        = list(),
    estimator      = "ols",
    alpha          = 0.05,
    correction     = "none",
    target_power   = 80,
    formula        = "y ~ x1",
    effect_skeleton = list(
      list(kind = "intercept"),                        # index 0
      list(kind = "continuous", predictor = "x1")     # index 1
    )
  )
  inner_raw <- make_ss_inner("default")
  res <- mcpower:::.make_result(inner_raw, meta, "find_sample_size")
  out <- paste(capture.output(print(res)), collapse = "\n")
  # The config template has {label} and {drop} placeholders.
  # The warning should mention the label "x1" and the drop "0.070".
  expect_match(out, "x1")
  expect_match(out, "0.070")
  # The non_monotone_warning config key must fire (contains "not monotone")
  expect_match(out, "not monotone")
})

# ── long summary: "Required N & 95% CI" section appears ──────────────────────

test_that("long summary: Required N & CI section appears for find_sample_size", {
  res <- MCPower$new("y ~ x1 + x2")$set_effects("x1=0.5, x2=0.3")$
    find_sample_size(from_size = 40, to_size = 200, by = 20, n_sims = 500,
                     verbose = FALSE, progress_callback = FALSE)
  txt <- paste(capture.output(print(summary(res))), collapse = "\n")
  expect_match(txt, "Required N & 95% CI")
  # Footnote: isotonic / Wilson text from config
  expect_match(txt, "isotonic")
  expect_match(txt, "Wilson")
  # CI brackets should appear in format [lo, hi]
  expect_match(txt, "\\[\\d+, \\d+\\]")
})

test_that("long summary: joint block uses fitted_joint chain (real run has entries)", {
  res <- MCPower$new("y ~ a + b")$set_effects("a=0.5, b=0.3")$
    find_sample_size(from_size = 20, to_size = 200, by = 20, n_sims = 500,
                     verbose = FALSE, progress_callback = FALSE)
  # fitted_joint must be non-empty (engine sends them for all joint targets)
  inner <- res
  expect_true(length(inner$fitted_joint) > 0L)
  expect_true(all(vapply(inner$fitted_joint, function(fj) fj$status %in%
    c("fitted", "at_or_below_min", "not_reached", "non_monotone"), logical(1L))))
  # The Joint detection block must still render in the long form
  txt <- paste(capture.output(print(summary(res))), collapse = "\n")
  expect_match(txt, "Joint detection")
})

# ──as.data.frame + as_latex / as_pdf stubs ─────────────────────────

test_that("as.data.frame for find_sample_size gives test x scenario x required_n + ci cols", {
  res <- MCPower$new("y ~ x1 + x2")$set_effects("x1=0.5, x2=0.3")$
    find_sample_size(from_size = 40, to_size = 200, by = 40, verbose = FALSE, progress_callback = FALSE)
  df <- as.data.frame(res)
  # Must have test / scenario / required_n (pre-existing) plus ci_lo / ci_hi (new)
  expect_true(all(c("test", "scenario", "required_n", "ci_lo", "ci_hi") %in% names(df)))
  # For fitted rows: required_n must be an integer, ci_lo <= required_n <= ci_hi
  fitted_rows <- df[!is.na(df$required_n), ]
  if (nrow(fitted_rows) > 0L) {
    expect_true(all(fitted_rows$ci_lo <= fitted_rows$required_n))
    expect_true(all(fitted_rows$required_n <= fitted_rows$ci_hi))
  }
})

test_that("as.data.frame: NA conventions for at_or_below_min and not_reached", {
  # at_or_below_min: required_n is NA (no single integer to export)
  inner_atmin <- list(
    status = "at_or_below_min", n_min = 40L)
  # Verify the dispatch path in as.data.frame via a hand-built result.
  # Build a single-target SSR where fitted has at_or_below_min.
  make_inner <- function(fitted_entry) {
    list(
      scenario = "default", n_sims = 200L, n_sample_sizes = 2L, n_targets = 1L,
      target_indices = 1L, target_power = 0.8,
      power_uncorrected = list(list(0.9), list(0.9)), power_corrected = list(list(0.9), list(0.9)),
      ci_uncorrected = list(list(c(0.8, 1.0)), list(c(0.8, 1.0))),
      ci_corrected = list(list(c(0.8, 1.0)), list(c(0.8, 1.0))),
      convergence_rate = list(1.0, 1.0),
      sample_sizes = c(40L, 80L),
      first_achieved = list(`0` = 40L),
      first_joint_achieved = list(`0` = 40L),
      success_count_histogram_uncorrected = c(10L, 190L),
      success_count_histogram_corrected   = c(10L, 190L),
      boundary_hit = integer(0L), boundary_hit_rate_tau_zero = 0.0,
      boundary_hit_rate_high_tau = 0.0, grid_warnings = character(0L),
      estimator_extras = list(estimator = "ols"),
      fitted = list(`0` = fitted_entry), fitted_joint = list(),
      cluster_atom = 1L)
  }
  # skeleton: intercept at index 0, x1 at index 1 (engine 0-based convention)
  meta <- list(
    effect_names = "x1", effect_sizes = list(0.5), factors = list(),
    estimator = "ols", alpha = 0.05, correction = "none",
    target_power = 80, formula = "y ~ x1",
    effect_skeleton = list(
      list(kind = "intercept"),
      list(kind = "continuous", predictor = "x1")))

  # at_or_below_min: required_n NA
  res_atmin <- mcpower:::.make_result(make_inner(list(status = "at_or_below_min", n_min = 40L)),
                                      meta, "find_sample_size")
  df_atmin <- as.data.frame(res_atmin)
  expect_true(is.na(df_atmin$required_n[1L]))
  expect_true(is.na(df_atmin$ci_lo[1L]))

  # fitted with both bounds: required_n == n_achievable; ci_lo/ci_hi present
  res_fit <- mcpower:::.make_result(
    make_inner(list(status = "fitted", n_star = 42.5, n_achievable = 43L, ci_lo = 38.1, ci_hi = 48.7)),
    meta, "find_sample_size")
  df_fit <- as.data.frame(res_fit)
  expect_equal(df_fit$required_n[1L], 43L)
  expect_equal(df_fit$ci_lo[1L], 38L)    # floor(38.1)
  expect_equal(df_fit$ci_hi[1L], 49L)    # ceiling(48.7)
})

test_that("as_latex / as_pdf are informative stubs", {
  res <- .r_ols_model()$find_power(sample_size = 120, verbose = FALSE, progress_callback = FALSE)
  expect_error(as_latex(res), "roadmap")
  expect_error(as_pdf(res, tempfile(fileext = ".pdf")), "roadmap")
})

# ── save_plot file-set: multi-scenario find_sample_size (SVG path derivation) ──
# Uses .unique_block_paths directly — no vegawidget/V8 needed — to verify that
# S>=2, m>=2 find_sample_size produces per-scenario + overlay + joint files and
# NO bare base path.

test_that("save_plot SVG file-set: multi-scenario find_sample_size S>=2 m>=2 produces correct paths", {
  res <- MCPower$new("y ~ a + b")$
    set_effects("a=0.5, b=0.3")$
    find_sample_size(from_size = 20, to_size = 200, by = 20,
                     target_power = 80, n_sims = 300,
                     scenarios = TRUE,
                     verbose = FALSE, progress_callback = FALSE)

  base_path <- file.path(tempdir(), "out.svg")
  blocks <- mcpower:::.plot_blocks(res, "find_sample_size")
  pairs  <- mcpower:::.unique_block_paths(base_path, blocks)

  derived_paths <- vapply(pairs, `[[`, character(1L), "path")
  stem <- tools::file_path_sans_ext(base_path)   # "<tmp>/out"

  # Must have >= 2 scenario:<label> blocks, one overlay, one at_least_k, one exactly_k
  block_keys <- vapply(pairs, `[[`, character(1L), "key")
  expect_true(sum(startsWith(block_keys, "scenario:")) >= 2L)
  expect_true("overlay"    %in% block_keys)
  expect_true("at_least_k" %in% block_keys)
  expect_true("exactly_k"  %in% block_keys)

  # Per-scenario files use sanitized label suffix (not bare base)
  scenario_paths <- derived_paths[startsWith(block_keys, "scenario:")]
  for (p in scenario_paths) {
    expect_true(startsWith(p, paste0(stem, "_")),
                info = sprintf("scenario path '%s' should start with '<stem>_'", p))
    expect_true(endsWith(p, ".svg"))
  }

  # Shared-block files use _overlay / _at_least_k / _exactly_k suffix
  expect_equal(derived_paths[block_keys == "overlay"],
               paste0(stem, "_overlay.svg"))
  expect_equal(derived_paths[block_keys == "at_least_k"],
               paste0(stem, "_at_least_k.svg"))
  expect_equal(derived_paths[block_keys == "exactly_k"],
               paste0(stem, "_exactly_k.svg"))

  # No bare out.svg (no power/curve block in multi-scenario find_sample_size)
  expect_false(base_path %in% derived_paths)
})

# ── .build_label_map / .relabel_walk: engine target_{idx} → real effect names ──

test_that(".relabel_walk replaces all target_ tokens with real effect names", {
  # Real find_power call: .build_label_map is populated from the registry skeleton,
  # .relabel_walk rewrites 0-based engine tokens (target_0, target_1, …) to the
  # host-side effect names before the spec reaches the caller.
  res <- MCPower$new("y ~ x1 + condition")$
    set_effects("x1=0.5, condition=0.4")$
    set_simulations(200)$
    find_power(sample_size = 120, verbose = FALSE, progress_callback = FALSE)
  blocks <- mcpower:::.plot_blocks(res, "find_power")
  spec_json <- jsonlite::toJSON(blocks[["power"]], auto_unbox = TRUE, null = "null")

  # (a) No residual target_ tokens anywhere in the spec JSON
  expect_false(grepl("target_", spec_json, fixed = TRUE),
               label = "no target_ token should survive relabeling")

  # (b) At least one real effect name appears as a data value in the spec
  expect_true(grepl('"x1"', spec_json, fixed = TRUE),
              label = "real effect name 'x1' must appear in spec data")
})

# ── .write_stacked_html: </script> script-breakout escape ───────────────────

test_that(".write_stacked_html escapes </script> in spec data", {
  # Scenario labels flow into spec data values (the 'scenario' column in the
  # data rows of a multi-scenario find_power spec).  Use the synthetic helper
  # pattern from .make_synthetic_multi above to inject a label containing
  # </script>, then verify the HTML file escapes it.
  make_inner <- function(power_val, label) {
    list(
      scenario              = label,
      target_indices        = 1L,
      sample_sizes          = 100L,
      n_sims                = 400L,
      power_uncorrected     = list(power_val),
      power_corrected       = list(power_val),
      ci_uncorrected        = list(list(c(power_val - 0.05, min(power_val + 0.05, 1)))),
      ci_corrected          = list(list(c(power_val - 0.05, min(power_val + 0.05, 1)))),
      convergence_rate      = list(1.0),
      boundary_hit          = integer(0L),
      boundary_hit_rate_tau_zero = 0.0,
      boundary_hit_rate_high_tau = 0.0,
      estimator_extras      = list(estimator = "ols"),
      overall_significant_rate  = NULL,
      overall_significant_ci    = NULL,
      success_count_histogram_uncorrected = integer(0L),
      success_count_histogram_corrected   = integer(0L)
    )
  }
  raw <- list(
    scenarios  = list(normal = make_inner(0.80, "normal"),
                      `</script>xss` = make_inner(0.75, "</script>xss")),
    comparison = list()
  )
  meta <- list(
    effect_names = "x1",
    factors      = list(),
    estimator    = "ols",
    alpha        = 0.05,
    correction   = "none",
    target_power = 80,
    formula      = "y ~ x1"
  )
  res <- mcpower:::.make_result(raw, meta, "find_power")

  # The scenario label "</script>xss" flows into spec data and then into the
  # HTML JSON payload.  .write_stacked_html must escape </ as <\/ so the
  # browser never sees a closing </script> tag inside the JSON block.
  f <- tempfile(fileext = ".html")
  save_plot(res, f, theme = NULL)
  html_text <- paste(readLines(f, warn = FALSE), collapse = "\n")

  # The escaped form <\/script> must appear (the payload's </script> is escaped).
  expect_true(grepl("<\\/script>", html_text, fixed = TRUE),
              label = "escaped form <\\/script> must appear in HTML")
  # Count unescaped </script>: must equal the template's own tag count (4),
  # meaning the injected "</script>xss" scenario label added no extra raw tag.
  template_count <- lengths(regmatches(mcpower:::plot_html_template(),
                                       gregexpr("</script>",
                                                mcpower:::plot_html_template(),
                                                fixed = TRUE)))
  raw_count <- lengths(regmatches(html_text,
                                  gregexpr("</script>", html_text, fixed = TRUE)))
  expect_equal(raw_count, template_count,
               label = "no extra unescaped </script> beyond the template's own tags")
})

# ── .diagnostic_warnings: factor exclusion / separation checks ───────────────

test_that(".diagnostic_warnings emits excluded warning when rate exceeds threshold", {
  # Fixture: 2 factors, first excluded 200/200 sims (100%), second clean.
  inner <- list(
    n_sims                     = 200L,
    convergence_rate           = 1.0,
    boundary_hit               = integer(0L),
    boundary_hit_rate_tau_zero = 0.0,
    boundary_hit_rate_high_tau = 0.0,
    factor_exclusion_counts    = c(200L, 0L),
    factor_separation_counts   = c(0L, 0L)
  )
  warns <- mcpower:::.diagnostic_warnings(inner)
  # threshold is 0.0, so 100% rate > threshold → warning expected
  expect_true(any(grepl("excluded 100.0%", warns)),
              info = paste("warns:", paste(warns, collapse = " | ")))
})

test_that(".diagnostic_warnings is silent when all factor counts are zero", {
  inner <- list(
    n_sims                     = 200L,
    convergence_rate           = 1.0,
    boundary_hit               = integer(0L),
    boundary_hit_rate_tau_zero = 0.0,
    boundary_hit_rate_high_tau = 0.0,
    factor_exclusion_counts    = c(0L, 0L),
    factor_separation_counts   = c(0L, 0L)
  )
  warns <- mcpower:::.diagnostic_warnings(inner)
  excl_warns <- warns[grepl("excluded|separation-dropped", warns)]
  expect_length(excl_warns, 0L)
})

test_that(".diagnostic_warnings uses factor_names when provided, falls back to 'factor N'", {
  inner <- list(
    n_sims                     = 200L,
    convergence_rate           = 1.0,
    boundary_hit               = integer(0L),
    boundary_hit_rate_tau_zero = 0.0,
    boundary_hit_rate_high_tau = 0.0,
    factor_exclusion_counts    = c(200L, 0L),
    factor_separation_counts   = c(0L, 0L)
  )
  # With factor_names: uses the name
  warns_named <- mcpower:::.diagnostic_warnings(inner, factor_names = c("dose", "site"))
  expect_true(any(grepl("^dose excluded", warns_named)),
              info = paste("named warns:", paste(warns_named, collapse = " | ")))

  # Without factor_names: falls back to "factor 1"
  warns_fallback <- mcpower:::.diagnostic_warnings(inner)
  expect_true(any(grepl("^factor 1 excluded", warns_fallback)),
              info = paste("fallback warns:", paste(warns_fallback, collapse = " | ")))
})

test_that(".diagnostic_warnings handles per-grid-point counts (sample_size shape)", {
  # Sample-size shape: factor_exclusion_counts is a list of vectors (one per N-point).
  # Max over N-points is used; factor 0 is excluded in first slot only.
  inner <- list(
    n_sims                     = 200L,
    convergence_rate           = c(1.0, 1.0),
    boundary_hit               = integer(0L),
    boundary_hit_rate_tau_zero = 0.0,
    boundary_hit_rate_high_tau = 0.0,
    # First N-point: factor 0 fully excluded; second N-point: all clear
    factor_exclusion_counts    = list(c(200L, 0L), c(0L, 0L)),
    factor_separation_counts   = list(c(0L, 0L), c(0L, 0L))
  )
  warns <- mcpower:::.diagnostic_warnings(inner, factor_names = c("grp", "site"))
  expect_true(any(grepl("grp excluded 100.0%", warns)),
              info = paste("warns:", paste(warns, collapse = " | ")))
})

# ── .diagnostic_warnings: boundary gate → high-τ̂ only (mirrors Python) ────────

test_that(".diagnostic_warnings trips on high-τ̂ boundary over threshold", {
  inner <- list(
    convergence_rate           = list(1.0),
    boundary_hit_rate_tau_zero = 0.0,
    boundary_hit_rate_high_tau = 0.05,  # 5% > 1% threshold
    estimator_extras           = list(estimator = "ols")
  )
  warns <- mcpower:::.diagnostic_warnings(inner)
  expect_true(any(grepl("high-τ̂ boundary", warns) & grepl("5.0%", warns)),
              info = paste("warns:", paste(warns, collapse = " | ")))
})

test_that(".diagnostic_warnings is silent on benign τ̂=0 alone (no high-τ̂)", {
  inner <- list(
    convergence_rate           = list(1.0),
    boundary_hit_rate_tau_zero = 0.40,  # 40% singular fits — benign
    boundary_hit_rate_high_tau = 0.0,
    estimator_extras           = list(estimator = "ols")
  )
  warns <- mcpower:::.diagnostic_warnings(inner)
  expect_length(warns[grepl("boundary", warns)], 0L)
})

# ── .diagnostic_warnings: GLM baseline drift → live (mirrors Python) ──────────

.glm_inner <- function(realized) list(
  convergence_rate           = list(1.0),
  boundary_hit_rate_tau_zero = 0.0,
  boundary_hit_rate_high_tau = 0.0,
  estimator_extras           = list(estimator = "glm", baseline_prob_realized = realized)
)

test_that(".diagnostic_warnings trips on GLM baseline drift over threshold", {
  warns <- mcpower:::.diagnostic_warnings(.glm_inner(0.50), baseline_prob_requested = 0.30)
  expect_true(any(grepl("GLM baseline drift 0.200", warns)),
              info = paste("warns:", paste(warns, collapse = " | ")))
})

test_that(".diagnostic_warnings is clean when GLM drift within threshold", {
  warns <- mcpower:::.diagnostic_warnings(.glm_inner(0.32), baseline_prob_requested = 0.30)
  expect_length(warns[grepl("drift", warns)], 0L)
})

test_that(".diagnostic_warnings is silent on drift when requested baseline is NULL", {
  warns <- mcpower:::.diagnostic_warnings(.glm_inner(0.90), baseline_prob_requested = NULL)
  expect_length(warns[grepl("drift", warns)], 0L)
})

test_that(".diagnostic_warnings is silent on drift for OLS", {
  inner <- list(
    convergence_rate           = list(1.0),
    boundary_hit_rate_tau_zero = 0.0,
    boundary_hit_rate_high_tau = 0.0,
    estimator_extras           = list(estimator = "ols")
  )
  warns <- mcpower:::.diagnostic_warnings(inner, baseline_prob_requested = 0.30)
  expect_length(warns[grepl("drift", warns)], 0L)
})

# ── .report_diagnostics: multi-scenario prefix + Laplace (mirrors Python) ─────

.clean_scen <- function(...) {
  # Replace top-level fields wholesale (NOT modifyList, which recurses into nested
  # lists and matches by name — an unnamed convergence_rate = list(x) override
  # would merge with the base instead of replacing it).
  base <- list(
    convergence_rate           = list(1.0),
    boundary_hit_rate_tau_zero = 0.0,
    boundary_hit_rate_high_tau = 0.0,
    estimator_extras           = list(estimator = "ols")
  )
  over <- list(...)
  for (nm in names(over)) base[[nm]] <- over[[nm]]
  base
}

test_that(".report_diagnostics prefixes a degraded scenario, leaves clean ones out", {
  result <- list(scenarios = list(
    baseline = .clean_scen(),
    stress   = .clean_scen(convergence_rate = list(0.50))
  ))
  out <- mcpower:::.report_diagnostics(result, list(factors = list()), mcpower:::.report_config())
  expect_match(out, "⚠ Diagnostics")
  expect_match(out, "stress: convergence")
  expect_false(grepl("baseline:", out))
})

test_that(".report_diagnostics leaves a single scenario un-prefixed", {
  result <- list(scenarios = list(default = .clean_scen(convergence_rate = list(0.50))))
  out <- mcpower:::.report_diagnostics(result, list(factors = list()), mcpower:::.report_config())
  expect_match(out, "! convergence")
  expect_false(grepl("default:", out))
})

test_that(".report_diagnostics surfaces the Laplace line for high τ̂² + small clusters", {
  inner <- .clean_scen(estimator_extras = list(
    estimator = "glm", baseline_prob_realized = 0.30, tau_squared_hat_mean = 2.0))
  result <- list(scenarios = list(default = inner))
  meta <- list(factors = list(), baseline_prob_requested = 0.30, min_cluster_size = 4L)
  out <- mcpower:::.report_diagnostics(result, meta, mcpower:::.report_config())
  expect_match(out, "Laplace-approximation bias likely")
})

test_that(".report_diagnostics has no Laplace line when clusters are large", {
  recommended <- mcpower:::.config()$limits$recommended_rows_per_cluster
  inner <- .clean_scen(estimator_extras = list(
    estimator = "glm", baseline_prob_realized = 0.30, tau_squared_hat_mean = 2.0))
  result <- list(scenarios = list(default = inner))
  meta <- list(factors = list(), baseline_prob_requested = 0.30,
               min_cluster_size = as.integer(recommended))
  out <- mcpower:::.report_diagnostics(result, meta, mcpower:::.report_config())
  expect_false(grepl("Laplace", out))
})

# ── .surface_grid_warnings: preflight warnings reach the user ────────────────

test_that("find_power surfaces grid_warnings as R warnings for sparse factor", {
  # [0.95, 0.05] at N=40: level 2 of the factor receives 2 observations (< 5);
  # engine emits a preflight warning with 'factor 1' in the text.
  expect_warning(
    MCPower$new("y ~ g")$
      set_effects("g[2]=0.5")$
      set_variable_type("g=(factor,0.95,0.05)")$
      find_power(sample_size = 40, n_sims = 50, seed = 2137,
                 verbose = FALSE, progress_callback = FALSE),
    regexp = "factor 1"
  )
})

test_that(".surface_grid_warnings dedupes across scenarios, preserves order", {
  # Two scenarios share 'shared warning'; each adds one unique string. The
  # shared one must fire exactly once, in first-seen order.
  result <- list(scenarios = list(
    list(grid_warnings = c("shared warning", "only in A")),
    list(grid_warnings = c("shared warning", "only in B"))
  ))
  warns <- character(0)
  withCallingHandlers(
    mcpower:::.surface_grid_warnings(result),
    warning = function(w) {
      warns <<- c(warns, conditionMessage(w))
      invokeRestart("muffleWarning")
    }
  )
  expect_identical(warns, c("shared warning", "only in A", "only in B"))
})
