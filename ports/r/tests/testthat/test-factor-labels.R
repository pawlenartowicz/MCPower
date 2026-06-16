# test-factor-labels.R — R mirror of Python test_factor_labels.py.
#
# Pins the port's half of the single-source-in-Rust design: the engine returns
# a FactorLevel { factor, level } skeleton (level = 0-based index into the
# factor's FULL label list, reference included), and the port renders
# factor[labels[[level + 1]]] — never re-deriving the factor-expansion layout.
#
# R indexing note: skeleton and levels lists are 1-based in R.  Engine emits
# 0-based target_indices and descriptor.level values.  All arithmetic below
# documents the +1L adjustments explicitly.

# ---------------------------------------------------------------------------
# _resolve_reference — reference-by-value resolution
# ---------------------------------------------------------------------------

test_that("_resolve_reference: string matches label exactly", {
  expect_equal(mcpower:::.resolve_reference("6", list("4", "6", "8")), "6")
})

test_that("_resolve_reference: integer matches via value_to_label", {
  # 6 → "6" (drops trailing .0)
  expect_equal(mcpower:::.resolve_reference(6L,  list("4", "6", "8")), "6")
  expect_equal(mcpower:::.resolve_reference(6.0, list("4", "6", "8")), "6")
})

test_that("_resolve_reference: NULL + labels → first label", {
  expect_equal(mcpower:::.resolve_reference(NULL, list("4", "6", "8")), "4")
})

test_that("_resolve_reference: NULL + no labels → 1L (parser-path default)", {
  expect_equal(mcpower:::.resolve_reference(NULL, NULL), 1L)
})

test_that("_resolve_reference: string not in labels raises", {
  expect_error(mcpower:::.resolve_reference("nope", list("4", "6", "8")))
})

test_that("_resolve_reference: numeric not in labels raises", {
  expect_error(mcpower:::.resolve_reference(99, list("4", "6", "8")))
})

# ---------------------------------------------------------------------------
# set_variable_type stores resolved reference and full levels
# ---------------------------------------------------------------------------

test_that("set_variable_type stores resolved reference and factor_levels accessor works", {
  reg <- mcpower:::RVariableRegistry$new("y = cyl")
  reg$set_variable_type("cyl", "factor", n_levels = 3L,
                        labels = list("4", "6", "8"), reference = 6)
  # reference resolved by value → the canonical label string
  expect_equal(reg$`_factors`[["cyl"]]$reference_level, "6")
  # factor_levels returns full ordered label list (reference included)
  expect_equal(reg$factor_levels("cyl"), c("4", "6", "8"))
})

# ---------------------------------------------------------------------------
# build_contract_from_spec returns a $skeleton element
# ---------------------------------------------------------------------------

test_that("build_contract_from_spec returns a non-empty $skeleton JSON string", {
  spec_json <- paste0(
    '{"formula":"y ~ x1",',
    '"predictors":[{"name":"x1","kind":"normal"}],',
    '"effects":[{"name":"x1","size":0.5}],',
    '"correlations":[],',
    '"alpha":0.05,',
    '"correction":"none",',
    '"targets":["overall"],',
    '"report_overall":true,',
    '"contrast_pairs":[],',
    '"heteroskedasticity":{"driver_var_index":null},',
    '"residual":{"distribution":"normal"},',
    '"max_failed_fraction":1.0,',
    '"scenarios":[{"name":"optimistic","heterogeneity":0.0,"heteroskedasticity_ratio":1.0,',
    '"correlation_noise_sd":0.0,"distribution_change_prob":0.0,',
    '"new_distributions":[2,3,5],"residual_change_prob":0.0,',
    '"residual_dists":[4,2],"residual_df":10.0,"lme":null}]}'
  )
  out <- mcpower:::build_contract_from_spec(spec_json, "continuous", "ols", 0.0, "[]")
  expect_type(out$skeleton, "character")
  expect_true(nzchar(out$skeleton))
  # Parse and confirm it is a non-empty JSON array
  sk <- jsonlite::fromJSON(out$skeleton, simplifyVector = FALSE)
  expect_true(is.list(sk))
  expect_true(length(sk) >= 1L)
})

# ---------------------------------------------------------------------------
# .build_rows renders from skeleton — string data labels
# ---------------------------------------------------------------------------

test_that(".build_rows renders string data labels from skeleton", {
  # origin with data labels; baseline "Europe" (index 0 in engine = position 1
  # in R list); dummies at engine levels 1 (Japan) and 2 (USA).
  meta <- list(
    effect_skeleton = list(
      list(kind = "intercept"),
      list(kind = "factor_level", factor = "origin", level = 1L),
      list(kind = "factor_level", factor = "origin", level = 2L)
    ),
    factors = list(
      origin = list(baseline = "Europe",
                    levels   = c("Europe", "Japan", "USA"))
    )
  )
  # target_indices 1 and 2 (0-based from engine) → skeleton positions 2 and 3
  rows <- mcpower:::.build_rows(c(1L, 2L), meta)
  expect_equal(rows[[1]], list(kind = "factor_header", label = "origin", baseline = "Europe"))
  expect_equal(rows[[2]], list(kind = "factor_level", label = "Japan", factor = "origin", pos = 1L))
  expect_equal(rows[[3]], list(kind = "factor_level", label = "USA",   factor = "origin", pos = 2L))
})

# ---------------------------------------------------------------------------
# .build_rows renders numeric labels + interaction from skeleton
# ---------------------------------------------------------------------------

test_that(".build_rows renders numeric labels and a continuous×factor interaction", {
  # cyl[6] (numeric label, engine level 1 = R index 2 in ["4","6","8"])
  # plus x1:cyl[6] interaction (a continuous row, label joined by ":")
  meta <- list(
    effect_skeleton = list(
      list(kind = "intercept"),
      list(kind = "continuous", predictor = "x1"),
      list(kind = "factor_level", factor = "cyl", level = 1L),
      list(kind = "interaction", components = list(
        list(kind = "continuous", predictor = "x1"),
        list(kind = "factor_level", factor = "cyl", level = 1L)
      ))
    ),
    factors = list(
      cyl = list(baseline = "4", levels = c("4", "6", "8"))
    )
  )
  # target_indices 1, 2, 3 (0-based)
  rows <- mcpower:::.build_rows(c(1L, 2L, 3L), meta)
  expect_equal(rows[[1]], list(kind = "continuous", label = "x1", pos = 1L))
  expect_equal(rows[[2]], list(kind = "factor_header", label = "cyl", baseline = "4"))
  expect_equal(rows[[3]], list(kind = "factor_level", label = "6", factor = "cyl", pos = 2L))
  expect_equal(rows[[4]], list(kind = "continuous", label = "x1:cyl[6]", pos = 3L))
})

# ---------------------------------------------------------------------------
# .build_rows falls back to integers when no labels stored
# ---------------------------------------------------------------------------

test_that(".build_rows falls back to integers without labels", {
  meta <- list(
    effect_skeleton = list(
      list(kind = "intercept"),
      list(kind = "factor_level", factor = "g", level = 1L)
    ),
    factors = list(
      g = list(baseline = "1", levels = character(0))
    )
  )
  rows <- mcpower:::.build_rows(c(1L), meta)
  # level 1 (0-based) with empty levels → fallback "2" (level + 1)
  expect_equal(rows[[2]], list(kind = "factor_level", label = "2", factor = "g", pos = 1L))
})

# ---------------------------------------------------------------------------
# End-to-end: uploaded factor renders data labels in the printed output
# ---------------------------------------------------------------------------

test_that("uploaded factor labels appear in find_power output (end-to-end)", {
  skip_if_not_installed("tibble")
  # 42 rows: sorted unique → bird (ref), cat, dog
  breeds <- rep(c("cat", "dog", "bird"), 14L)
  df <- data.frame(breed = breeds, stringsAsFactors = FALSE)

  m <- MCPower$new("y = breed")
  m$upload_data(df, verbose = FALSE)
  m$set_effects("breed[cat]=0.5, breed[dog]=0.5")
  m$set_simulations(200L)
  res <- m$find_power(sample_size = 80L, verbose = FALSE, progress_callback = FALSE)

  out <- paste(capture.output(print(res)), collapse = "\n")
  # Data labels must appear; legacy integer names must NOT appear
  expect_match(out, "breed")
  expect_match(out, "baseline: bird")
  expect_match(out, "cat")
  expect_match(out, "dog")
  expect_false(grepl("breed\\[1\\]", out))
  expect_false(grepl("breed\\[2\\]", out))
})

# ---------------------------------------------------------------------------
# .build_report_meta includes effect_skeleton and factor levels after find_power
# ---------------------------------------------------------------------------

test_that(".build_report_meta includes effect_skeleton and factor levels after find_power", {
  m <- MCPower$new("y ~ x1 + cyl")
  m$set_variable_type("cyl=(factor,3)")
  m$set_effects("x1=0.3, cyl[2]=0.5, cyl[3]=0.4")
  m$set_simulations(200L)

  # find_power triggers build_contract_bytes which captures the skeleton
  res <- m$find_power(sample_size = 100L, verbose = FALSE, progress_callback = FALSE)
  meta <- attr(res, "mcpower_meta")

  expect_false(is.null(meta$effect_skeleton))
  expect_true(is.list(meta$effect_skeleton))
  expect_true(length(meta$effect_skeleton) >= 2L)

  # factors must carry levels (not just baseline)
  cyl_meta <- meta$factors[["cyl"]]
  expect_false(is.null(cyl_meta$levels))
  expect_equal(cyl_meta$levels, c("1", "2", "3"))
})
