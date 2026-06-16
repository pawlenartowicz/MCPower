# output-report.R — S3 print methods for mcpower result objects.
# Content mirrors the Python port; delivery is cli/S3-native.
# Format constants come from .report_config().

# Engine field notes (verified against live output):
#   target_indices    — 1-based integer vector (no adjustment needed)
#   power_uncorrected — list of per-N vectors; [[1]] is the single-N vector
#   ci_uncorrected    — list of per-N lists; [[1]][[i]] is numeric[2] c(lo, hi)
#   sample_sizes      — numeric vector; [1] is N for find_power calls
#   overall_significant_rate — NULL unless overall test requested

# Percentage with `decimals` places, e.g. .fmt_pct(0.925, 1) == "92.5%".
# Exactly 100% drops the fractional part ("100%" not "100.0%") — mirrors the
# Python fmt_pct (results.py) so both ports render identically. Vectorised.
.fmt_pct <- function(x, decimals) {
  pct <- x * 100
  ifelse(round(pct, decimals) == 100,
         "100%",
         sprintf(paste0("%.", decimals, "f%%"), pct))
}

# CI cell padded so percent signs / decimals stack, e.g. "[99.0%,  100%]".
# Byte-for-byte identical to the Python fmt_ci (results.py): bounds go through
# .fmt_pct (100% drops its decimals), so the column reserves two integer digits
# and a 100% bound takes one leading space, not two. "" when ci is NULL/short/NA.
.fmt_ci <- function(ci, decimals) {
  if (is.null(ci) || length(ci) < 2L) return("")
  lo <- ci[[1]]; hi <- ci[[2]]
  if (is.na(lo) || is.na(hi)) return("")
  w <- 2L + (if (decimals > 0L) 1L + decimals else 0L) + 1L  # 2 int digits + '.dec' + '%'
  sprintf("[%s, %s]", formatC(.fmt_pct(lo, decimals), width = w),
                      formatC(.fmt_pct(hi, decimals), width = w))
}

# Largest N evaluated in a find_sample_size grid (the search ceiling); NULL when
# unknown. A required N reported as unreached lies above this -> "≥ <ceiling>".
.search_ceiling <- function(inner) {
  ss <- inner$sample_sizes
  if (is.null(ss) || length(ss) == 0L) return(NULL)
  as.integer(max(as.numeric(ss)))
}

# Headline display string + numeric value for one required-N cell: the
# model-based crossing fallback chain (tables.py:_required_n_headline).
#   fitted          -> (str(n_achievable), n_achievable)
#   at_or_below_min -> ("≤ n_min",         n_min)        used for footer max
#   not_reached     -> ("≥ ceiling",       NULL)          above grid, no numeric
#   non_monotone    -> first_achieved fallback (grid value or ≥ ceiling)
#   missing fit     -> first_achieved fallback (older payload, silent)
# Returns list(display=, numeric=).
.required_n_headline <- function(inner, pos) {
  # fitted is string-keyed by 0-based position index (IndexMap in Rust)
  key  <- as.character(pos - 1L)   # pos is 1-based (R); engine key is 0-based
  f    <- inner$fitted[[key]]

  if (!is.null(f)) {
    status <- f$status
    if (identical(status, "fitted"))
      return(list(display = as.character(f$n_achievable), numeric = as.integer(f$n_achievable)))
    if (identical(status, "at_or_below_min"))
      return(list(display = sprintf("≤ %d", f$n_min), numeric = as.integer(f$n_min)))
    if (identical(status, "not_reached")) {
      ceil <- .search_ceiling(inner)
      return(list(display = if (!is.null(ceil)) sprintf("≥ %d", ceil) else "—",
                  numeric = NULL))
    }
    # non_monotone: fall through to first_achieved
  }

  # Fallback: first_achieved (also covers non_monotone and missing fitted).
  # first_achieved is also string-keyed 0-based.
  v <- inner$first_achieved[[key]]
  if (!is.null(v) && !is.na(v))
    return(list(display = as.character(v), numeric = as.integer(v)))
  ceil <- .search_ceiling(inner)
  list(display = if (!is.null(ceil)) sprintf("≥ %d", ceil) else "—", numeric = NULL)
}

# Required-N cell (short form): thin wrapper over .required_n_headline.
# Mirrors Python fmt_required_n (tables.py). Returns the grid-empirical fallback
# display string ("≥ N" or "—").
.req_n_cell <- function(inner, pos) .required_n_headline(inner, pos)$display

`%||%` <- function(a, b) if (is.null(a)) b else a

# Split "condition[treatment]" -> list(factor="condition", level="treatment");
# NULL for a plain continuous predictor.
.split_factor <- function(name, factor_names) {
  m <- regmatches(name, regexec("^(.*)\\[(.*)\\]$", name))[[1]]
  if (length(m) == 3L && m[2] %in% factor_names) {
    list(factor = m[2], level = m[3])
  } else {
    NULL
  }
}

# results.py:362-368 _factor_label — render the display label for a factor dummy
# from the port's label store.  `level` is the 0-based index the engine emits in
# the EffectSkeleton (full label list, reference included).  Falls back to
# str(level + 1) when no labels are stored (unnamed factor renders 1..k).
.factor_label <- function(factors, fname, level) {
  lvls <- (factors[[fname]] %||% list())$levels %||% character(0)
  r_idx <- level + 1L   # engine level is 0-based; R lists are 1-based
  if (r_idx >= 1L && r_idx <= length(lvls)) lvls[[r_idx]] else as.character(level + 1L)
}

# results.py:371-382 _render_descriptor — render one EffectDescriptor from the
# engine's effect skeleton to a display name using the port's label store.
.render_descriptor <- function(desc, factors) {
  k <- desc$kind
  if (k == "continuous") return(desc$predictor)
  if (k == "factor_level") {
    return(sprintf("%s[%s]", desc$factor, .factor_label(factors, desc$factor, desc$level)))
  }
  if (k == "interaction") {
    return(paste(vapply(desc$components, function(c) .render_descriptor(c, factors), character(1)),
                 collapse = ":"))
  }
  "(Intercept)"
}

# Display label for the pairwise contrast β_p − β_n: both sides rendered from
# the skeleton, joined by the vs token (mirrors _contrast_label in Python's
# tables.py — change together). p/n are engine 0-based β-column indices.
.contrast_label <- function(skeleton, factors, p, n) {
  vs <- .report_config()$text[["vs_token"]] %||% "vs"
  sprintf("%s %s %s",
          .render_descriptor(skeleton[[p + 1L]], factors), vs,
          .render_descriptor(skeleton[[n + 1L]], factors))
}

# results.py:385-417 build_rows — ordered display rows from the engine's
# index-only effect skeleton + the port's label store.  Names are never
# re-derived from effect-name strings.  `contrast_pairs` (list of [positive,
# negative] engine column-index pairs — the result's contrast_pairs field)
# appends one "B vs A" contrast row per pair; the engine puts contrast entries
# after the marginals, so those rows get pos = length(target_indices) + j.
#
# Indexing note: the engine emits 0-based target_indices (intercept = 0).
# The skeleton is an R list (1-based), so skeleton element for engine index `idx`
# is at position `idx + 1L`.  desc$level is also 0-based (engine convention).
.build_rows <- function(target_indices, meta, contrast_pairs = list()) {
  skeleton <- meta$effect_skeleton
  factors  <- meta$factors %||% list()

  # Fall back to the old string-parse path when skeleton is absent (e.g. tests
  # that hand-build meta without a skeleton — those tests must be updated, but
  # we keep this guard to avoid a hard crash during transition).
  if (is.null(skeleton)) {
    effect_names <- meta$effect_names
    fnames <- names(factors)
    rows <- list()
    seen <- character(0)
    for (pos in seq_along(target_indices)) {
      idx <- target_indices[[pos]]   # engine 0-based; effect_names is 1-based-indexed
      nm <- effect_names[[idx]]
      sf <- .split_factor(nm, fnames)
      if (is.null(sf)) {
        rows[[length(rows) + 1L]] <- list(kind = "continuous", label = nm, pos = pos)
      } else {
        if (!(sf$factor %in% seen)) {
          rows[[length(rows) + 1L]] <- list(
            kind = "factor_header", label = sf$factor,
            baseline = (factors[[sf$factor]] %||% list())$baseline
          )
          seen <- c(seen, sf$factor)
        }
        rows[[length(rows) + 1L]] <- list(
          kind = "factor_level", label = sf$level,
          factor = sf$factor, pos = pos
        )
      }
    }
    return(rows)
  }

  rows <- list()
  seen <- character(0)
  for (pos in seq_along(target_indices)) {
    idx  <- target_indices[[pos]]   # 0-based from engine (intercept = 0)
    r_pos <- idx + 1L               # 1-based position in the R skeleton list
    if (r_pos < 1L || r_pos > length(skeleton)) {
      stop(sprintf(
        "target_indices entry %d is out of range for the effect skeleton (0..%d). ",
        idx, length(skeleton) - 1L))
    }
    desc <- skeleton[[r_pos]]
    if (desc$kind == "factor_level") {
      fname <- desc$factor
      if (!(fname %in% seen)) {
        rows[[length(rows) + 1L]] <- list(
          kind = "factor_header", label = fname,
          baseline = (factors[[fname]] %||% list())$baseline
        )
        seen <- c(seen, fname)
      }
      rows[[length(rows) + 1L]] <- list(
        kind = "factor_level",
        label = .factor_label(factors, fname, desc$level),
        factor = fname, pos = pos
      )
    } else {
      rows[[length(rows) + 1L]] <- list(
        kind = "continuous",
        label = .render_descriptor(desc, factors),
        pos = pos
      )
    }
  }
  n_marginals <- length(target_indices)
  for (j in seq_along(contrast_pairs)) {
    pair <- contrast_pairs[[j]]
    p <- as.integer(pair[[1L]]); n <- as.integer(pair[[2L]])
    if (p < 0L || p >= length(skeleton) || n < 0L || n >= length(skeleton)) {
      stop(sprintf(
        "contrast_pairs entry (%d, %d) is out of range for the effect skeleton (0..%d).",
        p, n, length(skeleton) - 1L))
    }
    rows[[length(rows) + 1L]] <- list(
      kind = "contrast",
      label = .contrast_label(skeleton, factors, p, n),
      pos = n_marginals + j
    )
  }
  rows
}

# Worst-case-over-the-grid exclusion rate per factor (worst across the grid).
# find_power:        counts is a flat integer vector — shape [n_factors].
# find_sample_size:  counts is a list of per-N vectors — shape [[n_grid]][n_factors].
# Returns a numeric vector of rates (count / n_sims), one per factor; length 0
# when counts is absent or n_sims is 0.
.max_exclusion_rates <- function(inner, key) {
  counts <- inner[[key]]
  n_sims <- inner$n_sims %||% 0L
  if (is.null(counts) || length(counts) == 0L || n_sims == 0L) return(numeric(0))
  # Normalise: list-of-vectors (sample_size path) vs flat vector (power path).
  per_n <- if (is.list(counts)) counts else list(counts)
  if (length(per_n) == 0L || length(per_n[[1L]]) == 0L) return(numeric(0))
  n_factors <- length(per_n[[1L]])
  vapply(seq_len(n_factors), function(f) {
    max(vapply(per_n, function(row) as.numeric(row[[f]]), numeric(1L)))
  }, numeric(1L)) / as.numeric(n_sims)
}

# Core diagnostic message per configured threshold that trips; empty when clean.
# Returns BARE messages — each caller decorates: the short form wraps them as
# "! <msg> — see summary()", the long form lists them as "! <msg>" under a
# "⚠ Diagnostics" heading. Faithful mirror of Python diagnostic_warnings()
# (ports/py/mcpower/output/tables.py) — gate set, message wording, and order
# change together. convergence reduces over N with the worst (min) point.
#
# baseline_prob_requested / min_cluster_size are meta-level (one per run): the
# requested GLM event probability (from set_baseline_probability) and the
# smallest cluster size at the evaluated N. Both NULL unless the run is a
# binary-outcome GLM / GLMM; they drive the GLM-drift and Laplace-bias gates.
.diagnostic_warnings <- function(inner, factor_names = NULL,
                                 baseline_prob_requested = NULL,
                                 min_cluster_size = NULL) {
  th <- .report_config()$thresholds
  warns <- character(0)
  conv <- inner$convergence_rate %||% 1
  conv_scalar <- min(unlist(conv))
  if (conv_scalar < th$convergence_min)
    warns <- c(warns, sprintf("convergence %s", .fmt_pct(conv_scalar, 1)))
  # Boundary-hit gates on high-τ̂ (boundary_hit==2: τ̂ pinned implausibly large,
  # or GLMM optimizer/Schur failure) ONLY. Benign τ̂=0 ("singular fit", common at
  # small ICC) is expected, not a red flag — informational via singular_fit_rate.
  bh_ht <- inner$boundary_hit_rate_high_tau
  if (!is.null(bh_ht)) {
    hit_rate <- if (length(bh_ht) > 0L) max(as.numeric(bh_ht)) else 0.0
    if (hit_rate > th$lme_boundary_hit_max)
      warns <- c(warns, sprintf("high-τ̂ boundary %s", .fmt_pct(hit_rate, 1)))
  }
  extras <- inner$estimator_extras
  if (!is.null(baseline_prob_requested) &&
      !is.null(extras) && identical(extras$estimator, "glm") &&
      !is.null(extras$baseline_prob_realized)) {
    drift <- abs(extras$baseline_prob_realized - baseline_prob_requested)
    # baseline_prob_realized is NaN when the engine does not realize a baseline
    # (e.g. non-clustered GLM): Python's `NaN > x` is False so no warning fires,
    # but R errors on `if (NaN)`. Guard with is.na (catches NaN) to mirror
    # Python. Mirrors tables.py diagnostic_warnings — change together.
    if (!is.na(drift) && drift > th$glm_baseline_drift_max)
      warns <- c(warns, sprintf("GLM baseline drift %.3f", drift))
  }
  th_excl <- th$factor_exclusion_max %||% 0.0
  for (pair in list(list(key = "factor_exclusion_counts", verb = "excluded"),
                    list(key = "factor_separation_counts", verb = "separation-dropped"))) {
    rates <- .max_exclusion_rates(inner, pair$key)
    for (f in seq_along(rates)) {
      if (rates[[f]] > th_excl) {
        name <- if (!is.null(factor_names) && f <= length(factor_names)) {
          factor_names[[f]]
        } else {
          sprintf("factor %d", f)
        }
        # Format mirrors diagnostic_warnings() in
        # ports/py/mcpower/output/tables.py — change together.
        warns <- c(warns, sprintf("%s %s %.1f%% of sims", name, pair$verb, rates[[f]] * 100))
      }
    }
  }
  # Laplace-approximation bias: GLMM with large τ̂² and small clusters. Reuses the
  # canonical helper so this report line matches the transient warn. min_cluster_size
  # is NULL for non-GLMM runs → check skipped. Mirrors Python.
  if (!is.null(min_cluster_size)) {
    lap <- .glmm_laplace_bias_warning(extras, min_cluster_size)
    if (!is.null(lap)) warns <- c(warns, lap)
  }
  warns
}

# Overall/omnibus row label for an estimator token (F-test for OLS, LRT for GLM,
# Wald for MLE). Mirrors tables.py:_overall_label_by_estimator — used by
# find_power (estimator off the result) and find_sample_size (off meta, the
# sample-size host dict carries no top-level estimator).
.overall_label_by_estimator <- function(est) {
  .report_config()$overall_label_by_estimator[[est %||% "ols"]] %||% "Overall"
}

.overall_label <- function(inner) {
  .overall_label_by_estimator((inner$estimator_extras$estimator) %||% "ols")
}

# Headline display + numeric for the overall-test required-N cell: the same
# model-based crossing fallback chain as .required_n_headline, on the single
# fitted_overall CrossingFit (string-keyed "0", an IndexMap of length 0/1) with
# first_overall_achieved (a scalar integer/NA) as the grid-empirical fallback.
# Mirrors tables.py:_overall_required_n_headline. Returns list(display=, numeric=).
.overall_required_n_headline <- function(inner) {
  f <- inner$fitted_overall[["0"]]
  if (!is.null(f)) {
    status <- f$status
    if (identical(status, "fitted"))
      return(list(display = as.character(f$n_achievable), numeric = as.integer(f$n_achievable)))
    if (identical(status, "at_or_below_min"))
      return(list(display = sprintf("≤ %d", f$n_min), numeric = as.integer(f$n_min)))
    if (identical(status, "not_reached")) {
      ceil <- .search_ceiling(inner)
      return(list(display = if (!is.null(ceil)) sprintf("≥ %d", ceil) else "—", numeric = NULL))
    }
    # non_monotone: fall through to first_overall_achieved
  }
  v <- inner$first_overall_achieved
  if (!is.null(v) && !is.na(v))
    return(list(display = as.character(v), numeric = as.integer(v)))
  ceil <- .search_ceiling(inner)
  list(display = if (!is.null(ceil)) sprintf("≥ %d", ceil) else "—", numeric = NULL)
}

# TRUE when the sample-size result carried an overall test (fitted_overall
# non-empty, or first_overall_achieved set). Mirrors tables.py:_has_overall_required_n.
.has_overall_required_n <- function(inner) {
  length(inner$fitted_overall) > 0L ||
    (!is.null(inner$first_overall_achieved) && !is.na(inner$first_overall_achieved))
}

# Post-hoc pairwise rows for the main per-test table: a "<factor>  (pairwise)"
# header followed by canonical pairwise contrast rows (mirrors Python
# posthoc_rows() in results.py). Each contrast carries 1-based block/contrast
# indices into inner$posthoc[[block]]. Empty list when meta has no
# posthoc_factors. Contrasts nest inline like factor levels — no separate table.
.posthoc_descriptors <- function(meta) {
  pf <- meta[["posthoc_factors"]]
  if (is.null(pf) || length(pf) == 0L) return(list())
  vs <- .report_config()$text[["vs_token"]] %||% "vs"
  out <- list()
  for (bi in seq_along(pf)) {
    levels <- pf[[bi]]$levels
    k      <- length(levels)
    out[[length(out) + 1L]] <- list(kind = "posthoc_header", label = pf[[bi]]$name)
    cc <- 1L
    for (a in seq_len(k - 1L)) {
      for (b in seq.int(a + 1L, k)) {
        out[[length(out) + 1L]] <- list(
          kind = "posthoc_contrast",
          label = sprintf("%s %s %s", levels[[b]], vs, levels[[a]]),
          block = bi, contrast = cc)
        cc <- cc + 1L
      }
    }
  }
  out
}

# ── Single-renderer table + section assembly (mirror Python tables.py/report.py) ──
# Every report table routes through .minimal_table; each section helper returns a
# string (never cat()s); the print methods assemble sections and emit once.
# Column widths are counted in CODE POINTS (nchar type="chars" == Python len()),
# not bytes, so the multi-byte glyphs the report emits (χ ² ≥ ≤ — ─) no longer
# shift later columns left.
# Line-for-line ports of the Python functions named in each comment — change
# together. Field extraction follows the R decode conventions already proven in
# this file (overall_significant_rate[[1]], power_uncorrected[[1]][[pos]], …).

# 2-space indent on nested rows; a factor header as a full-width span. Mirrors
# tables.label_of / the "(baseline: X)" span shared by every table.
.span_factor_row <- function(r) list(kind = "span",
  text = sprintf("%s  (baseline: %s)", r$label, r$baseline))
.indent_label <- function(r) paste0(if (identical(r$kind, "factor_level")) "  " else "", r$label)

# Port of tables.py:minimal_table — render a minimal-rules (booktabs-style) text
# table to a STRING (no trailing newline, no cat).
#   title:   heading above the table; NULL omits it (the short form has no caption,
#            the analysis header acts as the title).
#   columns: list of list(header=, align=) with align ∈ {"l","r"}; col 1 is the
#            left-aligned label column.
#   rows:    list of list(kind="row", cells=<character>) or
#            list(kind="span", text=<string>). A span prints verbatim and still
#            widens the label column so data lines up beneath it.
# Width per column = max(header, max cell) in code points; the label column is
# then clamped to [name_min, name_max]. Gap between columns = 3 spaces.
.minimal_table <- function(title, columns, rows, name_min = 18L, name_max = 44L) {
  headers <- vapply(columns, function(col) col$header, character(1L))
  aligns  <- vapply(columns, function(col) col$align,  character(1L))
  widths  <- pmax(nchar(headers, type = "chars"), 1L)
  for (row in rows) {
    if (identical(row$kind, "row")) {
      for (i in seq_along(row$cells))
        widths[i] <- max(widths[i], nchar(row$cells[[i]], type = "chars"))
    } else {
      widths[1] <- max(widths[1], nchar(row$text, type = "chars"))
    }
  }
  widths[1] <- max(name_min, min(widths[1], name_max))
  gap <- strrep(" ", 3L)

  pad <- function(x, w, align) {
    p <- max(0L, w - nchar(x, type = "chars"))
    if (identical(align, "l")) paste0(x, strrep(" ", p)) else paste0(strrep(" ", p), x)
  }
  render <- function(cells)
    paste(vapply(seq_along(cells),
                 function(i) pad(cells[[i]], widths[[i]], aligns[[i]]), character(1L)),
          collapse = gap)

  header_line <- render(headers)
  rule <- strrep("─", nchar(header_line, type = "chars"))
  lines <- c(if (!is.null(title)) title, rule, header_line, rule)
  for (row in rows)
    lines <- c(lines, if (identical(row$kind, "row")) render(row$cells) else row$text)
  lines <- c(lines, rule)
  paste(lines, collapse = "\n")
}

# Port of tables.py:main_power_tables — the §4 main result as a character vector
# of 1 or 2 table strings (two only when correction AND >1 scenario are both on).
# caption is the base section title (NULL for the short form).
.main_power_tables <- function(scen, meta, dec, tdec, target, caption) {
  cfg  <- .report_config(); cols <- cfg$text$columns
  corr <- !is.null(meta$correction) && meta$correction != "none"
  multi <- length(scen) > 1L
  inner0 <- scen[[1]]
  rows <- .build_rows(inner0$target_indices, meta, inner0$contrast_pairs %||% list())
  ph   <- if (!is.null(inner0$posthoc)) .posthoc_descriptors(meta) else list()
  tgt  <- .fmt_pct(target, tdec)
  posthoc_span <- function(r) list(kind = "span", text = sprintf("%s  (pairwise)", r$label))
  rowcells <- function(...) list(kind = "row", cells = c(...))

  if (!multi) {
    if (!corr) {
      columns <- list(list(header = cols$test,  align = "l"),
                      list(header = cols$power,  align = "r"),
                      list(header = cols$target, align = "r"))
      table <- list()
      if (!is.null(inner0$overall_significant_rate))
        table <- c(table, list(rowcells(.overall_label(inner0),
                     .fmt_pct(inner0$overall_significant_rate[[1]], dec), tgt)))
      pw <- inner0$power_uncorrected[[1]]
      for (r in rows) {
        if (identical(r$kind, "factor_header")) { table <- c(table, list(.span_factor_row(r))); next }
        table <- c(table, list(rowcells(.indent_label(r), .fmt_pct(pw[[r$pos]], dec), tgt)))
      }
      for (r in ph) {
        if (identical(r$kind, "posthoc_header")) { table <- c(table, list(posthoc_span(r))); next }
        val <- inner0$posthoc[[r$block]]$power_uncorrected[[r$contrast]]
        table <- c(table, list(rowcells(paste0("  ", r$label), .fmt_pct(val, dec), tgt)))
      }
      return(.minimal_table(caption, columns, table))
    }
    # correction only: Test | uncorrected | corrected | Target
    columns <- list(list(header = cols$test,        align = "l"),
                    list(header = cols$uncorrected, align = "r"),
                    list(header = cols$corrected,   align = "r"),
                    list(header = cols$target,      align = "r"))
    table <- list()
    pu <- inner0$power_uncorrected[[1]]; pc <- inner0$power_corrected[[1]]
    if (!is.null(inner0$overall_significant_rate))
      table <- c(table, list(rowcells(.overall_label(inner0),
                   .fmt_pct(inner0$overall_significant_rate[[1]], dec), "(same)", tgt)))
    for (r in rows) {
      if (identical(r$kind, "factor_header")) { table <- c(table, list(.span_factor_row(r))); next }
      table <- c(table, list(rowcells(.indent_label(r),
                   .fmt_pct(pu[[r$pos]], dec), .fmt_pct(pc[[r$pos]], dec), tgt)))
    }
    for (r in ph) {
      if (identical(r$kind, "posthoc_header")) { table <- c(table, list(posthoc_span(r))); next }
      blk <- inner0$posthoc[[r$block]]
      table <- c(table, list(rowcells(paste0("  ", r$label),
                   .fmt_pct(blk$power_uncorrected[[r$contrast]], dec),
                   .fmt_pct(blk$power_corrected[[r$contrast]], dec), tgt)))
    }
    return(.minimal_table(caption, columns, table))
  }

  # multi-scenario: one table per active correction state.
  names_v <- names(scen)
  build_scen_table <- function(pkey) {
    columns <- c(list(list(header = cols$test, align = "l")),
                 lapply(names_v, function(nm) list(header = nm, align = "r")),
                 list(list(header = cols$target, align = "r")))
    table <- list()
    scen_row <- function(label, pos, overall) {
      raw <- lapply(scen, function(s)
        if (overall) s$overall_significant_rate else s[[pkey]][[1]][[pos]])
      if (any(vapply(raw, is.null, logical(1L)))) return(NULL)
      vals <- vapply(raw, function(v) if (overall) v[[1]] else v, numeric(1L))
      rowcells(label, .fmt_pct(vals, dec), tgt)
    }
    if (!is.null(inner0$overall_significant_rate)) {
      r0 <- scen_row(.overall_label(inner0), NA_integer_, TRUE)
      if (!is.null(r0)) table <- c(table, list(r0))
    }
    for (r in rows) {
      if (identical(r$kind, "factor_header")) { table <- c(table, list(.span_factor_row(r))); next }
      rr <- scen_row(.indent_label(r), r$pos, FALSE)
      if (!is.null(rr)) table <- c(table, list(rr))
    }
    for (r in ph) {
      if (identical(r$kind, "posthoc_header")) { table <- c(table, list(posthoc_span(r))); next }
      vals <- numeric(0); ok <- TRUE
      for (s in scen) {
        blocks <- s$posthoc %||% list()
        if (r$block > length(blocks)) { ok <- FALSE; break }
        vals <- c(vals, blocks[[r$block]][[pkey]][[r$contrast]])
      }
      if (ok) table <- c(table, list(rowcells(paste0("  ", r$label), .fmt_pct(vals, dec), tgt)))
    }
    list(columns = columns, table = table)
  }

  if (!corr) {
    bt <- build_scen_table("power_uncorrected")
    return(.minimal_table(caption, bt$columns, bt$table))
  }
  # .strip() of the concatenated caption: short form (caption NULL) yields the
  # bare "— Uncorrected" / "— Corrected" title, exactly as Python.
  title_u <- trimws(paste0(caption %||% "", cfg$text$uncorrected_suffix))
  title_c <- trimws(paste0(caption %||% "", cfg$text$corrected_suffix))
  bu <- build_scen_table("power_uncorrected")
  bc <- build_scen_table("power_corrected")
  c(.minimal_table(if (nzchar(title_u)) title_u else NULL, bu$columns, bu$table),
    .minimal_table(if (nzchar(title_c)) title_c else NULL, bc$columns, bc$table))
}

# Port of tables.py:_render_sample_size_short — the find_sample_size short form
# (Required N column[s] + "First N achieving all targets" footer + non-monotone
# warnings). Returns the whole body as a string.
.render_sample_size_short <- function(result, meta, cfg) {
  cols <- cfg$text$columns
  scen <- .scenarios(result)
  inner0 <- scen[[1]]
  rows <- .build_rows(inner0$target_indices, meta, inner0$contrast_pairs %||% list())
  tdec <- cfg$format$target_decimals
  target <- (meta$target_power %||% (.sim_defaults()$target_power * 100)) / 100
  est_label <- toupper(meta$estimator %||% "ols")
  alpha <- meta$alpha %||% .sim_defaults()$alpha
  head <- sprintf("Power Analysis (sample size) — %s  target=%s  α=%s",
                  est_label, .fmt_pct(target, tdec), alpha)
  if (!is.null(meta$correction) && meta$correction != "none")
    head <- paste0(head, sprintf("\ncorrection: %s", meta$correction))
  if (length(scen) > 1L)
    head <- paste0(head, "\nscenarios: ", paste(names(scen), collapse = ", "))

  # Overall (omnibus) required-N row first, mirroring the find_power short form
  # and the long-form report. Estimator label from meta (the sample-size host
  # dict carries no top-level estimator).
  overall_label <- .overall_label_by_estimator(meta$estimator %||% "ols")
  table <- list()
  if (length(scen) == 1L) {
    columns <- list(list(header = cols$test, align = "l"),
                    list(header = cols$required_n, align = "r"))
    if (.has_overall_required_n(inner0))
      table <- c(table, list(list(kind = "row",
                   cells = c(overall_label, .overall_required_n_headline(inner0)$display))))
    for (r in rows) {
      if (identical(r$kind, "factor_header")) { table <- c(table, list(.span_factor_row(r))); next }
      table <- c(table, list(list(kind = "row", cells = c(.indent_label(r), .req_n_cell(inner0, r$pos)))))
    }
  } else {
    columns <- c(list(list(header = cols$test, align = "l")),
                 lapply(names(scen), function(nm) list(header = nm, align = "r")))
    if (any(vapply(scen, .has_overall_required_n, logical(1L)))) {
      cells <- c(overall_label, vapply(scen, function(s) .overall_required_n_headline(s)$display,
                                       character(1L)))
      table <- c(table, list(list(kind = "row", cells = cells)))
    }
    for (r in rows) {
      if (identical(r$kind, "factor_header")) { table <- c(table, list(.span_factor_row(r))); next }
      cells <- c(.indent_label(r), vapply(scen, function(s) .req_n_cell(s, r$pos), character(1L)))
      table <- c(table, list(list(kind = "row", cells = cells)))
    }
  }

  # footers: per-scenario max numeric headline (≥ ceiling when any row unreached);
  # collect non_monotone warnings across scenarios.
  footers <- character(0)
  non_monotone_items <- list()
  for (nm in names(scen)) {
    inner <- scen[[nm]]
    numerics <- list(); has_not_reached <- FALSE
    for (r in rows) {
      if (identical(r$kind, "factor_header")) next
      h <- .required_n_headline(inner, r$pos)
      numerics <- c(numerics, list(h$numeric))
      if (is.null(h$numeric)) has_not_reached <- TRUE
    }
    if (!has_not_reached && length(numerics) > 0L && all(!vapply(numerics, is.null, logical(1L)))) {
      footers <- c(footers, as.character(as.integer(max(unlist(numerics)))))
    } else {
      ceil <- .search_ceiling(inner)
      footers <- c(footers, if (!is.null(ceil)) sprintf("≥ %d", ceil) else "—")
    }
    fitted_map <- inner$fitted %||% list()
    for (r in rows) {
      if (identical(r$kind, "factor_header")) next
      f <- fitted_map[[as.character(r$pos - 1L)]]
      if (!is.null(f) && identical(f$status, "non_monotone"))
        non_monotone_items[[length(non_monotone_items) + 1L]] <-
          list(label = trimws(.indent_label(r)), drop = f$max_violation)
    }
  }
  body <- paste0(head, "\n\n", .minimal_table(NULL, columns, table),
                 "\n\nFirst N achieving all targets: ", paste(footers, collapse = " / "))
  if (length(non_monotone_items) > 0L) {
    warn_lines <- vapply(non_monotone_items, function(it) {
      w <- gsub("{label}", it$label, cfg$text$non_monotone_warning, fixed = TRUE)
      gsub("{drop}", sprintf("%.3f", it$drop), w, fixed = TRUE)
    }, character(1L))
    body <- paste0(body, "\n", paste(warn_lines, collapse = "\n"))
  }
  body
}

# ── Long-form section helpers (mirror report.py:Report._*) — each returns a
# string; "" means "omit this section". Assembled by print.mcpower_report. ──

# Report._header — the boxed header block.
.report_header <- function(result, meta, kind, cfg) {
  txt <- cfg$text
  scen <- .scenarios(result); inner0 <- scen[[1]]
  tdec <- cfg$format$target_decimals
  target <- (meta$target_power %||% (.sim_defaults()$target_power * 100)) / 100
  if (identical(kind, "find_sample_size")) {
    rows <- .build_rows(inner0$target_indices, meta, inner0$contrast_pairs %||% list())
    numerics <- list(); has_not_reached <- FALSE
    for (r in rows) {
      if (identical(r$kind, "factor_header")) next
      h <- .required_n_headline(inner0, r$pos)
      numerics <- c(numerics, list(h$numeric))
      if (is.null(h$numeric)) has_not_reached <- TRUE
    }
    ceil <- .search_ceiling(inner0)
    if (!has_not_reached && length(numerics) > 0L && all(!vapply(numerics, is.null, logical(1L)))) {
      n_label <- sprintf("N≥%d", as.integer(max(unlist(numerics))))
    } else if (!is.null(ceil)) {
      n_label <- sprintf("N≥%d (not all reached)", ceil)
    } else {
      n_label <- "N=— (target not reached)"
    }
  } else {
    n_raw <- if (!is.null(inner0$sample_sizes)) inner0$sample_sizes[[1]] else (inner0$n %||% "?")
    n_label <- sprintf("N=%s", n_raw)
  }
  title <- txt$long_title
  box <- strrep("=", max(nchar(title, type = "chars") + 4L, 50L))
  lines <- c(box, sprintf("  %s", title), box,
             sprintf("formula: %s", meta$formula %||% ""),
             sprintf("estimator: %s  %s  sims=%s  α=%s  target=%s",
                     toupper(meta$estimator %||% "ols"), n_label,
                     inner0$n_sims %||% "?", meta$alpha %||% .sim_defaults()$alpha,
                     .fmt_pct(target, tdec)))
  enames <- meta$effect_names; esizes <- meta$effect_sizes
  if (!is.null(enames) && !is.null(esizes) && length(enames) > 0L && length(esizes) > 0L) {
    pairs <- paste(vapply(seq_along(enames), function(i)
      sprintf("%s=%.2f", enames[[i]], as.numeric(esizes[[i]])), character(1L)), collapse = ", ")
    lines <- c(lines, sprintf("effects: %s", pairs))
  }
  if (!is.null(meta$correction) && meta$correction != "none")
    lines <- c(lines, sprintf("correction: %s", meta$correction))
  if (!is.null(meta$residual) && meta$residual != "normal")
    lines <- c(lines, sprintf("residual: %s", meta$residual))
  paste(lines, collapse = "\n")
}

# Report._per_test_power — main power table(s) for find_power; Required-N table
# for find_sample_size.
.report_per_test_power <- function(result, meta, kind, cfg) {
  if (identical(kind, "find_sample_size")) return(.report_required_n_table(result, meta, cfg))
  dec <- cfg$format$power_decimals_long
  tdec <- cfg$format$target_decimals
  scen <- .scenarios(result)
  target <- (meta$target_power %||% (.sim_defaults()$target_power * 100)) / 100
  paste(.main_power_tables(scen, meta, dec, tdec, target, cfg$text$main_caption), collapse = "\n\n")
}

# Report._required_n_table — long-form find_sample_size main table.
.report_required_n_table <- function(result, meta, cfg) {
  cols <- cfg$text$columns
  scen <- .scenarios(result)
  rows <- .build_rows(scen[[1]]$target_indices, meta, scen[[1]]$contrast_pairs %||% list())
  # Overall (omnibus) row first, positioned like the find_power overall row. The
  # estimator label comes from meta (the sample-size host dict has no estimator).
  overall_label <- .overall_label_by_estimator(meta$estimator %||% "ols")
  if (length(scen) == 1L) {
    s <- scen[[1]]
    table <- list()
    if (.has_overall_required_n(s))
      table <- c(table, list(list(kind = "row",
                   cells = c(overall_label, .overall_required_n_headline(s)$display))))
    for (r in rows) {
      if (identical(r$kind, "factor_header")) { table <- c(table, list(.span_factor_row(r))); next }
      table <- c(table, list(list(kind = "row", cells = c(.indent_label(r), .req_n_cell(s, r$pos)))))
    }
    return(.minimal_table(cfg$text$sample_size_caption,
                          list(list(header = cols$test, align = "l"),
                               list(header = cols$required_n, align = "r")), table))
  }
  columns <- c(list(list(header = cols$test, align = "l")),
               lapply(names(scen), function(nm) list(header = nm, align = "r")))
  table <- list()
  if (any(vapply(scen, .has_overall_required_n, logical(1L)))) {
    cells <- c(overall_label, vapply(scen, function(s) .overall_required_n_headline(s)$display,
                                     character(1L)))
    table <- c(table, list(list(kind = "row", cells = cells)))
  }
  for (r in rows) {
    if (identical(r$kind, "factor_header")) { table <- c(table, list(.span_factor_row(r))); next }
    cells <- c(.indent_label(r), vapply(scen, function(s) .req_n_cell(s, r$pos), character(1L)))
    table <- c(table, list(list(kind = "row", cells = cells)))
  }
  .minimal_table(cfg$text$sample_size_caption, columns, table)
}

# Report._ci_section — one "Power & 95% CI" table per scenario (find_power only),
# footnote directly under the bottom rule. Corrected values when correction is on.
.report_ci_section <- function(result, meta, kind, cfg) {
  if (identical(kind, "find_sample_size")) return("")
  txt <- cfg$text; cols <- txt$columns
  dec <- cfg$format$power_decimals_long
  scen <- .scenarios(result)
  corr <- !is.null(meta$correction) && meta$correction != "none"
  pkey <- if (corr) "power_corrected" else "power_uncorrected"
  ckey <- if (corr) "ci_corrected" else "ci_uncorrected"
  columns <- list(list(header = cols$test, align = "l"), list(header = cols$power, align = "r"),
                  list(header = cols$ci, align = "r"))
  blocks <- character(0)
  for (nm in names(scen)) {
    s <- scen[[nm]]
    rows <- .build_rows(s$target_indices, meta, s$contrast_pairs %||% list())
    table <- list()
    if (!is.null(s$overall_significant_rate))
      table <- c(table, list(list(kind = "row", cells = c(.overall_label(s),
                   .fmt_pct(s$overall_significant_rate[[1]], dec),
                   .fmt_ci(s$overall_significant_ci, dec)))))
    pws <- s[[pkey]][[1]]; cis <- s[[ckey]][[1]]
    for (r in rows) {
      if (identical(r$kind, "factor_header")) { table <- c(table, list(.span_factor_row(r))); next }
      pos <- r$pos
      ci_pair <- if (!is.null(cis) && length(cis) >= pos) {
        v <- cis[[pos]]; if (length(v) >= 2L) c(v[[1]], v[[2]]) else NULL
      } else NULL
      table <- c(table, list(list(kind = "row",
                   cells = c(.indent_label(r), .fmt_pct(pws[[pos]], dec), .fmt_ci(ci_pair, dec)))))
    }
    caption <- txt$ci_caption
    if (length(scen) > 1L) caption <- sprintf("%s — %s", caption, nm)
    footnote <- gsub("{n_sims}", s$n_sims %||% "?", txt$ci_footnote, fixed = TRUE)
    blocks <- c(blocks, paste0(.minimal_table(caption, columns, table), "\n", footnote))
  }
  paste(blocks, collapse = "\n\n")
}

# Report._required_n_ci_table — Required N & 95% CI from the model-based crossing
# fit (find_sample_size only); skipped when no scenario carries fitted data. CI
# bounds rounded outward; footnote clauses appended for appr / suppressed / floor.
.report_required_n_ci_table <- function(result, meta, kind, cfg) {
  if (!identical(kind, "find_sample_size")) return("")
  txt <- cfg$text; cols <- txt$columns
  scen <- .scenarios(result); inner0 <- scen[[1]]
  rows <- .build_rows(inner0$target_indices, meta, inner0$contrast_pairs %||% list())
  ceil_n <- .search_ceiling(inner0)
  floor_n <- if (!is.null(inner0$sample_sizes) && length(inner0$sample_sizes) > 0L)
    as.integer(min(as.numeric(inner0$sample_sizes))) else 0L
  if (!any(vapply(scen, function(s) length(s$fitted) > 0L, logical(1L)))) return("")
  columns <- list(list(header = cols$test, align = "l"), list(header = cols$required_n, align = "r"),
                  list(header = cols$ci, align = "r"))
  overall_label <- .overall_label_by_estimator(meta$estimator %||% "ols")
  blocks <- character(0)
  for (nm in names(scen)) {
    s <- scen[[nm]]
    fitted_map <- s$fitted
    if (is.null(fitted_map) || length(fitted_map) == 0L) next
    table <- list()
    has_appr <- FALSE; has_floor <- FALSE; non_mono_labels <- character(0)
    # Overall (omnibus) row first — same status dispatch as the per-target loop
    # below, on the single fitted_overall CrossingFit, folding its footnote flag
    # into the section flags. Mirrors report.py:_required_n_ci_table.
    if (.has_overall_required_n(s)) {
      of <- s$fitted_overall[["0"]]
      ostatus <- if (!is.null(of)) of$status else NULL
      if (is.null(of) || identical(ostatus, "non_monotone")) {
        table <- c(table, list(list(kind = "row",
                     cells = c(overall_label, .overall_required_n_headline(s)$display, "—"))))
        if (identical(ostatus, "non_monotone")) non_mono_labels <- c(non_mono_labels, overall_label)
      } else if (identical(ostatus, "fitted")) {
        o_lo <- of$ci_lo; o_hi <- of$ci_hi
        if (is.null(o_lo) && is.null(o_hi)) {
          o_cell <- if (!is.null(ceil_n)) sprintf("[≤ %d, ≥ %d]", floor_n, ceil_n) else "—"; has_floor <- TRUE
        } else if (is.null(o_lo)) {
          o_cell <- sprintf("[≤ %d, %d]", floor_n, ceiling(o_hi)); has_floor <- TRUE
        } else if (is.null(o_hi)) {
          o_cell <- if (!is.null(ceil_n)) sprintf("[%d, ≥ %d]", floor(o_lo), ceil_n)
                    else sprintf("[%d, —]", floor(o_lo))
        } else {
          o_cell <- sprintf("[%d, %d]", floor(o_lo), ceiling(o_hi))
        }
        table <- c(table, list(list(kind = "row",
                     cells = c(overall_label, as.character(of$n_achievable), o_cell))))
      } else if (identical(ostatus, "at_or_below_min")) {
        table <- c(table, list(list(kind = "row",
                     cells = c(overall_label, sprintf("≤ %d", of$n_min), "—")))); has_floor <- TRUE
      } else if (identical(ostatus, "not_reached")) {
        o_appr <- of$n_approx
        if (!is.null(o_appr) && !is.na(o_appr)) { o_cell <- sprintf("appr. %d", as.integer(o_appr)); has_appr <- TRUE }
        else o_cell <- "—"
        o_headline <- if (!is.null(ceil_n)) sprintf("≥ %d", ceil_n) else "—"
        table <- c(table, list(list(kind = "row", cells = c(overall_label, o_headline, o_cell))))
      }
    }
    for (r in rows) {
      if (identical(r$kind, "factor_header")) { table <- c(table, list(.span_factor_row(r))); next }
      label <- .indent_label(r)
      f <- fitted_map[[as.character(r$pos - 1L)]]
      if (is.null(f)) {
        table <- c(table, list(list(kind = "row",
                     cells = c(label, .required_n_headline(s, r$pos)$display, "—")))); next
      }
      status <- f$status
      if (identical(status, "fitted")) {
        headline <- as.character(f$n_achievable)
        ci_lo_r <- f$ci_lo; ci_hi_r <- f$ci_hi
        if (is.null(ci_lo_r) && is.null(ci_hi_r)) {
          ci_cell <- if (!is.null(ceil_n)) sprintf("[≤ %d, ≥ %d]", floor_n, ceil_n) else "—"; has_floor <- TRUE
        } else if (is.null(ci_lo_r)) {
          ci_cell <- sprintf("[≤ %d, %d]", floor_n, ceiling(ci_hi_r)); has_floor <- TRUE
        } else if (is.null(ci_hi_r)) {
          ci_cell <- if (!is.null(ceil_n)) sprintf("[%d, ≥ %d]", floor(ci_lo_r), ceil_n)
                     else sprintf("[%d, —]", floor(ci_lo_r))
        } else {
          ci_cell <- sprintf("[%d, %d]", floor(ci_lo_r), ceiling(ci_hi_r))
        }
        table <- c(table, list(list(kind = "row", cells = c(label, headline, ci_cell))))
      } else if (identical(status, "at_or_below_min")) {
        table <- c(table, list(list(kind = "row", cells = c(label, sprintf("≤ %d", f$n_min), "—")))); has_floor <- TRUE
      } else if (identical(status, "not_reached")) {
        n_appr <- f$n_approx
        if (!is.null(n_appr) && !is.na(n_appr)) { ci_cell <- sprintf("appr. %d", as.integer(n_appr)); has_appr <- TRUE }
        else ci_cell <- "—"
        headline_nr <- if (!is.null(ceil_n)) sprintf("≥ %d", ceil_n) else "—"
        table <- c(table, list(list(kind = "row", cells = c(label, headline_nr, ci_cell))))
      } else if (identical(status, "non_monotone")) {
        table <- c(table, list(list(kind = "row",
                     cells = c(label, .required_n_headline(s, r$pos)$display, "—"))))
        non_mono_labels <- c(non_mono_labels, trimws(label))
      }
    }
    caption <- txt$required_n_ci_caption
    if (length(scen) > 1L) caption <- sprintf("%s — %s", caption, nm)
    footnote <- txt$required_n_ci_footnote
    if (has_appr) footnote <- paste0(footnote, "  ", txt$required_n_ci_footnote_appr)
    if (length(non_mono_labels) > 0L)
      footnote <- paste0(footnote, "  ", gsub("{labels}", paste(non_mono_labels, collapse = ", "),
                                              txt$required_n_ci_footnote_suppressed, fixed = TRUE))
    if (has_floor) footnote <- paste0(footnote, "  ", txt$required_n_ci_footnote_floor)
    blocks <- c(blocks, paste0(.minimal_table(caption, columns, table), "\n", footnote))
  }
  paste(blocks, collapse = "\n\n")
}

# Shortest decimal that round-trips to x, with a forced ".0" on integral values —
# mirrors Python's float repr, used by str(list) when an extras field is a Vec<f64>
# (so 0.0 renders "0.0", not R's bare "0").
.py_float_repr <- function(x) {
  if (is.nan(x)) return("nan")
  if (is.infinite(x)) return(if (x > 0) "inf" else "-inf")
  s <- trimws(formatC(x, format = "g", digits = 17L))
  for (d in 1:17) {
    cand <- trimws(formatC(x, format = "g", digits = d))
    if (as.numeric(cand) == x) { s <- cand; break }
  }
  if (!grepl("[.eE]", s)) s <- paste0(s, ".0")
  s
}

# Report._estimator_extras — GLM/MLE numerics, one block per non-empty scenario;
# OLS carries only {estimator} so nothing is shown. fmt_val mirrors Python's
# `f"{v:.4g}" if isinstance(v, float) else str(v)`: a scalar float gets %.4g (with
# nan/inf lowercased to match Python), a Vec<f64> field renders as Python's
# str(list) ("[a, b]"). extendr flattens a length-1 Vec<f64> to a scalar numeric,
# so the one surfaced Vec field (EstimatorExtras::Mle.boundary_rate_per_component,
# result_host.rs) is named explicitly to recover the bracket in the length-1 case.
.report_estimator_extras <- function(result, meta, cfg) {
  scen <- .scenarios(result)
  caption <- cfg$text$estimator_extras_caption
  vec_keys <- "boundary_rate_per_component"
  fmt_val <- function(key, v) {
    if (is.double(v) && (length(v) != 1L || key %in% vec_keys))
      return(paste0("[", paste(vapply(v, .py_float_repr, character(1L)), collapse = ", "), "]"))
    if (is.double(v)) {
      if (is.nan(v)) return("nan")
      if (is.infinite(v)) return(if (v > 0) "inf" else "-inf")
      return(sprintf("%.4g", v))
    }
    as.character(v)
  }
  blocks <- character(0)
  for (nm in names(scen)) {
    extras <- as.list(scen[[nm]]$estimator_extras %||% list())
    extras[["estimator"]] <- NULL
    if (length(extras) == 0L) next
    head <- if (length(scen) > 1L) sprintf("%s — %s", caption, nm) else caption
    lines <- vapply(names(extras), function(k) sprintf("  %s: %s", k, fmt_val(k, extras[[k]])), character(1L))
    blocks <- c(blocks, paste0(head, "\n", paste(lines, collapse = "\n")))
  }
  paste(blocks, collapse = "\n\n")
}

# Report._joint_section — joint significance distribution (find_power); joint
# required-N table (find_sample_size).
.report_joint_section <- function(result, meta, kind, cfg) {
  if (identical(kind, "find_sample_size")) return(.report_joint_required_n_table(result, meta, cfg))
  inner0 <- .scenarios(result)[[1]]
  jd <- .joint_distribution(inner0$success_count_histogram_uncorrected, inner0$n_sims %||% 0)
  if (is.null(jd)) return("Joint significance distribution is unavailable for this result.")
  dec <- cfg$format$joint_table_decimals
  table <- lapply(seq_along(jd$exactly), function(i)
    list(kind = "row", cells = c(as.character(i - 1L),
         .fmt_pct(jd$exactly[[i]], dec), .fmt_pct(jd$at_least[[i]], dec))))
  .minimal_table("Joint significance distribution",
                 list(list(header = "k", align = "l"), list(header = "Exactly", align = "r"),
                      list(header = "At least", align = "r")), table, name_min = 3L)
}

# Report._joint_required_n_table — Joint detection → required N (find_sample_size).
.report_joint_required_n_table <- function(result, meta, cfg) {
  inner0 <- .scenarios(result)[[1]]
  fja <- inner0$first_joint_achieved
  if (is.null(fja) || length(fja) == 0L) return("")
  tdec <- cfg$format$target_decimals
  target <- (meta$target_power %||% (.sim_defaults()$target_power * 100)) / 100
  ceil <- .search_ceiling(inner0)
  fj_map <- inner0$fitted_joint %||% list()
  n_targets <- length(fja)
  fallback_cell <- function(jkey) {
    v <- fja[[jkey]]
    if (!is.null(v) && !is.na(v)) as.character(v) else if (!is.null(ceil)) sprintf("≥ %d", ceil) else "—"
  }
  table <- list()
  for (j in (n_targets - 1L):0L) {
    k <- j + 1L
    jkey <- as.character(j)
    fj <- fj_map[[jkey]]
    if (!is.null(fj)) {
      status <- fj$status
      cell <- if (identical(status, "fitted")) as.character(fj$n_achievable)
              else if (identical(status, "at_or_below_min")) sprintf("≤ %d", fj$n_min)
              else if (identical(status, "not_reached")) (if (!is.null(ceil)) sprintf("≥ %d", ceil) else "—")
              else fallback_cell(jkey)
    } else {
      cell <- fallback_cell(jkey)
    }
    table <- c(table, list(list(kind = "row", cells = c(sprintf("≥ %d of %d tests", k, n_targets), cell))))
  }
  .minimal_table(sprintf("Joint detection → required N (target %s)", .fmt_pct(target, tdec)),
                 list(list(header = "Joint target", align = "l"),
                      list(header = "Required N", align = "r")), table)
}

# Report._robustness — Δ power vs baseline table; tests as rows, non-baseline
# scenarios as columns. Shown only when >= 2 scenarios (no kind gate, as Python).
.report_robustness <- function(result, meta, cfg) {
  scen <- .scenarios(result)
  if (length(scen) < 2L) return("")
  names_v <- names(scen)
  prefer <- cfg$baseline_scenario$prefer_label
  base_idx <- if (!is.null(prefer) && prefer %in% names_v) which(names_v == prefer)[[1]] else 1L
  base_name <- names_v[[base_idx]]
  other_names <- names_v[names_v != base_name]
  dec <- cfg$format$drop_decimals
  corr <- !is.null(meta$correction) && meta$correction != "none"
  pkey <- if (corr) "power_corrected" else "power_uncorrected"
  rows <- .build_rows(scen[[1]]$target_indices, meta, scen[[1]]$contrast_pairs %||% list())
  base_pows <- scen[[base_idx]][[pkey]][[1]]
  columns <- c(list(list(header = "Test", align = "l")),
               lapply(other_names, function(nm) list(header = nm, align = "r")))
  table <- list()
  for (r in rows) {
    if (identical(r$kind, "factor_header")) { table <- c(table, list(.span_factor_row(r))); next }
    pos <- r$pos
    cells <- c(.indent_label(r), vapply(other_names, function(nm) {
      delta <- scen[[nm]][[pkey]][[1]][[pos]] - base_pows[[pos]]
      sprintf(paste0("%+.", dec, "f pp"), delta * 100)
    }, character(1L)))
    table <- c(table, list(list(kind = "row", cells = cells)))
  }
  .minimal_table(sprintf("Robustness  (Δ power vs baseline: %s)", base_name), columns, table)
}

# Report._diagnostics — gated: empty on a healthy run; else "⚠ Diagnostics" + the
# bare warnings each prefixed with "! ". Every scenario is checked (a degraded
# sweep scenario is the point of robustness runs); each message carries a
# "{scenario}: " prefix when >1 (mirrors tables.render_short / report._diagnostics).
.report_diagnostics <- function(result, meta, cfg) {
  scenarios <- .scenarios(result)
  factor_names <- names(meta$factors %||% list())
  multi <- length(scenarios) > 1L
  warns <- character(0)
  for (nm in names(scenarios)) {
    for (w in .diagnostic_warnings(scenarios[[nm]], factor_names,
                                   meta$baseline_prob_requested, meta$min_cluster_size)) {
      warns <- c(warns, if (multi) sprintf("%s: %s", nm, w) else w)
    }
  }
  if (length(warns) == 0L) return("")
  paste0("⚠ Diagnostics\n", paste(sprintf("! %s", warns), collapse = "\n"))
}

# Report._plot_footer — R idiom (the one permitted port difference: plot(result)
# rather than result.plot()).
.report_plot_footer <- function() "Plots: plot(result) to view, plot(result, 'chart.png') to save."

#' Print method for mcpower_result (single-scenario find_power output).
#'
#' Renders a short-form power summary to the console.
#' Main table shows Test | Power | Target (no CI, no pass/fail glyph).
#' For multi-scenario: Test | <scenario cols> | Target; correction adds uncorrected/corrected cols.
#'
#' @param x   An mcpower_result object returned by \code{find_power()}.
#' @param ... Ignored.
#' @return \code{x} invisibly.
#' @export
print.mcpower_result <- function(x, ...) {
  meta <- attr(x, "mcpower_meta")
  cfg  <- .report_config()
  dec  <- cfg$format$power_decimals_short
  tdec <- cfg$format$target_decimals
  scen <- .scenarios(x)
  inner0 <- scen[[1]]
  target <- (meta$target_power %||% (.sim_defaults()$target_power * 100)) / 100
  est_label <- toupper((meta$estimator %||% inner0$estimator_extras$estimator) %||% "ols")
  n <- if (!is.null(inner0$sample_sizes)) inner0$sample_sizes[[1]] else (inner0$n %||% "?")

  # Compact model-summary block (mirror tables.py:_model_summary_block).
  hlines <- c(sprintf("Power Analysis — %s  N=%s  sims=%s  α=%s  target=%s",
                      est_label, n, inner0$n_sims %||% "?",
                      meta$alpha %||% .sim_defaults()$alpha, .fmt_pct(target, tdec)),
              sprintf("formula: %s", meta$formula %||% ""))
  if (!is.null(meta$correction) && meta$correction != "none")
    hlines <- c(hlines, sprintf("correction: %s", meta$correction))
  if (length(scen) > 1L)
    hlines <- c(hlines, paste0("scenarios: ", paste(names(scen), collapse = ", ")))

  tables <- .main_power_tables(scen, meta, dec, tdec, target, NULL)
  out <- paste0(paste(hlines, collapse = "\n"), "\n\n", paste(tables, collapse = "\n\n"))
  # Short-form diagnostics: each bare warning decorated "! <msg> — see summary()",
  # appended with a single newline (no blank line), exactly as render_short. Every
  # scenario is checked, prefixed "{scenario}: " when >1 (mirrors render_short).
  factor_names <- names(meta$factors %||% list())
  multi <- length(scen) > 1L
  warns <- character(0)
  for (nm in names(scen)) {
    for (w in .diagnostic_warnings(scen[[nm]], factor_names,
                                   meta$baseline_prob_requested, meta$min_cluster_size)) {
      warns <- c(warns, if (multi) sprintf("%s: %s", nm, w) else w)
    }
  }
  if (length(warns) > 0L)
    out <- paste0(out, "\n", paste(sprintf("! %s — see summary()", warns), collapse = "\n"))
  cat(out, "\n", sep = "")
  invisible(x)
}

# .scenarios: normalise result to a named list of inner (per-scenario) results.
.scenarios <- function(result) {
  if (!is.null(result$scenarios)) return(result$scenarios)
  setNames(list(result), result$scenario %||% "default")
}

# .joint_distribution: compute exactly-k / at-least-k proportions from the
# success_count_histogram (bucket k = number of targets significant in that sim).
# Returns NULL when the histogram is empty or n_sims_used is 0.
.joint_distribution <- function(histogram, n_sims_used) {
  if (is.null(histogram) || length(histogram) == 0 || n_sims_used == 0) return(NULL)
  n <- as.numeric(n_sims_used)
  at_least <- rev(cumsum(rev(histogram))) / n
  list(exactly = histogram / n, at_least = at_least)
}

#' @export
summary.mcpower_result <- function(object, ...) .make_report(object)

#' @export
summary.mcpower_sample_size_result <- function(object, ...) .make_report(object)

.make_report <- function(result) {
  structure(
    list(result = result,
         meta   = attr(result, "mcpower_meta"),
         kind   = attr(result, "mcpower_kind")),
    class = "mcpower_report"
  )
}

#' Print method for mcpower_report (long-form summary output).
#'
#' Assembles the long-form report as ordered sections — boxed header, per-test
#' power table(s), Power & 95% CI (find_power) / Required N & 95% CI
#' (find_sample_size), joint significance, robustness (multi-scenario),
#' estimator details, gated diagnostics, and the plot footer — joined by blank
#' lines and emitted with cat() so capture.output() captures the text. Every
#' table routes through the shared .minimal_table renderer (code-point column
#' widths), making the output byte-identical to the Python port. The Diagnostics
#' section appears only when a configured threshold trips.
#'
#' @param x   An mcpower_report object returned by \code{summary()}.
#' @param ... Ignored.
#' @return \code{x} invisibly.
#' @export
print.mcpower_report <- function(x, ...) {
  cfg    <- .report_config()
  result <- x$result
  meta   <- x$meta
  kind   <- x$kind %||% "find_power"

  # Mirror report.py:Report.__str__ — header + per-test power are always present;
  # the remaining sections are appended only when non-empty; the plot footer is
  # always last. Sections are joined by one blank line.
  parts <- c(.report_header(result, meta, kind, cfg),
             .report_per_test_power(result, meta, kind, cfg))
  add <- function(parts, section) if (nzchar(section)) c(parts, section) else parts
  parts <- add(parts, .report_ci_section(result, meta, kind, cfg))
  parts <- add(parts, .report_required_n_ci_table(result, meta, kind, cfg))
  parts <- add(parts, .report_joint_section(result, meta, kind, cfg))
  parts <- add(parts, .report_robustness(result, meta, cfg))
  parts <- add(parts, .report_estimator_extras(result, meta, cfg))
  parts <- add(parts, .report_diagnostics(result, meta, cfg))
  parts <- c(parts, .report_plot_footer())
  cat(paste(parts, collapse = "\n\n"), "\n", sep = "")
  invisible(x)
}

#' Print method for mcpower_sample_size_result (find_sample_size output).
#'
#' Shows the required N per effect and the first N achieving all targets.
#' Header: Power Analysis (sample size) — <EST>  target=…  α=…
#'
#' @param x   An mcpower_sample_size_result returned by \code{find_sample_size()}.
#' @param ... Ignored.
#' @return \code{x} invisibly.
#' @export
print.mcpower_sample_size_result <- function(x, ...) {
  meta <- attr(x, "mcpower_meta")
  cat(.render_sample_size_short(x, meta, .report_config()), "\n", sep = "")
  invisible(x)
}
