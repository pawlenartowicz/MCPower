# output-plotting.R — plot spec emit + export helpers for mcpower results.
# Split from report.R; calls render helpers (.build_rows / .render_descriptor /
# .scenarios) defined in output-report.R, so it must load after it.

#' @include output-report.R
NULL

# ── Plot helpers ──────────────────────────────────────────────────────────────

# Overlay the host's real effect names onto the engine's generic `target_{idx}`
# plot tokens. Effect labels are host-owned, so the emitter ships `target_{idx}`
# (plot.rs target_label); this rewrites the spec's `target` data field to the
# same label the printed table uses (.render_descriptor), applied post-emit just
# like the theme overlay. No-op when the result carries no effect skeleton (so
# synthetic test results without one pass through unchanged).
.build_label_map <- function(result) {
  meta <- attr(result, "mcpower_meta")
  skeleton <- meta$effect_skeleton
  if (is.null(skeleton)) return(list())
  factors <- meta$factors %||% list()
  scen <- .scenarios(result)
  inner <- if (length(scen)) scen[[1]] else NULL
  ti <- inner$target_indices
  if (is.null(ti) || length(ti) == 0L) return(list())
  label_map <- list()
  for (idx in ti) {
    r_pos <- idx + 1L   # engine index is 0-based (intercept at 0); R lists 1-based
    if (r_pos >= 1L && r_pos <= length(skeleton)) {
      label_map[[paste0("target_", idx)]] <- .render_descriptor(skeleton[[r_pos]], factors)
    }
  }
  # Contrast entries carry target_{p}_vs_{n} tokens (plot.rs entry_label);
  # label them like the table's contrast rows (.contrast_label).
  for (pair in inner$contrast_pairs %||% list()) {
    p <- as.integer(pair[[1L]]); n <- as.integer(pair[[2L]])
    if (p >= 0L && p < length(skeleton) && n >= 0L && n < length(skeleton)) {
      label_map[[sprintf("target_%d_vs_%d", p, n)]] <-
        .contrast_label(skeleton, factors, p, n)
    }
  }
  # Overall/omnibus token for the sample-size curve series; label per estimator
  # (from meta — the sample-size result carries no top-level estimator).
  label_map[["overall"]] <- .overall_label_by_estimator(meta$estimator %||% "ols")
  label_map
}

# Recursively rewrite every `target` data-field value in a parsed Vega-Lite spec
# (top-level data, layered/marker data, vconcat children, facet spec) using
# `label_map` (target_{idx} → name). Vega-Lite keys the axis/legend off the
# field value, so relabelling the data relabels the chart.
.relabel_walk <- function(node, label_map) {
  if (!is.list(node)) return(node)
  if (!is.null(node$data) && is.list(node$data) && !is.null(node$data$values)) {
    vals <- node$data$values
    for (i in seq_along(vals)) {
      row <- vals[[i]]
      if (is.list(row) && !is.null(row$target)) {
        repl <- label_map[[as.character(row$target)]]
        if (!is.null(repl)) vals[[i]]$target <- repl
      }
    }
    node$data$values <- vals
  }
  for (nm in seq_along(node)) {
    # Single-bracket `node[nm] <- list(...)` so a NULL result (a JSON `null` in
    # the spec parses to a NULL element) is reassigned in place — `node[[nm]] <-
    # NULL` would instead DELETE the element, shrinking the list mid-loop and
    # making later seq_along indices out of bounds.
    node[nm] <- list(.relabel_walk(node[[nm]], label_map))
  }
  node
}

# In-place: rewrite every encoding axis title equal to exactly "Power"
# to "Power (<Correction>-corrected)". Joint-curve titles are not touched.
.rewrite_correction_axis_title_walk <- function(node, new_title) {
  if (!is.list(node)) return(node)
  if (!is.null(node$encoding) && is.list(node$encoding)) {
    for (fk in names(node$encoding)) {
      enc <- node$encoding[[fk]]
      if (is.list(enc)) {
        if (identical(enc$title, "Power")) {
          node$encoding[[fk]]$title <- new_title
        }
        if (is.list(enc$axis) && identical(enc$axis$title, "Power")) {
          node$encoding[[fk]]$axis$title <- new_title
        }
      }
    }
    # recurse into children but skip encoding (already handled)
    # Single-bracket reassignment preserves NULL elements — see `.relabel_walk`.
    for (nm in names(node)) {
      if (nm == "encoding") next
      node[nm] <- list(.rewrite_correction_axis_title_walk(node[[nm]], new_title))
    }
  } else {
    for (nm in seq_along(node)) {
      node[nm] <- list(.rewrite_correction_axis_title_walk(node[[nm]], new_title))
    }
  }
  node
}

# Make errorbar CIs legible. The engine emits errorbar marks with no colour, so
# they inherit the bar colour and vanish. Vega-Lite forbids `color` in
# config.errorbar, so the contrasting colour is set per-mark — host-applied
# colour/size, like the theme overlay. End ticks expose the interval bounds.
# Single-series error bars (no color encoding) get a foreground whisker;
# grouped/multi-series ones keep their colour and just gain ticks.
.style_ci_walk <- function(node, color) {
  if (!is.list(node)) return(node)
  mark <- node$mark
  mark_type <- if (is.list(mark)) mark$type else if (is.character(mark)) mark else NULL
  if (!is.null(mark_type) && identical(mark_type, "errorbar")) {
    md <- if (is.list(mark)) mark else list(type = "errorbar")
    has_color_enc <- is.list(node$encoding) && !is.null(node$encoding$color)
    if (has_color_enc) {
      md$ticks <- TRUE
    } else {
      md$ticks <- list(color = color)
      md$rule  <- list(color = color, strokeWidth = 1.5)
    }
    node$mark <- md
  }
  # Single-bracket reassignment preserves NULL elements — see `.relabel_walk`.
  for (nm in seq_along(node)) node[nm] <- list(.style_ci_walk(node[[nm]], color))
  node
}

# Recursively deep-merge `overlay` into `base` (mutating base). Nested dicts
# merge key-by-key; non-dict values overwrite.
.deep_merge <- function(base, overlay) {
  for (k in names(overlay)) {
    if (is.list(base[[k]]) && is.list(overlay[[k]])) {
      base[[k]] <- .deep_merge(base[[k]], overlay[[k]])
    } else {
      base[[k]] <- overlay[[k]]
    }
  }
  base
}

# Build the neutral plot envelope for the new engine plot-set functions.
# Correction-awareness: the host pre-selects corrected vs uncorrected arrays
# before serializing so the engine receives neutral `power`/`ci` keys.
.build_envelope_json <- function(result, kind) {
  meta <- attr(result, "mcpower_meta")
  corr <- !is.null(meta$correction) && meta$correction != "none"
  pkey <- if (corr) "power_corrected" else "power_uncorrected"
  ckey <- if (corr) "ci_corrected" else "ci_uncorrected"

  scen <- .scenarios(result)
  scenarios_list <- lapply(scen, function(inner) {
    # ci: list of per-N items, each a list of c(lo,hi) pairs.
    ci_rows <- lapply(inner[[ckey]], function(row) {
      lapply(row, function(pair) as.numeric(pair)[1:2])
    })
    # power: list of per-N numeric vectors
    power_rows <- lapply(inner[[pkey]], function(v) I(as.numeric(v)))
    out <- list(
      label          = inner$scenario %||% "default",
      sample_sizes   = I(as.integer(inner$sample_sizes)),
      target_indices = I(as.integer(inner$target_indices)),
      power          = power_rows,
      ci             = ci_rows
    )
    if (kind == "find_power") {
      # The overall/omnibus test is a first-class result: it draws one more bar
      # (last) on the power-at-N chart, matching the table. NULL when the family
      # suppressed the overall test (mixed/GLMM) → no bar.
      if (!is.null(inner$overall_significant_rate)) {
        out$overall_power <- inner$overall_significant_rate[[1]]
        if (!is.null(inner$overall_significant_ci)) {
          out$overall_ci <- I(as.numeric(inner$overall_significant_ci)[1:2])
        }
      }
    }
    if (kind == "find_sample_size") {
      # histogram: always corrected (for at_least_k / exactly_k curves)
      if (!is.null(inner$success_count_histogram_corrected)) {
        out$histogram <- lapply(inner$success_count_histogram_corrected,
                                function(v) I(as.integer(v)))
      }
    }
    out
  })

  # Build the envelope with label embedded into each scenario (array of objects)
  # engine-r bridge expects { "scenarios": [ { "label": ..., ... }, ... ] }
  # We need names to be the label field, but since we already set label inside,
  # pass as unnamed list (array):
  scen_names <- names(scenarios_list)
  for (i in seq_along(scenarios_list)) {
    scenarios_list[[i]]$label <- scen_names[[i]]
  }

  jsonlite::toJSON(
    list(scenarios = unname(scenarios_list)),
    auto_unbox = TRUE, null = "null"
  )
}

# Apply a named theme to a spec JSON string: deep-merge theme config into
# spec$config and re-style CI marks with the theme's axis.titleColor.
# Returns the new JSON string. theme=NULL → no-op (theme-naked).
.apply_theme <- function(spec_json, theme) {
  if (is.null(theme)) return(spec_json)
  theme_config <- jsonlite::fromJSON(plot_theme(theme), simplifyVector = FALSE)
  spec <- jsonlite::fromJSON(spec_json, simplifyVector = FALSE)
  if (is.null(spec$config)) spec$config <- list()
  spec$config <- .deep_merge(spec$config, theme_config)
  title_col <- spec$config$axis$titleColor %||% "#333333"
  spec <- .style_ci_walk(spec, title_col)
  jsonlite::toJSON(spec, auto_unbox = TRUE, null = "null")
}

CI_DEFAULT_COLOR <- "#333333"

# Build the ordered list of (block_key, spec_json) pairs for result.
# Builds the neutral envelope, calls the engine plot-set bridge, then applies
# relabeling, CI styling, and (when correction is active) rewrites the Power
# axis title to include the correction name.
.plot_blocks <- function(result, kind) {
  meta <- attr(result, "mcpower_meta")
  corr <- !is.null(meta$correction) && meta$correction != "none"
  correction_name <- if (corr) meta$correction else NULL
  target_power <- if (!is.null(meta$target_power)) meta$target_power / 100 else NULL

  envelope_json <- .build_envelope_json(result, kind)
  label_map <- .build_label_map(result)

  raw_blocks <- if (kind == "find_power") {
    power_plot_set_json(envelope_json, show_ci = TRUE,
                        target_power_line = target_power)
  } else {
    sample_size_plot_set_json(envelope_json, show_ci = TRUE,
                              target_power_line = target_power)
  }

  # raw_blocks is a named R list: names = block keys, values = spec JSON strings
  block_keys <- names(raw_blocks)
  blocks <- vector("list", length(block_keys))
  names(blocks) <- block_keys

  for (k in block_keys) {
    spec_json <- as.character(raw_blocks[[k]])
    spec <- jsonlite::fromJSON(spec_json, simplifyVector = FALSE)

    # Relabel target tokens to real effect names
    if (length(label_map) > 0L) {
      spec <- .relabel_walk(spec, label_map)
    }

    # Apply default CI styling (may be overridden by theme later)
    spec <- .style_ci_walk(spec, CI_DEFAULT_COLOR)

    # Rewrite "Power" axis title when correction is active
    if (!is.null(correction_name)) {
      cap <- paste0(toupper(substr(correction_name, 1, 1)),
                    substr(correction_name, 2, nchar(correction_name)))
      new_title <- sprintf("Power (%s-corrected)", cap)
      spec <- .rewrite_correction_axis_title_walk(spec, new_title)
    }

    blocks[[k]] <- spec
  }
  blocks
}

.mcpower_next_free <- function(path) {
  if (!file.exists(path)) return(path)
  stem <- tools::file_path_sans_ext(path); ext <- tools::file_ext(path)
  i <- 2L; repeat { cand <- sprintf("%s_%d.%s", stem, i, ext)
    if (!file.exists(cand)) return(cand); i <- i + 1L }
}

# Derive the output path for a single block given the user's base stem and ext.
# Block routing:
#   power / curve          -> <stem><ext>  (unchanged)
#   scenario:<label>       -> <stem>_<sanitized><ext>
#   overlay/at_least_k/exactly_k -> <stem>_<key><ext>
.derive_block_path <- function(stem, ext, block_key) {
  if (block_key %in% c("power", "curve")) {
    return(paste0(stem, ext))
  }
  if (startsWith(block_key, "scenario:")) {
    label <- substring(block_key, nchar("scenario:") + 1L)
    sanitized <- gsub("[^a-z0-9]+", "_", tolower(label))
    return(paste0(stem, "_", sanitized, ext))
  }
  paste0(stem, "_", block_key, ext)
}

# Pair each block with its output path; deduplicate in-call collisions.
.unique_block_paths <- function(user_path, blocks) {
  stem <- tools::file_path_sans_ext(user_path)
  ext  <- paste0(".", tools::file_ext(user_path))
  seen <- list()
  result <- list()
  for (k in names(blocks)) {
    base <- .derive_block_path(stem, ext, k)
    if (is.null(seen[[base]])) {
      seen[[base]] <- 1L
      path <- base
    } else {
      seen[[base]] <- seen[[base]] + 1L
      base_stem <- tools::file_path_sans_ext(base)
      base_ext  <- paste0(".", tools::file_ext(base))
      path <- paste0(base_stem, "_", seen[[base]], base_ext)
    }
    result[[length(result) + 1L]] <- list(key = k, spec = blocks[[k]], path = path)
  }
  result
}

# Write a single stacked HTML file with all block specs using the engine's
# CDN template. theme is applied to every spec; NULL = theme-naked.
# "</'" is escaped as "<\/" inside the JSON array to prevent script-tag breakout.
.write_stacked_html <- function(blocks, path, theme) {
  template <- plot_html_template()
  themed_specs <- lapply(names(blocks), function(k) {
    spec_json <- jsonlite::toJSON(blocks[[k]], auto_unbox = TRUE, null = "null")
    if (!is.null(theme)) {
      spec_json <- .apply_theme(spec_json, theme)
    }
    jsonlite::fromJSON(spec_json, simplifyVector = FALSE)
  })
  specs_json <- jsonlite::toJSON(themed_specs, auto_unbox = TRUE, null = "null")
  specs_json <- gsub("</", "<\\/", specs_json, fixed = TRUE)
  html <- sub("{{SPECS}}", specs_json, template, fixed = TRUE)
  writeLines(html, path, useBytes = TRUE)
  invisible(path)
}

#' Plot method for mcpower_result (power-at-N chart via Vega-Lite).
#'
#' No \code{file} argument: displays the chart. When \pkg{vegawidget} is
#' installed and the session is interactive, the chart renders as a bundled-JS
#' htmlwidget — it shows inline in the RStudio Viewer (or opens in the browser
#' under plain R) with no CDN, so it works in sandboxed panes that block the
#' fallback. Otherwise a self-contained CDN HTML file is written to the working
#' directory (\code{find_power.html}; successive calls use \code{_2}, \code{_3},
#' … suffixes) and opened; its path is reported so it can be opened manually.
#' With a \code{file} argument: delegates to \code{save_plot()} for
#' image/widget export (requires optional renderers).
#'
#' @param x    An mcpower_result object returned by \code{find_power()}.
#' @param file Optional output path; extension selects format (html/svg/png/pdf).
#' @param ...  Passed to \code{save_plot()} when \code{file} is supplied.
#' @return Invisibly: the rendered widget (vegawidget path) or the path written (CDN fallback).
#' @export
plot.mcpower_result <- function(x, file = NULL, ...) {
  kind <- attr(x, "mcpower_kind") %||% "find_power"
  if (!is.null(file)) return(save_plot(x, file, ...))
  blocks <- .plot_blocks(x, kind)

  # Preferred: render through vegawidget so the JS is bundled locally (no CDN).
  # The htmltools print method routes to getOption("viewer") in RStudio or a
  # temp file + browser under plain R, so it shows where CDN HTML stays blank.
  if (interactive() && requireNamespace("vegawidget", quietly = TRUE)) {
    widgets <- lapply(blocks, function(spec) {
      spec_json <- .apply_theme(
        jsonlite::toJSON(spec, auto_unbox = TRUE, null = "null"), "light-print")
      vegawidget::vegawidget(vegawidget::as_vegaspec(spec_json))
    })
    out <- htmltools::browsable(htmltools::tagList(widgets))
    print(out)
    return(invisible(out))
  }

  # Fallback: zero-dependency self-contained CDN HTML file, opened in a browser.
  base <- if (kind == "find_sample_size") "find_sample_size.html" else "find_power.html"
  path <- .mcpower_next_free(base)
  .write_stacked_html(blocks, path, theme = "light-print")
  open_ok <- interactive() &&
             (.Platform$OS.type == "windows" || nzchar(Sys.getenv("DISPLAY")) ||
              nzchar(Sys.getenv("WAYLAND_DISPLAY")) || Sys.info()[["sysname"]] == "Darwin")
  if (open_ok) {
    viewer <- getOption("viewer")
    if (is.function(viewer)) viewer(path) else utils::browseURL(path)
    message(sprintf("Wrote and opened %s", path))
  } else {
    message(sprintf("Wrote %s (no display — open it manually).", path))
  }
  invisible(path)
}

#' Plot method for mcpower_sample_size_result (power-vs-N curve via Vega-Lite CDN HTML).
#'
#' @param x    An mcpower_sample_size_result returned by \code{find_sample_size()}.
#' @param file Optional output path; extension selects format (html/svg/png/pdf).
#' @param ...  Passed to \code{save_plot()} when \code{file} is supplied.
#' @return Invisibly, the path written.
#' @export
plot.mcpower_sample_size_result <- plot.mcpower_result

#' knit_print method for mcpower_report (rich Quarto/RMarkdown output).
#'
#' Renders the text summary followed by a Vega-Lite chart widget when
#' \pkg{vegawidget} is available, or falls back to plain text otherwise.
#' Block selection: find_power -> "power" block; find_sample_size single-scenario
#' -> "curve" block; multi-scenario -> "overlay" block. light-print theme applied.
#'
#' @param x   An mcpower_report object returned by \code{summary()}.
#' @param ... Passed to \code{knitr::knit_print}.
#' @return An object suitable for knitr output.
#' @exportS3Method knitr::knit_print
knit_print.mcpower_report <- function(x, ...) {
  txt <- paste(utils::capture.output(print(x)), collapse = "\n")
  has_vega  <- requireNamespace("vegawidget", quietly = TRUE)
  has_knitr <- requireNamespace("knitr",      quietly = TRUE)
  if (!has_vega) {
    if (has_knitr) return(knitr::knit_print(txt, ...))
    return(txt)
  }
  kind <- x$kind %||% "find_power"
  blocks <- .plot_blocks(x$result, kind)

  # Select the single representative block
  if (kind == "find_power") {
    spec <- blocks[["power"]] %||% blocks[[1]]
  } else {
    scen <- .scenarios(x$result)
    if (length(scen) >= 2L) {
      spec <- blocks[["overlay"]] %||% blocks[["curve"]] %||% blocks[[1]]
    } else {
      spec <- blocks[["curve"]] %||% blocks[[1]]
    }
  }

  # Apply light-print theme
  spec_json <- .apply_theme(
    jsonlite::toJSON(spec, auto_unbox = TRUE, null = "null"),
    "light-print"
  )
  widget <- vegawidget::vegawidget(vegawidget::as_vegaspec(spec_json))
  out <- htmltools::tagList(
    htmltools::tags$pre(txt),
    widget
  )
  if (has_knitr) knitr::knit_print(out, ...) else out
}

# ──as.data.frame + as_latex / as_pdf stubs ─────────────────────────

#' @export
as.data.frame.mcpower_sample_size_result <- function(x, ...) {
  # required_n: n_achievable (fitted), first_achieved value (non_monotone or
  # missing fit), NA_integer_ for at_or_below_min and not_reached (these are
  # ≤ / ≥ sentinel rows with no single exportable integer).
  # ci_lo/ci_hi: floor/ceiling-rounded when status == fitted and bound present;
  # NA_integer_ otherwise (mirrors Python SampleSizeResult.to_dataframe()).
  meta <- attr(x, "mcpower_meta")
  recs <- list()
  for (nm in names(.scenarios(x))) {
    inner      <- .scenarios(x)[[nm]]
    fitted_map <- inner$fitted
    for (r in .build_rows(inner$target_indices, meta, inner$contrast_pairs %||% list())) {
      if (r$kind == "factor_header") next
      key <- as.character(r$pos - 1L)   # 0-based engine key
      f   <- if (!is.null(fitted_map)) fitted_map[[key]] else NULL
      if (!is.null(f) && identical(f$status, "fitted")) {
        req   <- as.integer(f$n_achievable)
        ci_lo <- if (!is.null(f$ci_lo)) as.integer(floor(f$ci_lo))   else NA_integer_
        ci_hi <- if (!is.null(f$ci_hi)) as.integer(ceiling(f$ci_hi)) else NA_integer_
      } else if (is.null(f) || identical(f$status, "non_monotone")) {
        # Missing fit or non_monotone: use first_achieved grid value
        v   <- inner$first_achieved[[key]]
        req <- if (!is.null(v) && !is.na(v)) as.integer(v) else NA_integer_
        ci_lo <- NA_integer_; ci_hi <- NA_integer_
      } else {
        # at_or_below_min / not_reached: no meaningful integer to export
        req <- NA_integer_; ci_lo <- NA_integer_; ci_hi <- NA_integer_
      }
      recs[[length(recs) + 1L]] <- data.frame(
        test = r$label, scenario = nm,
        required_n = req, ci_lo = ci_lo, ci_hi = ci_hi,
        stringsAsFactors = FALSE)
    }
  }
  do.call(rbind, recs)
}

#' @export
as_latex <- function(x, ...) UseMethod("as_latex")
#' @export
as_latex.default <- function(x, ...)
  stop("as_latex(): not implemented yet — on the roadmap.", call. = FALSE)
#' @export
as_pdf <- function(x, file, ...) UseMethod("as_pdf")
#' @export
as_pdf.default <- function(x, file, ...)
  stop("as_pdf(): not implemented yet — on the roadmap.", call. = FALSE)

#' Save an MCPower result's chart to a file
#'
#' Renders the plot-set for the result to one or more files. HTML suffix writes a
#' single stacked file with all blocks; other suffixes write one file per block
#' with derived names (see Details). Format is chosen by the file extension:
#' \code{.html}, \code{.svg}, \code{.png}, or \code{.pdf}.
#'
#' For HTML, the engine's CDN template is used — no \pkg{vegawidget}/\pkg{htmlwidgets}
#' dependency. SVG export requires \pkg{vegawidget} + \pkg{V8}. PNG/PDF require
#' \pkg{rsvg} (system \code{librsvg}: \code{apt librsvg2-dev} /
#' \code{dnf librsvg2-devel} on Linux).
#'
#' Default theme is \code{"light-print"}; pass \code{theme = NULL} for theme-naked output.
#'
#' @param x An \code{mcpower_result}, \code{mcpower_sample_size_result}, or \code{mcpower_report}.
#' @param file Output path; extension selects the format.
#' @param theme Optional theme name (one of \code{list_plot_themes()}), or \code{NULL} for theme-naked.
#' @param scale Raster scale for PNG (default 2). The PNG is rasterised at \code{scale × SVG pixel width}.
#' @export
save_plot <- function(x, file, theme = "light-print", scale = 2) {
  if (inherits(x, "mcpower_report")) {
    result <- x$result
    kind <- x$kind %||% "find_power"
  } else {
    result <- x
    kind <- attr(x, "mcpower_kind") %||% "find_power"
  }

  ext <- tolower(tools::file_ext(file))
  if (!ext %in% c("html", "svg", "png", "pdf")) {
    stop(sprintf("unsupported plot format '.%s'; use one of: html, svg, png, pdf", ext),
         call. = FALSE)
  }

  blocks <- .plot_blocks(result, kind)

  if (ext == "html") {
    .write_stacked_html(blocks, file, theme = theme)
    return(invisible(file))
  }

  # Non-HTML: one file per block
  if (!requireNamespace("vegawidget", quietly = TRUE)) {
    stop("Image export needs 'vegawidget': install.packages('vegawidget').", call. = FALSE)
  }

  if (ext == "svg") {
    if (!requireNamespace("V8", quietly = TRUE)) {
      stop("SVG export needs 'V8': install.packages('V8').", call. = FALSE)
    }
  }

  if (ext %in% c("png", "pdf")) {
    if (!requireNamespace("rsvg", quietly = TRUE)) {
      stop(paste0(
        sprintf("%s export needs the 'rsvg' package: install.packages('rsvg'). ", toupper(ext)),
        "On Linux also install the system librsvg (apt: librsvg2-dev / dnf: librsvg2-devel)."),
        call. = FALSE)
    }
  }

  block_paths <- .unique_block_paths(file, blocks)
  for (item in block_paths) {
    spec_json <- jsonlite::toJSON(item$spec, auto_unbox = TRUE, null = "null")
    if (!is.null(theme)) {
      spec_json <- .apply_theme(spec_json, theme)
    }
    vspec <- vegawidget::as_vegaspec(spec_json)

    if (ext == "svg") {
      vegawidget::vw_write_svg(vspec, item$path)
    } else {
      # png or pdf via rsvg
      svg_str <- vegawidget::vw_to_svg(vspec)
      svg_raw <- charToRaw(svg_str)
      if (ext == "png") {
        # Honor scale: rasterize at scale * SVG pixel width
        w_match <- regmatches(svg_str,
                              regexpr('width="([0-9.]+)"', svg_str, perl = TRUE))
        w_px <- if (length(w_match) > 0L) {
          as.numeric(sub('width="([0-9.]+)"', "\\1", w_match[[1]]))
        } else NULL
        target_w <- if (!is.null(w_px) && !is.na(w_px)) as.integer(round(scale * w_px)) else NULL
        rsvg::rsvg_png(svg_raw, item$path, width = target_w)
      } else {
        rsvg::rsvg_pdf(svg_raw, item$path)
      }
    }
  }
  invisible(file)
}

#' List the embedded plot theme names
#'
#' Returns the names of all built-in themes that can be passed to
#' \code{save_plot(theme = ...)}, \code{plot_theme()}, or the \code{theme}
#' argument of any plotting function.  Themes are embedded at build time from
#' \code{configs/plot-themes.json}.
#'
#' @return A character vector of theme names (e.g. \code{"light-print"},
#'   \code{"dark-print"}, \code{"light-app"}, \code{"dark-app"}).
#' @name list_plot_themes
#' @export
NULL

#' Convert an mcpower_result to a long-format tibble.
#'
#' Each row corresponds to one tested effect in one scenario. Factor header
#' rows (baseline labels) are omitted. Columns: \code{test}, \code{scenario},
#' \code{power}, \code{ci_lo}, \code{ci_hi}.
#'
#' @param x   An mcpower_result object returned by \code{find_power()}.
#' @param ... Ignored.
#' @return A tibble with one row per non-header effect per scenario.
#' @importFrom tibble as_tibble
#' @export
as_tibble.mcpower_result <- function(x, ...) {
  meta <- attr(x, "mcpower_meta")
  corr <- !is.null(meta$correction) && meta$correction != "none"
  pkey <- if (corr) "power_corrected" else "power_uncorrected"
  ckey <- if (corr) "ci_corrected" else "ci_uncorrected"
  recs <- list()
  for (nm in names(.scenarios(x))) {
    inner <- .scenarios(x)[[nm]]
    # For find_power single N: [[1]] selects the single-N vector/list
    powers <- inner[[pkey]][[1]]  # numeric vector, one value per target
    cis    <- inner[[ckey]][[1]]  # list of numeric[2] c(lo, hi), one per target
    for (r in .build_rows(inner$target_indices, meta, inner$contrast_pairs %||% list())) {
      if (r$kind == "factor_header") next
      pos <- r$pos
      ci_pair <- cis[[pos]]  # numeric[2]: c(lo, hi)
      recs[[length(recs) + 1L]] <- data.frame(
        test     = r$label,
        scenario = nm,
        power    = powers[[pos]],
        ci_lo    = ci_pair[[1]],
        ci_hi    = ci_pair[[2]],
        stringsAsFactors = FALSE
      )
    }
  }
  tibble::as_tibble(do.call(rbind, recs))
}
