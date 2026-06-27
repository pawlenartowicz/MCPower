# Registry state from Rust parse output; reshapes parse_formula / parse_assignments JSON into Python-parity VariableRegistry fields — byte-for-byte match verified against a Python golden test.

.fromjson <- function(j) jsonlite::fromJSON(j, simplifyVector = FALSE)


# ---------------------------------------------------------------------------
# parsers.py mirror — _terms_to_fixed_string / _parse_equation /
# _parse_independent_variables / _re_to_legacy_dict
# ---------------------------------------------------------------------------

# Mirror of parsers.py _terms_to_fixed_string.
.terms_to_fixed_string <- function(terms) {
  parts <- character(0)
  for (t in terms) {
    if (t$kind == "main") {
      parts <- c(parts, t$name)
    } else if (t$kind == "interaction") {
      parts <- c(parts, paste(unlist(t$vars), collapse = ":"))
    }
  }
  paste(parts, collapse = "+")
}

# Mirror of parsers.py _re_to_legacy_dict.
.re_to_legacy_dict <- function(re_dict) {
  kind <- re_dict$kind
  if (kind == "intercept") {
    out <- list(type = "random_intercept", grouping_var = re_dict$group)
    parent <- re_dict$parent
    if (!is.null(parent)) out$parent_var <- parent
    return(out)
  }
  if (kind == "slope") {
    return(list(
      type = "random_slope",
      grouping_var = re_dict$group,
      slope_vars = as.list(unlist(re_dict$vars))
    ))
  }
  stop(sprintf("unknown random effect kind from engine: %s", sQuote(kind)))
}

# Mirror of parsers.py _parse_equation. Returns list(dependent,
# fixed_formula, random_effects). Tolerates `=` as a synonym for `~`.
.parse_equation <- function(equation) {
  rewritten <- if (!grepl("~", equation, fixed = TRUE) &&
                   grepl("=", equation, fixed = TRUE)) {
    sub("=", "~", equation, fixed = TRUE)
  } else {
    equation
  }
  out <- .fromjson(parse_formula(rewritten))
  fixed_formula <- .terms_to_fixed_string(out$terms)
  random_effects <- lapply(out$random_effects, .re_to_legacy_dict)
  list(
    dependent = out$dependent,
    fixed_formula = fixed_formula,
    random_effects = random_effects
  )
}

# Mirror of parsers.py _parse_independent_variables. Returns
# list(variables = <ordered names>, effects = <list of effect dicts>).
# The Python numbered keys (`variable_1`, `effect_1`, ...) are an
# implementation detail of the dict shape; only iteration order matters here,
# so we keep ordered lists.
.parse_independent_variables <- function(formula) {
  out <- .fromjson(parse_formula(paste0("explained_variable ~ ", formula)))
  predictors <- unlist(out$predictors)
  if (is.null(predictors)) predictors <- character(0)

  effects <- list()
  for (term in out$terms) {
    if (term$kind == "main") {
      name <- term$name
      effects[[length(effects) + 1L]] <- list(
        name = name,
        type = "main",
        column_index = match(name, predictors) - 1L
      )
    } else if (term$kind == "interaction") {
      vars_list <- unlist(term$vars)
      name <- paste(vars_list, collapse = ":")
      effects[[length(effects) + 1L]] <- list(
        name = name,
        type = "interaction",
        var_names = as.list(vars_list),
        column_indices = as.integer(match(vars_list, predictors) - 1L)
      )
    } else {
      stop(sprintf("unknown term kind from engine: %s", sQuote(term$kind)))
    }
  }
  list(variables = as.list(predictors), effects = effects)
}


# Variable-type value parsing (tuple syntax + bare-form defaults) is
# single-sourced in Rust (engine-spec-builder `parse_assignments`); routed
# through `.parse_assignment_kind(spec, "variable_type", ...)` below.

# Mirror of parsers.py _normalise_correlation_input — rewrite the bare
# `(a,b)=v` form to `corr(a,b)=v` so the Rust parser accepts it.
.normalise_correlation_input <- function(input_string) {
  parts <- split_assignments(input_string)
  rewritten <- character(0)
  for (part in parts) {
    stripped <- sub("^\\s+", "", part)
    if (startsWith(stripped, "(") && !startsWith(stripped, "corr")) {
      indent <- substr(part, 1L, nchar(part) - nchar(stripped))
      rewritten <- c(rewritten, paste0(indent, "corr", stripped))
    } else {
      rewritten <- c(rewritten, part)
    }
  }
  paste(rewritten, collapse = ", ")
}

# Mirror of parsers.py _AssignmentParser._parse for all three kinds (effect,
# correlation, variable_type) — each delegates to the Rust parser.
# Returns list(parsed, errors). For effects, `parsed` is a named list keyed by
# variable name; for variable_type, by name with the info list as value; for
# correlations, an unnamed list of list(pair = c(a, b) sorted, value = v)
# entries (R has no tuple keys).
.parse_assignment_kind <- function(input_string, parse_type, available_items) {
  if (parse_type == "correlation") {
    input_string <- .normalise_correlation_input(input_string)
  }
  known <- jsonlite::toJSON(list(
    predictors = as.list(available_items),
    interaction_terms = list()
  ), auto_unbox = TRUE)
  out <- tryCatch(
    .fromjson(parse_assignments(input_string, parse_type, known)),
    error = function(e) NULL
  )
  if (is.null(out)) {
    return(list(parsed = list(), errors = list("parse error")))
  }

  if (parse_type == "correlation") {
    parsed <- list()
    for (it in out$items) {
      pair <- sort(unlist(it$key$pair))
      parsed[[length(parsed) + 1L]] <- list(pair = pair, value = it$value$correlation)
    }
    return(list(parsed = parsed, errors = unlist(out$errors)))
  }

  if (parse_type == "variable_type") {
    # value$variable_type is the engine's legacy info list
    # list(type=, [proportion] | [n_levels, proportions]), consumed by set_variable_type.
    parsed <- list()
    for (it in out$items) {
      parsed[[it$key$name]] <- it$value$variable_type
    }
    return(list(parsed = parsed, errors = unlist(out$errors)))
  }

  # effect
  parsed <- list()
  for (it in out$items) {
    parsed[[it$key$name]] <- it$value$effect
  }
  list(parsed = parsed, errors = unlist(out$errors))
}


# ---------------------------------------------------------------------------
# variables.py _resolve_reference — resolve a by-VALUE factor reference
# to its canonical label string.  Called by set_variable_type when labels= is
# supplied.  Mirrors the Python helper exactly:
#   NULL + no labels → 1L (parser-path default)
#   NULL + labels    → labels[[1]]
#   character        → must be in labels (exact match)
#   numeric          → converted via .value_to_label, then checked
.resolve_reference <- function(reference, labels) {
  if (is.null(labels)) {
    # Parser path: integer labels 1..k; bare default becomes level 1L.
    return(if (!is.null(reference)) reference else 1L)
  }
  if (is.null(reference)) return(labels[[1]])
  # Resolve value → canonical label string.
  resolved <- if (is.character(reference)) {
    reference
  } else {
    .value_to_label(reference)
  }
  if (!(resolved %in% unlist(labels))) {
    stop(sprintf(
      "reference %s (resolved to %s) is not among the factor labels %s",
      deparse(reference), sQuote(resolved),
      paste(unlist(labels), collapse = ", ")),
      call. = FALSE)
  }
  resolved
}

# upload_data_utils.py value_to_label: integer-valued numeric → string without
# decimal ("6" not "6.0"); otherwise as.character.
.value_to_label <- function(v) {
  if (is.numeric(v) && !is.na(v) && v == as.integer(v)) {
    as.character(as.integer(v))
  } else {
    as.character(v)
  }
}

# RVariableRegistry — mirror of variables.py VariableRegistry
# ---------------------------------------------------------------------------

#' @importFrom R6 R6Class
RVariableRegistry <- R6::R6Class(
  "RVariableRegistry",
  active = list(
    # Property-style accessors mirroring VariableRegistry @property methods.
    # Accessed without parens (e.g. reg$non_factor_names), matching Python.

    # variables.py dependent
    dependent = function() private$dependent_,

    # variables.py predictor_names — sorted by column_index.
    predictor_names = function() {
      if (length(private$predictors) == 0L) return(character(0))
      idx <- vapply(private$predictors,
                    function(p) if (is.null(p$column_index)) 0L else p$column_index,
                    numeric(1))
      names(private$predictors)[order(idx)]
    },

    # variables.py effect_names
    effect_names = function() names(private$effects),

    # variables.py factor_names
    factor_names = function() {
      n <- names(self$`_factors`)
      if (is.null(n)) character(0) else n
    },

    # variables.py non_factor_names
    non_factor_names = function() {
      if (length(private$predictors) == 0L) return(character(0))
      keep <- vapply(private$predictors, function(p) {
        !isTRUE(p$is_factor) && !isTRUE(p$is_dummy) && p$var_type != "cluster_effect"
      }, logical(1))
      names(private$predictors)[keep]
    },

    # variables.py cluster_effect_names
    cluster_effect_names = function() {
      if (length(private$predictors) == 0L) return(character(0))
      keep <- vapply(private$predictors, function(p) p$var_type == "cluster_effect", logical(1))
      names(private$predictors)[keep]
    },

    # variables.py dummy_names
    dummy_names = function() {
      if (length(private$predictors) == 0L) return(character(0))
      keep <- vapply(private$predictors, function(p) isTRUE(p$is_dummy), logical(1))
      names(private$predictors)[keep]
    }
  ),

  public = list(
    # Public Python-parity fields read directly by downstream code.
    equation = NULL,
    `_factors` = NULL,
    `_random_effects_parsed` = NULL,

    # variables.py factor_levels — full ordered label list for factor
    # `name` (reference included), as character strings.  This is the port's
    # label store the engine's EffectSkeleton FactorLevel.level index resolves
    # against, and the same list sent to the contract as `levels`.
    factor_levels = function(name) {
      info <- self$`_factors`[[name]]
      level_labels <- info$level_labels
      if (is.null(level_labels) || length(level_labels) == 0L) {
        return(as.character(seq_len(as.integer(info$n_levels %||% 0L))))
      }
      as.character(unlist(level_labels))
    },

    initialize = function(equation) {
      self$equation <- trimws(equation)
      private$dependent_ <- ""
      private$predictors <- list()       # ordered: name -> PredictorVar list
      private$effects <- list()          # ordered: name -> Effect list
      self$`_factors` <- list()
      private$factor_dummies <- list()
      private$correlation_matrix <- NULL
      self$`_random_effects_parsed` <- list()

      eq <- .parse_equation(self$equation)
      private$dependent_ <- eq$dependent
      self$`_random_effects_parsed` <- eq$random_effects

      if (!nzchar(trimws(eq$fixed_formula))) {
        if (length(eq$random_effects) > 0L) {
          grouping_vars <- paste(
            vapply(eq$random_effects, function(re) re$grouping_var, character(1)),
            collapse = ", "
          )
          stop(sprintf(paste0(
            "Model has random effects (1|%s) but no fixed effects. ",
            "Power analysis requires at least one fixed effect to test. ",
            "Example: 'y ~ treatment + (1|cluster)'"), grouping_vars))
        }
        stop("Equation cannot be empty. Expected format: 'y = x1 + x2'")
      }

      ve <- .parse_independent_variables(eq$fixed_formula)
      if (length(ve$effects) == 0L) {
        stop("No predictor variables found in equation")
      }

      # Initialize predictors (column_index = insertion order, 0-based).
      for (name in ve$variables) {
        private$predictors[[name]] <- private$.predictor_var(
          name = name, column_index = length(private$predictors)
        )
      }

      # Initialize effects.
      for (effect_info in ve$effects) {
        name <- effect_info$name
        eff <- private$.effect(
          name = name,
          effect_type = effect_info$type,
          var_names = if (!is.null(effect_info$var_names)) effect_info$var_names else list(name)
        )
        if (effect_info$type == "main") {
          eff$column_index <- effect_info$column_index
        } else {
          eff$column_indices <- effect_info$column_indices
        }
        private$effects[[name]] <- eff
      }
    },

    # -- methods (mirror VariableRegistry methods) -------------------------

    # variables.py get_predictor
    get_predictor = function(name) private$predictors[[name]],

    # variables.py get_effect
    get_effect = function(name) private$effects[[name]],

    # variables.py set_variable_type
    # Public params renamed to `labels=` / `reference=` (Python parity).
    # Internal _factors fields stay `level_labels` / `reference_level`.
    set_variable_type = function(name, var_type, proportion = NULL, n_levels = NULL,
                                 proportions = NULL, labels = NULL,
                                 reference = NULL, sampled_proportions = NULL) {
      if (is.null(private$predictors[[name]])) stop(sprintf("Unknown variable: %s", name))
      pred <- private$predictors[[name]]
      pred$var_type <- var_type

      # Continuous synthetic kinds explicitly set by the user (including "normal")
      # are pinned — scenario distribution swaps leave the column alone.
      # binary and factor have their own semantics, never pinned this way.
      .continuous_synth_kinds <- c("normal", "right_skewed", "left_skewed", "high_kurtosis", "uniform")
      if (var_type %in% .continuous_synth_kinds) {
        pred$pinned <- TRUE
      }

      if (var_type == "binary") {
        pred$proportion <- if (!is.null(proportion)) proportion else 0.5
      } else if (var_type == "factor") {
        pred$is_factor <- TRUE
        pred$n_levels <- if (!is.null(n_levels)) n_levels else 2L
        pred$proportions <- proportions
        if (!is.null(labels)) pred$level_labels <- labels

        # `reference` is by VALUE: resolve it to a canonical label string.
        ref <- .resolve_reference(reference, labels)
        self$`_factors`[[name]] <- list(
          n_levels = pred$n_levels,
          proportions = pred$proportions,
          level_labels = labels,
          reference_level = ref,
          sampled_proportions = sampled_proportions  # NULL | TRUE | FALSE
        )
      }
      private$predictors[[name]] <- pred
      invisible(NULL)
    },

    # variables.py set_effect_size
    set_effect_size = function(name, value) {
      if (is.null(private$effects[[name]])) stop(sprintf("Unknown effect: %s", name))
      private$effects[[name]]$effect_size <- value
      invisible(NULL)
    },

    # variables.py expand_factors
    expand_factors = function() {
      if (length(self$`_factors`) == 0L) return(invisible(NULL))

      original_effects <- private$effects
      new_predictors <- list()
      new_effects <- list()

      # Keep non-factor predictors, re-index.
      col_idx <- 0L
      for (name in names(private$predictors)) {
        pred <- private$predictors[[name]]
        if (!isTRUE(pred$is_factor)) {
          pred$column_index <- col_idx
          new_predictors[[name]] <- pred
          col_idx <- col_idx + 1L
        }
      }

      # Keep non-factor effects.
      for (name in names(private$effects)) {
        eff <- private$effects[[name]]
        has_factor <- any(vapply(eff$var_names, function(vn) vn %in% names(self$`_factors`),
                                 logical(1)))
        if (!has_factor) new_effects[[name]] <- eff
      }

      # Create dummies + main effects per factor.
      for (factor_name in names(self$`_factors`)) {
        factor_info <- self$`_factors`[[factor_name]]
        n_levels <- factor_info$n_levels
        level_labels <- factor_info$level_labels
        reference_level <- if (!is.null(factor_info$reference_level)) factor_info$reference_level else 1L

        if (!is.null(level_labels)) {
          non_ref <- Filter(function(lb) lb != as.character(reference_level),
                            unlist(level_labels))
        } else {
          non_ref <- seq.int(2L, n_levels)
        }

        for (level in non_ref) {
          dummy_name <- sprintf("%s[%s]", factor_name, level)
          dummy_pred <- private$.predictor_var(
            name = dummy_name, var_type = "factor_dummy", is_dummy = TRUE,
            factor_source = factor_name, factor_level = level,
            column_index = col_idx,
            level_labels = if (!is.null(level_labels)) level_labels else NULL
          )
          new_predictors[[dummy_name]] <- dummy_pred

          dummy_eff <- private$.effect(
            name = dummy_name, effect_type = "main", var_names = list(dummy_name)
          )
          dummy_eff$column_index <- col_idx
          dummy_eff$factor_source <- factor_name
          dummy_eff$factor_level <- level
          new_effects[[dummy_name]] <- dummy_eff

          private$factor_dummies[[dummy_name]] <- list(
            factor_name = factor_name, level = level
          )
          col_idx <- col_idx + 1L
        }
      }

      # Expand factor interactions — Cartesian product of non-reference levels.
      for (name in names(original_effects)) {
        eff <- original_effects[[name]]
        if (eff$effect_type == "interaction") {
          factor_vars <- Filter(function(vn) vn %in% names(self$`_factors`),
                                unlist(eff$var_names))
          if (length(factor_vars) > 0L) {
            level_options <- list()
            for (vn in unlist(eff$var_names)) {
              if (vn %in% names(self$`_factors`)) {
                fi <- self$`_factors`[[vn]]
                n_levels <- fi$n_levels
                level_labels <- fi$level_labels
                reference_level <- if (!is.null(fi$reference_level)) fi$reference_level else 1L
                if (!is.null(level_labels)) {
                  opts <- vapply(
                    Filter(function(lb) lb != as.character(reference_level), unlist(level_labels)),
                    function(lb) sprintf("%s[%s]", vn, lb), character(1)
                  )
                } else {
                  opts <- vapply(seq.int(2L, n_levels),
                                 function(lvl) sprintf("%s[%s]", vn, lvl), character(1))
                }
                level_options[[length(level_options) + 1L]] <- opts
              } else {
                level_options[[length(level_options) + 1L]] <- vn
              }
            }
            # Cartesian product preserving component order (expand.grid
            # varies the first factor fastest; iterate rows to match Python's
            # itertools.product which varies the LAST fastest -> use rev order
            # then reorder columns). We instead build the product manually to
            # match itertools.product ordering exactly.
            combos <- private$.cartesian(level_options)
            for (combo in combos) {
              new_var_names <- as.list(combo)
              new_interaction_name <- paste(combo, collapse = ":")
              new_eff <- private$.effect(
                name = new_interaction_name, effect_type = "interaction",
                var_names = new_var_names
              )
              new_eff$column_indices <- integer(0)
              new_effects[[new_interaction_name]] <- new_eff
            }
          }
        }
      }

      private$predictors <- new_predictors
      private$effects <- new_effects
      private$update_effect_indices()
      invisible(NULL)
    },

    # variables.py get_effect_sizes — in effect order.
    get_effect_sizes = function() {
      vapply(private$effects, function(e) e$effect_size, numeric(1), USE.NAMES = FALSE)
    },

    # variables.py get_correlation_matrix
    get_correlation_matrix = function() private$correlation_matrix,

    # variables.py set_correlation_matrix
    set_correlation_matrix = function(matrix) {
      private$correlation_matrix <- matrix
      invisible(NULL)
    },

    # variables.py set_correlation
    set_correlation = function(var1, var2, value) {
      non_factor <- self$non_factor_names
      if (!(var1 %in% non_factor) || !(var2 %in% non_factor)) {
        stop("Can only set correlations between non-factor variables")
      }
      if (is.null(private$correlation_matrix)) {
        private$correlation_matrix <- diag(length(non_factor))
      }
      idx1 <- match(var1, non_factor)
      idx2 <- match(var2, non_factor)
      private$correlation_matrix[idx1, idx2] <- value
      private$correlation_matrix[idx2, idx1] <- value
      invisible(NULL)
    },

    # -- public setters mirroring model.py _apply() driving the registry ---

    # model.py — variable types branch of _apply().
    set_variable_types = function(spec) {
      res <- .parse_assignment_kind(spec, "variable_type", self$predictor_names)
      if (length(res$errors) > 0L) {
        stop(paste0("Variable type validation failed:\n",
                    paste(sprintf("- %s", res$errors), collapse = "\n")))
      }
      for (name in names(res$parsed)) {
        info <- res$parsed[[name]]
        kwargs <- info[setdiff(names(info), "type")]
        do.call(self$set_variable_type,
                c(list(name = name, var_type = info$type), kwargs))
      }
      invisible(NULL)
    },

    # model.py — effects branch of _apply() (validation + set).
    set_effects = function(spec) {
      assignments <- split_assignments(spec)
      effect_names <- self$effect_names
      errors <- character(0)
      for (assignment in assignments) {
        if (!grepl("=", assignment, fixed = TRUE)) {
          errors <- c(errors, sprintf("Invalid format '%s'. Expected: 'name=value'", assignment))
          next
        }
        nv <- regmatches(assignment, regexpr("=", assignment, fixed = TRUE),
                         invert = TRUE)[[1]]
        name <- trimws(nv[1])
        value_str <- trimws(paste(nv[-1], collapse = "="))
        value <- suppressWarnings(as.numeric(value_str))
        if (is.na(value)) {
          errors <- c(errors, sprintf("Invalid value '%s' for '%s'", value_str, name))
          next
        }
        if (!(name %in% effect_names)) {
          errors <- c(errors, sprintf("Effect '%s' not found. Available: %s",
                                      name, paste(effect_names, collapse = ", ")))
          next
        }
        self$set_effect_size(name, value)
      }
      if (length(errors) > 0L) {
        stop(paste0("Effect validation failed:\n",
                    paste(sprintf("- %s", errors), collapse = "\n")))
      }
      invisible(NULL)
    },

    # model.py — correlation branch of _apply() (string path).
    set_correlation_spec = function(spec) {
      non_factor <- self$non_factor_names
      n <- length(non_factor)
      if (is.null(private$correlation_matrix)) {
        private$correlation_matrix <- diag(n)
      }
      if (nzchar(trimws(spec))) {
        if (n < 2L) stop("Need at least 2 non-factor variables for correlations")
        res <- .parse_assignment_kind(spec, "correlation", non_factor)
        if (length(res$errors) > 0L) {
          stop(paste0("Error setting correlations:\n",
                      paste(sprintf("- %s", res$errors), collapse = "\n")))
        }
        for (entry in res$parsed) {
          self$set_correlation(entry$pair[1], entry$pair[2], entry$value)
        }
      }
      invisible(NULL)
    }
  ),

  private = list(
    dependent_ = NULL,
    predictors = NULL,
    effects = NULL,
    factor_dummies = NULL,
    correlation_matrix = NULL,

    # Mirror of variables.py PredictorVar dataclass defaults.
    .predictor_var = function(name, var_type = "normal", proportion = 0.5,
                              n_levels = NULL, proportions = NULL,
                              is_factor = FALSE, is_dummy = FALSE,
                              factor_source = NULL, factor_level = NULL,
                              column_index = NULL, level_labels = NULL,
                              pinned = FALSE) {
      list(name = name, var_type = var_type, proportion = proportion,
           n_levels = n_levels, proportions = proportions,
           is_factor = is_factor, is_dummy = is_dummy,
           factor_source = factor_source, factor_level = factor_level,
           column_index = column_index, level_labels = level_labels,
           pinned = pinned)
    },

    # Mirror of variables.py Effect dataclass defaults.
    .effect = function(name, effect_type, effect_size = 0.0,
                       var_names = list(), column_index = NULL,
                       column_indices = integer(0), factor_source = NULL,
                       factor_level = NULL) {
      list(name = name, effect_type = effect_type, effect_size = effect_size,
           var_names = var_names, column_index = column_index,
           column_indices = column_indices, factor_source = factor_source,
           factor_level = factor_level)
    },

    # Cartesian product matching itertools.product ordering (last list varies
    # fastest). Each `lists` element is a character vector of options.
    .cartesian = function(lists) {
      result <- list(character(0))
      for (lst in lists) {
        new_result <- list()
        for (prefix in result) {
          for (item in lst) {
            new_result[[length(new_result) + 1L]] <- c(prefix, item)
          }
        }
        result <- new_result
      }
      result
    },

    # variables.py _update_effect_indices.
    update_effect_indices = function() {
      predictor_order <- self$predictor_names
      for (name in names(private$effects)) {
        eff <- private$effects[[name]]
        if (eff$effect_type == "main") {
          if (eff$name %in% predictor_order) {
            eff$column_index <- match(eff$name, predictor_order) - 1L
          }
        } else {
          vns <- unlist(eff$var_names)
          present <- vns[vns %in% predictor_order]
          eff$column_indices <- as.integer(match(present, predictor_order) - 1L)
        }
        private$effects[[name]] <- eff
      }
    }
  )
)


# ---------------------------------------------------------------------------
# model.py mirror — _scenario_dict (nested inner fn, model.py)
# ---------------------------------------------------------------------------

# Mirror of model.py _encode_dist_list. Code table from `.dist_codes()` (zzz.R).
.encode_dist_list <- function(names, scenario_name) {
  dist_codes <- .dist_codes()
  out <- integer(0)
  for (n in names) {
    code <- dist_codes[[n]]
    if (is.null(code)) {
      stop(sprintf(
        "scenario %s: unknown distribution name %s; valid: %s",
        sQuote(scenario_name), sQuote(n),
        paste(sort(names(dist_codes)), collapse = ", ")
      ))
    }
    out <- c(out, code)
  }
  out
}

# Mirror of model.py _encode_residual_list. All five canonical residual names
# are valid in scenario residual_dists pools. Code table from `.residual_codes()`.
.encode_residual_list <- function(names, scenario_name) {
  residual_codes <- .residual_codes()
  out <- integer(0)
  for (n in names) {
    code <- residual_codes[[n]]
    if (is.null(code)) {
      stop(sprintf(
        "scenario %s: unknown residual distribution %s; valid: %s",
        sQuote(scenario_name), sQuote(n),
        paste(sort(names(residual_codes)), collapse = ", ")
      ))
    }
    out <- c(out, code)
  }
  out
}

# Mirror of model.py _scenario_dict. `scenario_configs` defaults to
# `.scenario_defaults()`; callers may pass merged per-model configs.
# Integer code lists are wrapped in I() so length-1 lists still serialize as
# JSON arrays under auto_unbox. The three RE knobs (random_effect_dist/df,
# icc_noise_sd) are always emitted here; .to_linear_spec_list zeros them for
# non-LME families so the spec-builder produces lme: None for those calls.
.scenario_dict <- function(name, scenario_configs = .scenario_defaults()) {
  if (!(name %in% names(scenario_configs))) {
    stop(sprintf("Unknown scenario %s; configured: %s",
                 sQuote(name), paste(sort(names(scenario_configs)), collapse = ", ")))
  }
  cfg <- scenario_configs[[name]]

  getf <- function(key, default) {
    v <- cfg[[key]]
    if (is.null(v)) default else v
  }

  new_dists <- .encode_dist_list(unlist(getf("new_distributions", list())), name)
  residual_dists <- .encode_residual_list(unlist(getf("residual_dists", list())), name)

  # random_effect_dist: single RE dist name -> integer code via the RE-dist
  # table (normal/heavy_tailed vocabulary, NOT the residual-pool space).
  re_dist_name <- getf("random_effect_dist", "normal")
  re_dist_code <- {
    re_codes <- .re_dist_codes()
    v <- re_codes[[re_dist_name]]
    if (is.null(v)) stop(sprintf(
      "scenario %s: unknown random_effect_dist %s; valid: %s",
      sQuote(name), sQuote(re_dist_name),
      paste(sort(names(re_codes)), collapse = ", ")
    ))
    as.integer(v)
  }

  list(
    name = name,
    heterogeneity = as.numeric(getf("heterogeneity", 0.0)),
    heteroskedasticity_ratio = as.numeric(getf("heteroskedasticity_ratio", 1.0)),
    correlation_noise_sd = as.numeric(getf("correlation_noise_sd", 0.0)),
    distribution_change_prob = as.numeric(getf("distribution_change_prob", 0.0)),
    new_distributions = I(as.integer(new_dists)),
    residual_change_prob = as.numeric(getf("residual_change_prob", 0.0)),
    residual_dists = I(as.integer(residual_dists)),
    residual_df = as.numeric(getf("residual_df", 0.0)),
    sampled_factor_proportions = as.logical(getf("sampled_factor_proportions", FALSE)),
    random_effect_dist = re_dist_code,
    random_effect_df = as.numeric(getf("random_effect_df", 0.0)),
    icc_noise_sd = as.numeric(getf("icc_noise_sd", 0.0))
  )
}


# ---------------------------------------------------------------------------
# model.py mirror — _resolve_tests (model.py)
# ---------------------------------------------------------------------------
# Parses a target_test DSL string into list(targets, contrast_pairs,
# report_overall). Mirrors the Python tokenizer/keyword-expansion/validation.

# Whether the omnibus / overall test is defined for this fit. The overall test
# is the OLS F-test / GLM likelihood-ratio test, exposed for an *unclustered*
# OLS or GLM fit only. Mixed-effects fits suppress it: LME (estimator "mle") and
# clustered-logistic GLMM (estimator "glm" with a (1|group) random effect) have
# no exposed omnibus (the engine path for them is parked, not wired up). An OLS
# fit on clustered data (family='lme', estimator='ols') keeps the F-test — the
# naive omnibus matches the naive fit. Mirrors Python overall_test_available().
.overall_test_available <- function(estimator, reg) {
  if (identical(estimator, "ols")) return(TRUE)
  if (identical(estimator, "glm")) return(length(reg$`_random_effects_parsed`) == 0L)
  FALSE  # mle (LME)
}

.resolve_tests <- function(reg, target_test, overall_available = TRUE) {
  cluster_effects <- reg$cluster_effect_names

  tokens <- trimws(strsplit(target_test, ",", fixed = TRUE)[[1]])
  tokens <- tokens[nzchar(tokens)]

  # "all-contrasts" and "all-posthoc" are synonymous (posthoc keyword).
  # They produce a posthoc_factors list — NOT expanded into contrast_pairs strings.
  posthoc_keywords <- c("all-contrasts", "all-posthoc")

  keywords <- character(0)
  exclusions <- character(0)
  explicit_tests <- character(0)
  posthoc_keyword_seen <- FALSE
  for (tok in tokens) {
    tok_lower <- tolower(tok)
    if (tok_lower == "all") {
      keywords <- c(keywords, tok_lower)
    } else if (tok_lower %in% posthoc_keywords) {
      posthoc_keyword_seen <- TRUE
    } else if (startsWith(tok, "-")) {
      exclusions <- c(exclusions, trimws(substring(tok, 2L)))
    } else {
      explicit_tests <- c(explicit_tests, tok)
    }
  }

  keyword_expansion <- character(0)
  if ("all" %in% keywords) {
    fixed_effects <- setdiff(reg$effect_names, cluster_effects)
    # The omnibus rides along with "all" only where it is defined (OLS /
    # unclustered GLM). For mixed-effects fits "all" means every fixed-effect β
    # with no omnibus — silently dropped, not an error.
    omnibus <- if (overall_available) "overall" else character(0)
    keyword_expansion <- c(keyword_expansion, omnibus, fixed_effects)
  }

  # Resolve posthoc keyword: collect factor names (factor-only, any n_levels > 0).
  posthoc_factors <- character(0)
  if (posthoc_keyword_seen) {
    for (factor_name in reg$factor_names) {
      factor_info <- reg$`_factors`[[factor_name]]
      if (!is.null(factor_info$n_levels) && factor_info$n_levels > 0L) {
        posthoc_factors <- c(posthoc_factors, factor_name)
      }
    }
    if (length(posthoc_factors) == 0L && length(keyword_expansion) == 0L &&
        length(explicit_tests) == 0L) {
      stop(paste0("'all-contrasts'/'all-posthoc' was specified but the model has no factor ",
                  "variables. Post-hoc contrasts require at least one factor."))
    }
  }

  expanded <- c(keyword_expansion, explicit_tests)

  dep_var_name <- reg$dependent
  alias_set <- c(dep_var_name, "y")
  expanded <- ifelse(expanded %in% alias_set, "overall", expanded)
  exclusions <- ifelse(exclusions %in% alias_set, "overall", exclusions)

  for (excl in exclusions) {
    if (!(excl %in% expanded)) {
      stop(sprintf(paste0(
        "Exclusion '-%s' does not match any test in the expanded set. ",
        "Available: %s"), excl, paste(expanded, collapse = ", ")))
    }
    expanded <- expanded[-match(excl, expanded)]
  }

  if (length(expanded) == 0L && length(posthoc_factors) == 0L) {
    stop("All tests were excluded - nothing left to analyse.")
  }

  dup <- unique(expanded[duplicated(expanded)])
  if (length(dup) > 0L) {
    stop(sprintf(paste0(
      "Duplicate target test(s): %s. Each test may appear only once. If using ",
      "keyword expansion (e.g. 'all'), do not also list tests that are already ",
      "included."), paste(dup, collapse = ", ")))
  }

  vs_pattern <- "^([A-Za-z_][[:alnum:]_]*)\\[([^]]+)\\]\\s+vs\\s+([A-Za-z_][[:alnum:]_]*)\\[([^]]+)\\]$"

  targets <- character(0)
  contrast_pairs <- list()
  report_overall <- FALSE
  valid_effect_names <- setdiff(reg$effect_names, cluster_effects)

  for (t in expanded) {
    if (t == "overall") {
      if (!overall_available) {
        stop(paste0("The overall/omnibus test is not available for ",
                    "mixed-effects models (LME / clustered GLMM). Request ",
                    "specific fixed effects instead, e.g. target_test='x1'."))
      }
      report_overall <- TRUE
      next
    }
    m <- regmatches(t, regexec(vs_pattern, t))[[1]]
    if (length(m) == 5L) {
      factor_a <- m[2]; level_a_str <- m[3]; factor_b <- m[4]; level_b_str <- m[5]
      if (factor_a != factor_b) {
        stop(sprintf(paste0(
          "Post-hoc comparison must be between levels of the same factor, ",
          "got '%s' vs '%s'"), factor_a, factor_b))
      }
      factor_name <- factor_a
      if (!(factor_name %in% names(reg$`_factors`))) {
        avail <- names(reg$`_factors`)
        stop(sprintf("Factor '%s' not found. Available: %s",
                     factor_name, if (length(avail)) paste(avail, collapse = ", ") else "none"))
      }
      factor_info <- reg$`_factors`[[factor_name]]
      level_labels <- factor_info$level_labels
      if (!is.null(level_labels) && length(level_labels) > 0L) {
        avail <- as.character(unlist(level_labels))
        if (!(level_a_str %in% avail)) {
          stop(sprintf("Level '%s' not found for factor '%s'. Available: %s",
                       level_a_str, factor_name, paste(avail, collapse = ", ")))
        }
        if (!(level_b_str %in% avail)) {
          stop(sprintf("Level '%s' not found for factor '%s'. Available: %s",
                       level_b_str, factor_name, paste(avail, collapse = ", ")))
        }
      } else {
        n_levels <- factor_info$n_levels
        lvl_a <- suppressWarnings(as.integer(level_a_str))
        lvl_b <- suppressWarnings(as.integer(level_b_str))
        if (is.na(lvl_a) || is.na(lvl_b) ||
            grepl("[^0-9+-]", level_a_str) || grepl("[^0-9+-]", level_b_str)) {
          stop(sprintf(paste0(
            "Factor '%s' has no named levels; use integer indices 1..%d"),
            factor_name, n_levels))
        }
        if (lvl_a < 1L || lvl_a > n_levels) {
          stop(sprintf("Level %d out of range for factor '%s' (valid: 1 to %d)",
                       lvl_a, factor_name, n_levels))
        }
        if (lvl_b < 1L || lvl_b > n_levels) {
          stop(sprintf("Level %d out of range for factor '%s' (valid: 1 to %d)",
                       lvl_b, factor_name, n_levels))
        }
        if (lvl_a == lvl_b) {
          stop(sprintf("Cannot compare a level to itself: %s", t))
        }
      }
      positive_name <- sprintf("%s[%s]", factor_name, level_a_str)
      negative_name <- sprintf("%s[%s]", factor_name, level_b_str)
      contrast_pairs[[length(contrast_pairs) + 1L]] <- c(positive_name, negative_name)
    } else {
      if (!(t %in% valid_effect_names)) {
        stop(sprintf("Unknown test name '%s'. Available effects: %s",
                     t, paste(sort(valid_effect_names), collapse = ", ")))
      }
      targets <- c(targets, t)
    }
  }

  list(targets = targets, contrast_pairs = contrast_pairs, report_overall = report_overall,
       posthoc_factors = posthoc_factors)
}


# ---------------------------------------------------------------------------
# model.py mirror — _to_linear_spec_dict (model.py)
# ---------------------------------------------------------------------------
# Config that lives on `self` in Python is passed explicitly here (the registry
# carries only the predictor/effect/correlation state). The result is a plain R
# list ready for jsonlite::toJSON(payload, auto_unbox = TRUE, null = "null", digits = NA)
# (the production serialization in MCPower's build_contract_bytes).
.to_linear_spec_list <- function(reg, scenario_names, alpha, correction,
                                  wald_se = NULL, target_test, heteroskedasticity,
                                  residual_name, residual_pinned = FALSE, max_failed,
                                  test_formula, scenario_configs = .scenario_defaults(),
                                  pending_data = NULL, cluster_level_vars = NULL,
                                  estimator = "ols") {
  # Predictors — non-factors first (formula order), then factors. (model.py _to_linear_spec_dict)
  predictors <- list()
  non_factor_names <- reg$non_factor_names
  for (name in non_factor_names) {
    pred <- reg$get_predictor(name)
    if (is.null(pred)) stop(sprintf("missing predictor metadata for %s", sQuote(name)))
    var_type <- pred$var_type
    kind_name <- if (is.null(.dist_codes()[[var_type]])) NULL else var_type
    if (is.null(kind_name)) {
      stop(sprintf(paste0(
        "predictor %s has var_type %s, which is not supported by the Rust ",
        "spec builder"), sQuote(name), sQuote(var_type)))
    }
    entry <- list(name = name, kind = kind_name)
    if (kind_name == "binary") {
      entry$proportion <- as.numeric(pred$proportion)
    }
    # Emit pinned=TRUE when the user explicitly chose this continuous synthetic
    # distribution (incl. explicit "normal") so scenario swaps leave it alone.
    # Omit when FALSE (the default) to keep the JSON minimal.
    if (isTRUE(pred$pinned)) {
      entry$pinned <- TRUE
    }
    predictors[[length(predictors) + 1L]] <- entry
  }

  for (name in reg$factor_names) {
    info <- reg$`_factors`[[name]]
    n_levels <- as.integer(info$n_levels)
    proportions <- info$proportions
    if (is.null(proportions)) {
      proportions <- as.list(rep(1.0 / n_levels, n_levels))
    }
    level_labels <- info$level_labels
    reference_level <- info$reference_level
    if (is.null(level_labels)) {
      # Parser-driven path: integer labels 1..n_levels, reference=1.
      levels <- as.character(seq_len(n_levels))
      reference <- if (!is.null(reference_level)) as.character(reference_level) else "1"
    } else {
      levels <- as.character(unlist(level_labels))
      reference <- if (!is.null(reference_level)) as.character(reference_level) else levels[1]
    }
    entry <- list(
      name = name,
      kind = "factor",
      levels = I(levels),
      proportions = I(as.numeric(unlist(proportions))),
      reference = reference
    )
    sampled <- info$sampled_proportions
    if (!is.null(sampled)) {
      # Omit when NULL (inherit); jsonlite auto_unbox emits a scalar bool.
      entry$sampled_proportions <- as.logical(sampled)
    }
    predictors[[length(predictors) + 1L]] <- entry
  }

  # Effects — skip cluster effects (random, not tested). (model.py _to_linear_spec_dict)
  cluster_effect_names <- reg$cluster_effect_names
  effects <- list()
  for (name in reg$effect_names) {
    if (name %in% cluster_effect_names) next
    eff <- reg$get_effect(name)
    effects[[length(effects) + 1L]] <- list(name = name, size = as.numeric(eff$effect_size))
  }

  # Correlations — non-zero off-diagonal only. (model.py _to_linear_spec_dict)
  correlations <- list()
  corr_matrix <- reg$get_correlation_matrix()
  if (!is.null(corr_matrix)) {
    n_nf <- length(non_factor_names)
    if (n_nf >= 2L) {
      for (i in seq_len(n_nf - 1L)) {
        for (j in seq.int(i + 1L, n_nf)) {
          value <- as.numeric(corr_matrix[i, j])
          if (value != 0.0) {
            correlations[[length(correlations) + 1L]] <- list(
              a = non_factor_names[i], b = non_factor_names[j], value = value
            )
          }
        }
      }
    }
  }

  # Correction (model.py _to_linear_spec_dict).
  correction_wire <- .correction_for_rust(correction)

  # wald_se: normalise and validate; only affects the GLMM estimator, but the
  # spec field is always emitted so the engine can apply it uniformly.
  wald_se_wire <- .wald_se_for_rust(wald_se)

  # Scenarios (model.py _to_linear_spec_dict).
  scenarios <- lapply(scenario_names, function(nm) .scenario_dict(nm, scenario_configs))
  # RE knobs (random_effect_dist/df, icc_noise_sd) are LME-only: the engine
  # contract rejects lme: Some(...) when the estimator is not Mle.  Zero them
  # for non-LME families so the spec-builder produces lme: None for those calls.
  if (length(reg$`_random_effects_parsed`) == 0L) {
    for (i in seq_along(scenarios)) {
      scenarios[[i]]$random_effect_dist <- 0L
      scenarios[[i]]$random_effect_df   <- 0.0
      scenarios[[i]]$icc_noise_sd       <- 0.0
    }
  }

  # Formula — strip `(1|group)` when random effects present. (model.py _to_linear_spec_dict)
  formula_for_builder <- reg$equation
  if (length(reg$`_random_effects_parsed`) > 0L) {
    eq <- .parse_equation(reg$equation)
    formula_for_builder <- paste0(eq$dependent, " ~ ", eq$fixed_formula)
  }

  # targets / contrast_pairs / report_overall. (model.py _to_linear_spec_dict)
  # The omnibus is added only where it is defined (OLS / unclustered GLM);
  # mixed-effects fits report the marginals without it (see
  # .overall_test_available).
  overall_available <- .overall_test_available(estimator, reg)
  if (!is.null(target_test)) {
    tests_resolved <- .resolve_tests(reg, target_test, overall_available = overall_available)
  } else {
    tests_resolved <- NULL
  }

  if (is.null(tests_resolved)) {
    wire_targets <- list("overall")
    wire_contrast_pairs <- list()
    wire_report_overall <- overall_available
    wire_posthoc_factors <- character(0)
  } else {
    user_targets <- tests_resolved$targets
    if (length(user_targets) > 0L) {
      wire_targets <- as.list(user_targets)
    } else {
      wire_targets <- list()
    }
    wire_contrast_pairs <- lapply(tests_resolved$contrast_pairs, function(pair) as.list(pair))
    wire_report_overall <- tests_resolved$report_overall
    wire_posthoc_factors <- tests_resolved$posthoc_factors %||% character(0)
  }

  posthoc_requests <- lapply(wire_posthoc_factors, function(f) list(factor = f))

  # Build the residual block: distribution + pinned flag (no df — df is
  # scenario-only via residual_df in ScenarioInput).
  residual_block <- list(distribution = residual_name)
  if (isTRUE(residual_pinned)) {
    residual_block$pinned <- TRUE
  }

  payload <- list(
    formula = formula_for_builder,
    predictors = predictors,
    effects = effects,
    correlations = correlations,
    alpha = as.numeric(alpha),
    correction = correction_wire,
    wald_se = wald_se_wire,
    targets = wire_targets,
    report_overall = wire_report_overall,
    contrast_pairs = wire_contrast_pairs,
    posthoc_requests = posthoc_requests,
    heteroskedasticity = heteroskedasticity,
    residual = residual_block,
    max_failed_fraction = as.numeric(max_failed),
    scenarios = scenarios
  )
  if (!is.null(test_formula)) {
    payload$test_formula <- test_formula
  }
  # Emit cluster_level_vars only when non-NULL (engine ignores absent key).
  # I() so a length-1 character vector still serialises as a JSON array
  # (engine GenerationSpec.cluster_level_columns is Vec<_>), not a bare string.
  if (!is.null(cluster_level_vars)) {
    payload$cluster_level_vars <- I(cluster_level_vars)
  }

  # Upload block: emit when pending_data is set. (model.py _to_linear_spec_dict)
  if (!is.null(pending_data)) {
    upload_columns <- list()
    for (entry in pending_data$columns_typed) {
      col_name   <- entry$name
      col_type   <- entry$col_type
      raw_vals   <- unlist(entry$raw_vals)
      col_labels <- entry$col_labels

      if (identical(col_type, "factor")) {
        # Encode raw values as integer level codes 0..k-1 using .value_to_label.
        label_to_code <- setNames(seq_along(col_labels) - 1L, col_labels)
        values_encoded <- vapply(raw_vals, function(v) {
          lbl <- .value_to_label(v)
          code <- label_to_code[lbl]            # single bracket → NA if absent
          if (is.na(code)) {
            stop(sprintf("factor column %s: value %s was not seen during type detection; re-upload the full dataset",
                         sQuote(col_name), sQuote(lbl)), call. = FALSE)
          }
          as.numeric(code)
        }, numeric(1))
        upload_columns[[length(upload_columns) + 1L]] <- list(
          name     = col_name,
          col_type = "factor",
          values   = as.list(unname(values_encoded)),  # unname: Rust expects array not object
          labels   = as.list(col_labels)
        )
      } else {
        # binary or continuous: raw numeric values
        upload_columns[[length(upload_columns) + 1L]] <- list(
          name     = col_name,
          col_type = col_type,
          values   = as.list(as.numeric(raw_vals)),
          labels   = list()
        )
      }
    }
    payload$upload <- list(
      mode   = pending_data$mode,
      n_rows = as.integer(pending_data$uploaded_n),
      columns = upload_columns
    )
  }

  payload
}


# ---------------------------------------------------------------------------
# model.py mirror — _encode_outcome_and_clusters (model.py) +
# _build_cluster_spec_dict (model.py)
# ---------------------------------------------------------------------------
# `pending_clusters` is a (possibly empty) named list keyed by grouping var,
# each entry list(icc = <num>, n_clusters = <int>). Returns the four wire
# values matching `_engine.build_contract_from_spec`'s signature; clusters_json
# is a JSON string serialized with jsonlite to match Python's json.dumps.
.encode_outcome_and_clusters <- function(family, estimator, intercept, pending_clusters) {
  outcome_kind <- if (identical(family, "logit")) "binary" else "continuous"

  if (length(pending_clusters) > 0L) {
    grouping_var <- names(pending_clusters)[1]
    cfg <- pending_clusters[[grouping_var]]
    icc <- as.numeric(cfg$icc)
    n_clusters <- cfg$n_clusters
    if (is.null(n_clusters)) {
      stop(paste0("cluster_size-only LME specs need a runtime sample_size; ",
                  "call find_power / find_sample_size or set n_clusters directly"))
    }
    denom <- 1.0 - icc
    # Latent-scale logistic conversion (Snijders & Bosker): binary outcome
    # multiplies the Gaussian ICC->tau^2 by pi^2/3 (logistic variance ~= 3.29).
    # Gaussian path (lme/continuous) uses tau^2 = ICC/(1-ICC) unchanged.
    # Mirrors the Python port's _build_cluster_spec_dict latent branch — change together.
    tau_raw     <- if (denom > 0) icc / denom else 0.0
    tau_squared <- if (identical(outcome_kind, "binary")) tau_raw * (pi^2 / 3) else tau_raw
    cluster_spec <- list(
      sizing      = list(FixedClusters = list(n_clusters = as.integer(n_clusters))),
      tau_squared = as.numeric(tau_squared)
    )

    # Build extra_groupings from the 2nd..Nth entries in pending_clusters.
    # Each additional set_cluster call contributes one entry in the Crossed or
    # NestedWithin relation shape (engine GroupingRelation; mirrors the Python
    # _build_cluster_spec_dict). Gaussian ICC -> tau^2 = ICC/(1-ICC), sigma^2=1.
    extra_from_public <- list()
    gvars <- names(pending_clusters)
    if (length(gvars) > 1L) {
      for (gv in gvars[-1L]) {
        gcfg  <- pending_clusters[[gv]]
        gicc  <- as.numeric(gcfg$icc %||% 0.0)
        gdenom <- 1.0 - gicc
        # Same latent-scale branch as the primary grouping — binary outcome
        # requires the pi^2/3 factor; mirrors primary conversion above.
        gtau_raw <- if (gdenom > 0) gicc / gdenom else 0.0
        gtau     <- if (identical(outcome_kind, "binary")) gtau_raw * (pi^2 / 3) else gtau_raw
        # Relation: NestedWithin when n_per_parent supplied, Crossed otherwise.
        if (!is.null(gcfg$n_per_parent)) {
          relation <- list(NestedWithin = list(n_per_parent = as.integer(gcfg$n_per_parent)))
        } else {
          gn <- gcfg$n_clusters
          if (is.null(gn)) stop(sprintf(
            "set_cluster(%s): n_clusters required for extra groupings", gv), call. = FALSE)
          relation <- list(Crossed = list(n_clusters = as.integer(gn)))
        }
        extra_from_public[[length(extra_from_public) + 1L]] <- list(
          relation    = relation,
          tau_squared = as.numeric(gtau)
        )
      }
    }

    # Merge public-built extra_groupings with any debug-seam override.
    # Debug seam (set_extra_groupings_debug) writes directly into cfg$extra_groupings;
    # the public path builds extra_from_public. We keep the debug seam working for
    # the validation harness by letting it take precedence when present.
    combined_extra <- if (!is.null(cfg$extra_groupings)) {
      cfg$extra_groupings          # debug seam (harness use only)
    } else if (length(extra_from_public) > 0L) {
      extra_from_public
    } else {
      NULL
    }
    if (!is.null(combined_extra)) {
      cluster_spec$extra_groupings <- combined_extra
    }

    # Debug-only seam (validation harness) + public path: forward primary
    # random slopes in the contract's serde JSON shape. The debug seam sets
    # cfg$slopes directly via MCPowerDebug$set_slopes_debug(); the public path
    # sets it via resolved_clusters[[gv]]$slopes in build_contract_bytes.
    # corr_with must serialize as a JSON array even when length-1 (serde
    # expects Vec<f64>); wrap each entry with I() to suppress auto_unbox.
    if (!is.null(cfg$slopes)) {
      cluster_spec$slopes <- lapply(cfg$slopes, function(s) {
        s$corr_with <- I(as.numeric(s$corr_with))
        s
      })
    }
    clusters_json <- jsonlite::toJSON(list(cluster_spec), auto_unbox = TRUE, digits = NA)
  } else {
    clusters_json <- "[]"
  }

  list(
    outcome_kind = outcome_kind,
    estimator = estimator,
    intercept = as.numeric(intercept),
    clusters_json = as.character(clusters_json)
  )
}
