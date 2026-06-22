# data-upload.R — R-side ingest helpers for upload_data() / get_effects_from_data().
#
# Mirrors mcpower/ports/py/mcpower/upload_data_utils.py:
#   .value_to_label    ↔ value_to_label
#   .detect_column_types ↔ detect_column_types
#   .normalize_upload_input ↔ normalize_upload_input (data.frame / matrix / list / file path)
#
# File I/O lives entirely on the host (R). The engine receives
# only validated typed columns.

# ── .value_to_label ─────────────────────────────────────────────────────────
# Mirror of upload_data_utils.py:value_to_label.
# Integer-valued numerics render without a decimal (4.0 → "4");
# non-numeric values pass through as character.
.value_to_label <- function(v) {
  if (is.numeric(v) || (is.character(v) && !is.na(suppressWarnings(as.numeric(v))))) {
    fv <- suppressWarnings(as.numeric(v))
    if (!is.na(fv) && fv == as.integer(fv)) {
      return(as.character(as.integer(fv)))
    } else if (!is.na(fv)) {
      return(as.character(fv))
    }
  }
  as.character(v)
}


# ── .detect_column_types ────────────────────────────────────────────────────
# Mirror of upload_data_utils.py:detect_column_types.
#
# Args:
#   mat       2-D matrix or data.frame (n_rows x n_cols).
#   names_vec character vector length n_cols.
#   max_k     integer — soft upper bound on factor levels (max_factor_k_soft from config).
#   max_ratio numeric — min rows-per-level ratio for a numeric column to be a factor.
#
# Returns list(types = <character vector>, labels = <list of character vectors>).
.detect_column_types <- function(mat, names_vec, max_k, max_ratio) {
  if (is.data.frame(mat)) mat <- as.matrix(mat)
  n_rows <- nrow(mat)
  n_cols <- ncol(mat)
  types  <- character(n_cols)
  labels <- vector("list", n_cols)

  for (j in seq_len(n_cols)) {
    col <- mat[, j]

    # Attempt numeric cast to distinguish string vs numeric columns.
    # A column is "numeric" iff every NON-missing value casts to a number;
    # genuine NA values are simply excluded (mirrors Python upload_data_utils.py).
    orig_na <- is.na(col)
    num <- suppressWarnings(as.numeric(col))
    coercion_failed <- is.na(num) & !orig_na      # a real string like "abc"
    col_is_numeric <- !any(coercion_failed)

    if (!col_is_numeric) {
      # String / mixed column → always factor; gather distinct string labels
      distinct <- sort(unique(as.character(col)))
      types[j]  <- "factor"
      labels[[j]] <- distinct
      next
    }

    # Numeric column: count distinct on non-missing values only
    vals <- num[!is.na(num)]
    n_distinct <- length(unique(vals))

    if (n_distinct == 2L) {
      types[j]  <- "binary"
      labels[[j]] <- character(0)
      next
    }

    # Factor guard: few distinct AND enough rows per level
    if (n_distinct <= max_k && (length(vals) / n_distinct) >= max_ratio) {
      # Labels via .value_to_label (integer-valued floats → "4")
      distinct_sorted <- sort(unique(vapply(vals, .value_to_label, character(1))))
      types[j]  <- "factor"
      labels[[j]] <- distinct_sorted
    } else {
      types[j]  <- "continuous"
      labels[[j]] <- character(0)
    }
  }

  list(types = types, labels = labels)
}


# ── .normalize_upload_input ─────────────────────────────────────────────────
# Mirror of upload_data_utils.py:normalize_upload_input.
# Accepted inputs:
#   - character file path (.csv / .tsv): read with read.csv / read.delim
#   - data.frame: used directly
#   - matrix: used directly
#   - named list of vectors: keys become column names
#
# When columns is not NULL it overrides the inferred column names (only for
# matrix / unnamed-list inputs that lack inherent names).
#
# Returns list(matrix = <matrix>, names = <character vector>).
.normalize_upload_input <- function(data, columns = NULL) {
  # --- file path ---
  if (is.character(data) && length(data) == 1L && file.exists(data)) {
    suffix <- tolower(tools::file_ext(data))
    df <- if (suffix == "tsv") {
      read.delim(data, stringsAsFactors = FALSE, check.names = FALSE)
    } else {
      read.csv(data, stringsAsFactors = FALSE, check.names = FALSE)
    }
    if (!is.null(columns)) {
      if (length(columns) != ncol(df)) {
        stop(sprintf("columns length (%d) must match file columns (%d)",
                     length(columns), ncol(df)))
      }
      colnames(df) <- columns
    }
    return(.normalize_upload_input(df, NULL))
  }

  # --- data.frame ---
  if (is.data.frame(data)) {
    col_names <- colnames(data)
    mat <- as.matrix(data)
    return(list(matrix = mat, names = col_names))
  }

  # --- named list ---
  if (is.list(data) && !is.null(names(data))) {
    col_names <- names(data)
    mat <- do.call(cbind, lapply(data, function(x) {
      if (is.factor(x)) as.character(x) else x
    }))
    if (!is.matrix(mat)) mat <- matrix(mat, ncol = length(col_names))
    return(list(matrix = mat, names = col_names))
  }

  # --- matrix ---
  if (is.matrix(data)) {
    if (is.null(columns)) {
      if (!is.null(colnames(data))) {
        columns <- colnames(data)
      } else {
        columns <- paste0("column_", seq_len(ncol(data)))
      }
    }
    if (length(columns) != ncol(data)) {
      stop(sprintf("columns length (%d) must match data columns (%d)",
                   length(columns), ncol(data)))
    }
    return(list(matrix = data, names = columns))
  }

  # --- vector (single column) ---
  if (is.numeric(data) || is.character(data) || is.logical(data)) {
    mat <- matrix(data, ncol = 1)
    col_names <- if (!is.null(columns)) {
      if (length(columns) != 1L) stop("columns length must be 1 for a single vector")
      columns
    } else {
      "column_1"
    }
    return(list(matrix = mat, names = col_names))
  }

  stop("data must be a matrix, data.frame, named list, character file path, or vector")
}
