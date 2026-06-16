# D2 — host-side type detection: detect_column_types against shared golden fixture.
# Mirrors mcpower/ports/py/tests/test_upload_type_detection.py exactly.

GOLDEN_PATH <- testthat::test_path("_golden", "upload_type_detection.json")

# Build the test array from the fixture spec (mirror of Python's build_array_from_fixture).
# seed=42 matches Python's np.random.default_rng(42) output order for the continuous column.
.build_array_from_fixture <- function(fx) {
  n_rows <- fx$n_rows
  set.seed(42)  # match Python rng(42) for the continuous column
  arrays <- list()
  names_vec <- character(0)

  for (col in fx$columns) {
    names_vec <- c(names_vec, col$name)
    sample_key <- if (!is.null(col$sample)) col$sample else ""

    if (identical(sample_key, "continuous_60_distinct")) {
      # 60 fully distinct floats (standard normal draw, seed 42)
      col_data <- rnorm(n_rows)
    } else if (identical(sample_key, "20_distinct_over_60_rows_ratio_3")) {
      # 20 distinct numeric values spread over 60 rows → ratio 3 < 15 → continuous
      vals <- as.numeric(0:19)
      col_data <- rep(vals, 3)  # 60 rows total
    } else {
      # Use the explicit values list, repeated/tiled to fill n_rows
      raw <- unlist(col$values)
      if (length(raw) < n_rows) {
        repeats <- ceiling(n_rows / length(raw))
        raw <- rep(raw, repeats)[seq_len(n_rows)]
      }
      col_data <- raw[seq_len(n_rows)]
    }
    arrays[[length(arrays) + 1L]] <- col_data
  }

  mat <- do.call(cbind, lapply(arrays, function(x) x))
  list(matrix = mat, names = names_vec)
}

test_that("golden fixture exists", {
  expect_true(file.exists(GOLDEN_PATH))
})

test_that("all columns match shared golden fixture", {
  fx <- jsonlite::fromJSON(GOLDEN_PATH, simplifyVector = FALSE)
  built <- .build_array_from_fixture(fx)
  mat <- built$matrix
  names_vec <- built$names
  max_k     <- fx$max_factor_k_soft
  max_ratio <- fx$max_factor_ratio

  result <- mcpower:::.detect_column_types(mat, names_vec, max_k, max_ratio)
  types_got  <- result$types
  labels_got <- result$labels

  expect_equal(length(types_got), length(fx$columns))
  expect_equal(length(labels_got), length(fx$columns))

  for (i in seq_along(fx$columns)) {
    col_spec <- fx$columns[[i]]
    expect_equal(
      types_got[[i]], col_spec$expect,
      info = sprintf("Column %s: expected %s, got %s",
                     col_spec$name, col_spec$expect, types_got[[i]])
    )
    if (!is.null(col_spec$expect_levels)) {
      expect_equal(
        labels_got[[i]], unlist(col_spec$expect_levels),
        info = sprintf("Column %s levels", col_spec$name)
      )
    }
  }
})

test_that("binary detection: exactly 2 distinct numeric values", {
  mat <- matrix(c(0, 1, 0, 1, 1), ncol = 1)
  res <- mcpower:::.detect_column_types(mat, "x", 7L, 15)
  expect_equal(res$types[[1]], "binary")
  expect_equal(res$labels[[1]], character(0))
})

test_that("string column is always factor", {
  mat <- matrix(c("cat", "dog", "cat", "bird", "dog"), ncol = 1)
  res <- mcpower:::.detect_column_types(mat, "pet", 7L, 15)
  expect_equal(res$types[[1]], "factor")
  expect_equal(res$labels[[1]], c("bird", "cat", "dog"))  # sorted
})

test_that("numeric many distinct is continuous", {
  mat <- matrix(as.numeric(1:8), ncol = 1)  # 8 distinct > max_k=7
  res <- mcpower:::.detect_column_types(mat, "z", 7L, 15)
  expect_equal(res$types[[1]], "continuous")
})

test_that("numeric few distinct enough rows is factor — integer labels", {
  # 3 distinct values over 60 rows → ratio 20 >= 15 → factor
  # Integer-valued floats (1.0, 2.0, 3.0) render without decimals
  vals <- rep(c(1.0, 2.0, 3.0), 20)  # 60 rows, 3 distinct
  mat  <- matrix(vals, ncol = 1)
  res <- mcpower:::.detect_column_types(mat, "x", 7L, 15)
  expect_equal(res$types[[1]], "factor")
  expect_equal(res$labels[[1]], c("1", "2", "3"))
})

test_that("numeric few distinct too sparse is continuous", {
  # 3 distinct over 6 rows → ratio 2 < 15 → continuous
  vals <- c(1.0, 2.0, 3.0, 1.0, 2.0, 3.0)
  mat  <- matrix(vals, ncol = 1)
  res <- mcpower:::.detect_column_types(mat, "x", 7L, 15)
  expect_equal(res$types[[1]], "continuous")
})

test_that("factor labels are sorted and deduped", {
  raw <- rep(c("Japan", "USA", "Europe"), 20)  # 60 rows
  mat <- matrix(raw, ncol = 1)
  res <- mcpower:::.detect_column_types(mat, "origin", 7L, 15)
  expect_equal(res$types[[1]], "factor")
  expect_equal(res$labels[[1]], c("Europe", "Japan", "USA"))
})

test_that("value_to_label renders integers without decimals", {
  expect_equal(mcpower:::.value_to_label(4.0), "4")
  expect_equal(mcpower:::.value_to_label(6.0), "6")
  expect_equal(mcpower:::.value_to_label(6.5), "6.5")
  expect_equal(mcpower:::.value_to_label("USA"), "USA")
})

# Regression test for CRITICAL 3: numeric column containing a genuine NA must
# classify as continuous (not factor), matching Python upload_data_utils.py
# behaviour where non-missing values are checked for numeric-ness.
test_that("numeric column with genuine NA classifies as continuous, not factor", {
  # 8 distinct non-missing floats + 2 NAs: every non-NA value is numeric,
  # so the column is continuous.
  vals <- c(1.0, 2.5, NA, 4.0, 5.5, 6.1, 7.2, 8.3, 9.4, NA)
  mat  <- matrix(as.character(vals), ncol = 1)
  res  <- mcpower:::.detect_column_types(mat, "x", 7L, 15)
  expect_equal(res$types[[1]], "continuous",
               info = "numeric column with NA must be continuous, not factor")
})

test_that("genuine string column still classifies as factor when it contains NA", {
  # Real string values (non-numeric) → must remain factor even if NA present.
  vals <- c("a", "b", "a", NA, "b")
  mat  <- matrix(as.character(vals), ncol = 1)
  res  <- mcpower:::.detect_column_types(mat, "pet", 7L, 15)
  expect_equal(res$types[[1]], "factor",
               info = "string column must still be factor")
})
