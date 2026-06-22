# Multi-call accumulation for the three assignment-string setters — mirrors the
# Python suite (test_setter_accumulation.py). set_variable_type / set_effects /
# set_correlation accumulate fragments and replay them in call order (last-wins
# per key) instead of overwriting. Chained/separate calls (one predictor at a
# time) must each survive; pre-fix overwrite kept only the last, silently
# demoting earlier factors to continuous columns. (A combined single string also
# parses correctly — the parser is paren-aware — but separate calls must work.)
#
# Each test is strictly stronger than the pre-fix code: with overwrite the
# earlier fragment is lost, so the surviving-factor / both-effects /
# both-correlations assertions fail.

priv <- function(m) m$.__enclos_env__$private

two_factor_model_separate_calls <- function() {
  # y ~ g1*g2 with both factors declared via SEPARATE calls (the chained
  # one-factor-at-a-time pattern that overwrite used to break).
  MCPower$new("y ~ g1*g2")$
    set_variable_type("g1=(factor, 0.5, 0.5)")$
    set_variable_type("g2=(factor, 0.6, 0.4)")$
    set_effects("g1[2]=0.5, g2[2]=0.4, g1[2]:g2[2]=0.3")
}

# ---- direct regression: multi-call accumulation through the public API

test_that("two factors via separate set_variable_type calls both survive", {
  m <- two_factor_model_separate_calls()
  priv(m)$apply()
  expect_setequal(priv(m)$registry$factor_names, c("g1", "g2"))
})

test_that("two separate factor calls produce [level]-suffixed effect names", {
  m <- two_factor_model_separate_calls()
  priv(m)$apply()
  expect_equal(priv(m)$registry$effect_names,
               c("g1[2]", "g2[2]", "g1[2]:g2[2]"))
})

test_that("two set_effects calls accumulate — no effect lost", {
  m <- MCPower$new("y ~ x1 + x2")$set_effects("x1=0.3")$set_effects("x2=0.5")
  priv(m)$apply()
  reg <- priv(m)$registry
  sizes <- reg$get_effect_sizes()
  names(sizes) <- reg$effect_names
  expect_equal(sizes[["x1"]], 0.3)  # x1 survived the first call
  expect_equal(sizes[["x2"]], 0.5)
})

test_that("two set_correlation string calls accumulate — both pairs present", {
  m <- MCPower$new("y ~ x1 + x2 + x3")$set_effects("x1=0.3, x2=0.5, x3=0.1")$
    set_correlations("corr(x1,x2)=0.3")$set_correlations("corr(x1,x3)=0.2")
  priv(m)$apply()
  mat <- priv(m)$registry$get_correlation_matrix()
  expect_equal(mat[1, 2], 0.3)  # x1,x2 from the first call
  expect_equal(mat[1, 3], 0.2)  # x1,x3 from the second call
})

# ---- call-grouping independence: bytes(N separate) == bytes(1 combined)
#
# Uses bare `factor` (3-level default) for brevity; the combined call is a
# single legal string that must equal the separate calls.

contract_bytes <- function(build) {
  m <- MCPower$new("y ~ a + b")
  build(m)
  m$set_effects("a[2]=0.3, a[3]=0.2, b[2]=0.4, b[3]=0.1")
  priv(m)$build_contract_bytes("optimistic")
}

test_that("variable_type: combined call == separate calls", {
  separate <- contract_bytes(function(m) {
    m$set_variable_type("a=factor"); m$set_variable_type("b=factor")
  })
  combined <- contract_bytes(function(m) m$set_variable_type("a=factor, b=factor"))
  expect_identical(separate, combined)
})

test_that("variable_type: last-wins per key (clean continuous transition)", {
  lw <- MCPower$new("y ~ x1 + x2")$
    set_variable_type("x1=normal")$set_variable_type("x1=right_skewed")$
    set_effects("x1=0.3, x2=0.5")
  dedup <- MCPower$new("y ~ x1 + x2")$
    set_variable_type("x1=right_skewed")$set_effects("x1=0.3, x2=0.5")
  expect_identical(priv(lw)$build_contract_bytes("optimistic"),
                   priv(dedup)$build_contract_bytes("optimistic"))
})

test_that("effects: combined call == separate calls", {
  eff <- function(build) {
    m <- MCPower$new("y ~ x1 + x2"); build(m)
    priv(m)$build_contract_bytes("optimistic")
  }
  separate <- eff(function(m) { m$set_effects("x1=0.3"); m$set_effects("x2=0.5") })
  combined <- eff(function(m) m$set_effects("x1=0.3, x2=0.5"))
  expect_identical(separate, combined)
})

test_that("effects: last-wins per key", {
  eff <- function(build) {
    m <- MCPower$new("y ~ x1 + x2"); build(m)
    priv(m)$build_contract_bytes("optimistic")
  }
  lw    <- eff(function(m) { m$set_effects("x1=0.3, x2=0.5"); m$set_effects("x1=0.9") })
  dedup <- eff(function(m) m$set_effects("x1=0.9, x2=0.5"))
  expect_identical(lw, dedup)
})

test_that("string correlations: combined call == separate calls", {
  corr <- function(build) {
    m <- MCPower$new("y ~ x1 + x2 + x3")$set_effects("x1=0.3, x2=0.5, x3=0.1")
    build(m)
    priv(m)$build_contract_bytes("optimistic")
  }
  separate <- corr(function(m) {
    m$set_correlations("corr(x1,x2)=0.3"); m$set_correlations("corr(x1,x3)=0.2")
  })
  combined <- corr(function(m) m$set_correlations("corr(x1,x2)=0.3, corr(x1,x3)=0.2"))
  expect_identical(separate, combined)
})

test_that("a matrix call resets the correlation accumulator", {
  corr <- function(build) {
    m <- MCPower$new("y ~ x1 + x2 + x3")$set_effects("x1=0.3, x2=0.5, x3=0.1")
    build(m)
    priv(m)$build_contract_bytes("optimistic")
  }
  string_then_matrix <- corr(function(m) {
    m$set_correlations("corr(x1,x2)=0.4"); m$set_correlations(diag(3))
  })
  matrix_only <- corr(function(m) m$set_correlations(diag(3)))
  expect_identical(string_then_matrix, matrix_only)
})

test_that("a string fragment layers onto a prior matrix", {
  m <- MCPower$new("y ~ x1 + x2 + x3")$set_effects("x1=0.3, x2=0.5, x3=0.1")$
    set_correlations(diag(3))$set_correlations("corr(x1,x2)=0.4")
  priv(m)$apply()
  mat <- priv(m)$registry$get_correlation_matrix()
  expect_equal(mat[1, 2], 0.4)  # layered on top of the identity base
  expect_equal(mat[1, 3], 0.0)
})

# ---- cross-port parity: R bytes are byte-identical to the Python golden for a
# 2-factor model built by separate calls (proves accumulation is consistent
# across the Python and R ports — the strong cross-port parity guard).

test_that("two-factor accumulation contract matches the Python golden", {
  golden_path <- test_path("_golden/two_factor_accumulation_contract.msgpack")
  golden <- readBin(golden_path, "raw", n = file.info(golden_path)$size)
  m <- two_factor_model_separate_calls()
  blob <- priv(m)$build_contract_bytes("optimistic")
  expect_identical(blob, golden)
})
