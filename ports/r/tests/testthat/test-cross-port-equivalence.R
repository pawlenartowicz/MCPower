test_that("R contract bytes are byte-identical to the Python golden", {
  golden_path <- test_path("_golden/ols_contract.msgpack")
  golden <- readBin(golden_path, "raw", n = file.info(golden_path)$size)

  m <- MCPower$new("y ~ x1 + x2")$set_effects("x1=0.5, x2=0.3")
  blob <- m$.__enclos_env__$private$build_contract_bytes("optimistic")

  expect_identical(blob, golden)
})
