test_that("scenario defaults load from the embedded JSON", {
  sc <- mcpower:::.scenario_defaults()
  expect_setequal(names(sc), c("optimistic", "realistic", "doomer"))
  # configs/scenarios.json now uses canonical names (no alias layer).
  expect_equal(sc$optimistic$residual_dists, c("high_kurtosis", "right_skewed"))
  expect_equal(sc$realistic$random_effect_dist, "heavy_tailed")
  expect_equal(sc$optimistic$random_effect_dist, "normal")
  expect_null(sc$optimistic$lme)
})
