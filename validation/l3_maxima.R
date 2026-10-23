#!/usr/bin/env Rscript
# l3_maxima.R — largest MCPower-vs-frozen-R-SOTA discrepancy per model class,
# over the same frozen goldens and saved bytes that regression.R gates. Writes
# data/l3_maxima.csv with columns model_class, check, n_cells, max_abs_diff:
#   coefficients_and_statistics — max over every coefficient |beta_MC - beta_R|
#                                 and every target ||stat_MC| - |stat_R||
#   critical_value              — max over every target |crit_MC - crit_R|
# n_cells is the number of L3 cases (formula cells) behind the row. Differences
# are absolute, while the gate's bands are relative; this reports, it gates nothing.
#
# Model classes: OLS = family "ols"; GLM = an unclustered logit/probit/poisson
# case; LMM = family "lme" (CASES via check_golden, EXTRA_CASES via the extras
# golden); GLMM = a non-"lme" EXTRA_CASES case. The extras goldens freeze no R
# critical value, so their cases add nothing to critical_value; a class with no
# such case gets n_cells = 0 and max_abs_diff = NA.
suppressMessages(library(mcpower))

script_dir <- local({
  fa <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
  if (length(fa)) dirname(normalizePath(sub("^--file=", "", fa[1]))) else getwd()
})
setwd(script_dir)
source("common.R"); source("tolerances.R"); source("formulas.R")

model_class <- function(case, extras) {
  if (identical(case$family, "ols")) "OLS"
  else if (identical(case$family, "lme")) "LMM"
  else if (extras) "GLMM"
  else "GLM"
}

rows <- list()
add <- function(case, extras, check, diffs)
  rows[[length(rows) + 1L]] <<- data.frame(
    model_class = model_class(case, extras), check = check, label = case$label,
    max_abs_diff = max(diffs), stringsAsFactors = FALSE)

# Same comparator table regression.R gates on (compare_sides via check_golden).
for (case in CASES) {
  tb  <- check_golden(case)$table
  tgt <- !is.na(tb[["R t"]])
  add(case, FALSE, "coefficients_and_statistics",
      c(abs(tb[["MCPower beta"]] - tb[["R beta"]]),
        abs(abs(tb[["MCPower stat"]][tgt]) - abs(tb[["R t"]][tgt]))))
  add(case, FALSE, "critical_value", abs(tb[["MCPower crit"]][tgt] - tb[["R crit"]][tgt]))
}

for (case in EXTRA_CASES) {
  g <- check_extras_golden(case)
  add(case, TRUE, "coefficients_and_statistics", c(g$beta_diff, g$stat_diff))
}

per_case <- do.call(rbind, rows)
out <- do.call(rbind, lapply(c("OLS", "GLM", "LMM", "GLMM"), function(cl)
  do.call(rbind, lapply(c("coefficients_and_statistics", "critical_value"), function(ck) {
    r <- per_case[per_case$model_class == cl & per_case$check == ck, ]
    data.frame(model_class = cl, check = ck, n_cells = nrow(r),
               max_abs_diff = if (nrow(r)) max(r$max_abs_diff) else NA_real_)
  }))))
utils::write.csv(out, file.path("data", "l3_maxima.csv"), row.names = FALSE)
print(out, row.names = FALSE)
cat("Wrote data/l3_maxima.csv\n")
