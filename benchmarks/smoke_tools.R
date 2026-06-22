#!/usr/bin/env Rscript
# On-request smoke for the dedicated-tool adapters: cell-means goldens + one
# case per tool at nsim = 4, asserting finite power in [0, 1].
# Run from mcpower/benchmarks:  Rscript smoke_tools.R
source("harness.R")   # load_cases, seed_for, sources loops_r.R + tools_r.R; no run (guarded)

cases <- load_cases("benchmark_cases.json")
by_id <- function(id) Filter(function(c) c$id == id, cases)[[1]]

# --- cell-means goldens (2x2, one-way 4, 2x3) ---
stopifnot(all.equal(superpower_mu(loop_design(by_id("anova_2x2"))),     c(0, 0.5, 0.5, 1.5), check.attributes = FALSE))
stopifnot(all.equal(superpower_mu(loop_design(by_id("anova_oneway4"))), c(0, 0.5, 0.5, 0.5), check.attributes = FALSE))
stopifnot(all.equal(superpower_mu(loop_design(by_id("anova_2x3"))),     c(0, 0.5, 0.5, 0.5, 1.5, 1.5), check.attributes = FALSE))
cat("cell-means goldens ok\n")

# --- one case per tool at nsim = 4 (10 for Superpower, which enforces a minimum of 10) ---
smoke <- list(c("lme_simple", "simr"), c("anova_2x2", "superpower"), c("ols_simple", "simglm"))
for (s in smoke) {
  case <- by_id(s[1])
  if (!identical(case$tool, s[2])) { cat(sprintf("%s no longer %s — skipped\n", s[1], s[2])); next }
  n <- case$n_grid[[length(case$n_grid)]]
  nsim_smoke <- if (s[2] == "superpower") 10L else 4L
  out <- TOOLS[[case$tool]](case, n, nsim_smoke, seed_for(n))
  stopifnot(all(is.finite(out$power)), all(out$power >= 0 & out$power <= 1))
  cat(sprintf("%-14s %-11s ok  power=[%s]\n", s[1], s[2],
              paste(sprintf("%.2f", out$power), collapse = ", ")))
}
cat("tool smoke ok\n")
