#!/usr/bin/env Rscript
# Does simr's scatter around MCPower come from its fixed predictor draw?
# Reruns the agreement harness's own simr adapter (tool_simr) at the mid-range
# simr points under the original seed (k = 0, reproduces the snapshot) and
# K new seeds (each a new predictor draw). One row per (case, n, k) in
# simr_seeds.csv.
# Usage: Rscript simr_seeds.R <agreement_dir> <out_csv> <cores> [K] [nsim]

args  <- commandArgs(trailingOnly = TRUE)
AG    <- normalizePath(args[1])
OUT   <- args[2]
CORES <- as.integer(args[3])
K     <- if (length(args) >= 4) as.integer(args[4]) else 16L
NSIM  <- if (length(args) >= 5) as.integer(args[5]) else 800L

Sys.setenv(OMP_NUM_THREADS = "1", OPENBLAS_NUM_THREADS = "1")
source(file.path(AG, "loops_r.R"))
source(file.path(AG, "tools_r.R"))

`%||%` <- function(a, b) if (is.null(a)) b else a
doc <- jsonlite::fromJSON(file.path(AG, "cases.json"), simplifyVector = FALSE)
# Case fields tool_simr reads, merged with family defaults as harness.R's load_cases does.
cases <- lapply(Filter(function(c) identical(c$tool, "simr"), doc$cases), function(c) {
  d <- doc$defaults[[c$family]]
  list(id = c$id, family = c$family, formula = c$formula, effects = c$effects,
       targets = unlist(c$targets),
       variable_types = c$variable_types %||% list(),
       cluster = c$cluster, crossed = c$crossed, tool = c$tool,
       correlations = c$correlations,
       baseline_p = c$baseline_p %||% d$baseline_p)
})
names(cases) <- vapply(cases, `[[`, "", "id")

# Mid-range simr points (MCPower power in [.2, .9] in the 1.5.0 snapshot);
# lme_ex2 thinned to three of its six points for run time.
points <- list(
  list("glmm_multislope", c(150, 300)),
  list("glmm_simple",     c(100, 200)),
  list("glmm_slope",      c(100, 200, 300)),
  list("lme_ex2",         c(360, 600, 840)),
  list("lme_factor_inter", c(100, 200, 300, 400, 500)),
  list("lme_multi",       c(100, 200, 300, 400)),
  list("lme_simple",      c(100, 200, 300, 400))
)
jobs <- list()
for (p in points) for (n in p[[2]]) for (k in 0:K)
  jobs[[length(jobs) + 1L]] <- list(case = p[[1]], n = n, k = k,
                                    seed = 2137L + as.integer(n) + 100000L * k)

run_job <- function(j) {
  t0 <- proc.time()[["elapsed"]]
  r <- tryCatch(tool_simr(cases[[j$case]], j$n, NSIM, j$seed)$power,
                error = function(e) NA_real_)
  data.frame(case = j$case, target = cases[[j$case]]$targets[1], n = j$n, k = j$k,
             seed = j$seed, nsim = NSIM, power = r,
             secs = proc.time()[["elapsed"]] - t0)
}

res <- parallel::mclapply(jobs, run_job, mc.cores = CORES, mc.preschedule = FALSE)
out <- do.call(rbind, res)
write.csv(out, OUT, row.names = FALSE)
cat(sprintf("%d jobs, %d NA, written to %s\n", nrow(out), sum(is.na(out$power)), OUT))
