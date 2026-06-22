#!/usr/bin/env Rscript
# Parameterized R benchmark harness — mirrors harness.py.
# Run from mcpower/benchmarks with:  Rscript harness.R --case ols_simple --out results/r_smoke.json

# ---- arg parsing (do this first, before loading mcpower) ----
parse_args <- function(argv) {
  out <- list(case = "all", methods = NULL, out = "results/r.json", threads = "auto")
  i <- 1
  while (i <= length(argv)) {
    key <- argv[i]
    if (!key %in% c("--case", "--methods", "--out", "--threads", "--scale"))
      stop(sprintf("unknown arg: %s", key))
    if (i + 1 > length(argv)) stop(sprintf("missing value for %s", key))
    val <- argv[i + 1]
    if (key == "--case")         out$case <- val
    else if (key == "--methods") out$methods <- strsplit(val, ",")[[1]]
    else if (key == "--out")     out$out <- val
    else if (key == "--scale")   out$scale <- val
    else                         out$threads <- val
    i <- i + 2
  }
  out
}
ARGS <- parse_args(commandArgs(trailingOnly = TRUE))

# ---- single-thread re-exec (must precede library(mcpower)) ----
if (ARGS$threads == "1" && Sys.getenv("RAYON_NUM_THREADS") != "1") {
  self <- normalizePath(sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE)[1]))
  code <- system2("Rscript", args = c(shQuote(self), commandArgs(trailingOnly = TRUE)),
                  env = "RAYON_NUM_THREADS=1")
  quit(save = "no", status = code)
}

# Pin lazily-initialized BLAS pools for the loop tiers (fairness — mirrors
# harness.py's env pinning); harmless if the BLAS backend ignores it. The
# engine is unaffected (rayon reads RAYON_NUM_THREADS).
Sys.setenv(OMP_NUM_THREADS = "1", OPENBLAS_NUM_THREADS = "1")

suppressMessages({ library(mcpower); library(jsonlite) })
source("loops_r.R")
source("tools_r.R")

RECORDED_METHODS <- c("mcpower_find_power", "mcpower_find_sample_size",
                      "tool", "loop_naive", "loop_best")
REPS <- list(mcpower_find_power = 3L, mcpower_find_sample_size = 3L,
             loop_best = 3L, loop_naive = 1L)

# Scaling for every tier's sim count (mcpower / loops / tools). 1.0 = the real
# (publishable) run; set e.g. 0.1 for a 1/10-sims quick preview. Recorded rows
# carry the scaled n_sims, so per-sim normalization in combine.py stays honest
# either way; meta records the scale.
# Keep in sync with harness.py's N_SIMS_SCALE.
N_SIMS_SCALE <- 1.0
if (!is.null(ARGS$scale)) N_SIMS_SCALE <- as.numeric(ARGS$scale)  # --scale 0.1 = 10% sims
scaled_sims <- function(count) max(1L, as.integer(round(N_SIMS_SCALE * count)))

seed_for <- function(n) 2137L + as.integer(n)

`%||%` <- function(a, b) if (is.null(a)) b else a

# ---- load_cases: mirror cases.py (merge family defaults) ----
load_cases <- function(path) {
  doc <- jsonlite::fromJSON(path, simplifyVector = FALSE)
  defaults <- doc$defaults
  lapply(doc$cases, function(c) {
    d <- defaults[[c$family]]
    n <- c$n %||% d$n                       # per-case grid override
    n_sims <- d$n_sims                      # key-level merge with per-case overrides
    for (k in names(c$n_sims)) n_sims[[k]] <- c$n_sims[[k]]
    list(
      id = c$id, family = c$family, formula = c$formula, effects = c$effects,
      targets = unlist(c$targets),
      n_grid = seq(n$from, n$to, by = n$by),
      n_sims = n_sims,                      # list(mcpower=, best=, naive=, tool=)
      target_power = d$target_power,
      variable_types = if (is.null(c$variable_types)) list() else c$variable_types,
      cluster = c$cluster,
      tool = c$tool,                        # NULL = cliff band
      correlations = c$correlations,
      baseline_p = if (!is.null(c$baseline_p)) c$baseline_p else d$baseline_p,
      max_failed_frac = if (!is.null(c$max_failed_frac)) c$max_failed_frac else d$max_failed_frac
    )
  })
}

build_model <- function(case) {
  # The benchmark "glmm" family is a logistic model with clusters — mcpower has
  # no separate GLMM family; it dispatches GLMM off logit + set_cluster.
  mc_family <- if (identical(case$family, "glmm")) "logit" else case$family
  m <- MCPower$new(case$formula, family = mc_family)
  for (nm in names(case$variable_types)) m$set_variable_type(paste0(nm, "=", case$variable_types[[nm]]))
  m$set_effects(case$effects)
  if (!is.null(case$correlations)) m$set_correlations(case$correlations)
  if (!is.null(case$baseline_p)) m$set_baseline_probability(case$baseline_p)
  if (!is.null(case$cluster)) {
    cl <- case$cluster
    # Random slopes ride inside the cluster dict (uncorrelated REs — see
    # harness.py). R's set_cluster takes a per-slope list; the shared JSON
    # carries one scalar slope_variance, so each slope gets the same variance
    # and corr_with_intercept = 0. Keep in sync with harness.py.
    if (length(cl$random_slopes) > 0L) {
      sic <- if (is.null(cl$slope_intercept_corr)) 0 else cl$slope_intercept_corr
      # corr_with = one zero per EARLIER slope (uncorrelated); the engine
      # requires slope k to list k correlations. Mirrors harness.py / model.py.
      sl <- lapply(seq_along(cl$random_slopes), function(i)
        list(predictor = cl$random_slopes[[i]], variance = cl$slope_variance,
             corr_with_intercept = sic, corr_with = rep(0, i - 1L)))
      m$set_cluster(cl$var, ICC = cl$ICC, n_clusters = cl$n_clusters, random_slopes = sl)
    } else {
      m$set_cluster(cl$var, ICC = cl$ICC, n_clusters = cl$n_clusters)
    }
  }
  if (!is.null(case$max_failed_frac)) m$set_max_failed_simulations(case$max_failed_frac)
  # Honest-compare override (benchmark-only): the loops and dedicated tools
  # draw factor levels randomly per sim, while mcpower's optimistic baseline
  # uses exact-count allocation. Flip to sampled so all tiers answer the
  # same random-allocation question. No-op for factor-free cases; the
  # engine default is unchanged. Keep in sync with harness.py.
  m$set_scenario_configs(list(optimistic = list(sampled_factor_proportions = TRUE)))
  m
}

time_call <- function(fn, warmup, reps) {
  if (warmup) fn()
  times <- numeric(reps); result <- NULL
  for (i in seq_len(reps)) {
    t0 <- proc.time()[["elapsed"]]; result <- fn(); times[i] <- proc.time()[["elapsed"]] - t0
  }
  list(time = median(times), result = result)
}

run_loop <- function(case, kind, n, n_sims, seed) {
  as.numeric(LOOPS[[case$family]][[kind]](case, n, n_sims, seed)$power)
}

run_mcpower_find_power <- function(case, n, n_sims, seed) {
  m <- build_model(case)
  r <- m$find_power(sample_size = n, n_sims = n_sims, seed = seed, progress_callback = FALSE, verbose = FALSE)
  as.numeric(r$power_uncorrected[[1]])
}

run_mcpower_find_sample_size <- function(case, n_sims, seed) {
  m <- build_model(case); g <- case$n_grid
  r <- m$find_sample_size(target_power = case$target_power, from_size = g[1], to_size = g[length(g)],
                          by = g[2] - g[1], mode = "linear", n_sims = n_sims, seed = seed,
                          progress_callback = FALSE, verbose = FALSE)
  fa <- r$first_achieved
  fa <- fa[order(as.integer(names(fa)))]                   # names are 0-based target indices
  # Full per-grid-point power curve (diagnostic: must agree with the find_power
  # tier at the same n within MC noise — seeds differ). I() keeps each point a
  # JSON array even with one target, matching harness.py's nested shape.
  curve <- lapply(r$power_uncorrected, function(v) I(as.numeric(v)))
  list(first_achieved = fa,
       n_pts = as.integer(r$n_sample_sizes %||% 0),
       power = curve)
}

record <- function(case, method, n, n_sims, elapsed, power) {
  list(case_id = case$id, family = case$family, lang = "r", method = method,
       n = n, n_sims = n_sims, time_s = elapsed, per_sim_s = elapsed / n_sims,
       power = I(as.numeric(power)))   # I() keeps it a JSON array even when length 1
}

build_meta <- function(threads_mode) {
  cpu_model <- "unknown"
  ci <- tryCatch(readLines("/proc/cpuinfo"), error = function(e) character(0),
                 warning = function(w) character(0))
  mn <- grep("^model name", ci, value = TRUE)
  if (length(mn)) cpu_model <- trimws(sub("^[^:]*:", "", mn[1]))
  pkg_ver <- function(p) tryCatch(as.character(utils::packageVersion(p)),
                                  error = function(e) "not installed")
  list(
    lang = "r",
    timestamp_utc = format(Sys.time(), "%Y-%m-%dT%H:%M:%S", tz = "UTC"),
    os = paste(Sys.info()[["sysname"]], Sys.info()[["release"]]),
    cpu_model = cpu_model,
    cores_physical = parallel::detectCores(logical = FALSE),
    cores_logical = parallel::detectCores(logical = TRUE),
    threads_mode = threads_mode,
    n_sims_scale = N_SIMS_SCALE,
    lang_version = paste(R.version$major, R.version$minor, sep = "."),
    packages = sapply(c("mcpower", "lme4", "lmerTest", "simr", "Superpower", "simglm"),
                      pkg_ver, simplify = FALSE)
  )
}

main <- function(args) {
  all_cases <- load_cases("benchmark_cases.json")   # relative to cwd (run from benchmarks/)
  cases <- if (args$case == "all") all_cases else Filter(function(c) c$id == args$case, all_cases)
  if (length(cases) == 0) stop(sprintf("Case '%s' not found", args$case))
  methods <- if (is.null(args$methods)) RECORDED_METHODS else args$methods

  records <- list()
  for (case in cases) {
    for (method in methods) {
      if (method == "mcpower_find_power") {
        n_sims <- scaled_sims(case$n_sims$mcpower)
        cat(sprintf("\n=== %s mcpower find_power (%d sims/n, threads=%s) ===\n", case$id, n_sims, args$threads))
        cat(sprintf("%6s | %10s | %12s | power\n", "n", "time(s)", "per-sim(s)")); cat(strrep("-", 60), "\n")
        for (n in case$n_grid) {
          tc <- time_call(function() run_mcpower_find_power(case, n, n_sims, seed_for(n)), warmup = TRUE, reps = REPS$mcpower_find_power)
          records[[length(records) + 1]] <- record(case, method, n, n_sims, tc$time, tc$result)
          cat(sprintf("%6d | %10.4f | %12.6f | [%s]\n", n, tc$time, tc$time / n_sims, paste(sprintf("%.3f", tc$result), collapse = ", ")))
        }
      } else if (method == "mcpower_find_sample_size") {
        # One call evaluates the FULL grid from n_sims shared draws (budget =
        # n_sims total, NOT x grid — see orchestrator find_sample_size.rs).
        # Same sims/point as the find_power tier; combine.py compares the two
        # grid-vs-grid. Mirrors harness.py.
        n_sims <- scaled_sims(case$n_sims$mcpower)
        cat(sprintf("\n=== %s mcpower find_sample_size (%d sims, full grid, threads=%s) ===\n", case$id, n_sims, args$threads))
        tc <- time_call(function() run_mcpower_find_sample_size(case, n_sims, 2137L),
                        warmup = TRUE, reps = REPS$mcpower_find_sample_size)
        ss <- tc$result
        # One row per case: n = recommended n (max over targets; 0 if any target
        # never achieves); power = the full per-point curve (diagnostic).
        fa_vals <- vapply(ss$first_achieved,
                          function(v) if (is.null(v) || is.na(v)) NA_integer_ else as.integer(v),
                          integer(1))
        n_rec <- if (length(fa_vals) && !anyNA(fa_vals)) max(fa_vals) else 0L
        rec <- record(case, method, n_rec, n_sims, tc$time, NA)
        rec$power <- ss$power                              # nested per-point curve, not a flat vector
        records[[length(records) + 1]] <- rec
        cat(sprintf("target=%.2f  time=%.4fs  grid_pts=%d  n*=%d\n",
                    case$target_power, tc$time, ss$n_pts, n_rec))
      } else if (method %in% c("loop_best", "loop_naive")) {
        kind   <- if (method == "loop_best") "best" else "naive"
        tier   <- kind
        n_sims <- scaled_sims(case$n_sims[[tier]])
        reps   <- REPS[[method]]
        cat(sprintf("\n=== %s %s (%d sims/n) ===\n", case$id, method, n_sims))
        cat(sprintf("%6s | %10s | %12s | power\n", "n", "time(s)", "per-sim(s)")); cat(strrep("-", 60), "\n")
        for (n in case$n_grid) {
          tc <- time_call(function() run_loop(case, kind, n, n_sims, seed_for(n)), warmup = TRUE, reps = reps)
          records[[length(records) + 1]] <- record(case, method, n, n_sims, tc$time, tc$result)
          cat(sprintf("%6d | %10.4f | %12.6f | [%s]\n", n, tc$time, tc$time / n_sims, paste(sprintf("%.3f", tc$result), collapse = ", ")))
        }
      } else if (method == "tool") {
        if (is.null(case$tool)) {
          cat(sprintf("\n=== %s tool: skipped (cliff — no dedicated tool covers this design) ===\n", case$id))
        } else {
          fn <- TOOLS[[case$tool]]
          resolved <- paste0("tool_", case$tool)
          n_sims <- scaled_sims(case$n_sims$tool)
          cat(sprintf("\n=== %s %s (%d sims/n, micro warm-up + 1 rep) ===\n", case$id, resolved, n_sims))
          cat(sprintf("%6s | %10s | %12s | power\n", "n", "time(s)", "per-sim(s)")); cat(strrep("-", 60), "\n")
          warmup_nsim <- if (identical(case$tool, "superpower")) 10L else 4L
          for (n in case$n_grid) {
            # tryCatch backstop: a tool crash at one n (e.g. simglm drawing a
            # level-free factor sample at small n) drops that point, not the run.
            res <- tryCatch({
              invisible(fn(case, n, warmup_nsim, seed_for(n)))  # micro warm-up
              t0 <- proc.time()[["elapsed"]]
              out <- fn(case, n, n_sims, seed_for(n))
              list(el = proc.time()[["elapsed"]] - t0, power = out$power)
            }, error = function(e) e)
            if (inherits(res, "error")) {
              cat(sprintf("%6d | FAILED: %s\n", n, conditionMessage(res)))
              next
            }
            records[[length(records) + 1]] <- record(case, resolved, n, n_sims, res$el, res$power)
            cat(sprintf("%6d | %10.4f | %12.6f | [%s]\n", n, res$el, res$el / n_sims,
                        paste(sprintf("%.3f", res$power), collapse = ", ")))
          }
        }
      } else {
        cat(sprintf("WARNING: unknown/unsupported method '%s', skipping\n", method))
      }
    }
  }

  dir.create(dirname(args$out), showWarnings = FALSE, recursive = TRUE)
  jsonlite::write_json(list(meta = build_meta(args$threads), records = records),
                       args$out, auto_unbox = TRUE, digits = NA)
  cat(sprintf("\nWrote %d records to %s\n", length(records), args$out))
}

if (sys.nframe() == 0L) main(ARGS)
