#!/usr/bin/env Rscript
# Summarises simr_seeds.csv against an agreement run's agreement_r_points.csv.
# Per point: k = 0 against the agreement run's simr value (reproduction), the mean of
# the K new seeds against MCPower, and the spread across seeds beyond binomial
# noise (the predictor-draw component). Writes simr_seeds_points.csv.
# Usage: Rscript simr_seeds_summary.R <simr_seeds.csv> <agreement_r_points.csv> <out_points.csv>

a <- commandArgs(trailingOnly = TRUE)
runs <- read.csv(a[1])
snap <- read.csv(a[2])
snap <- snap[snap$comparator == "simr", c("case", "target", "n", "mc_power", "ref_power", "mc_n_sims", "ref_n_sims")]
d <- merge(runs, snap, by = c("case", "target", "n"))
fam <- function(case) ifelse(grepl("^glmm", case), "GLMM", "LMM")

pts <- do.call(rbind, lapply(split(d, list(d$case, d$n), drop = TRUE), function(g) {
  s  <- g[g$k > 0, ]
  K  <- nrow(s)
  m  <- mean(s$power)
  binvar <- mean(s$power * (1 - s$power)) / g$nsim[1]          # per-run binomial variance
  mcvar  <- g$mc_power[1] * (1 - g$mc_power[1]) / g$mc_n_sims[1]
  data.frame(family = fam(g$case[1]), case = g$case[1], n = g$n[1],
             mc = g$mc_power[1], snap_simr = g$ref_power[1],
             rerun_k0 = g$power[g$k == 0],
             seeds_mean = m, seeds_sd = sd(s$power), K = K,
             draw_sd = sqrt(max(0, var(s$power) - binvar)),
             mean_minus_mc = m - g$mc_power[1],
             z_mean = (m - g$mc_power[1]) / sqrt(var(s$power) / K + mcvar),
             snap_excess2 = (g$ref_power[1] - g$mc_power[1])^2 - binvar - mcvar)
}))
pts <- pts[order(pts$family, pts$case, pts$n), ]
write.csv(pts, a[3], row.names = FALSE)

cat(sprintf("k = 0 reproduces the agreement run at %d of %d points (max |diff| %.4f)\n",
            sum(abs(pts$rerun_k0 - pts$snap_simr) < 1e-9), nrow(pts), max(abs(pts$rerun_k0 - pts$snap_simr))))
for (f in c("LMM", "GLMM", "all")) {
  p <- if (f == "all") pts else pts[pts$family == f, ]
  cat(sprintf("%-4s points %2d | draw SD (RMS over points) %.2f pp | agreement-run excess scatter on these points %.2f pp | mean(seeds - MCPower) %+.2f pp, |z| > 1.96 at %d points, max |z| %.2f\n",
              f, nrow(p), 100 * sqrt(mean(p$draw_sd^2)), 100 * sqrt(max(0, mean(p$snap_excess2))),
              100 * mean(p$mean_minus_mc), sum(abs(p$z_mean) > 1.96), max(abs(p$z_mean))))
}
print(transform(pts[, c("family", "case", "n", "mc", "snap_simr", "seeds_mean", "draw_sd", "z_mean")],
                draw_sd = round(draw_sd, 3), seeds_mean = round(seeds_mean, 4), z_mean = round(z_mean, 2)),
      row.names = FALSE)
