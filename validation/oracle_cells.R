# oracle_cells.R — the shared HARSH GLMM Wald-SE cell grid (Tasks 15+16).
#
# Six clustered-logistic cells deliberately parked at the GLMM Wald-SE failure
# regimes — few clusters, high ICC, tiny clusters, near-separation — where the
# per-fit SE coefficient-of-variation is largest and the asymp fixed-denominator
# assumption is most likely to strain. Each cell is a case-like list consumable
# by build_m4_model (common.R:90): formula / effects / baseline_probability /
# cluster=list(var,ICC,n_clusters) [+ slopes] [+ extra] / n / seed, PLUS the
# oracle-only fields target_term (the fixed effect under test), beta_star (its
# alt magnitude) and target_col (its 0-based design-column index).
#
# slopes / extra serde shapes mirror M4_GLMM_CASES (formulas.R:928-992) verbatim:
#   slopes = list(list(column=<0-based gen col>, variance=, corr_with_intercept=,
#                      corr_with=numeric(0)))
#   extra  = list(list(var=, kind="crossed", n_clusters=, tau_squared=))
# (build_m4_model translates `extra` to the contract relation/tau_squared shape.)
#
# n = n_clusters × cluster_size (the engine derives cluster_size = n / n_clusters).
# target_col = 1: every cell is `y ~ x1 + (1|grp)`, so the design columns are
# [intercept(0), x1(1)] and x1 is the only fixed-effect target.

ORACLE_CELLS <- list(

  # 1. few6_int — 6 clusters, intercept-only RE. Few-cluster baseline.
  few6_int = list(
    label = "few6_int", structure = "intercept",
    formula = "y ~ x1 + (1|grp)",
    effects = "x1=0.50", target_term = "x1", beta_star = 0.50, target_col = 1L,
    baseline_probability = 0.3,
    cluster = list(var = "grp", ICC = 0.3, n_clusters = 6L, cluster_size = 20L),
    n = 120L, seed = 71000
  ),

  # 2. few8_slope — 8 clusters, random slope on x1. §11.1 PRIMARY decision cell.
  few8_slope = list(
    label = "few8_slope", structure = "slope",
    formula = "y ~ x1 + (1|grp)",
    effects = "x1=0.50", target_term = "x1", beta_star = 0.50, target_col = 1L,
    baseline_probability = 0.3,
    cluster = list(var = "grp", ICC = 0.4, n_clusters = 8L, cluster_size = 15L),
    slopes = list(list(column = 0L, variance = 0.10,
                       corr_with_intercept = 0.3, corr_with = numeric(0))),
    n = 120L, seed = 71001
  ),

  # 3. few10_crossed — 10 clusters, slope RE + crossed (1|item). §11.1 PRIMARY
  #    crossed cell. Mirrors glmm_slope_crossed (formulas.R:966).
  few10_crossed = list(
    label = "few10_crossed", structure = "slope+crossed",
    formula = "y ~ x1 + (1|grp)",
    effects = "x1=0.40", target_term = "x1", beta_star = 0.40, target_col = 1L,
    baseline_probability = 0.3,
    cluster = list(var = "grp", ICC = 0.3, n_clusters = 10L, cluster_size = 12L),
    slopes = list(list(column = 0L, variance = 0.10,
                       corr_with_intercept = 0.3, corr_with = numeric(0))),
    extra = list(list(var = "item", kind = "crossed", n_clusters = 6L, tau_squared = 0.15)),
    n = 120L, seed = 71002
  ),

  # 4. hiICC_int — 12 clusters, intercept RE at HIGH ICC=0.6.
  hiICC_int = list(
    label = "hiICC_int", structure = "intercept",
    formula = "y ~ x1 + (1|grp)",
    effects = "x1=0.50", target_term = "x1", beta_star = 0.50, target_col = 1L,
    baseline_probability = 0.3,
    cluster = list(var = "grp", ICC = 0.6, n_clusters = 12L, cluster_size = 20L),
    n = 240L, seed = 71003
  ),

  # 5. smallclust — 20 clusters of 5 obs (tiny clusters), intercept RE.
  smallclust = list(
    label = "smallclust", structure = "intercept",
    formula = "y ~ x1 + (1|grp)",
    effects = "x1=0.50", target_term = "x1", beta_star = 0.50, target_col = 1L,
    baseline_probability = 0.3,
    cluster = list(var = "grp", ICC = 0.3, n_clusters = 20L, cluster_size = 5L),
    n = 100L, seed = 71004
  ),

  # 6. boundary — sparse near-separation (baseline 0.1, x1=0.7), intercept RE.
  boundary = list(
    label = "boundary", structure = "intercept",
    formula = "y ~ x1 + (1|grp)",
    effects = "x1=0.70", target_term = "x1", beta_star = 0.70, target_col = 1L,
    baseline_probability = 0.1,
    cluster = list(var = "grp", ICC = 0.3, n_clusters = 10L, cluster_size = 20L),
    n = 200L, seed = 71005
  )
)
names(ORACLE_CELLS) <- vapply(ORACLE_CELLS, function(c) c$label, character(1))

# Build the paired alt / null MCPowerDebug models for one cell. The alt model
# plants effects target_term=beta_star (power); the null sets target_term=0
# (Type-I); everything else (cluster spec, slopes, crossed extra, baseline) is
# identical so all flavours share one DGP. seed_off shifts the per-draw stream.
oracle_models <- function(cell, seed_off = 0L) {
  alt_case  <- cell; alt_case$effects  <- sprintf("%s=%g", cell$target_term, cell$beta_star)
  null_case <- cell; null_case$effects <- sprintf("%s=0",  cell$target_term)
  list(
    alt        = build_m4_model(alt_case,  seed_off),
    null       = build_m4_model(null_case, seed_off),
    target_term = cell$target_term,
    target_col  = cell$target_col
  )
}

# Pull the per-target list for the cell's target column from a load_data() fit.
# Falls back to the first target when the index is absent (single-predictor
# cells always carry x1 as a target, so the match is normally exact).
oracle_target <- function(fit, target_col) {
  for (tg in fit$targets) if (isTRUE(tg$target_index == target_col)) return(tg)
  fit$targets[[1]]
}
