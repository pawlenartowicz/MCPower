# data_generation.r — generate + save one dataset per validation case.
#
# Drives MCPowerDebug$create_data() for every case in formulas.R and writes the
# generated (design, outcome, cluster_ids) to data/<label>.rds with a content
# hash. This is the formula -> data half of L3. The saved datasets are the load-back
# artifact for the
# validation .rmd reports:
#   - A<->B (validation_data_generation.rmd): descriptives + R-SOTA recovery;
#   - B<->C (validation_*_solving.rmd): R refit vs MCPower's load_data() refit,
#     asserting the loaded bytes match the hash stored here.
#
# Data is git-TRACKED (a golden: a missing dataset silently re-freezes from the
# current engine, so deletion must be visible in git — see validation/.gitignore). Run:
#   Rscript mcpower/validation/data_generation.r
# Requires R with mcpower, digest.

# Resolve paths relative to this script, so cwd doesn't matter.
script_dir <- local({
  fa <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
  if (length(fa)) dirname(normalizePath(sub("^--file=", "", fa[1]))) else getwd()
})
DATA_DIR <- file.path(script_dir, "data")
dir.create(DATA_DIR, showWarnings = FALSE, recursive = TRUE)
source(file.path(script_dir, "common.R"))     # build_model, content_hash, %||%
source(file.path(script_dir, "formulas.R"))   # CASES

mcpower_ver <- as.character(utils::packageVersion("mcpower"))
cat(sprintf("Generating %d cases -> %s  (mcpower %s)\n",
            length(ALL_GEN_CASES), DATA_DIR, mcpower_ver))

for (case in ALL_GEN_CASES) {
  needs_extras <- !is.null(case$slopes) || !is.null(case$extra) || !is.null(case$extra_groupings)
  m <- if (needs_extras) build_extras_model(case) else build_model(case)
  m$.debug_n      <- as.integer(case$n)
  m$.debug_n_sims <- 1L
  m$.debug_seed   <- as.numeric(case$seed)
  d <- m$create_data()
  h <- content_hash(d$design, d$outcome, d$cluster_ids)

  saveRDS(
    list(
      label              = case$label,
      spec               = case,                   # full DGP spec (for the .rmd reports)
      columns            = d$columns,
      design             = d$design,
      outcome            = d$outcome,
      cluster_ids        = d$cluster_ids,
      extra_grouping_ids = d$extra_grouping_ids,   # NULL for non-extras; needed by R-SOTA fit
      hash               = h,
      n                  = case$n,
      n_predictors       = ncol(d$design),
      mcpower            = mcpower_ver
    ),
    file = file.path(DATA_DIR, paste0(case$label, ".rds"))
  )

  cat(sprintf("  %-20s n=%-4d cols=%-2d [%s] clusters=%-4s hash=%s\n",
              case$label, nrow(d$design), ncol(d$design),
              paste(d$columns, collapse = ","),
              if (is.null(d$cluster_ids)) "-" else length(unique(d$cluster_ids)),
              substr(h, 1, 12)))
}

cat(sprintf("Done: %d datasets written.\n", length(ALL_GEN_CASES)))
