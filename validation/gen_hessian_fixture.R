# Regenerates crates/engine-core/tests/fixtures/glmm_hessian_vcov.json.
# Provenance only — the JSON is the committed source of truth for the Rust test.
# A small fixed clustered-binary GLMM fit with both glmer vcov conventions
# (use.hessian = TRUE / FALSE) extracted, so the Rust fd_hessian_cov kernel can
# match lme4's use.hessian=TRUE block exactly (the factor-of-2 deviance→cov).
suppressMessages({library(lme4); library(jsonlite)})
set.seed(2137)
n_grp <- 12; per <- 8; n <- n_grp * per
grp <- rep(seq_len(n_grp), each = per)
x1  <- rnorm(n)
eta <- -0.4 + 0.5 * x1 + rnorm(n_grp, sd = sqrt(0.3))[grp]
y   <- rbinom(n, 1, plogis(eta))
df  <- data.frame(y, x1, grp = factor(grp))
fit <- glmer(y ~ x1 + (1 | grp), data = df, family = binomial)
vc_h <- as.matrix(vcov(fit, use.hessian = TRUE))
vc_r <- as.matrix(vcov(fit, use.hessian = FALSE))
theta <- getME(fit, "theta"); beta <- as.numeric(fixef(fit))
jsonlite::write_json(
  list(case = "intercept", n = n,
       x = cbind(1, x1), y = y, cluster_ids = grp - 1L,
       theta = unname(theta), beta = beta,
       vcov_hessian = vc_h, vcov_rx = vc_r),
  "../crates/engine-core/tests/fixtures/glmm_hessian_vcov.json",
  auto_unbox = TRUE, digits = 12, matrix = "rowmajor")
