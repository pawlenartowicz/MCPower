# External oracle for the posthoc contrast golden test
# (engine-core/src/posthoc.rs).
#
# A small 3-level-factor + continuous-covariate OLS design. Computes, for each
# contrast c, the squared t-statistic  t^2 = (c'b)^2 / (sigma^2 * c'(X'X)^-1 c)
# with sigma^2 = RSS/(n-p) -- exactly the estimator engine-core's
# `evaluate_posthoc` uses (se^2 = sigma_hat^2 * ||L^-1 c||^2). Prints the exact
# x/y doubles to paste into the Rust fixture plus the golden t^2 per contrast.
#
# Run: Rscript scripts/posthoc_contrast_golden.R
set.seed(2137)
g <- factor(rep(c("A", "B", "C"), each = 4))                 # 12 rows, 3 levels
x <- rnorm(12)
y <- 1.0 + 1.5 * (g == "B") + 0.2 * (g == "C") + 0.8 * x + rnorm(12) * 0.5

X <- model.matrix(~ x + g)            # cols: (Intercept), x, gB, gC  -> p = 4
fit <- lm(y ~ x + g)
b <- coef(fit)
n <- length(y); p <- ncol(X)
sigma2 <- sum(resid(fit)^2) / (n - p)
XtXinv <- solve(crossprod(X))

contr <- list(
  gB          = c(0, 0, 1,  0),       # level B vs A
  gC          = c(0, 0, 0,  1),       # level C vs A
  gB_minus_gC = c(0, 0, 1, -1)        # level B vs C
)

repr <- function(v) paste(sprintf("%.17g", v), collapse = ", ")
cat("// --- paste into Rust fixture (column order: 1, x, dB, dC) ---\n")
cat(sprintf("let x_cont = [%s];\n", repr(x)))
cat(sprintf("let y = [%s];\n", repr(y)))
cat("// group: rows 0-3 = A (ref), 4-7 = B, 8-11 = C\n")
cat("// --- golden t^2 per contrast ---\n")
for (nm in names(contr)) {
  cc  <- contr[[nm]]
  est <- sum(cc * b)
  se2 <- sigma2 * as.numeric(t(cc) %*% XtXinv %*% cc)
  cat(sprintf("// %-12s est=% .10f  t_sq=%.12f\n", nm, est, est^2 / se2))
}
