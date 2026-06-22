# External oracle for the C12 profiled-deviance test
# (engine-core/src/lme.rs :: profiled_deviance_value_and_curvature_at_theta_1).
#
# Builds lme4's REML deviance function for the 6-row fixture (2 clusters of 3)
# via mkLmerDevfun and evaluates it at theta = 1 WITHOUT optimising. The engine
# pins the deviance at theta = 1, which is NOT this fixture's optimum (its
# optimum is the tau_hat = 0 boundary), so we evaluate devfun at a fixed theta
# rather than fitting.
#
# lme4's REML deviance is
#     devfun(theta) = log|L_theta|^2 + log|R_X|^2 + (N-P)*[1 + log(2*pi*sigma_hat^2)].
# The engine's profiled_deviance returns the same quantity minus the additive
# constant (N-P)*(1 + log(2*pi)) -- it returns  ... + (N-P)*log(sigma_hat^2).
# Hence:
#     engine_expected(theta) = devfun(theta) - (N-P)*(1 + log(2*pi)).
#
# Run: Rscript scripts/lme_devfun_theta1.R
suppressMessages(library(lme4))

x1 <- c(0.1257302210933933, -0.1321048632913019, 0.6404226504432821,
        0.10490011715303971, -0.535669373161111, 0.36159505490948474)
y  <- c(0.7718630718197979, -0.09922526468089643, 1.4699547603093044,
        1.266192714345762, -1.8688474280172964, 1.3141089963737067)
cluster <- factor(c(0, 0, 0, 1, 1, 1))
df <- data.frame(y = y, x1 = x1, cluster = cluster)

ctrl <- lmerControl(check.nobs.vs.nlev = "ignore",
                    check.nobs.vs.nRE = "ignore",
                    check.conv.singular = "ignore")
lmod   <- lFormula(y ~ x1 + (1 | cluster), data = df, REML = TRUE, control = ctrl)
devfun <- do.call(mkLmerDevfun, lmod)

N <- 6L
P <- 2L
const <- (N - P) * (1 + log(2 * pi))

for (theta in c(1.0, 1e-4)) {
  d <- devfun(theta)
  cat(sprintf("theta=%-9g devfun=%.15f  engine_expected=%.15f\n",
              theta, d, d - const))
}
