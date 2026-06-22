# tolerances.R — central L3 validation tolerances.
#
# One place to tune every L3 gate. The .rmd reports source this (after common.R)
# so a tolerance change is a one-line edit here and all reports stay in lockstep.
# These are validation-harness thresholds, consumed only by the R reports /
# common.R — NOT cross-port engine config (that lives in mcpower/configs/).
#
# Two campaigns, two gates:
#   - B<->C (model solving): the same saved bytes fit by R-SOTA and by MCPower's
#     load_data(); shared data cancels sampling error, so the gate is tight. The
#     estimate band splits by solver regime: `estimate_rel_ols` for closed-form OLS
#     (agrees with R to ~machine precision), `estimate_rel_iter` for the iterative
#     GLM (IRLS) / MLE (REML-Brent) fits (can't beat R to machine precision). Both
#     bound beta AND the decision statistic. The critical-value band splits the
#     same way: `crit_abs` for the closed-form OLS t (engine vs `qt` ~9e-14),
#     `crit_abs_iter` for the GLM/MLE z, where the engine's own normal/chi2
#     inverse-CDF agrees with R's `qnorm`/`qchisq` only to ~1.6e-9 (negligible for
#     power, but coarser than `qt`).
#   - A<->B (data generation): generated moments vs the spec values, averaged over
#     many draws; `moment_abs` for the absolute moment checks, `var_rel` for the
#     residual variance, `bias_fdr_q`/`bias_logit_abs` for beta recovery (split by
#     estimator regime — see DGP_TOL comments).

SOLVING_TOL <- list(
  estimate_rel_ols  = 1e-11,  # beta + statistic, closed-form OLS (B<->C).
                              # Measured worst over the 14 OLS cases: 1.17e-12
                              # (beta) / 1.18e-14 (stat) -> ~8.5x margin. 1e-12
                              # would FAIL ols_corr_b; 1e-11 is the tightest safe gate.
  estimate_rel_iter = 1e-4,   # beta + statistic, iterative GLM/MLE (B<->C).
                              # IRLS / REML-Brent agree with R only to ~1e-4--1e-6,
                              # not machine precision -- this is their floor.
  crit_abs          = 1e-9,   # critical value, closed-form OLS t (engine vs qt
                              # ~9e-14 -> ~5 orders of margin).
  crit_abs_iter     = 1e-8,   # critical value, GLM/MLE z. The engine's normal/chi2
                              # inverse-CDF agrees with R's qnorm/qchisq to ~1.58e-9
                              # (measured, constant across z-cases); 1e-8 is ~6x margin
                              # and still trips a real crit regression.
  vc_rel            = 1e-3,   # variance components, B<->C vs lme4 VarCorr on the
                              # same bytes. Two REML optimizers: BOBYQA rho_end
                              # 1e-6 on theta => tau-hat^2 noise ~ 2*theta*1e-6*sigma^2;
                              # lme4 carries its own convergence tolerance. Amend
                              # only with measured evidence + user approval (Gate-0 rule).
  vc_abs            = 1e-5,   # absolute floor for near-zero components.
  slope_var_rel  = 1e-3,   # random-slope variance vs lme4 VarCorr, same bytes
                           # (two REML optimizers; same basis as vc_rel).
  slope_corr_abs = 2e-3    # intercept–slope correlation, abs band. ρ̂ is a
                           # ratio of θ̂ entries (λ₁₀/√(λ₁₀²+λ₁₁²)); BOBYQA
                           # rho_end 1e-6 placement + lme4 convergence noise.
                           # Amend only with measured evidence + user approval.
)

DGP_TOL <- list(
  moment_abs   = 0.01,   # predictor mean/sd, correlation, factor proportion, ICC (A<->B)
  var_rel      = 0.01,   # residual / within-cluster variance (A<->B)
  icc_marginal_abs = 0.01,  # observed (marginal) ICC vs predicted tau^2/(tau^2+sigma^2+Var(Xbeta))
                         # (A<->B). The set ICC is CONDITIONAL; the raw outcome's ICC is lower by
                         # the fixed-effect variance. The prediction uses each draw's OWN variance
                         # components (between=tau^2, within=sigma^2) + empirical Var(Xbeta), so the
                         # check tests that the variance budget composes -- NOT the conditional ICC's
                         # finite-sample bias (that is gated by the ICC row). Measured worst K-draw
                         # gap over the 8 cluster cases ~0.0022 (lme_factor_b, K=200) -> ~4.5x margin;
                         # the K=800 report is tighter still.
  # Beta recovery gate (A<->B), SPLIT by estimator regime.
  #   OLS/LME: the per-coefficient z-of-the-mean (mean beta_hat vs true, in
  #     standard-errors-of-the-mean) is ~N(0,1) under a correct DGP. The gate is a
  #     HARD stopifnot over ~103 coefficients pooled across all cases, so a fixed
  #     z-threshold below ~4 aborts nearly every clean render on multiplicity alone
  #     (P(any |z|>2) ~ 1). Instead control the FALSE-DISCOVERY RATE across the
  #     pooled OLS/LME recovery tests via Benjamini-Hochberg at bias_fdr_q: under a
  #     clean DGP P(any rejection) ~ q, while a genuine systematic bias still gets
  #     flagged with power.
  #   logit (GLM): finite-sample MLE bias is SYSTEMATIC (~3-3.8 SE at K=1600), not a
  #     detectable-vs-null effect, so gate it on an ABSOLUTE band (mirrors
  #     SCENARIO_TOL$glm_beta_abs): |mean beta_hat - true|.
  bias_fdr_q     = 0.001,  # OLS/LME: Benjamini-Hochberg FDR level on pooled z-recovery p-values
  bias_logit_abs = 0.02,   # logit: |mean beta_hat - true| (absolute, like glm_beta_abs)
  parabola_abs = 1e-5    # strict-bootstrap parabola-preservation: max |x2_std - f(x1_std)|
                         # per draw. Strict floor ~3e-8 (CSV float precision); NORTA would
                         # be O(1). 1e-5 is fully discriminating between the two paths.
)

# L3 model-based crossing validation (validation_crossing.rmd).
# COARSE: auto-grid ~12 points at 1600 sims.  DENSE: by=2, 100 000 sims.
# Gate: |coarse n_achievable - dense n_achievable| / dense <= n_rel.
# ci_covers is a boolean gate (not a tolerance): CI must bracket the dense
# crossing. It is stated here for documentation but enforced in the rmd.
#
# Calibration (ols_two_b, x1=0.40/x2=0.25, range 20-200):
#   measured worst at smoke scale (2000 sims / 12 pts): 0.023 (x1).
#   full-size run (100 000 sims) is much tighter; 4× safety margin → 0.10.
CROSSING_TOL <- list(
  n_rel = 0.10   # relative band: |coarse fitted - dense| / dense crossing.
                 # Smoke measured worst: 0.023 (x1 at reduced sims);
                 # 4× safety factor gives 0.09 -> rounded to 0.10.
)

# get_effects_from_data round-trip — the acceptance gate for the standardized-effect
# convention per estimator (continuous main effects; factor/interaction scaling is a
# separate, still-open item). Simulate at size s, recover via get_effects_from_data,
# compare to the *convention-predicted* value (not the raw input s): OLS z-scores the
# outcome and so recovers the standardized coefficient beta / sqrt(sum beta^2 + 1)
# (N(0,1) residual, independent continuous predictors); GLM (logit) and MLE (mixed)
# fit the native outcome and recover beta directly. The gate is on the K-draw MEAN —
# single-draw noise ~1/sqrt(n) is averaged down (the A<->B harness pattern). Measured
# worst over the 6 cases: 0.007 at n=4000, K=20 -> ~2.8x margin. A wrong estimator,
# sign, or scale shifts the mean by 0.1-0.5, far outside this band.
GETEFFECTS_TOL <- list(
  mean_abs = 0.02   # |mean recovered - analytic expected|, K-draw average
)

# L5 scenario-perturbation gates (validation_scenarios.rmd). The L5 oracle is
# each knob's documented perturbation LAW (a distribution), so the gate
# primitive is the SE-of-mean z-score band: per-draw statistics vs the law,
# z = |mean - law| / (sd/sqrt(K)) <= z_c (the beta_rows pattern — common.R
# zgate/zgate_se/sdgate). Laws are analytic (clamp truncation, censored-t3
# table moments, NORTA oracle, binomial counts) — never tuned constants; the
# few non-z constants below are derived bounds, documented at the point of
# derivation.
SCENARIO_TOL <- list(
  z_c        = 4,     # shared z-band, all SE-of-mean / SD / binomial gates.
                      # |z| > 4 by chance is < 1e-4 per gate, so one run's ~70
                      # gates stay below ~1% family-wise false alarm while a
                      # mis-scaled knob (s vs s/sqrt(2), lambda vs sqrt(lambda),
                      # h vs h^2) lands 10-100 sigma out.
  alloc_dev_max = 1.0,# sampled_factor_proportions (exact): max |count_g - n*p_g| per draw. The
                      # largest-remainder walk is exact (deviation <= 1) for
                      # 2-level factors and stays within ~1 for the mild
                      # proportion vectors L5 uses (engine fixed_level_next
                      # property); counts must also be IDENTICAL across draws
                      # (the walk consumes no RNG).
  glm_beta_abs = 0.02,# B1 logit absolute leak band: |K-mean beta-hat - true|.
                      # Logistic MLE's finite-sample bias (O(p/n), here ~1e-3)
                      # swamps any scenario effect at this K, so the z-band is
                      # the wrong gate; 0.02 (the GETEFFECTS_TOL magnitude)
                      # catches gross mean leaks while admitting the bias.
                      # beta-unbiased is NOT the GLM gate — Tier A calibration
                      # and the flip rate are.
  golden_abs = 1e-9   # MCPower-golden reproducibility: |stat - frozen stat|.
                      # Statistics are deterministic given the committed seeds,
                      # so any engine/DGP drift moves them by orders more than
                      # this; the band only absorbs libm/platform last-bit
                      # noise (the L3 hash-tripwire role, numeric form).
)

# MLMM_TOL — M2/M3/M4 LMM/GLMM gate bands moved out of validation_MLE_solving.rmd
# (G4: "one place to tune every L3 gate"). B<->C floors are optimizer-vs-optimizer
# (BOBYQA placement noise on the SAME bytes); A<->B pads are Monte-Carlo slop added
# to 4*MC_SE so a K-draw mean within sampling noise of the spec passes.
MLMM_TOL <- list(
  beta_abs_floor   = 1e-5,  # B<->C |beta_hat_R - beta_hat_MCPower| floor (below this, rel band is noise)
  stat_abs_floor   = 1e-4,  # B<->C |z_R - z_MCPower| floor (Wald-z optimizer placement)
  ab_beta_pad      = 0.005, # A<->B fixed-effect recovery pad (lightest tail)
  ab_vc_pad        = 0.01,  # A<->B intercept tau^2 / residual sigma^2 pad (M2; heavier tail than beta)
  ab_slope_var_pad = 0.02,  # A<->B random-SLOPE variance pad (M3/M4; harder to recover at moderate K)
  ab_corr_pad      = 0.05,  # A<->B RE-correlation pad (noisiest VC quantity: ratio of theta entries)
  # M4 GLMM B<->C gross-regression backstop (the rmd's XFAIL_BACKSTOP, a 4-field list).
  # M4 B<->C is a documented XFAIL (RX/PLS Schur SE ~3% anticonservative vs glmer), so the
  # gate gives M4 beta/z THIS, not the tight floors, or it would FAIL every run. The rmd keeps
  # `XFAIL_BACKSTOP <- MLMM_TOL$xfail_backstop` as a thin alias so campaign + gate read ONE value.
  xfail_backstop   = list(z_abs = 2.0, beta_abs = 0.05, vc_abs = 0.1, corr_abs = 0.1)
)

# M4 GLMM solver tolerances — clustered logistic (glmer Laplace vs MCPower-GLMM).
#
# B↔C reference is lme4::glmer with family = binomial (Laplace approximation,
# like MCPower-GLMM). The two iterative optimizers (BOBYQA in both) agree to
# the same band as the LMM case — two optima on the same bytes, not machine
# precision. The τ²→0 collapse gate and the Laplace-bias cell use wider bands
# reflecting the physical quantity being asserted (plain GLM agreement /
# approximation gap) rather than solver precision.
GLMM_TOL <- list(
  beta_rel       = SOLVING_TOL$estimate_rel_iter,  # β̂ B↔C vs glmer (Laplace), iterative floor
  beta_abs_floor = 1e-3,    # absolute β̂ floor (BOBYQA rho_end 1e-6 + glmer convergence)
  tau_rel        = SOLVING_TOL$slope_var_rel,      # τ̂² vs glmer VarCorr, same bytes
  corr_abs       = SOLVING_TOL$slope_corr_abs,     # RE correlation, same bytes
  collapse_beta  = 1e-3,    # τ²→0: MCPower-GLMM β̂ vs plain glm() β̂, same bytes
  collapse_stat  = 1e-2,    # τ²→0: Wald z agreement, same bytes
  laplace_bias_beta_abs = 0.05  # Laplace vs nAGQ=7 β̂: the approximation gap itself
)
