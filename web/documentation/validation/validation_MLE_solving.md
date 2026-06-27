---
title: "MCPower — Validating the MLE (Mixed-Effects) Solver"
description: "MCPower MLE solver validation: byte-for-byte comparison of mixed-model REML fixed-effect coefficients and Wald z-statistics against R lme4::lmer."
right_sidebar: body
---

# MCPower validation — MLE solver (data → results)

# What this report shows

MCPower generates a clustered dataset from a formula, then **fits it
back** to recover the fixed-effect coefficients. This report checks the
*fitting* half: given one fixed dataset, does MCPower’s own mixed-model
solver return the same numbers a trusted, independent R fit returns?

For each formula we take the exact bytes MCPower saved (the same design
matrix, outcome, and cluster ids, byte-for-byte), fit them two ways —

- **R (the reference):** `lme4::lmer` with REML on the design matrix
  through the origin plus a random intercept per cluster — the textbook
  restricted-maximum- likelihood mixed-model fit.
- **MCPower:** the engine’s own `load_data()` path, which runs the
  *same* REML profile solver (Brent on the variance component) the power
  simulation uses.

— and compare the fitted fixed-effect coefficients, the test statistics,
and the decision threshold. The statistic is the **Wald *z*** on the
fixed effect (`fixef / sqrt(diag(vcov))`), *not* lmerTest’s
Satterthwaite *t* — MCPower’s own test definition is the Wald z. Because
both fit the identical bytes, sampling noise cancels: any difference is
a solver discrepancy.

The chosen cases sit at ICC 0.1–0.2 with 20–30 clusters, well clear of
the **τ̂ ≈ 0 boundary**. If a case ever collapsed to that boundary,
MCPower’s OLS-fallback fit would diverge from `lmer`’s REML fit and the
statistic band would **FAIL** vs R — surfacing the boundary indirectly.
None of the cases below are expected to hit it.

# How the check works

1.  **Provenance.** Re-generate the dataset from the committed seed and
    confirm its content hash matches the saved file — proof the bytes
    haven’t drifted.
2.  **B (R) vs C (MCPower)** on those bytes: fixed-effect coefficients,
    the Wald *z* statistics (β̂/se), and the critical value
    `qnorm(1 − α/2)` ≈ 1.96.
3.  **C vs A** (a readable sanity overlay): MCPower’s recovered
    coefficients next to the formula’s true values.

# The thresholds

| Quantity | Allowed difference | Why |
|----|----|----|
| Coefficient (β) | 10^{-4} relative | REML profile optimisation — agrees with `lmer` inside this band |
| Statistic (*z*) | 10^{-4} relative | derived from the same β and variance |
| Critical value | 10^{-8} absolute | engine’s own normal/χ² quantile vs R’s `qnorm` ≈1.6e-9 |

These are the B↔C gates from `tolerances.R`. MLE is a **REML profile
optimisation** (Brent on the variance component) — the loosest of the
three fits against the iterative `estimate_rel_iter` band, since two
independent optimisers (MCPower’s and `lmer`’s) need only agree to that
band, not to machine precision. The per-formula tables below show each
formula’s actual margin; a case sitting near the band edge is called out
in the summary.

| Formula                |   n | B↔C  | Converged | Reproduces |
|:-----------------------|----:|:-----|:----------|:-----------|
| y ~ x1 + (1\|grp)      | 600 | PASS | yes       | yes        |
| y ~ x1 + (1\|grp)      | 600 | PASS | yes       | yes        |
| y ~ x1 + x2 + (1\|grp) | 750 | PASS | yes       | yes        |
| y ~ x1 + x2 + (1\|grp) | 750 | PASS | yes       | yes        |
| y ~ x1\*x2 + (1\|grp)  | 750 | PASS | yes       | yes        |
| y ~ x1\*x2 + (1\|grp)  | 900 | PASS | yes       | yes        |
| y ~ x1 + g + (1\|grp)  | 750 | PASS | yes       | yes        |
| y ~ x1 + g + (1\|grp)  | 900 | PASS | yes       | yes        |

> **All MLE formulas pass:** MCPower’s REML solver matches `lme4::lmer`
> well inside the iterative band on every saved dataset, and every file
> reproduces from its seed.

png 2 ![B↔C relative agreement of every coefficient and statistic across
the base MLE formulas, against the tolerance
gate.](figures/mle_agreement.png)

*Each point is one fitted quantity on one saved dataset; the red line is
the gate. Two independent REML optimisers (MCPower’s and `lme4`’s) agree
inside the iterative band on every quantity.*

## y = 0.5·x1 + per-grp random intercept (ICC 0.20) + noise

R formula `y ~ x1 + (1|grp)` · linear mixed model (random intercept) ·
n=600, seed=2137. File reproduces from seed: **yes**.

| Term | True (formula) | β (R = MCPower) | t / z | crit | rel. diff (β, stat) | crit abs. diff | Verdict |
|:---|---:|---:|---:|---:|:---|:---|:---|
| intercept | 0.0 | 0.078383 | NA | NA | 3.0e-08 | — | PASS |
| x1 | 0.5 | 0.404979 | 10.34034 | 1.959964 | 8.2e-07 | 1.6e-09 | PASS |

## y = 0.3·x1 + per-grp random intercept (ICC 0.30) + noise

R formula `y ~ x1 + (1|grp)` · linear mixed model (random intercept) ·
n=600, seed=2138. File reproduces from seed: **yes**.

| Term | True (formula) | β (R = MCPower) | t / z | crit | rel. diff (β, stat) | crit abs. diff | Verdict |
|:---|---:|---:|---:|---:|:---|:---|:---|
| intercept | 0.0 | 0.131583 | NA | NA | 3.6e-09 | — | PASS |
| x1 | 0.3 | 0.347749 | 8.595397 | 1.959964 | 4.3e-07 | 1.6e-09 | PASS |

## y = 0.5·x1 + 0.3·x2 + per-grp random intercept (ICC 0.20) + noise

R formula `y ~ x1 + x2 + (1|grp)` · linear mixed model (random
intercept) · n=750, seed=2137. File reproduces from seed: **yes**.

| Term | True (formula) | β (R = MCPower) | t / z | crit | rel. diff (β, stat) | crit abs. diff | Verdict |
|:---|---:|---:|---:|---:|:---|:---|:---|
| intercept | 0.0 | -0.007542 | NA | NA | 5.6e-07 | — | PASS |
| x1 | 0.5 | 0.440708 | 12.927637 | 1.959964 | 7.6e-07 | 1.6e-09 | PASS |
| x2 | 0.3 | 0.341816 | 9.183627 | 1.959964 | 9.0e-07 | 1.6e-09 | PASS |

## y = 0.3·x1 + 0.5·x2 + per-grp random intercept (ICC 0.30) + noise

R formula `y ~ x1 + x2 + (1|grp)` · linear mixed model (random
intercept) · n=750, seed=2138. File reproduces from seed: **yes**.

| Term | True (formula) | β (R = MCPower) | t / z | crit | rel. diff (β, stat) | crit abs. diff | Verdict |
|:---|---:|---:|---:|---:|:---|:---|:---|
| intercept | 0.0 | 0.102675 | NA | NA | 1.9e-09 | — | PASS |
| x1 | 0.3 | 0.357704 | 9.980624 | 1.959964 | 1.1e-07 | 1.6e-09 | PASS |
| x2 | 0.5 | 0.497794 | 13.989819 | 1.959964 | 9.8e-08 | 1.6e-09 | PASS |

## y = 0.5·x1 + 0.3·x2 + 0.3·x1:x2 + per-grp random intercept (ICC 0.20) + noise

R formula `y ~ x1*x2 + (1|grp)` · linear mixed model (random intercept)
· n=750, seed=2137. File reproduces from seed: **yes**.

| Term | True (formula) | β (R = MCPower) | t / z | crit | rel. diff (β, stat) | crit abs. diff | Verdict |
|:---|---:|---:|---:|---:|:---|:---|:---|
| intercept | 0.0 | -0.007002 | NA | NA | 3.6e-07 | — | PASS |
| x1 | 0.5 | 0.447848 | 12.953272 | 1.959964 | 8.3e-07 | 1.6e-09 | PASS |
| x2 | 0.3 | 0.341794 | 9.187226 | 1.959964 | 9.3e-07 | 1.6e-09 | PASS |
| x1:x2 | 0.3 | 0.254123 | 6.738949 | 1.959964 | 5.7e-07 | 1.6e-09 | PASS |

## y = 0.4·x1 + 0.3·x2 + 0.2·x1:x2 + per-grp random intercept (ICC 0.30) + noise

R formula `y ~ x1*x2 + (1|grp)` · linear mixed model (random intercept)
· n=900, seed=2138. File reproduces from seed: **yes**.

| Term | True (formula) | β (R = MCPower) | t / z | crit | rel. diff (β, stat) | crit abs. diff | Verdict |
|:---|---:|---:|---:|---:|:---|:---|:---|
| intercept | 0.0 | 0.202794 | NA | NA | 7.4e-10 | — | PASS |
| x1 | 0.4 | 0.428858 | 12.586038 | 1.959964 | 8.1e-08 | 1.6e-09 | PASS |
| x2 | 0.3 | 0.289511 | 8.587036 | 1.959964 | 1.6e-07 | 1.6e-09 | PASS |
| x1:x2 | 0.2 | 0.251104 | 7.673815 | 1.959964 | 1.7e-07 | 1.6e-09 | PASS |

## y = 0.3·x1 + 0.3·g\[2\] + per-grp random intercept (ICC 0.20) + noise

R formula `y ~ x1 + g + (1|grp)` · linear mixed model (random intercept)
· n=750, seed=2137. File reproduces from seed: **yes**.

| Term | True (formula) | β (R = MCPower) | t / z | crit | rel. diff (β, stat) | crit abs. diff | Verdict |
|:---|---:|---:|---:|---:|:---|:---|:---|
| intercept | 0.0 | -0.042469 | NA | NA | 6.6e-08 | — | PASS |
| x1 | 0.3 | 0.241373 | 7.079191 | 1.959964 | 8.8e-07 | 1.6e-09 | PASS |
| g\[2\] | 0.3 | 0.375982 | 5.275995 | 1.959964 | 6.6e-07 | 1.6e-09 | PASS |

## y = 0.4·x1 + 0.5·g\[2\] + 0.8·g\[3\] + per-grp random intercept (ICC 0.30) + noise

R formula `y ~ x1 + g + (1|grp)` · linear mixed model (random intercept)
· n=900, seed=2138. File reproduces from seed: **yes**.

| Term | True (formula) | β (R = MCPower) | t / z | crit | rel. diff (β, stat) | crit abs. diff | Verdict |
|:---|---:|---:|---:|---:|:---|:---|:---|
| intercept | 0.0 | 0.123802 | NA | NA | 6.8e-09 | — | PASS |
| x1 | 0.4 | 0.435261 | 12.890829 | 1.959964 | 5.2e-08 | 1.6e-09 | PASS |
| g\[2\] | 0.5 | 0.612951 | 2.210123 | 1.959964 | 3.2e-06 | 1.6e-09 | PASS |
| g\[3\] | 0.8 | 1.000041 | 3.147536 | 1.959964 | 3.2e-06 | 1.6e-09 | PASS |

| item | value |
|:---|:---|
| generated | 2026-06-26 |
| R | R version 4.5.3 (2026-03-11) |
| mcpower | 1.0.3 |
| MLE solving bands (beta/stat rel · crit abs) | 0.0001 rel / 0.0001 rel / 1e-08 abs |

## M2 — crossed & nested groupings (general lmm path)

A↔B: lme4 recovers the spec’s β and variance components from MCPower’s
own draws (the spec is the oracle; K independent draws, Monte-Carlo
bands). B↔C: MCPower and lme4 fit the SAME saved bytes; β̂ at
`estimate_rel_iter` with the optimizer-vs-optimizer abs floors (stat
1e-4 / β̂ 1e-5), variance components at `vc_rel`/`vc_abs`. Wald z on both
sides — never lmerTest t.

**lmm_crossed_a**: PASS (25/25 lme4-clean draws)

**lmm_crossed_b**: PASS (24/25 lme4-clean draws)

**lmm_nested_a**: PASS (25/25 lme4-clean draws)

**lmm_nested_b**: PASS (25/25 lme4-clean draws)

**lmm_crossed_nested_a**: PASS (25/25 lme4-clean draws)

**lmm_crossed_a**: A\<-\>B PASS (K=200)

**lmm_crossed_b**: A\<-\>B PASS (K=200)

**lmm_nested_a**: A\<-\>B PASS (K=200)

**lmm_nested_b**: A\<-\>B PASS (K=200)

**lmm_crossed_nested_a**: A\<-\>B PASS (K=200)

Reproduce: `Rscript mcpower/validation/data_generation.r` then
`rmarkdown::render("mcpower/validation/validation_MLE_solving.rmd", output_dir = "mcpower/web/documentation/validation")`.

## M3 — random slopes (general lmm path: standalone, multi-slope, composition)

B↔C: MCPower and lme4 fit the SAME saved bytes; β̂ at `estimate_rel_iter`
with the optimizer-vs-optimizer abs floors (stat 1e-4 / β̂ 1e-5); every
RE variance at `slope_var_rel`/`vc_abs`; the full RE correlation vector
(`re_corr`) at `slope_corr_abs`. A↔B: lme4 recovers the spec’s β, the
per-component variances, and the set correlations from MCPower’s own
draws over K draws (the spec is the oracle). Per-component boundary: a
τ→0 component must drive its boundary rate up while the others stay low.
Wald z on both sides — never lmerTest t.

**lmm_slope_a**: PASS (25/25 lme4-clean draws)

**lmm_slope_b**: PASS (25/25 lme4-clean draws)

**lmm_multislope**: PASS (25/25 lme4-clean draws)

**lmm_slope_crossed**: PASS (24/25 lme4-clean draws)

**lmm_slope_a**: A↔B PASS (K=300)

**lmm_slope_b**: A↔B PASS (K=300)

**lmm_multislope**: A↔B PASS (K=300)

**lmm_slope_crossed**: A↔B PASS (K=300)

    ## Power Analysis — MLE  N=480  sims=800  α=0.05  target=80%
    ## formula: y ~ x1 + (1|grp)
    ## 
    ## ───────────────────────────────────
    ## Test                 Power   Target
    ## ───────────────────────────────────
    ## x1                    100%      80%
    ## ───────────────────────────────────

    ## Power Analysis — MLE  N=480  sims=800  α=0.05  target=80%
    ## formula: y ~ x1 + (1|grp)
    ## 
    ## ───────────────────────────────────
    ## Test                 Power   Target
    ## ───────────────────────────────────
    ## x1                   95.0%      80%
    ## ───────────────────────────────────

    ## Slope τ²=0: singular_fit_rate = 0.626 (expect high)

    ## Slope τ²=0.10: singular_fit_rate = 0.010 (expect low)

## M4 — clustered logistic GLMM (intercept, slope, multi-slope, composition, Laplace-bias)

B↔C: MCPower-GLMM and `lme4::glmer(family = binomial)` fit the SAME
generated bytes; β̂ at `GLMM_TOL$beta_rel` / `beta_abs_floor`; RE
variances at `GLMM_TOL$tau_rel`; RE correlations at `GLMM_TOL$corr_abs`.
Wald z (β̂/se) on both sides. Draws where `glmer` reports convergence
warnings or a singular fit are skipped (same policy as M3: two
optimizers on the same bytes need not share the optimum at a boundary).

> **B↔C is a documented XFAIL, not a hard gate.** MCPower-GLMM diverges
> from `glmer`’s *default* summary on two known, deferred axes (full
> rationale in the workspace doc
> `docs/ideas-features/mixed-model-significance-reference.md`): **(z)**
> MCPower’s Wald SE is the RX/PLS Schur
> (`= glmer vcov(use.hessian = FALSE)`), ~3% anticonservative vs glmer’s
> default `use.hessian = TRUE` — a deliberate, documented, deferred
> choice (that doc’s §“Second axis”), *not* a regression; **(β̂/τ̂²/ρ̂)**
> MCPower-BOBYQA vs glmer-Laplace agree on the same bytes only to ~2–3×
> the LMM-inherited `GLMM_TOL` bands, plus glmer’s few-cluster
> convergence noise. The cell logs the measured gaps and renders green,
> tripping only a generous gross-regression backstop; the engine’s own
> brute-force-Laplace unit tests are the fine fit-regression guard.

A↔B: `glmer` recovers the spec’s β and variance components from
MCPower’s draws over K draws (the spec is the oracle; Monte-Carlo
bands).

Section 8.4 asserts τ²→0 collapse: using a near-boundary config
(ICC=0.05, 10 clusters × 5 obs) where the non-negative variance
estimator pins at τ̂²=0 in ~55–60% of draws, MCPower-GLMM must agree with
plain `glm()` on those boundary-pinned draws within
`GLMM_TOL$collapse_beta` / `collapse_stat`. Non-pinned draws are
excluded from the tight gate (a fitted RE legitimately makes the GLMM
diverge from glm).

Section 8.5 is the Laplace-bias cell: MCPower-GLMM (Laplace) vs
`glmer(nAGQ = 7)`; the gap must stay within
`GLMM_TOL$laplace_bias_beta_abs`.

**glmm_intercept**: XFAIL — 60 z (SE convention, 2.6% max) + 13 β̂/τ̂²/ρ̂
(optimizer band: β̂ 1.7e-03, τ̂² 2.8e-03, ρ̂ 0.0e+00) (60/60 glmer-clean
draws)

**glmm_slope**: XFAIL — 40 z (SE convention, 8.6% max) + 91 β̂/τ̂²/ρ̂
(optimizer band: β̂ 2.9e-03, τ̂² 4.0e-03, ρ̂ 5.1e-03) (40/60 glmer-clean
draws)

**glmm_multislope**: XFAIL — 30 z (SE convention, 7.8% max) + 62 β̂/τ̂²/ρ̂
(optimizer band: β̂ 2.6e-03, τ̂² 2.9e-03, ρ̂ 6.1e-03) (15/60 glmer-clean
draws)

**glmm_slope_crossed**: XFAIL — 37 z (SE convention, 9.5% max) + 127
β̂/τ̂²/ρ̂ (optimizer band: β̂ 3.6e-03, τ̂² 6.6e-03, ρ̂ 2.6e-02) (37/60
glmer-clean draws)

## 8.3b Wald-SE convention (Oracle-1)

The engine ships two GLMM Wald-SE conventions reachable per fit: `rx`
(the internal RX/PLS Schur SE — identical to what `glmer` reports under
`vcov(use.hessian = FALSE)`) and `hessian` (a finite-difference
observed-Hessian SE, matching `glmer`’s default
`vcov(use.hessian = TRUE)`). **Oracle-1** pins both conventions against
`glmer` on the *same generated bytes*: for the shipped `M4_GLMM_CASES`
plus the harsh `ORACLE_CELLS` (few clusters, high ICC, tiny clusters,
near-separation), it refits each clean draw in `glmer` and compares the
engine’s target SEs to the two `vcov` flavours. This is a CONVENTION
check (does the engine compute the SE `glmer` would?), distinct from the
statistical Oracle-2 (is that SE the *right* finite-sample SE?).
Expectation: `rx` matches `vcov(FALSE)` to 4–5 significant figures (same
Schur computation); `hessian` sits within the FD band (~1e-3 relative on
the diagonal). At the harshest cells the FD-Hessian may strain — that
residual is reported as a finding, not gated.

| Cell | clean draws | rx vs vcov(F) mean | rx vs vcov(F) max | hess vs vcov(T) mean | hess vs vcov(T) max |
|:---|---:|---:|---:|---:|---:|
| glmm_intercept | 30 | 0.0000980 | 0.0001896 | 0.00540 | 0.00915 |
| glmm_slope | 22 | 0.0004707 | 0.0010742 | 0.01220 | 0.02050 |
| glmm_multislope | 11 | 0.0006205 | 0.0019027 | 0.01232 | 0.02584 |
| glmm_slope_crossed | 17 | 0.0004877 | 0.0013428 | 0.01563 | 0.02685 |
| few6_int | 29 | 0.0001968 | 0.0007165 | 0.04320 | 0.98032 |
| few8_slope | 7 | 0.0010401 | 0.0063649 | 0.02123 | 0.12168 |
| few10_crossed | 6 | 0.0006336 | 0.0018790 | 0.01464 | 0.04054 |
| hiICC_int | 29 | 0.0004461 | 0.0013959 | 0.02746 | 0.05822 |
| smallclust | 26 | 0.0000986 | 0.0008882 | 0.00283 | 0.02885 |
| boundary | 24 | 0.0003738 | 0.0009031 | 0.01575 | 0.11231 |

Oracle-1 — Wald-SE convention gaps (relative). rx should match glmer
vcov(use.hessian=FALSE) to ~4-5 sig figs; hessian within the FD band
(~1e-3 diag).

**glmm_intercept**: A↔B PASS (K=200)

**glmm_slope**: A↔B PASS (K=200)

**glmm_multislope**: A↔B PASS (K=200)

**glmm_slope_crossed**: A↔B PASS (K=200)

    ## **τ²→0 collapse (ICC=0.05, 10×5)**: PASS — 14/25 draws pinned; worst |Δβ̂| 7.92e-08, |Δz| 3.39e-04 (25 draws, τ̂² mean 0.309)

    ## **Laplace-bias cell**: max |Δβ̂_x1| = 0.0136 (limit 0.0500), mean = 0.0067 — within limit
