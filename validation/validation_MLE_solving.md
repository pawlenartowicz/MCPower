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

## y = 0.5·x1 + per-grp random intercept (ICC 0.20) + noise

R formula `y ~ x1 + (1|grp)` · linear mixed model (random intercept) ·
n=600, seed=2137. File reproduces from seed: **yes**.

| Term | True (formula) | R beta | MCPower beta | R t | MCPower stat | R crit | MCPower crit | Verdict |
|:---|---:|---:|---:|---:|---:|---:|---:|:---|
| intercept | 0.0 | 0.078383 | 0.078383 | NA | NA | NA | NA | PASS |
| x1 | 0.5 | 0.404979 | 0.404979 | 10.34034 | 10.34033 | 1.959964 | 1.959964 | PASS |

## y = 0.3·x1 + per-grp random intercept (ICC 0.30) + noise

R formula `y ~ x1 + (1|grp)` · linear mixed model (random intercept) ·
n=600, seed=2138. File reproduces from seed: **yes**.

| Term | True (formula) | R beta | MCPower beta | R t | MCPower stat | R crit | MCPower crit | Verdict |
|:---|---:|---:|---:|---:|---:|---:|---:|:---|
| intercept | 0.0 | 0.131583 | 0.131583 | NA | NA | NA | NA | PASS |
| x1 | 0.3 | 0.347749 | 0.347749 | 8.595397 | 8.595401 | 1.959964 | 1.959964 | PASS |

## y = 0.5·x1 + 0.3·x2 + per-grp random intercept (ICC 0.20) + noise

R formula `y ~ x1 + x2 + (1|grp)` · linear mixed model (random
intercept) · n=750, seed=2137. File reproduces from seed: **yes**.

| Term | True (formula) | R beta | MCPower beta | R t | MCPower stat | R crit | MCPower crit | Verdict |
|:---|---:|---:|---:|---:|---:|---:|---:|:---|
| intercept | 0.0 | -0.007542 | -0.007542 | NA | NA | NA | NA | PASS |
| x1 | 0.5 | 0.440708 | 0.440708 | 12.927637 | 12.927627 | 1.959964 | 1.959964 | PASS |
| x2 | 0.3 | 0.341816 | 0.341815 | 9.183627 | 9.183619 | 1.959964 | 1.959964 | PASS |

## y = 0.3·x1 + 0.5·x2 + per-grp random intercept (ICC 0.30) + noise

R formula `y ~ x1 + x2 + (1|grp)` · linear mixed model (random
intercept) · n=750, seed=2138. File reproduces from seed: **yes**.

| Term | True (formula) | R beta | MCPower beta | R t | MCPower stat | R crit | MCPower crit | Verdict |
|:---|---:|---:|---:|---:|---:|---:|---:|:---|
| intercept | 0.0 | 0.102675 | 0.102675 | NA | NA | NA | NA | PASS |
| x1 | 0.3 | 0.357704 | 0.357704 | 9.980624 | 9.980625 | 1.959964 | 1.959964 | PASS |
| x2 | 0.5 | 0.497794 | 0.497794 | 13.989819 | 13.989820 | 1.959964 | 1.959964 | PASS |

## y = 0.5·x1 + 0.3·x2 + 0.3·x1:x2 + per-grp random intercept (ICC 0.20) + noise

R formula `y ~ x1*x2 + (1|grp)` · linear mixed model (random intercept)
· n=750, seed=2137. File reproduces from seed: **yes**.

| Term | True (formula) | R beta | MCPower beta | R t | MCPower stat | R crit | MCPower crit | Verdict |
|:---|---:|---:|---:|---:|---:|---:|---:|:---|
| intercept | 0.0 | -0.007002 | -0.007002 | NA | NA | NA | NA | PASS |
| x1 | 0.5 | 0.447848 | 0.447848 | 12.953272 | 12.953262 | 1.959964 | 1.959964 | PASS |
| x2 | 0.3 | 0.341794 | 0.341794 | 9.187226 | 9.187218 | 1.959964 | 1.959964 | PASS |
| x1:x2 | 0.3 | 0.254123 | 0.254123 | 6.738949 | 6.738949 | 1.959964 | 1.959964 | PASS |

## y = 0.4·x1 + 0.3·x2 + 0.2·x1:x2 + per-grp random intercept (ICC 0.30) + noise

R formula `y ~ x1*x2 + (1|grp)` · linear mixed model (random intercept)
· n=900, seed=2138. File reproduces from seed: **yes**.

| Term | True (formula) | R beta | MCPower beta | R t | MCPower stat | R crit | MCPower crit | Verdict |
|:---|---:|---:|---:|---:|---:|---:|---:|:---|
| intercept | 0.0 | 0.202794 | 0.202794 | NA | NA | NA | NA | PASS |
| x1 | 0.4 | 0.428858 | 0.428858 | 12.586038 | 12.586039 | 1.959964 | 1.959964 | PASS |
| x2 | 0.3 | 0.289511 | 0.289511 | 8.587036 | 8.587037 | 1.959964 | 1.959964 | PASS |
| x1:x2 | 0.2 | 0.251104 | 0.251104 | 7.673815 | 7.673816 | 1.959964 | 1.959964 | PASS |

## y = 0.3·x1 + 0.3·g\[2\] + per-grp random intercept (ICC 0.20) + noise

R formula `y ~ x1 + g + (1|grp)` · linear mixed model (random intercept)
· n=750, seed=2137. File reproduces from seed: **yes**.

| Term | True (formula) | R beta | MCPower beta | R t | MCPower stat | R crit | MCPower crit | Verdict |
|:---|---:|---:|---:|---:|---:|---:|---:|:---|
| intercept | 0.0 | -0.042469 | -0.042469 | NA | NA | NA | NA | PASS |
| x1 | 0.3 | 0.241373 | 0.241373 | 7.079191 | 7.079184 | 1.959964 | 1.959964 | PASS |
| g\[2\] | 0.3 | 0.375982 | 0.375982 | 5.275995 | 5.275992 | 1.959964 | 1.959964 | PASS |

## y = 0.4·x1 + 0.5·g\[2\] + 0.8·g\[3\] + per-grp random intercept (ICC 0.30) + noise

R formula `y ~ x1 + g + (1|grp)` · linear mixed model (random intercept)
· n=900, seed=2138. File reproduces from seed: **yes**.

| Term | True (formula) | R beta | MCPower beta | R t | MCPower stat | R crit | MCPower crit | Verdict |
|:---|---:|---:|---:|---:|---:|---:|---:|:---|
| intercept | 0.0 | 0.123802 | 0.123802 | NA | NA | NA | NA | PASS |
| x1 | 0.4 | 0.435261 | 0.435261 | 12.890829 | 12.890830 | 1.959964 | 1.959964 | PASS |
| g\[2\] | 0.5 | 0.612951 | 0.612951 | 2.210123 | 2.210116 | 1.959964 | 1.959964 | PASS |
| g\[3\] | 0.8 | 1.000041 | 1.000041 | 3.147536 | 3.147526 | 1.959964 | 1.959964 | PASS |

| item | value |
|:---|:---|
| generated | 2026-06-15 |
| R | R version 4.5.3 (2026-03-11) |
| mcpower | 0.0.0.9000 |
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
`rmarkdown::render("mcpower/validation/validation_MLE_solving.rmd", output_dir = "mcpower/documentation/validation")`.

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
> on this GLMM path MCPower’s Wald SE is the RX/PLS Schur
> (`= glmer vcov(use.hessian = FALSE)`), ~3% anticonservative vs glmer’s
> default `use.hessian = TRUE` (GLMM-only; the linear LMM path above
> uses the standard GLS `(XᵀV̂⁻¹X)⁻¹`, lme4-equivalent, with no
> Schur/Hessian fork) — a deliberate, documented, deferred
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

**glmm_intercept**: A↔B PASS (K=200)

**glmm_slope**: A↔B PASS (K=200)

**glmm_multislope**: A↔B PASS (K=200)

**glmm_slope_crossed**: A↔B PASS (K=200)

    ## **τ²→0 collapse (ICC=0.05, 10×5)**: PASS — 14/25 draws pinned; worst |Δβ̂| 7.92e-08, |Δz| 3.39e-04 (25 draws, τ̂² mean 0.309)

    ## **Laplace-bias cell**: max |Δβ̂_x1| = 0.0136 (limit 0.0500), mean = 0.0067 — within limit
