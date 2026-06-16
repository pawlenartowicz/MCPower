# MCPower — Validating the Scenario Perturbations

# What this report shows

MCPower’s **scenario analysis** deliberately stresses a planned study:
instead of simulating the idealised design you specified, each scenario
*perturbs* it — slopes wobble between observations, residual variance
trends with the predictors, correlations differ from the plan, predictor
and residual distributions get swapped, group allocation becomes random.
Each perturbation is controlled by a knob in `configs/scenarios.json`,
and each knob documents a precise statistical *law* for the disturbance
it injects.

This report checks **that every knob generates exactly its documented
law** — no more, no less. It is the L5 layer of the validation charter:
the L3 reports prove the *optimistic* (unperturbed) generator faithful
against the spec as a point oracle; here the spec value is deliberately
randomised, so the oracle is the perturbation law itself (including its
documented distortions: the ±0.8 correlation clamp, PSD repair, the
censored t(3) table, the stale heteroskedasticity anchor).

**The gate doctrine.** Every gate is *set == get on the realised
magnitude*: the knob’s value, recovered from the generated data by a
probe, must match the documented law within an SE-of-mean z-band of 4
(each case draws K independent perturbation blocks; a mis-scaled knob
lands tens of σ out, chance alone exceeds the band once in ~10,000
gates). Two checks are explicitly **not** gates: monotone power across
presets (a readable summary that washes out real faults) and a global
error-variance invariant (heterogeneity is *supposed* to inflate it).
The β̂-unbiasedness backstop (B1) — **including the intercept** — is kept
as a cheap mean-leak tripwire: a mis-centred swapped distribution lands
in β̂₀ while every effect estimate stays clean.

# Results at a glance

> **72 of 72 gates pass.** Golden reproducibility: baseline frozen this
> run.

| Case            | Verdict  | Golden check    |
|:----------------|:---------|:----------------|
| fg_glm_flip     | all PASS | frozen this run |
| fg_glm_ident    | all PASS | frozen this run |
| scen_b0         | all PASS | frozen this run |
| scen_b1_glm     | all PASS | frozen this run |
| scen_b1_ols     | all PASS | frozen this run |
| scen_b2         | all PASS | frozen this run |
| scen_b3         | all PASS | frozen this run |
| scen_b4         | all PASS | frozen this run |
| scen_b5         | all PASS | frozen this run |
| scen_co_high    | all PASS | frozen this run |
| scen_co_low     | all PASS | frozen this run |
| scen_co_psd     | all PASS | frozen this run |
| scen_fa_mle     | all PASS | frozen this run |
| scen_fa_mle_fp  | all PASS | frozen this run |
| scen_fa_ols     | all PASS | frozen this run |
| scen_he         | all PASS | frozen this run |
| scen_hs         | all PASS | frozen this run |
| scen_px         | all PASS | frozen this run |
| scen_px_t3      | all PASS | frozen this run |
| scen_re         | all PASS | frozen this run |
| scen_re_replace | all PASS | frozen this run |

# Tier A — one knob at a time

Each case turns on a single knob on an otherwise-optimistic design,
draws K independent perturbation blocks, and recovers the knob’s
magnitude with a probe matched to its law.

## Heterogeneity (slope wobble)

Effects vary per observation: βⱼ + N(0, (h·βⱼ)²). The probe regresses
squared true-β residuals on each squared predictor — the slope recovers
h²βⱼ² per predictor, separating He from anything that only moves pooled
moments.

|  | Case | Gate | Measured | Law | z | Verdict |
|:---|:---|:---|---:|---:|---:|:---|
| jitter slope on x1² = h²β² (h=0.4) | scen_he | jitter slope on x1² = h²β² (h=0.4) | 0.02491 | 0.0256 | -0.25106 | PASS |
| jitter slope on x2² = h²β² (h=0.4) | scen_he | jitter slope on x2² = h²β² (h=0.4) | 0.00830 | 0.0100 | -1.07315 | PASS |

## Heteroskedasticity (residual-variance trend)

Residual variance follows Var(εᵢ) = σ²·exp(γzᵢ)/exp(γ²/2) with γ =
ln(λ)/4, z the standardised driver. log e² is then *linear* in z with
slope exactly γ — shape-blind, so the same probe serves B3. The realised
±2σ ratio is exp(4γ̂).

|  | Case | Gate | Measured | Law | z | Verdict |
|:---|:---|:---|---:|---:|---:|:---|
| log-e² slope γ = ln(λ)/4 | scen_hs | log-e² slope γ = ln(λ)/4 | 0.34797 | 0.34657 | 0.60302 | PASS |

Realised λ̂ = exp(4γ̂) = **4.022** (set: 4); the raw ±2σ binned variance
ratio reads 3.95 (reported only — finite bins make its law approximate;
the slope is the gate).

## Correlation noise

Per block the off-diagonals get symmetrised Gaussian noise —
symmetrisation halves the variance, so the per-block ρ law is N(ρ, s²/2)
**censored at ±0.8**, plus finite-n sampling noise in quadrature. The
low-ρ case gates the exact law; the high-ρ case sits against the clamp,
gating the censored-normal truncation law itself (mean visibly below the
naive ρ).

|  | Case | Gate | Measured | Law | z | Verdict |
|:---|:---|:---|---:|---:|---:|:---|
| block-r mean = ±0.8-clamp censored law | scen_co_low | block-r mean = ±0.8-clamp censored law | 0.29620 | 0.30000 | -0.68832 | PASS |
| block-r SD = √(s²/2 + (1−ρ²)²/n) | scen_co_low | block-r SD = √(s²/2 + (1−ρ²)²/n) | 0.11048 | 0.10704 | 0.88024 | PASS |
| block-r mean = ±0.8-clamp censored law1 | scen_co_high | block-r mean = ±0.8-clamp censored law | 0.73059 | 0.72807 | 0.94848 | PASS |
| block-r SD = √(s²/2 + (1−ρ²)²/n)1 | scen_co_high | block-r SD = √(s²/2 + (1−ρ²)²/n) | 0.07522 | 0.07873 | -1.86440 | PASS |

At ρ = 0.75, s = 0.15: the censored law predicts mean 0.7281 — the
−0.0219 shift below the nominal ρ **is** the documented clamp
truncation, and the realised mean lands on it.

### PSD repair (3 predictors, high ρ)

Repair cannot fire at p = 2 (any clamped 2×2 is PD), so a 3-variable
all-0.6 design under s = 0.3 is where eigenvalue-floor +
diagonal-renormalisation distortion lives. There is no closed-form law;
the empirical per-pair moments are frozen as MCPower-golden (table
below) and re-checked on every run.

| r12_mean | r13_mean | r23_mean |  r12_sd |  r13_sd |  r23_sd |
|---------:|---------:|---------:|--------:|--------:|--------:|
|  0.58466 |  0.56899 |  0.55664 | 0.17938 | 0.18598 | 0.19204 |

## Distribution swaps (predictors)

Each continuous-normal column is swapped per block with probability q to
a uniform pick from the pool; every pool candidate is standardised (mean
0, var 1), so a swap perturbs *shape only* — a mis-centred candidate
would be a mean leak straight into β̂₀.

|  | Case | Gate | Measured | Law | z | Verdict |
|:---|:---|:---|---:|---:|---:|:---|
| swap frequency = q | scen_px | swap frequency = q | 0.53000 | 0.50000 | 1.65979 | PASS |
| pick share right_skewed = 1/3 | scen_px | pick share right_skewed = 1/3 | 0.37264 | 0.33333 | 1.71701 | PASS |
| pick share left_skewed = 1/3 | scen_px | pick share left_skewed = 1/3 | 0.29245 | 0.33333 | -1.78569 | PASS |
| pick share uniform = 1/3 | scen_px | pick share uniform = 1/3 | 0.33491 | 0.33333 | 0.06868 | PASS |
| right_skewed mean = 0 | scen_px | right_skewed mean = 0 | -0.00186 | 0.00000 | -1.44183 | PASS |
| right_skewed var = 1 | scen_px | right_skewed var = 1 | 0.99895 | 1.00000 | -0.32336 | PASS |
| left_skewed mean = 0 | scen_px | left_skewed mean = 0 | 0.00075 | 0.00000 | 0.51932 | PASS |
| left_skewed var = 1 | scen_px | left_skewed var = 1 | 0.99996 | 1.00000 | -0.01066 | PASS |
| uniform mean = 0 | scen_px | uniform mean = 0 | -0.00133 | 0.00000 | -0.98269 | PASS |
| uniform var = 1 | scen_px | uniform var = 1 | 1.00042 | 1.00000 | 0.34051 | PASS |

Block classifications: left_skewed 124, normal 376, right_skewed 158,
uniform 142.

### Custom pool: `high_kurtosis`

The only swappable marginal outside the presets. Its engine identity is
a **censored** standardised t(3): a 2048-knot inverse-CDF table on
percentiles \[0.00121, 0.99879\], normalised at build to the censored
table’s own SD (1.5958 raw, vs √3 ≈ 1.7321 for the full t(3)) — so the
marginal has **exactly unit variance**, excess kurtosis ≈ 6.39, and
support ±6.0 SD. The censoring is deliberate: it bounds every synthetic
marginal at ±6 SD while keeping this the heaviest-tailed shape (t(3)’s
own kurtosis is infinite). v1 — and this engine until 2026-06 — divided
by √3, which standardises the *full* t(3) and left the censored marginal
at var ≈ 0.858, a silent ~14% effect-size shrink for every high-kurtosis
predictor; the L5 gate below caught it, and the table is now normalised
by construction.

|  | Case | Gate | Measured | Law | z | Verdict |
|:---|:---|:---|---:|---:|---:|:---|
| every block swapped (q = 1) | scen_px_t3 | every block swapped (q = 1) | 1.00000 | 1 | NA | PASS |
| t3 mean = 0 | scen_px_t3 | t3 mean = 0 | -0.00194 | 0 | -1.67897 | PASS |
| t3 var = 1 (table-normalized) | scen_px_t3 | t3 var = 1 (table-normalized) | 0.99882 | 1 | -0.37682 | PASS |

## Residual swaps

With probability q_r the block’s residual distribution is **replaced** —
distribution *and* df — by a pool pick (t(df)·√((df−2)/df) or
(χ²(df)−df)/√(2df)). The shape laws (skew = √(8/df), excess kurtosis =
6/(df−4)) recover the df, proving the scenario’s df is carried, not the
spec’s.

|  | Case | Gate | Measured | Law | z | Verdict |
|:---|:---|:---|---:|---:|---:|:---|
| swap frequency = q_r | scen_re | swap frequency = q_r | 0.51000 | 0.50000 | 0.39958 | PASS |
| pick share heavy_tailed = 1/2 | scen_re | pick share heavy_tailed = 1/2 | 0.51471 | 0.50000 | 0.42008 | PASS |
| pick share skewed = 1/2 | scen_re | pick share skewed = 1/2 | 0.48529 | 0.50000 | -0.42008 | PASS |
| skewed: skew = √(8/df), df = 10 | scen_re | skewed: skew = √(8/df), df = 10 | 0.88878 | 0.89443 | -0.85425 | PASS |
| heavy: excess kurtosis = 6/(df−4), df = 10 | scen_re | heavy: excess kurtosis = 6/(df−4), df = 10 | 1.04341 | 1.00000 | 1.59313 | PASS |
| pin holds: residual stays high_kurtosis (right_skewed swap inert) | scen_re_replace | pin holds: residual stays high_kurtosis (right_skewed swap inert) | 0.98000 | 1.00000 | NA | PASS |
| symmetric: skew = 0 (pinned t(6), not the χ² swap) | scen_re_replace | symmetric: skew = 0 (pinned t(6), not the χ² swap) | -0.01593 | 0.00000 | -0.95569 | PASS |
| mean = 0 | scen_re_replace | mean = 0 | 0.00140 | 0.00000 | 1.20001 | PASS |
| var = 1 (t(6) standardized) | scen_re_replace | var = 1 (t(6) standardized) | 0.99766 | 1.00000 | -0.92347 | PASS |

The pinned case (`scen_re_replace`) verifies the swap-eligibility rule:
it *pins* the spec residual with
`set_residual_distribution("high_kurtosis")`, then configures a forced
right_skewed swap (q_r = 1). Because `pick_residual` only swaps an
**unpinned default-normal** residual, the swap is inert — every draw
keeps the pinned, symmetric censored-t3 high_kurtosis residual (skew ≈
0, table-normalised to var 1), never the χ²(6) the config asks for. The
skew = 0 gate is the tripwire: a fired swap would force skew = √(8/6) ≈
1.15.

## Factor-proportion sampling

`sampled_factor_proportions = FALSE` (the optimistic default) assigns
factor levels by a deterministic largest-remainder walk — counts are a
pure function of (n, p), identical across draws, each within 1 of
n·p. `TRUE` draws levels per row: counts are Binomial(n, p) with
variance n·p(1−p).

|  | Case | Gate | Measured | Law | z | Verdict |
|:---|:---|:---|---:|---:|---:|:---|
| fixed: counts identical across draws (no RNG) | scen_fa_ols | fixed: counts identical across draws (no RNG) | 1.0000 | 1 | NA | PASS |
| fixed: max \|count − n·p\| ≤ 1 (largest remainder) | scen_fa_ols | fixed: max \|count − n·p\| ≤ 1 (largest remainder) | 0.0000 | 1 | NA | PASS |
| sampled: Var(count) level 1 = n·p(1−p) | scen_fa_ols | sampled: Var(count) level 1 = n·p(1−p) | 243.7875 | 250 | -0.3935 | PASS |
| sampled: Var(count) level 2 = n·p(1−p) | scen_fa_ols | sampled: Var(count) level 2 = n·p(1−p) | 182.2775 | 210 | -2.1712 | PASS |
| sampled: Var(count) level 3 = n·p(1−p) | scen_fa_ols | sampled: Var(count) level 3 = n·p(1−p) | 175.0250 | 160 | 1.3196 | PASS |
| fixed: counts identical across draws (no RNG)1 | scen_fa_mle | fixed: counts identical across draws (no RNG) | 1.0000 | 1 | NA | PASS |
| fixed: max \|count − n·p\| ≤ 1 (largest remainder)1 | scen_fa_mle | fixed: max \|count − n·p\| ≤ 1 (largest remainder) | 0.0000 | 1 | NA | PASS |
| sampled: Var(count) level 1 = n·p(1−p)1 | scen_fa_mle | sampled: Var(count) level 1 = n·p(1−p) | 220.4100 | 250 | -1.3942 | PASS |
| sampled: Var(count) level 2 = n·p(1−p)1 | scen_fa_mle | sampled: Var(count) level 2 = n·p(1−p) | 194.0300 | 210 | -0.9094 | PASS |
| sampled: Var(count) level 3 = n·p(1−p)1 | scen_fa_mle | sampled: Var(count) level 3 = n·p(1−p) | 158.4900 | 160 | -0.1044 | PASS |
| find_power accepts Fa toggle under estimator = Mle | scen_fa_mle_fp | find_power accepts Fa toggle under estimator = Mle | 1.0000 | 1 | NA | PASS |

Fixed counts at n = 1000: 500/300/200 against expected 500/300/200. The
MLE rows are the one scenario knob the mixed-model estimator admits
(every other knob is rejected by the engine’s estimator gate — a
deterministic L1 assertion owned by the engine test suite, not re-tested
here).

# Tier B — knob interactions

The shipped presets co-vary every knob, so B2–B4 isolate single
interactions through custom scenario pairs on **shared seeds** (P1
pairing: two scenarios at the same seed draw the same raw noise streams,
so cross-scenario deltas are knob-attributable).

## B0 — Optimistic ≡ baseline

|  | Case | Gate | Measured | Law | z | Verdict |
|:---|:---|:---|---:|---:|---:|:---|
| optimistic ≡ non-scenario path (bit-identical PowerResult) | scen_b0 | optimistic ≡ non-scenario path (bit-identical PowerResult) | 1 | 1 | NA | PASS |

The optimistic member of a three-scenario paired call is bit-identical
to the plain non-scenario call — the DGP-level companion to the
orchestrator’s call-seed test (power here: 0.948, 0.28).

## B1 — β̂ unbiased across the presets (leak backstop)

|  | Case | Gate | Measured | Law | z | Verdict |
|:---|:---|:---|---:|---:|---:|:---|
| optimistic: β̂\[intercept\] unbiased | scen_b1_ols | optimistic: β̂\[intercept\] unbiased | -0.0025 | 0.0000 | -2.1096 | PASS |
| optimistic: β̂\[x1\] unbiased | scen_b1_ols | optimistic: β̂\[x1\] unbiased | 0.3994 | 0.4000 | -0.4687 | PASS |
| optimistic: β̂\[x2\] unbiased | scen_b1_ols | optimistic: β̂\[x2\] unbiased | 0.2499 | 0.2500 | -0.1007 | PASS |
| realistic: β̂\[intercept\] unbiased | scen_b1_ols | realistic: β̂\[intercept\] unbiased | 0.0001 | 0.0000 | 0.0560 | PASS |
| realistic: β̂\[x1\] unbiased | scen_b1_ols | realistic: β̂\[x1\] unbiased | 0.3952 | 0.4000 | -1.0430 | PASS |
| realistic: β̂\[x2\] unbiased | scen_b1_ols | realistic: β̂\[x2\] unbiased | 0.2500 | 0.2500 | 0.0007 | PASS |
| doomer: β̂\[intercept\] unbiased | scen_b1_ols | doomer: β̂\[intercept\] unbiased | 0.0009 | 0.0000 | 0.7089 | PASS |
| doomer: β̂\[x1\] unbiased | scen_b1_ols | doomer: β̂\[x1\] unbiased | 0.4034 | 0.4000 | 0.3641 | PASS |
| doomer: β̂\[x2\] unbiased | scen_b1_ols | doomer: β̂\[x2\] unbiased | 0.2496 | 0.2500 | -0.0609 | PASS |
| optimistic: β̂\[intercept\] = per-study pseudo-true | scen_b1_glm | optimistic: β̂\[intercept\] = per-study pseudo-true | -0.4050 | -0.4055 | 0.1720 | PASS |
| optimistic: β̂\[x1\] = per-study pseudo-true | scen_b1_glm | optimistic: β̂\[x1\] = per-study pseudo-true | 0.5007 | 0.5000 | 0.2342 | PASS |
| optimistic: β̂\[x2\] = per-study pseudo-true | scen_b1_glm | optimistic: β̂\[x2\] = per-study pseudo-true | 0.2988 | 0.3000 | -0.4579 | PASS |
| realistic: β̂\[intercept\] = per-study pseudo-true | scen_b1_glm | realistic: β̂\[intercept\] = per-study pseudo-true | -0.3905 | -0.4055 | 1.1979 | PASS |
| realistic: β̂\[x1\] = per-study pseudo-true | scen_b1_glm | realistic: β̂\[x1\] = per-study pseudo-true | 0.5066 | 0.5000 | 1.0446 | PASS |
| realistic: β̂\[x2\] = per-study pseudo-true | scen_b1_glm | realistic: β̂\[x2\] = per-study pseudo-true | 0.2992 | 0.3000 | -0.1913 | PASS |
| doomer: β̂\[intercept\] = per-study pseudo-true | scen_b1_glm | doomer: β̂\[intercept\] = per-study pseudo-true | -0.3944 | -0.4055 | 0.4703 | PASS |
| doomer: β̂\[x1\] = per-study pseudo-true | scen_b1_glm | doomer: β̂\[x1\] = per-study pseudo-true | 0.4939 | 0.5004 | -0.5737 | PASS |
| doomer: β̂\[x2\] = per-study pseudo-true | scen_b1_glm | doomer: β̂\[x2\] = per-study pseudo-true | 0.2983 | 0.3002 | -0.2780 | PASS |

OLS rows gate on the z-band, **intercept included** — the mean-leak
tripwire (linear averaging keeps OLS β̂ exactly unbiased under every
knob). Logit rows need a different law: the heterogeneity β-jitter is
drawn **once per study**, so each study’s data is a clean logit at its
own β_eff and the MLE recovers it — averaged over the K studies the
fitted coefficient → E\[β_eff\], **not** the attenuated coefficient a
per-observation population-averaged marginal would show. The clip toward
zero (s_j = h·\|β_j\|) nudges each slope’s magnitude *up* by
×(Φ(1/h)+h·φ(1/h)) ≈ ×1.0008 at doomer’s h = 0.4; the symmetric
unclipped intercept jitter leaves β_0 unchanged. So the logit law is
this per-study pseudo-true value (`glm_perstudy_beta`), gated on the
absolute band ±0.02. The GLM calibration gates remain Tier A and the
flip rate below.

## B2 — He × Hs separation

The β-jitter variance ∝ xᵢⱼ²βⱼ² must be present at λ = 1 and *unchanged*
by a λ toggle (λ is driven by the clean linear predictor, never the
jittered one). The λ driver is pinned to x2, so the x1²-decomposed
jitter variance is uncontaminated by the λ channel’s even cosh
component; the paired Δ across {h = 0.4, λ = 1} vs {h = 0.4, λ = 4}
isolates the interaction.

|  | Case | Gate | Measured | Law | z | Verdict |
|:---|:---|:---|---:|---:|---:|:---|
| x1 jitter slope at λ=1 = h²β₁² | scen_b2 | x1 jitter slope at λ=1 = h²β₁² | 0.02174 | 0.0256 | -1.77201 | PASS |
| x1 jitter slope λ-invariant (paired Δ = 0) | scen_b2 | x1 jitter slope λ-invariant (paired Δ = 0) | 0.00002 | 0.0000 | 0.04726 | PASS |

The *driver* column shows why the pin matters: x2²-slope reads 0.0095 at
λ = 1 (law h²β₂² = 0.01) but inflates to 0.0689 at λ = 4 — the cosh(γz)
contamination the naive probe would misread as an He × Hs interaction.

## B3 — Hs × Re preservation

|  | Case | Gate | Measured | Law | z | Verdict |
|:---|:---|:---|---:|---:|---:|:---|
| γ = ln(λ)/4 under forced high_kurtosis residual | scen_b3 | γ = ln(λ)/4 under forced high_kurtosis residual | 0.34263 | 0.34657 | -1.62754 | PASS |
| γ = ln(λ)/4 under forced right_skewed residual | scen_b3 | γ = ln(λ)/4 under forced right_skewed residual | 0.35076 | 0.34657 | 1.69386 | PASS |

Under a forced t(10) or χ²(10) residual the log-e² slope still reads
ln(λ)/4 — the multiplicative variance trend amplifies the swapped tails
without bending the ratio.

## B4 — Co × Px (NORTA)

Correlation is induced on the *latent* normals; a swapped marginal
transforms them, so the realised Pearson r follows the NORTA law, not
the latent spec value. The oracle is computed numerically in R
(Gauss–Hermite over the latent bivariate normal; it matches the closed
forms (e^ρ−1)/(e−1) and (6/π)asin(ρ/2) to 7 digits).

|  | Case | Gate | Measured | Law | z | Verdict |
|:---|:---|:---|---:|---:|---:|:---|
| right_skewed pair: r = NORTA 0.4541 (latent ρ = 0.50) | scen_b4 | right_skewed pair: r = NORTA 0.4541 (latent ρ = 0.50) | 0.45459 | 0.45410 | 0.41461 | PASS |
| uniform pair: r = NORTA 0.4826 (latent ρ = 0.50) | scen_b4 | uniform pair: r = NORTA 0.4826 (latent ρ = 0.50) | 0.48296 | 0.48258 | 0.44450 | PASS |

## B5 — Heteroskedasticity-anchor drift (P6), measured and bounded

`het_coeffs` (the driver moments that standardise the λ driver) is
computed once from the **base spec**; correlation noise moves the
realised driver SD σ′ per block while the anchor σ₀ stays put, so the
realised ±2σ′ ratio drifts to λ′ = exp(4γσ′/σ₀). This is an accepted
approximation — the gates *prove the mechanism* and *bound the drift*;
they do not fail on the drift itself.

|  | Case | Gate | Measured | Law | z | Verdict |
|:---|:---|:---|---:|---:|---:|:---|
| slope magnitude = γ/σ₀ (stale spec anchor) | scen_b5 | slope magnitude = γ/σ₀ (stale spec anchor) | 0.65938 | 0.65206 | 1.59849 | PASS |
| staleness: lm(λ̂′ ~ λ′ predicted) slope = 1 | scen_b5 | staleness: lm(λ̂′ ~ λ′ predicted) slope = 1 | 0.90100 | 1.00000 | -0.95987 | PASS |
| drift bounded: max λ′ ≤ clamp-range bound | scen_b5 | drift bounded: max λ′ ≤ clamp-range bound | 5.04465 | 5.39376 | NA | PASS |

| Preset    | λ set | λ′ 5% | λ′ median | λ′ 95% |
|:----------|------:|------:|----------:|-------:|
| realistic |     2 | 1.900 |     1.988 |  2.080 |
| doomer    |     4 | 3.343 |     3.960 |  4.627 |

The staleness gate is the discriminating one: per block, the measured
ratio λ̂′ tracks the moment-predicted λ′ = exp(4γσ̂′/σ₀) with regression
slope ≈ 1. (A per-block recompute of the anchor would pin λ̂′ at λ, slope
≈ 0 — ~11σ away.) Note the *mean* slope alone cannot tell the two
designs apart (γ·E\[1/σ′\] ≈ γ/σ₀ to 0.1%); the per-block tracking is
what evidences the stale anchor. The preset table quantifies the
documented drift: under realistic/doomer the effective λ′ wobbles around
the set λ by the quantiles shown — second-order next to the
perturbations themselves.

# Family gating

Knobs must be live or inert exactly per family.

|  | Case | Gate | Measured | Law | z | Verdict |
|:---|:---|:---|---:|---:|---:|:---|
| bit-identity under logit: optimistic ≡ glm_l4 | fg_glm_ident | bit-identity under logit: optimistic ≡ glm_l4 | 1.00000 | 1 | NA | PASS |
| bit-identity under logit: glm_l4 ≡ glm_l4re | fg_glm_ident | bit-identity under logit: glm_l4 ≡ glm_l4re | 1.00000 | 1 | NA | PASS |
| h-toggle flip rate = MC prediction (paired Δ = 0) | fg_glm_flip | h-toggle flip rate = MC prediction (paired Δ = 0) | 0.00234 | 0 | 0.6365 | PASS |

- **GLM λ / residual swaps — inert by bit-identity.** `apply_hsk`
  requires a continuous outcome and consumes no RNG, so a λ toggle is an
  exact no-op under logit; a forced residual swap only consumes
  scenario-stream draws after every other consumer, leaving the
  Bernoulli outcomes bit-identical.
- **GLM heterogeneity — live, latent.** The per-row log-odds jitter is
  hidden behind a single Bernoulli draw, so the observable is the paired
  h-toggle **flip rate**: X and the uniforms are drawn before the jitter
  normals, the pair shares them bit-identically, and P(flip) = E\|p_h −
  p₀\| — predicted by numerical integration row by row. Realised flip
  rate 0.0827 vs predicted 0.0804. The Jensen mean-rate shift (+0.0051)
  is expected and reported, not gated — median latent-rate invariance is
  unobservable.
- **MLE — only `sampled_factor_proportions` is live** (see the Fa
  section above); every other knob is rejected by the engine’s estimator
  gate.

# Verdict table — every gate

| Case | Gate | Measured | Law | z | Verdict |
|:---|:---|---:|---:|---:|:---|
| scen_he | jitter slope on x1² = h²β² (h=0.4) | 0.0249 | 0.0256 | -0.2511 | PASS |
| scen_he | jitter slope on x2² = h²β² (h=0.4) | 0.0083 | 0.0100 | -1.0731 | PASS |
| scen_hs | log-e² slope γ = ln(λ)/4 | 0.3480 | 0.3466 | 0.6030 | PASS |
| scen_co_low | block-r mean = ±0.8-clamp censored law | 0.2962 | 0.3000 | -0.6883 | PASS |
| scen_co_low | block-r SD = √(s²/2 + (1−ρ²)²/n) | 0.1105 | 0.1070 | 0.8802 | PASS |
| scen_co_high | block-r mean = ±0.8-clamp censored law | 0.7306 | 0.7281 | 0.9485 | PASS |
| scen_co_high | block-r SD = √(s²/2 + (1−ρ²)²/n) | 0.0752 | 0.0787 | -1.8644 | PASS |
| scen_co_psd | PSD-repaired r ∈ \[0.45, 0.60\] (shrinks below ρ=0.6 input, stays positive) | 0.5566 | 0.4500 | NA | PASS |
| scen_px | swap frequency = q | 0.5300 | 0.5000 | 1.6598 | PASS |
| scen_px | pick share right_skewed = 1/3 | 0.3726 | 0.3333 | 1.7170 | PASS |
| scen_px | pick share left_skewed = 1/3 | 0.2925 | 0.3333 | -1.7857 | PASS |
| scen_px | pick share uniform = 1/3 | 0.3349 | 0.3333 | 0.0687 | PASS |
| scen_px | right_skewed mean = 0 | -0.0019 | 0.0000 | -1.4418 | PASS |
| scen_px | right_skewed var = 1 | 0.9989 | 1.0000 | -0.3234 | PASS |
| scen_px | left_skewed mean = 0 | 0.0007 | 0.0000 | 0.5193 | PASS |
| scen_px | left_skewed var = 1 | 1.0000 | 1.0000 | -0.0107 | PASS |
| scen_px | uniform mean = 0 | -0.0013 | 0.0000 | -0.9827 | PASS |
| scen_px | uniform var = 1 | 1.0004 | 1.0000 | 0.3405 | PASS |
| scen_px_t3 | every block swapped (q = 1) | 1.0000 | 1.0000 | NA | PASS |
| scen_px_t3 | t3 mean = 0 | -0.0019 | 0.0000 | -1.6790 | PASS |
| scen_px_t3 | t3 var = 1 (table-normalized) | 0.9988 | 1.0000 | -0.3768 | PASS |
| scen_re | swap frequency = q_r | 0.5100 | 0.5000 | 0.3996 | PASS |
| scen_re | pick share heavy_tailed = 1/2 | 0.5147 | 0.5000 | 0.4201 | PASS |
| scen_re | pick share skewed = 1/2 | 0.4853 | 0.5000 | -0.4201 | PASS |
| scen_re | skewed: skew = √(8/df), df = 10 | 0.8888 | 0.8944 | -0.8543 | PASS |
| scen_re | heavy: excess kurtosis = 6/(df−4), df = 10 | 1.0434 | 1.0000 | 1.5931 | PASS |
| scen_re_replace | pin holds: residual stays high_kurtosis (right_skewed swap inert) | 0.9800 | 1.0000 | NA | PASS |
| scen_re_replace | symmetric: skew = 0 (pinned t(6), not the χ² swap) | -0.0159 | 0.0000 | -0.9557 | PASS |
| scen_re_replace | mean = 0 | 0.0014 | 0.0000 | 1.2000 | PASS |
| scen_re_replace | var = 1 (t(6) standardized) | 0.9977 | 1.0000 | -0.9235 | PASS |
| scen_fa_ols | fixed: counts identical across draws (no RNG) | 1.0000 | 1.0000 | NA | PASS |
| scen_fa_ols | fixed: max \|count − n·p\| ≤ 1 (largest remainder) | 0.0000 | 1.0000 | NA | PASS |
| scen_fa_ols | sampled: Var(count) level 1 = n·p(1−p) | 243.7875 | 250.0000 | -0.3935 | PASS |
| scen_fa_ols | sampled: Var(count) level 2 = n·p(1−p) | 182.2775 | 210.0000 | -2.1712 | PASS |
| scen_fa_ols | sampled: Var(count) level 3 = n·p(1−p) | 175.0250 | 160.0000 | 1.3196 | PASS |
| scen_fa_mle | fixed: counts identical across draws (no RNG) | 1.0000 | 1.0000 | NA | PASS |
| scen_fa_mle | fixed: max \|count − n·p\| ≤ 1 (largest remainder) | 0.0000 | 1.0000 | NA | PASS |
| scen_fa_mle | sampled: Var(count) level 1 = n·p(1−p) | 220.4100 | 250.0000 | -1.3942 | PASS |
| scen_fa_mle | sampled: Var(count) level 2 = n·p(1−p) | 194.0300 | 210.0000 | -0.9094 | PASS |
| scen_fa_mle | sampled: Var(count) level 3 = n·p(1−p) | 158.4900 | 160.0000 | -0.1044 | PASS |
| scen_fa_mle_fp | find_power accepts Fa toggle under estimator = Mle | 1.0000 | 1.0000 | NA | PASS |
| scen_b0 | optimistic ≡ non-scenario path (bit-identical PowerResult) | 1.0000 | 1.0000 | NA | PASS |
| scen_b1_ols | optimistic: β̂\[intercept\] unbiased | -0.0025 | 0.0000 | -2.1096 | PASS |
| scen_b1_ols | optimistic: β̂\[x1\] unbiased | 0.3994 | 0.4000 | -0.4687 | PASS |
| scen_b1_ols | optimistic: β̂\[x2\] unbiased | 0.2499 | 0.2500 | -0.1007 | PASS |
| scen_b1_ols | realistic: β̂\[intercept\] unbiased | 0.0001 | 0.0000 | 0.0560 | PASS |
| scen_b1_ols | realistic: β̂\[x1\] unbiased | 0.3952 | 0.4000 | -1.0430 | PASS |
| scen_b1_ols | realistic: β̂\[x2\] unbiased | 0.2500 | 0.2500 | 0.0007 | PASS |
| scen_b1_ols | doomer: β̂\[intercept\] unbiased | 0.0009 | 0.0000 | 0.7089 | PASS |
| scen_b1_ols | doomer: β̂\[x1\] unbiased | 0.4034 | 0.4000 | 0.3641 | PASS |
| scen_b1_ols | doomer: β̂\[x2\] unbiased | 0.2496 | 0.2500 | -0.0609 | PASS |
| scen_b1_glm | optimistic: β̂\[intercept\] = per-study pseudo-true | -0.4050 | -0.4055 | 0.1720 | PASS |
| scen_b1_glm | optimistic: β̂\[x1\] = per-study pseudo-true | 0.5007 | 0.5000 | 0.2342 | PASS |
| scen_b1_glm | optimistic: β̂\[x2\] = per-study pseudo-true | 0.2988 | 0.3000 | -0.4579 | PASS |
| scen_b1_glm | realistic: β̂\[intercept\] = per-study pseudo-true | -0.3905 | -0.4055 | 1.1979 | PASS |
| scen_b1_glm | realistic: β̂\[x1\] = per-study pseudo-true | 0.5066 | 0.5000 | 1.0446 | PASS |
| scen_b1_glm | realistic: β̂\[x2\] = per-study pseudo-true | 0.2992 | 0.3000 | -0.1913 | PASS |
| scen_b1_glm | doomer: β̂\[intercept\] = per-study pseudo-true | -0.3944 | -0.4055 | 0.4703 | PASS |
| scen_b1_glm | doomer: β̂\[x1\] = per-study pseudo-true | 0.4939 | 0.5004 | -0.5737 | PASS |
| scen_b1_glm | doomer: β̂\[x2\] = per-study pseudo-true | 0.2983 | 0.3002 | -0.2780 | PASS |
| scen_b2 | x1 jitter slope at λ=1 = h²β₁² | 0.0217 | 0.0256 | -1.7720 | PASS |
| scen_b2 | x1 jitter slope λ-invariant (paired Δ = 0) | 0.0000 | 0.0000 | 0.0473 | PASS |
| scen_b3 | γ = ln(λ)/4 under forced high_kurtosis residual | 0.3426 | 0.3466 | -1.6275 | PASS |
| scen_b3 | γ = ln(λ)/4 under forced right_skewed residual | 0.3508 | 0.3466 | 1.6939 | PASS |
| scen_b4 | right_skewed pair: r = NORTA 0.4541 (latent ρ = 0.50) | 0.4546 | 0.4541 | 0.4146 | PASS |
| scen_b4 | uniform pair: r = NORTA 0.4826 (latent ρ = 0.50) | 0.4830 | 0.4826 | 0.4445 | PASS |
| scen_b5 | slope magnitude = γ/σ₀ (stale spec anchor) | 0.6594 | 0.6521 | 1.5985 | PASS |
| scen_b5 | staleness: lm(λ̂′ ~ λ′ predicted) slope = 1 | 0.9010 | 1.0000 | -0.9599 | PASS |
| scen_b5 | drift bounded: max λ′ ≤ clamp-range bound | 5.0446 | 5.3938 | NA | PASS |
| fg_glm_ident | bit-identity under logit: optimistic ≡ glm_l4 | 1.0000 | 1.0000 | NA | PASS |
| fg_glm_ident | bit-identity under logit: glm_l4 ≡ glm_l4re | 1.0000 | 1.0000 | NA | PASS |
| fg_glm_flip | h-toggle flip rate = MC prediction (paired Δ = 0) | 0.0023 | 0.0000 | 0.6365 | PASS |

> **All 72 gates pass and every case reproduces its golden baseline.**
> Each scenario knob generates its documented law: the realised slope
> wobble, variance trend, correlation noise (with its clamp truncation),
> swap frequencies, pool moments, allocation behaviour, NORTA coupling,
> and heteroskedasticity-anchor drift all land on their predicted
> values, and the β̂ backstop shows no mean leak.

# scen_re_multi — multi-grouping RE knobs (M2)

The §2.5 uniformity law on a crossed extra beside the primary:
`random_effect_dist` / `random_effect_df` apply to *every* grouping’s
draw, and `icc_noise_sd` jitters every grouping’s τ² **independently**.
Measurement mirrors `scen_re`’s level-mean-variance recovery
(solver-free); β̂ unbiased is the leak tripwire. The jitter-presence
check is paired against a no-jitter (`optimistic`) control: the realised
τ̂² spread under the knob must exceed the control’s Monte-Carlo floor on
the *extra* grouping too.

**scen_re_multi**: set == get per grouping, independent jitter present,
beta unbiased — PASS

# How this was produced

| item                     | value                        |
|:-------------------------|:-----------------------------|
| Report generated         | 15 June 2026                 |
| R version                | R version 4.5.3 (2026-03-11) |
| mcpower                  | 0.0.0.9000                   |
| Gate band (SE-of-mean z) | 4                            |
| Golden tolerance         | 1e-09                        |
| Cases                    | 21                           |
| Gates                    | 72                           |

Cases live in `mcpower/validation/formulas.R` (`SCENARIO_CASES`), probes
in `mcpower/validation/common.R`, gates in
`mcpower/validation/tolerances.R` (`SCENARIO_TOL`). Golden baseline:
`mcpower/validation/data/scenario_golden.rds` (delete it to re-freeze
after a *deliberate* DGP change). To reproduce, from the repository
root:

``` r
rmarkdown::render("mcpower/validation/validation_scenarios.rmd",
                  output_dir = "mcpower/documentation/validation")
```
