# Scenario analysis

Traditional power analysis assumes ideal conditions: effects exactly as specified, perfectly normal distributions, constant error variance. Real studies never match those assumptions. Scenario analysis reports how robust your power is when they break down — a *range* of power estimates instead of one optimistic number.

This page is the reference for every scenario knob: what assumption violation it represents, how MCPower simulates it, and what the number means in your simulated data.

## The three built-in scenarios

- **Optimistic** — your exact settings, no perturbations. The best case.
- **Realistic** — moderate violations: effects vary between participants, correlations fluctuate, distributions drift from normal.
- **Doomer** — severe violations, a worst case.

> [!tip] Plan around Realistic
> If even **Doomer** clears your target, the design is highly robust. If only **Optimistic** reaches it, increase the sample size — the design is fragile.

Each scenario is a named set of knob values:

| Knob | Optimistic | Realistic | Doomer |
|---|---|---|---|
| `heterogeneity` | 0 | 0.20 | 0.40 |
| `heteroskedasticity_ratio` (λ) | 1.0 (off) | 2.0 | 4.0 |
| `correlation_noise_sd` | 0 | 0.15 | 0.30 |
| `distribution_change_prob` | 0 | 0.5 | 0.8 |
| `residual_change_prob` | 0 | 0.5 | 0.8 |
| `residual_df` | 10 | 8 | 5 |
| `sampled_factor_proportions` | false | true | true |
| `random_effect_dist` (mixed models) | normal | heavy_tailed | heavy_tailed |
| `random_effect_df` (mixed models) | 5 | 10 | 5 |
| `icc_noise_sd` (mixed models) | 0 | 0.15 | 0.30 |

Every knob is documented in **The knobs** below. The replacement pools (`new_distributions`, `residual_dists`) are the same in all three presets; they matter when you build your own scenario (see **Custom scenarios**).

## Reading the output

Each scenario adds one column to the per-test power table, one power-and-CI block, and one column to the **Robustness** section, which reports Δ power versus the baseline (optimistic) in percentage points.

Scenario columns are **paired runs on the same draws**: every scenario in a call runs on the same random-number stream, so wherever a knob doesn't perturb the data, the scenarios see identical datasets. The Δ-power table is therefore attributable to the knobs, not simulation noise.

Read the spread, not just the levels: a test whose power barely moves across columns is robust to the perturbations; one that falls steeply depends on the violated assumption. For a worked example, see the custom-scenarios tutorial for your port.

## The knobs

Three groups: one **effect knob** (`heterogeneity`), three **design knobs** (`correlation_noise_sd`, `distribution_change_prob` with its pool `new_distributions`, `sampled_factor_proportions`), and the **error knobs** (`heteroskedasticity_ratio`, the residual trio) — plus the mixed-model group (`random_effect_dist`, `random_effect_df`, `icc_noise_sd`). Each follows the same shape: the assumption violation, what MCPower does, what the value means, and the empirical values from the literature that the preset numbers are calibrated to (full citations in **References**).

### Heterogeneity

Your model assumes one true effect for everyone, but in real replications the effect varies study to study. The `heterogeneity` knob (`h`) draws a fresh effect per simulated study with SD = `h`·|β|, so `h = 0.2` lets an effect set at 0.50 wander between roughly 0.40 and 0.60. Set `0` to turn it off; presets use 0.20 (realistic) and 0.40 (doomer).

**Definition.** The model treats each effect as one fixed number for everyone. In reality the true effect varies from study to study — across people, sites, and protocols a different effect is realized each time. This is effect heterogeneity, the routine finding of multisite replications.

**How MCPower simulates it.** Each simulated dataset is treated as **one realized study** whose true effect is a single draw: for every effect, one coefficient is drawn from a normal centred on the specified value with SD = h·|β|, and that *same* coefficient applies to every observation in the dataset. This is genuine *per-study* variation — the between-study SD τ = h·|β| that multisite replications report — not per-observation noise. The draw is clipped so a realized effect is pulled toward zero but never reverses sign: a study that would flip becomes a true null. Linear models jitter every effect but leave the intercept fixed. Logistic models jitter on the log-odds scale, and the baseline is jittered too: SD = h on the baseline log-odds (a nuisance shift, never clipped), so the baseline odds wobble by roughly ±h·100%, independent of the intercept's size.

**What the value means.** h = 0.2: the effect you specified as 0.50 varies from study to study with SD ≈ 0.10 — most simulated studies land between about 0.40 and 0.60, i.e. 80–120% of the effect you set. Because each study draws its effect once, this variation doesn't average away as the sample grows: at large h it sets a hard ceiling on achievable power that more data can't beat (see [[concepts/limitations|Limitations]]). To pin heterogeneity — including to 0 — in a particular scenario, set it in that scenario's config (see **Custom scenarios**).

> [!note] Empirical values in research
> The anchor is the between-study effect SD (τ) from multisite replications, divided by the effect size (h = τ/|β|) — exactly the quantity the per-study draw simulates.
> - Klein et al. (2018; Many Labs 2) — 28 effects replicated at 125 sites: 68% of effects had τ ≈ 0, eight ≈ 0.10, and only one above 0.20 (τ = 0.24). Against typical replication effect sizes this places realistic 0.2 mid-distribution and doomer 0.4 in the upper range.
> - Holzmeister et al. (2024) — 70 meta-analyses: *population* heterogeneity (same design, different samples — the directly matching source) has median τ = 0.06; *design* heterogeneity (different protocols) median τ = 0.23. For typical effect sizes these map to h ≈ 0.2–0.4.
> - Klein et al. (2014; Many Labs 1) — across 36 standardised sites, between-site variance is near zero for most effects, grounding h = 0 for tightly matched, single-protocol designs.
> - Gelman et al. (2026) — proposes planning from a *distribution* of individual effects with a generic default SD ≈ |β| (h ≈ 1.0) plus a large share of pure nulls; together with Bryan, Tipton & Yeager (2021) — researchers systematically *underestimate* heterogeneity — it suggests that for broad, mixed-population interventions even doomer 0.4 is conservative.

### Correlation noise

The correlations you set between predictors are estimates, not fixed truths — each sample's measured correlation wobbles around the real one. The `correlation_noise_sd` knob adds Gaussian noise of that SD to your specified correlations on each draw: at `0.15`, a specified r = 0.30 is simulated from roughly r ≈ 0.20–0.40. Set `0` to turn it off; presets use 0.15 (realistic) and 0.30 (doomer).

**Definition.** The correlations you specify between predictors are estimates, not constants — a correlation measured in any one sample fluctuates around the population value, so the design's correlation structure is itself uncertain.

**How MCPower simulates it.** Symmetric Gaussian noise with this SD is added to the specified correlations at each design draw; perturbed values are kept within ±0.8 and the matrix is repaired so it remains a valid correlation matrix. The noise applies on the latent scale used to build the joint distribution — before each variable takes its target shape — so it perturbs the structure, not the marginals. A correlation that was **measured** from [[concepts/upload-data|uploaded data]] (`partial` mode) enters this same latent scale — the upload measures the rank (Spearman) correlation and converts it to the latent scale before generation — so the noise perturbs it exactly as it does a value you set by hand.

**What the value means.** sd = 0.15: a specified r = 0.30 is simulated from roughly r ≈ 0.20–0.40 (±1 SD) from one design draw to the next.

> [!note] Empirical values in research
> - Gnambs (2023) — the standard error of a correlation is ≈ 1/√(n−3): about 0.15 at n ≈ 45–50 and 0.30 at n ≈ 14. Realistic 0.15 is the per-sample SD of r in a typical mid-sized study; doomer 0.30 is that of a severely underpowered one.
> - Schönbrodt & Perugini (2013) — a ±0.15 corridor is their boundary between "stable" and "unstable" correlation estimates, reached only around n ≈ 100; smaller samples — common in psychology — fluctuate by more than 0.15 per draw.
> - Open Science Collaboration (2015) — across 100 replicated studies, the SD of replication correlations was 0.257, combining sampling error, lab effects, and true moderation. The sampling-error component alone (≈ 0.10–0.15 at typical n) matches realistic; the full spread supports doomer.
> - Stanley & Spence (2014) — measurement error and sampling error each independently move observed correlations by ≈ 0.07–0.13; combined, at small n and low reliability, replication intervals imply SDs near 0.24–0.30.

### Distribution swaps

Predictors you assume normal often arrive skewed or otherwise non-normal once data are collected. The `distribution_change_prob` knob sets the per-simulation chance that an unpinned normal predictor is reshaped into a random pick from the `new_distributions` pool (right-skewed, left-skewed, uniform): at `0.5`, about half the draws reshape a given predictor. Set `0` to turn it off; presets use 0.5 (realistic) and 0.8 (doomer).

**Definition.** Variables assumed normal often turn out skewed, bounded, or otherwise non-normal once the data are collected.

**How MCPower simulates it.** At each simulation, every predictor whose distribution is the unpinned default (normal) is independently swapped, with probability `distribution_change_prob`, to a shape drawn from the `new_distributions` pool (default pool: right-skewed, left-skewed, uniform). Declaring any explicit continuous distribution for a predictor — including explicit `normal` — pins it, and a pinned predictor is never swapped. Binary predictors and uploaded data are also never touched. The pool may also include `high_kurtosis`, and `normal` itself (a swap to normal changes nothing, so listing it dilutes the effective swap rate); `binary` is rejected — see **Custom scenarios**.

**What the value means.** prob = 0.5: in about half the design draws, a given normal predictor is generated skewed or uniform instead. Specified correlations are preserved as far as the new shape allows; the realized correlations bend a little (see **How the knobs interact**).

> [!note] Empirical values in research
> - Micceri (1989) — of 440 large-sample achievement and psychometric distributions, 100% were significantly non-normal and only 4.3% were reasonable Gaussian approximations.
> - Cain, Zhang & Yuan (2017) — 1,567 distributions from published psychology/education work: 74% significantly non-normal, rising to 95% at n > 106.
> - Blanca et al. (2013) — 693 real distributions, only 5.5% near-normal; skewness splits roughly evenly between left (31%) and right (35%), and 46% are platykurtic — matching a pool of right-skewed, left-skewed, and uniform shapes. The pool's skewed shape (skew ±1.9 — see [[concepts/variable-types|variable types]]) sits at the upper edge of the skewness they observe in real data (−2.49 to +2.33).
> - Bono et al. (2017) — the most common non-normal families in empirical data are right-skewed (gamma, negative binomial, lognormal, exponential); the pool's skewed shape is exponential-based.
>
> Against observed prevalence of 74–100%, realistic 0.5 is conservative and doomer 0.8 mid-range — though these surveys measure outcome variables and scale scores, not predictors specifically.

### Factor allocation

This knob decides whether each simulated dataset hits your requested group proportions exactly, or lets realized group sizes vary the way random assignment does. The `sampled_factor_proportions` knob is `false` for exact planned counts (optimistic) and `true` for simple randomization, where counts jitter around the target each simulation — at N = 100 with a 50/50 factor, roughly 45–55 per group. Realistic and doomer use `true`.

**Definition.** Planned allocation versus simple randomization: does each simulated dataset hit the requested group proportions exactly, or do realized group sizes vary the way per-participant random assignment makes them vary?

**How MCPower simulates it.** `false`: factor cell counts match the requested proportions exactly in every simulation, the way a planned allocation would. `true`: each observation is assigned independently with the requested probabilities, so realized counts jitter multinomially around the target from simulation to simulation. The optimistic baseline — and any custom scenario that doesn't set it — uses `false`; realistic and doomer use `true`. Unlike every other knob, this one applies to all model types, mixed models included.

**What the value means.** At N = 100 with a 50/50 binary factor, `true` gives roughly 45–55 per group (±1 SD) across simulations instead of exactly 50/50. Under either allocation mode, a factor whose level falls below the 5-observation minimum is excluded from that run's model and reported in the diagnostics — its effects record power 0 while the rest of the model is still analysed. See [[limitations#Sparse factor levels at small N]].

**With uploaded data.** In the distribution-mapping upload modes (`none` / `partial`), a factor from your data obeys the knob like any other — its empirical proportions are the target, hit exactly (`false`) or with multinomial jitter (`true`). In **strict bootstrap** mode the knob has no effect on uploaded predictors: every simulated row is a resampled real row, so group sizes follow the data — planning exact cell counts would break apart the very joint structure strict exists to preserve. A factor you declare on top of the upload (one that is not a column of your data) is still simulated and still obeys the knob. See [[concepts/upload-data|Using empirical data]].

> [!note] Empirical values in research
> - Lim & In (2019) — under simple randomization, the probability of landing outside a 45/55 split is 52.7% at n = 40, 15.7% at n = 200, and 4.6% at n = 400. At typical power-analysis sample sizes imbalance is more likely than not, so sampled counts are the realistic case.
> - Kang et al. (2008) — the power cost is modest: at n = 40 (d = 0.91), an exact 20/20 split gives 80% power, a realized 25/15 gives 77%, an extreme 30/10 gives 67%.
> - Lachin (1988) — group sizes under simple randomization are binomial; power is materially reduced only beyond about a 70/30 split, a remote event.
> - Schulz & Grimes (2002a, b) — exact balance is power-optimistic but bias-pessimistic: in non-blinded trials a forced-balance schedule makes upcoming assignments predictable, and published "simple randomized" trials report exactly equal groups far more often than chance allows (54–71%).

### Heteroskedasticity

Linear models assume the error variance is constant, but real error often grows or shrinks along a predictor. The `heteroskedasticity_ratio` knob (λ) sets how lopsided the variance is: λ is the residual-variance ratio between the high and low ends of the driver, so λ = 4 means 4× the variance at the high end. Total noise is held fixed — only its pattern tilts. λ = 1 turns it off; presets use 2.0 (realistic) and 4.0 (doomer). Linear models only.

**Definition.** Linear models assume constant error variance (homoskedasticity). In real data the error variance often depends on the predicted value — or on a specific predictor — violating that assumption.

**How MCPower simulates it.** A monotone variance pattern along a driver (by default the predicted value): λ is the ratio of residual variance between observations 2 SD above and 2 SD below the driver's mean. The total amount of noise is preserved — dialing λ changes the *pattern* of the noise, not its overall amount. λ = 1 disables it. Linear models only.

To make the driver part of the *model* rather than a perturbation, call `set_heteroskedasticity_driver(var)` — the `var` is the predicted value or any continuous predictor. The λ ratio itself always comes from the active scenario's `heteroskedasticity_ratio`; there is no model-level pin for λ. To run with a known λ, override it in your scenario configs:

```python
model.set_scenario_configs({"optimistic": {"heteroskedasticity_ratio": 2.0}})
```

```r
model$set_scenario_configs(list(optimistic = list(heteroskedasticity_ratio = 2.0)))
```

With scenarios off, MCPower runs the single merged optimistic scenario — so the override above applies to that run too. With scenarios on, a known λ must be set in every scenario the run includes (there is no single-pin that spans all scenarios at once).

**What the value means.** λ = 4: residuals at the high end of predictions have 4× the variance of those at the low end. Because the total noise is unchanged, the cost shows up through the *mismatch* between the assumed and actual error structure, not through extra noise.

> [!note] Empirical values in research
> - Gelfand (2015) — across 42 real regression datasets, 25 were heteroskedastic with a median SD-ratio of 3.08, i.e. a variance ratio ≈ 9.5×; named examples reach 10–28×.
> - Long & Ervin (2000) — the "moderate" error structures in their reference simulation reach variance ratios ≈ 5–8; the mild ones (≈ 1.4–1.7×) align with λ = 2.
> - Hayes & Cai (2007); Cook & Weisberg (1983); White (1980) — establish how routinely real datasets fail homoskedasticity tests, without pinning a magnitude.
>
> Real heteroskedastic data is typically *worse* than the doomer setting: the observed median (≈ 9.5×) exceeds λ = 4, so the presets are conservative. If your driver strongly governs the variance, λ = 8–10 is empirically defensible — set it in a custom scenario.

### Residual swaps

Residuals assumed normal are often heavy-tailed or skewed in practice. The `residual_change_prob` knob is the per-simulation chance the residual shape is replaced by a draw from the `residual_dists` pool (heavy-tailed t, right-skewed); `residual_df` sets how heavy the t tails are — lower df means more outlier-dominated runs (5 is severe, 10 is mild). Presets pair prob 0.5/0.8 with df 8/5 (realistic/doomer). Linear models only.

**Definition.** Residuals assumed normal are, in practice, often heavy-tailed (outlier-prone) or skewed.

**How MCPower simulates it.** At each simulation, with probability `residual_change_prob`, the residual distribution is *replaced* — not blended — by a shape drawn from the `residual_dists` pool. The canonical pool entries are `high_kurtosis` (Student t, with df from the active scenario's `residual_df`) and `right_skewed`; the built-in presets carry `["high_kurtosis", "right_skewed"]`. A swap fires only when the model's residual distribution is the unpinned default — calling `set_residual_distribution` with any name (including explicit `"normal"`) pins the residual, and a pinned residual is never swapped. `residual_df` also supplies the df when a `high_kurtosis` residual is pinned. On no-swap simulations the model's residual setting stays. `normal` is also a valid pool entry (a swap to normal changes nothing — the same dilution trick as in `new_distributions`). Linear models only.

**What the value means.** df = 5: t residuals with four times the excess kurtosis of df = 8 (6 vs 1.5) — the lower the df, the more simulations are dominated by outliers. A custom scenario that doesn't set `residual_df` inherits 10 from the optimistic baseline, a mild shape. This df also applies when `high_kurtosis` is pinned as the model's residual distribution.

> [!note] Empirical values in research
> - Cain, Zhang & Yuan (2017) — calibrates the df ladder: the 75th-percentile excess kurtosis observed in real data (1.62) maps to t-df ≈ 7.7 ≈ realistic df = 8; the 95th percentile (9.48) maps to df ≈ 4.6 ≈ doomer df = 5.
> - Lange, Little & Taylor (1989) — fitted t degrees of freedom across real scientific datasets cluster in 3–10, with df = 4 recommended as a robust default; df ≈ 10 fits near-normal data, the optimistic baseline.
> - Ng, Zhu, Zhang & Reid (2026) — robust-regression work treating df ∈ {2, 5, 10} as the operative heavy-to-mild range; the 5/8/10 ladder sits within its moderate band.
> - Micceri (1989); Blanca et al. (2013) — the prevalence backdrop: 49% of real distributions have at least one heavy tail, and the maximum excess kurtosis Blanca et al. observe (7.41) corresponds to df ≈ 4.8.
>
> Those prevalence rates are for raw observed variables; residuals are pushed toward normality by averaging over predictors, so the true swap rate is plausibly nearer 0.3–0.4. The preset 0.5 therefore errs slightly toward severity — the upper edge of "realistic" — while the df ladder is well-calibrated conditional on a swap.

### Mixed-model knobs

Mixed models add three knobs that stress-test the random-effect structure. `random_effect_dist` swaps the cluster-effect shape from `normal` to `heavy_tailed` (more extreme clusters) or `right_skewed`, with `random_effect_df` controlling the heavy-tailed t's df (5 is heavy, 10 is mild). `icc_noise_sd` jitters the between-cluster variance to probe ICC uncertainty. They apply to LMM and clustered-logistic (GLMM) models and are inert elsewhere.

Mixed models support three scenario knobs for stress-testing the random-effect structure:

**`random_effect_dist`** — the shape of the group-level random effects. Values: `"normal"` (default), `"heavy_tailed"` (Student t, parameterised by `random_effect_df`; fatter tails mean more extreme cluster-to-cluster variation), `"right_skewed"` (asymmetric — some clusters have systematically higher baselines). In all cases the random effects are scaled to unit variance so that `tau_squared` (set via ICC) remains the between-cluster variance component; the shape changes only the tail behaviour. Heavier tails have a modest power cost: very large or very small cluster random effects add noise around the within-cluster fixed effects.

**`random_effect_df`** — degrees of freedom for `heavy_tailed`. Minimum 3. At df = 5 the excess kurtosis is 6 (pronounced tails); at df = 10 it is 1.5 (mild). Shared between `heavy_tailed` and `right_skewed` when both are in a pool (analogous to `residual_df`).

**`icc_noise_sd`** — block-to-block Gaussian jitter on the between-cluster variance. At each design draw the random-intercept variance `τ²` (the ICC→τ² conversion of your specified ICC) is perturbed by a draw from N(0, `icc_noise_sd`²) and clamped at 0. Useful for probing sensitivity to ICC uncertainty — a common issue when the ICC is estimated from a pilot study.

> [!note] `icc_noise_sd` on the logistic path
> The jitter is applied in raw τ²-space (the random-intercept variance on the scale the link uses — latent log-odds for logistic, identity for Gaussian). Because the latent-scale τ² for a logistic GLMM at the same ICC is ≈3.3× larger than for a Gaussian LMM (the π²/3 latent factor — see the latent-scale ICC convention in the mixed-effects concept), a given `icc_noise_sd` perturbs the GLMM τ² by a *smaller relative amount*. This is intentional: `icc_noise_sd` is a single cross-path knob and its absolute τ²-space magnitude is the controlled quantity. For the same *relative* perturbation on the logistic path, multiply your `icc_noise_sd` by √(π²/3) ≈ 1.81.

These knobs apply to every grouping in the model (primary plus any crossed/nested extras), on both the Gaussian (LMM) and clustered-logistic (GLMM) mixed-model paths. They are independent of the other knobs (heterogeneity, correlation noise, residual swaps), which target the residual/linear-predictor structure of OLS and non-clustered logistic models.

> [!note] Empirical values in research
> - Thompson et al. (2012) — within-practice ICCs across 61 primary-care practices span 0.007 to 0.265 for different outcomes in the *same* dataset; realistic `icc_noise_sd` 0.15 sits inside that spread, and doomer 0.30 covers its upper (demographic) tail.
> - Adams et al. (2004) — 1,039 ICC estimates from primary-care research: the typical ICC is small (median ≈ 0.01), but the same measure varies 10–50× across datasets (e.g. diastolic blood pressure 0–0.108) — exactly the planning uncertainty the jitter models. (These anchors are ICC-scale; at small ICCs the τ²-space jitter the engine applies has closely similar magnitude on the Gaussian path.)
> - Hedges & Hedberg (2007) — the canonical reference for ICC values when planning group-randomized designs.
> - McCulloch & Neuhaus (2011); Verbeke & Lesaffre (1997) — fixed-effect estimates are largely robust to a misspecified random-effects shape; what degrades is the precision of the variance-component/ICC estimates that feed a power plan. Read the heavy-tailed presets as a stress test on the planning input, not a warning that your fixed-effect power is biased.

**Valid values** for `random_effect_dist`: `"normal"`, `"heavy_tailed"`, `"right_skewed"`.

Example — stress-test a multi-site trial with uncertain ICC and non-normal site effects:

```python
model.set_scenario_configs({
    "site_stress": {
        "random_effect_dist": "heavy_tailed",
        "random_effect_df": 5,
        "icc_noise_sd": 0.07
    }
})
result = model.find_power(sample_size=600, target_test="treatment",
                          scenarios=["optimistic", "site_stress"])
```

```r
model$set_scenario_configs(list(
    site_stress = list(
        random_effect_dist = "heavy_tailed",
        random_effect_df = 5,
        icc_noise_sd = 0.07
    )
))
result <- model$find_power(sample_size = 600, target_test = "treatment",
                           scenarios = c("optimistic", "site_stress"))
```

## Custom scenarios

Scenarios are configured per model with `set_scenario_configs` (see the custom-scenarios tutorial for your port for the exact call). The merge has two branches:

- **Overriding a built-in name** (`optimistic`, `realistic`, `doomer`) **updates that preset.** Knobs you don't set keep their preset values — a custom `realistic` that only raises λ still carries realistic heterogeneity, distribution changes, and residual swaps.
- **A new name inherits the optimistic baseline.** Every knob starts at its optimistic (no-perturbation) value and only your overrides apply — including `residual_df = 10` and the standard pools, so arming a probability knob without restating its pool works out of the box.

After configuring, select scenarios to run by name; names without a custom config run the built-in presets.

**Valid knob names** — exactly these twelve keys: `heterogeneity`, `heteroskedasticity_ratio`, `correlation_noise_sd`, `distribution_change_prob`, `new_distributions`, `residual_change_prob`, `residual_dists`, `residual_df`, `sampled_factor_proportions`, `random_effect_dist`, `random_effect_df`, `icc_noise_sd`.

**Valid `new_distributions` entries:** `normal`, `right_skewed`, `left_skewed`, `high_kurtosis`, `uniform`.

**Valid `residual_dists` entries:** `high_kurtosis` (Student t, df from `residual_df`), `right_skewed`, `left_skewed`, `normal`, `uniform`. `binary` is rejected.

Configurations are validated up front, and the errors are loud by design:

- An unknown or misspelled knob raises an error naming the key and listing the valid set. In particular, the key `heteroskedasticity` taught by older versions of these docs is now `heteroskedasticity_ratio` — the old name errors rather than silently doing nothing.
- The mixed-model keys (`random_effect_dist`, `random_effect_df`, `icc_noise_sd`) take effect only on mixed models — both Gaussian (LMM) and clustered-logistic (GLMM); see **Mixed-model knobs** above. On OLS and non-clustered logistic models they are inert.
- `binary` in `new_distributions` is rejected: a swapped binary column would collapse to a constant.
- A scenario that arms the residual swap (`residual_change_prob > 0`) with a `high_kurtosis` or `right_skewed` pool entry must carry `residual_df >= 3`, or it errors.

## How the knobs interact

The perturbations are designed to compose. Effect jitter and residual swaps are independent noise channels added together: the jitter stays per-observation Gaussian even when the residual shape has been swapped, both inflate the total noise, and their power costs compound.

The heteroskedasticity multiplier rescales whatever residual was drawn — normal or swapped — so the λ pattern holds under high-kurtosis and right-skewed residuals too, with their tails amplified at the high end of the driver. One accepted approximation: λ is calibrated against the *unperturbed* design, so when distribution swaps or correlation noise change the spread of the predicted values, the realized variance ratio deviates slightly from the nominal λ.

Distribution swaps and correlation noise interact through the correlation structure: correlations are built on a latent scale and each variable then takes its target shape, so a swapped shape bends the realized correlations — on top of the noise already added to the latent matrix. Under doomer, realized correlations differ from the specification through both channels at once.

## References

The studies behind the preset values, cited throughout **The knobs**. Three are pre-publication versions (Gelman et al. and Ng et al. are arXiv preprints; Gelfand is a master's thesis).

- Adams, G., Gulliford, M. C., Ukoumunne, O. C., Eldridge, S., Chinn, S., & Campbell, M. J. (2004). Patterns of intra-cluster correlation from primary care research to inform study design and analysis. *Journal of Clinical Epidemiology*. https://doi.org/10.1016/j.jclinepi.2003.12.013
- Blanca, M. J., Arnau, J., López-Montiel, D., Bono, R., & Bendayan, R. (2013). Skewness and kurtosis in real data samples. *Methodology*. https://doi.org/10.1027/1614-2241/a000057
- Bono, R., Blanca, M. J., Arnau, J., & Gómez-Benito, J. (2017). Non-normal distributions commonly used in health, education, and social sciences: A systematic review. *Frontiers in Psychology*. https://doi.org/10.3389/fpsyg.2017.01602
- Bryan, C. J., Tipton, E., & Yeager, D. S. (2021). Behavioural science is unlikely to change the world without a heterogeneity revolution. *Nature Human Behaviour*. https://doi.org/10.1038/s41562-021-01143-3
- Cain, M. K., Zhang, Z., & Yuan, K.-H. (2017). Univariate and multivariate skewness and kurtosis for measuring nonnormality: Prevalence, influence and estimation. *Behavior Research Methods*. https://doi.org/10.3758/s13428-016-0814-1
- Cook, R. D., & Weisberg, S. (1983). Diagnostics for heteroscedasticity in regression. *Biometrika*. https://doi.org/10.1093/biomet/70.1.1
- Gelfand, S. J. (2015). *Understanding the impact of heteroscedasticity on the predictive ability of modern regression methods* (MSc thesis, Simon Fraser University). https://summit.sfu.ca/item/15679
- Gelman, A., Krefman, A., Kennedy, L., & Hullman, J. (2026). Hypothesizing an effect size by considering individual variation. *arXiv*. https://arxiv.org/abs/2604.08421
- Gnambs, T. (2023). A brief note on the standard error of the Pearson correlation. *Collabra: Psychology*. https://doi.org/10.1525/collabra.87615
- Hayes, A. F., & Cai, L. (2007). Using heteroskedasticity-consistent standard error estimators in OLS regression: An introduction and software implementation. *Behavior Research Methods*. https://doi.org/10.3758/BF03192961
- Hedges, L. V., & Hedberg, E. C. (2007). Intraclass correlation values for planning group-randomized trials in education. *Educational Evaluation and Policy Analysis*. https://doi.org/10.3102/0162373707299706
- Holzmeister, F., Johannesson, M., Böhm, R., Dreber, A., Huber, J., & Kirchler, M. (2024). Heterogeneity in effect size estimates. *Proceedings of the National Academy of Sciences*. https://doi.org/10.1073/pnas.2403490121
- Kang, M., Ragan, B. G., & Park, J.-H. (2008). Issues in outcomes research: An overview of randomization techniques for clinical trials. *Journal of Athletic Training*. https://doi.org/10.4085/1062-6050-43.2.215
- Klein, R. A., Ratliff, K. A., Vianello, M., Adams, R. B., Jr., Bahník, Š., Bernstein, M. J., … Nosek, B. A. (2014). Investigating variation in replicability: A "many labs" replication project. *Social Psychology*. https://doi.org/10.1027/1864-9335/a000178
- Klein, R. A., Vianello, M., Hasselman, F., Adams, B. G., Adams, R. B., Jr., Alper, S., … Nosek, B. A. (2018). Many Labs 2: Investigating variation in replicability across samples and settings. *Advances in Methods and Practices in Psychological Science*. https://doi.org/10.1177/2515245918810225
- Lachin, J. M. (1988). Properties of simple randomization in clinical trials. *Controlled Clinical Trials*. https://doi.org/10.1016/0197-2456(88)90046-3
- Lange, K. L., Little, R. J. A., & Taylor, J. M. G. (1989). Robust statistical modeling using the t distribution. *Journal of the American Statistical Association*. https://doi.org/10.1080/01621459.1989.10478834
- Lim, C.-Y., & In, J. (2019). Randomization in clinical studies. *Korean Journal of Anesthesiology*. https://doi.org/10.4097/kja.19049
- Long, J. S., & Ervin, L. H. (2000). Using heteroscedasticity consistent standard errors in the linear regression model. *The American Statistician*. https://doi.org/10.1080/00031305.2000.10474549
- McCulloch, C. E., & Neuhaus, J. M. (2011). Misspecifying the shape of a random effects distribution: Why getting it wrong may not matter. *Statistical Science*. https://doi.org/10.1214/11-STS361
- Micceri, T. (1989). The unicorn, the normal curve, and other improbable creatures. *Psychological Bulletin*. https://doi.org/10.1037/0033-2909.105.1.156
- Ng, A., Zhu, S., Zhang, A. G., & Reid, N. (2026). Robust regression with Student's t: The role of degrees of freedom. *arXiv*. https://arxiv.org/abs/2603.00269
- Open Science Collaboration. (2015). Estimating the reproducibility of psychological science. *Science*. https://doi.org/10.1126/science.aac4716
- Schönbrodt, F. D., & Perugini, M. (2013). At what sample size do correlations stabilize? *Journal of Research in Personality*. https://doi.org/10.1016/j.jrp.2013.05.009
- Schulz, K. F., & Grimes, D. A. (2002a). Generation of allocation sequences in randomised trials: Chance, not choice. *The Lancet*. https://doi.org/10.1016/S0140-6736(02)07683-3
- Schulz, K. F., & Grimes, D. A. (2002b). Unequal group sizes in randomised trials: Guarding against guessing. *The Lancet*. https://doi.org/10.1016/S0140-6736(02)08029-7
- Stanley, D. J., & Spence, J. R. (2014). Expectations for replications: Are yours realistic? *Perspectives on Psychological Science*. https://doi.org/10.1177/1745691614528518
- Thompson, D. M., Fernald, D. H., & Mold, J. W. (2012). Intraclass correlation coefficients typical of cluster-randomized studies: Estimates from the Robert Wood Johnson Prescription for Health projects. *Annals of Family Medicine*. https://doi.org/10.1370/afm.1347
- Verbeke, G., & Lesaffre, E. (1997). The effect of misspecifying the random-effects distribution in linear mixed models for longitudinal data. *Computational Statistics & Data Analysis*. https://doi.org/10.1016/S0167-9473(96)00047-3
- White, H. (1980). A heteroskedasticity-consistent covariance matrix estimator and a direct test for heteroskedasticity. *Econometrica*. https://doi.org/10.2307/1912934
