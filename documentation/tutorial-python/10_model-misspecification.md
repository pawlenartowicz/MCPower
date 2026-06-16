# Model misspecification

Every rung so far fit the **same** model that generated the data — you set effects, then measured power for those exact effects. Real analyses are rarely so tidy: you choose which covariates to include, and that choice is itself a modelling decision with a power cost. This rung uses `test_formula` to generate data from one model and measure power on a *different* one, so you can see what misspecifying your analysis does to power — in both directions.

## One truth, three fits

The story: students who study more also drink more coffee. Studying genuinely raises the exam score; caffeine does nothing to it — it only rides along with studying because the two are correlated (`corr = 0.6`). We write that truth as the generation model, with **caffeine's effect set to 0**:

```python
model = MCPower("score = study + caffeine")
model.set_effects("study=0.3, caffeine=0")
model.set_correlations("corr(study, caffeine)=0.6")
```

Because caffeine's coefficient is 0, the **genuinely correct model is `score = study`** — caffeine is in the generation formula only so it can shape the data and carry the correlation. `test_formula` lets us fit three analysis models against this one truth. (The rule: every term in a test formula must already exist in the generation formula, which is why caffeine sits there at effect 0 — so we can still name it.)

## The correct model

First, fit exactly what drives the data — `score = study`:

<!-- example:10-misspec-correct -->
```python
from mcpower import MCPower

# Truth: studying raises the score; caffeine only rides along via correlation.
model = MCPower("score = study + caffeine")
model.set_effects("study=0.3, caffeine=0")
model.set_correlations("corr(study, caffeine)=0.6")

# Correct model — fit what actually drives the data.
result = model.find_power(sample_size=100, target_test="study",
                          test_formula="score = study", verbose=False)
print(result.summary())
```

```
==================================================
  MCPower · Power Analysis
==================================================
formula: score = study + caffeine
estimator: OLS  N=100  sims=1600  α=0.05  target=80%
effects: study=0.30, caffeine=0.00

Per-test power
───────────────────────────────────
Test                 Power   Target
───────────────────────────────────
study                84.9%      80%
───────────────────────────────────

Power & 95% CI
───────────────────────────────────────────
Test                 Power           CI 95%
───────────────────────────────────────────
study                84.9%   [83.1%, 86.6%]
───────────────────────────────────────────
95% CIs are Monte-Carlo (Wilson), n_sims=1600.

Joint significance distribution
────────────────────────
k     Exactly   At least
────────────────────────
0       15.1%       100%
1       84.9%      84.9%
────────────────────────

Plots: result.plot() to view, result.plot('chart.png') to save.
```
<!-- /example -->

At N = 100, `study` reaches **84.9%** power. That is the honest reference point — the power you actually have when your analysis matches reality. The next two fits each break that match, in opposite directions.

## Dropping the real cause

Now omit the real cause and test only its correlated proxy — fit `score = caffeine`:

<!-- example:10-misspec-confounded -->
```python
from mcpower import MCPower

model = MCPower("score = study + caffeine")
model.set_effects("study=0.3, caffeine=0")
model.set_correlations("corr(study, caffeine)=0.6")

# Mis-specified — drop the real cause, keep its correlated proxy.
result = model.find_power(sample_size=100, target_test="caffeine",
                          test_formula="score = caffeine", verbose=False)
print(result.summary())
```

```
==================================================
  MCPower · Power Analysis
==================================================
formula: score = study + caffeine
estimator: OLS  N=100  sims=1600  α=0.05  target=80%
effects: study=0.30, caffeine=0.00

Per-test power
───────────────────────────────────
Test                 Power   Target
───────────────────────────────────
caffeine             40.2%      80%
───────────────────────────────────

Power & 95% CI
───────────────────────────────────────────
Test                 Power           CI 95%
───────────────────────────────────────────
caffeine             40.2%   [37.9%, 42.7%]
───────────────────────────────────────────
95% CIs are Monte-Carlo (Wilson), n_sims=1600.

Joint significance distribution
────────────────────────
k     Exactly   At least
────────────────────────
0       59.8%       100%
1       40.2%      40.2%
────────────────────────

Plots: result.plot() to view, result.plot('chart.png') to save.
```
<!-- /example -->

caffeine — a variable with **no real effect** — now tests significant **40.2%** of the time, eight times the 5% you would expect from pure noise. This is **omitted-variable confounding**: studying drives the score, caffeine is correlated with studying, so with studying left out of the model caffeine absorbs its signal and looks important. Dropping a correlated true cause manufactures a spurious effect on an innocent bystander — and the stronger the [[concepts/correlations|correlation]], the worse it gets.

## Adding a null covariate

The opposite mistake: keep the real cause but pad the model with the null covariate — fit the full `score = study + caffeine`:

<!-- example:10-misspec-overspec -->
```python
from mcpower import MCPower

model = MCPower("score = study + caffeine")
model.set_effects("study=0.3, caffeine=0")
model.set_correlations("corr(study, caffeine)=0.6")

# Over-specified — keep the real cause but add the null covariate.
result = model.find_power(sample_size=100, target_test="study, caffeine",
                          test_formula="score = study + caffeine", verbose=False)
print(result.summary())
```

```
==================================================
  MCPower · Power Analysis
==================================================
formula: score = study + caffeine
estimator: OLS  N=100  sims=1600  α=0.05  target=80%
effects: study=0.30, caffeine=0.00

Per-test power
───────────────────────────────────
Test                 Power   Target
───────────────────────────────────
study                65.8%      80%
caffeine              5.2%      80%
───────────────────────────────────

Power & 95% CI
───────────────────────────────────────────
Test                 Power           CI 95%
───────────────────────────────────────────
study                65.8%   [63.4%, 68.0%]
caffeine              5.2%   [ 4.3%,  6.5%]
───────────────────────────────────────────
95% CIs are Monte-Carlo (Wilson), n_sims=1600.

Joint significance distribution
────────────────────────
k     Exactly   At least
────────────────────────
0       31.8%       100%
1       65.4%      68.2%
2        2.8%       2.8%
────────────────────────

Plots: result.plot() to view, result.plot('chart.png') to save.
```
<!-- /example -->

caffeine behaves now — **5.2%**, right at α, correctly flagged as null. But `study` has dropped from 84.9% to **65.8%**. Nothing about the truth changed; only the analysis model grew. A correlated null predictor shares variance with `study`, absorbing some of its *unique* variance and inflating its standard error, so the real effect gets harder to detect. Padding a model with junk covariates is not free — it costs power on the effects you care about.

> [!note]
> This over-specified test, `score = study + caffeine`, is the generation formula itself — the model MCPower fits when you give no `test_formula` at all. It is named explicitly here only to line all three cases up side by side.

## The lesson

| Test formula | What it is | study | caffeine |
|---|---|---|---|
| `score = study` | correct (matches truth) | **84.9%** | — |
| `score = caffeine` | omits the real cause | — | **40.2%** (should be ~5%) |
| `score = study + caffeine` | adds a null covariate | **65.8%** | **5.2%** |

**The model you test, not just the data you generate, decides what you can detect.** Estimate power for the model you will actually fit — and when you are unsure which covariates belong, test the misspecified versions too, as above, to see what each choice costs. The [[concepts/model-misspecification|model misspecification]] concept page works through the same two directions in more depth.

next → [[concepts/limitations|When to be cautious]]
