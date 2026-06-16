# Model misspecification

The other concepts here are about the **data** you generate — effect sizes, correlations, sample size. This page is about the **model you fit to it**. They are not the same thing: MCPower can generate data from one model and then measure power on a *different* model that you name with `test_formula`. When the model you analyse does not match the truth, the power you get can mislead you badly — sometimes far too low, sometimes spuriously high.

The mechanic — how to pass `test_formula`, and the rule that every term in it must already exist in your generation formula — lives in [[concepts/model-specification#test-formula-misspecification|model specification]]. This page is about *why it matters*.

## One truth, three fits

Students who study more also drink more coffee. Studying genuinely raises the exam score; caffeine does nothing to it — it only rides along with studying because the two are correlated. Write that truth as a generation model:

```python
model = MCPower("score = study + caffeine")
model.set_effects("study=0.3, caffeine=0")          # caffeine has NO real effect
model.set_correlations("corr(study, caffeine)=0.6")
```

```r
model <- MCPower$new("score ~ study + caffeine")
model$set_effects("study=0.3, caffeine=0")          # caffeine has NO real effect
model$set_correlations("corr(study, caffeine)=0.6")
```

Because caffeine's coefficient is `0`, the **genuinely correct model is `score = study`** — caffeine sits in the generation formula only so it can shape the data and carry the correlation, not because it belongs in the answer. It *has* to sit there: `test_formula` terms must be a subset of the generation formula, so caffeine being present at effect 0 is the only way to make it available to a model that adds it while keeping it out of the truth.

Now fit three models against that one truth:

```python
# 1. Correct — matches the truth
model.find_power(sample_size=100, target_test="study", test_formula="score = study")
# 2. Mis-specified — omit the real cause
model.find_power(sample_size=100, target_test="caffeine", test_formula="score = caffeine")
# 3. Over-specified — add the null covariate
model.find_power(sample_size=100, target_test="study, caffeine", test_formula="score = study + caffeine")
```

```r
# 1. Correct — matches the truth
model$find_power(sample_size = 100, target_test = "study", test_formula = "score ~ study")
# 2. Mis-specified — omit the real cause
model$find_power(sample_size = 100, target_test = "caffeine", test_formula = "score ~ caffeine")
# 3. Over-specified — add the null covariate
model$find_power(sample_size = 100, target_test = "study, caffeine", test_formula = "score ~ study + caffeine")
```

The **correct** model — fitting exactly what drives the data — gives `study` about **85%** power at N = 100. That is the honest reference point. (Continuous effects benchmark at small 0.10 / medium 0.25 / large 0.40, so 0.30 is a solid medium-to-large effect.) The other two fits each go wrong, in opposite directions.

## Direction A — confounding (omit a real cause)

Drop the real cause and test only its correlated proxy, and caffeine comes back significant about **40%** of the time — eight times the 5% you would expect for a variable with no real effect at all. Studying drives the score; caffeine is correlated with studying; so when studying is left out of the model, caffeine absorbs studying's signal and looks important.

This is omitted-variable bias seen through the lens of power: **dropping a correlated true cause manufactures a spurious, significant effect on an innocent bystander.** The stronger the [[concepts/correlations|correlation]] between the dropped cause and the proxy, the more signal the proxy steals — at `corr = 0.9` the spurious "effect" would look stronger still.

## Direction B — over-specification (add a null covariate)

Keep the real cause but pad the model with the null covariate, and the damage runs the other way: `study` falls to about **66%** (down from ~85% in the correct model), while caffeine sits at about **5%**, right at α — correctly flagged as null. Nothing about the truth changed; only the analysis model grew.

A correlated null predictor shares variance with `study`, so it absorbs some of study's *unique* variance and inflates its standard error — the real effect becomes harder to detect. **Padding a model with junk covariates is not free; it costs power on the effects you actually care about.**

> [!note] The over-specified test equals the default fit
> `score = study + caffeine` is the generation formula itself — the model MCPower fits when you pass no `test_formula` at all. It is named explicitly here only to line the three cases up side by side.

## The lesson

| Test formula | What it is | Result | Moral |
|---|---|---|---|
| `score = study` | correct (matches truth) | study ≈ 85% | the honest reference |
| `score = caffeine` | omits the real cause | caffeine ≈ 40% | a correlated proxy fakes an effect |
| `score = study + caffeine` | adds a null covariate | study ≈ 66%, caffeine ≈ 5% | a null covariate drains real power |

**The model you test, not just the data you generate, decides what you can detect.** A power analysis is only as honest as the model behind it: estimate power for the model you will actually fit, and when you are unsure which covariates belong, test the misspecified versions too — MCPower will show you what each choice costs.

See [[concepts/model-specification|model specification]] for formula syntax and the `test_formula` mechanic, and [[concepts/correlations|correlations]] for why correlated predictors move power in the first place.
