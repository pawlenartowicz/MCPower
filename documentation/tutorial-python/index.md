# MCPower for Python

MCPower answers two questions about a study *before* you run it: **how much
statistical power** a given sample size buys, and **what sample size** reaches a
power target. It works by simulating the study many times on a native engine, so
the same short workflow covers OLS, logistic, mixed, and factorial models.

## Install

```bash
pip install mcpower
```

## A first analysis

Describe the model as a formula, mark which predictors are binary, set the effect
sizes you want to be able to detect, and ask for power:

<!-- example:index-power -->
```python
from mcpower import MCPower

# Does a treatment lift satisfaction, controlling for age?
model = MCPower("satisfaction = treatment + age")
model.set_variable_type("treatment=binary")
model.set_effects("treatment=0.5, age=0.3")

result = model.find_power(sample_size=120, target_test="treatment")
```

```
Power Analysis — OLS  N=120  sims=1600  α=0.05  target=80%
formula: satisfaction = treatment + age

───────────────────────────────────
Test                 Power   Target
───────────────────────────────────
treatment            77.4%      80%
───────────────────────────────────
```
<!-- /example -->

That is the whole loop. At `n = 120` this design has **77.4%** power to detect the
treatment effect — just under the conventional 80% target, so you would want a
few more participants. The next page asks both questions properly.

## The tutorial ladder

Each rung adds one modelling idea and powers it. Climb from the top:

1. [[01_first-analysis|First analysis]] — the two questions, in full
2. [[02_interactions|Interactions]] — `a:b` terms
3. [[03_correlations|Correlations]] — correlated predictors
4. [[04_logistic-regression|Logistic regression]] — a binary outcome
5. [[05_mixed-models|Mixed models]] — grouped / hierarchical data
6. [[06_anova-posthoc|ANOVA & post-hoc]] — factors and pairwise contrasts
7. [[07_multiple-testing|Multiple testing]] — corrections across many tests
8. [[08_custom-scenarios|Custom scenarios]] — robustness sweeps
9. [[09_upload-data|Upload data]] — drive simulation from a CSV or dataframe
10. [[10_model-misspecification|Model misspecification]] — what testing the wrong model costs

New to power analysis? The [[concepts/index|Concepts]] pages explain effect sizes,
model specification, and what the power number actually means.
