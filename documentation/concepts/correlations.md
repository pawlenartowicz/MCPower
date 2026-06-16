# Correlations

Setting a correlation tells MCPower to generate two predictors that move together rather than independently, the way real predictors usually do (income and education, age and experience). This cuts both ways for power: shared variance gives each predictor less *unique* variance, so a **main effect** needs a larger sample to reach the same power — but it makes their product more variable, so an **interaction** often needs a smaller one. Only the strength $|r|$ matters, not the sign.

## Why correlation cuts both ways

The effect of correlated predictors on power depends on **what you are testing**:

- **Main effects** — correlation *reduces* power. Shared variance means less unique variance per predictor, so standard errors grow.
- **Interaction effects** — correlation *increases* power. Correlated predictors produce a more variable product term, and a more variable interaction is easier to detect.

> [!note] Only the magnitude matters
> The sign of the correlation (and of the effects) does not change this — only $|r|$ matters. In a model with both main effects and an interaction, correlation creates a trade-off: the main effects want more observations while the interaction wants fewer.

MCPower generates correlated predictors and then transforms each to its target [[concepts/variable-types|distribution]] while preserving the correlation structure.

A correlation you set yourself is treated as a *latent* (Gaussian) correlation — exact for normally-distributed predictors. When MCPower instead **measures** a correlation from [[concepts/upload-data|uploaded data]] (`partial` mode), it targets the **rank (Spearman)** correlation of your data, not the Pearson one, because redrawing each predictor through its empirical shape preserves rank order rather than linear scale.

## The matrix must be consistent (PSD)

MCPower validates every correlation you supply *before* any simulation runs, and rejects impossible input on three levels:

- **Each pair must be in range.** A correlation must lie in $[-1, 1]$; a stray `corr(a, b)=1.5` fails with `correlation pair (a, b): value must be in [-1, 1], got 1.5`.
- **A full matrix must be well-formed.** If you pass a matrix it must be square, symmetric, and have an all-1s diagonal — MCPower errors loudly rather than silently coercing a malformed one.
- **The whole structure must be consistent.** Even when every pair is individually in range, the combination can still be impossible. A valid matrix is **positive semi-definite** (PSD) — i.e. the correlations could actually co-occur in real data. A non-PSD matrix fails with `correlation matrix is not positive semi-definite`.

> [!example] An impossible matrix
> If A–B are strongly correlated ($r = 0.9$) and A–C are strongly correlated ($r = 0.9$), then B and C cannot be strongly *negatively* correlated ($r = -0.9$) — that contradicts the first two relationships. If you hit a PSD error, reduce the most extreme correlations or check that the signs are logically consistent.

## How much does it matter?

| $r$ | Label | Sample-size impact | Action |
|---|---|---|---|
| 0.00–0.20 | Negligible | ~1.0× | Safe to ignore |
| 0.20–0.40 | Small | 1.07–1.20× | Include if known |
| 0.40–0.60 | Moderate | 1.20–1.53× | Always include |
| 0.60–0.70 | Large | 1.53–1.87× | Include; consider dropping a predictor |
| 0.70+ | Very large | >1.87×; multicollinearity risk | Check both predictors are needed |

### Main effects — required sample size grows with r

| Correlation $r$ | Multiplier vs. $r = 0$ |
|---|---|
| 0.00–0.20 | 1.00× |
| 0.30 | 1.07× |
| 0.40 | 1.20× |
| 0.50 | 1.33× |
| 0.60 | 1.53× |
| 0.70 | 1.87× |

### Interactions — required sample size shrinks with r

| Correlation $|r|$ | Multiplier vs. $r = 0$ |
|---|---|
| 0.00 | 1.00× |
| 0.30 | 0.82× |
| 0.50 | 0.76× |

### Three correlated predictors — the impact compounds

| Pairwise correlation | Multiplier vs. all $r = 0$ |
|---|---|
| All 0.00 | 1.00× |
| All 0.30 | 1.14× |
| All 0.50 | 1.57× |

*Multipliers are illustrative, from simulations of two- and three-predictor models with small effects. The relationship — not the absolute sample size — is what carries across designs.*

## Typical correlations by domain

| Domain | Predictor pair | Typical $r$ |
|---|---|---|
| Education | SES and test scores | 0.30–0.50 |
| Psychology | Anxiety and depression | 0.40–0.70 |
| Medicine | Age and blood pressure | 0.20–0.40 |
| Social science | Income and education | 0.40–0.60 |
| Marketing | Ad spend and brand awareness | 0.20–0.40 |

## Binary and factor predictors can't be correlated directly

Correlation is continuous-only by design. Binary predictors and factor predictors (3+ levels) are generated from their marginals, so naming either one in a correlation pair is rejected. When you need correlated categorical *and* continuous variables, simulate from [[concepts/upload-data|empirical data]], which preserves the joint structure of your dataset directly.

Correlations below ~0.30 have negligible impact; above ~0.50 the sample-size increase becomes substantial. Use [[concepts/scenario-analysis|scenario analysis]] to see how a noisy or uncertain correlation structure moves your power.
