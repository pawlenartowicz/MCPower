# How required N is estimated

When you ask MCPower for a sample size, it simulates your design over a **grid** of candidate sizes and reports, per tested effect, the N that reaches your target power. As of the model-based crossing release, the headline number is not simply the first grid point whose simulated power happened to clear the target — it is read off a **fitted power curve**.

## The search grid

MCPower finds the required N by evaluating power on a **grid** of candidate sample sizes — it walks from a `from` size up to a `to` size in steps, simulates your design at each one, and reports where power crosses your target. A grid search is the only method MCPower offers, and it works identically on every face: Python, R, the desktop app, and the browser all sweep the same grid and read the same crossing.

You set the range with `from`/`to`; `by="auto"` (the default) places about 12 points between them, or pass `by=<n>` for a fixed step. Defaults are `from=30`, `to=200`. Because the grid points are chosen up front — never adjusted in response to observed power — the search is fully reproducible run to run, and that is also what lets the browser split the work across CPU cores and recombine it. A grid that's too coarse near your target widens the reported interval; refine it by lowering `by` or adding simulations.

## The model-based crossing

Simulated power at each grid point carries Monte-Carlo noise, so the raw "first N that hit 80%" can land a whole grid step early or late on the strength of a lucky draw. Instead, MCPower fits a non-decreasing curve (an **isotonic fit**) through the simulated power values across the *whole* grid and reports the N where that curve crosses the target. The fit uses every grid point, so the headline is more stable than any single point — and it can land *between* grid points, which is why a search "from 30 to 300 by 30" can answer **84**.

The headline is always an integer, rounded **up** — and for clustered designs, up to the nearest whole-cluster size — so the reported N is achievable as designed. Confidence-interval bounds are likewise rounded **outward** (lower bound down, upper bound up): the printed interval never overstates precision.

## The 95% CI on required N

The full report adds a **Required N & 95% CI** section. The interval reflects the Monte-Carlo uncertainty of the simulation (it is derived from the same Wilson intervals shown for power), so more simulations narrow it. A wide interval is information, not a malfunction: it means the power curve is flat near your target, and small changes in assumptions move the required N a lot — consider a finer grid, more simulations, or [[concepts/scenario-analysis|scenario analysis]].

## Reading the result table

The required-N **table** is where you read the answer — the power-vs-N **plot** is just a line and a shaded confidence band, with no points or flags marking the crossing on the curve itself. In the table, a required-N cell is not always a plain number; each annotation is a statement about where the answer sits relative to the range you searched:

- **`≤ 40`** — the target is already met at the smallest N searched. The true requirement is at or below the search floor; rerun with a lower `from` to localise it.
- **`≥ 300`** — the target is not reached anywhere in the searched range. The honest answer is "more than your ceiling"; rerun with a higher `to`.
- **`appr. 330`** — shown next to a `≥` cell when the curve's shape supports a cautious extrapolation beyond the ceiling. It is a hint, not a result: confirm it by rerunning with a higher `to`.
- **Non-monotone warning** (`⚠ … power not monotone in N`) — simulated power for that effect genuinely *decreased* with larger N, by more than noise can explain. A power curve should rise with N; a real decrease usually signals a model issue worth investigating (for example an interaction working against a main effect). The curve fit is suppressed for that effect and the raw grid value is shown instead.

## The overall (omnibus) row

When an overall test is requested — the default for OLS and (unclustered) logistic models — the table opens with an **Overall** row (labelled *Overall F* for OLS, *LR χ²* for logistic) before the per-effect rows. It reports the required N for the omnibus test (the model as a whole) using the same crossing fit, annotations, and CI as every other row. Mixed-effects and clustered-logistic models have no omnibus test, so the row is absent there. (The power-vs-N **plot** also draws the overall test as one more curve with its own confidence band.)

## Joint detection

The joint rows ("≥ k of m tests") get the same treatment: their curves are fitted the same way and their cells use the same annotations, so the N you budget for *all* planned tests is estimated with the same stability as the per-effect numbers.
