# Fast, parallel, reproducible

The native engine is built to make simulation-based power analysis quick and
trustworthy. Three things make that work — and one of them is a knob you control.

## Automatic multi-core

On Python, R, and the desktop app the engine spreads its simulations across **every
CPU core on your machine, with no configuration**. You ask for power; it uses the
hardware it has. (In the browser the same parallelism comes from a pool of workers
instead — see [[internals/engine-architecture|one engine, four ports]].)

## A reproducible random-number stream

Simulation needs randomness, but randomness that you can't repeat would make a power
analysis impossible to check. So the engine uses a **reproducible random-number
stream**: the same seed plus the same inputs produce **identical numbers on every
run**. The default seed is **2137**. You normally never touch it — change it only
when you want to confirm a result is stable across different random draws. See
[[concepts/simulation-settings|simulation settings]].

## Why Grid is the trustworthy sample-size search

When you search for a sample size, the engine reuses the *same* simulated rows as it
moves between nearby sample sizes. The data it uses at N = 35 is literally the data
at N = 30 with five more rows appended — a technique called **common random
numbers**. Because adjacent sample sizes share their underlying data, their power
estimates move *together* rather than independently: the power-versus-N curve comes
out **smooth and near-monotone**, and the sample size where it crosses your target is
**steady from run to run**.

A search that drew fresh data at every sample size would instead produce a jagged
curve and an answer that jumps around between runs even at the same seed. That is why
**Grid** is the reliable sample-size method, and the one the browser app uses.
(The alternative, Bisection, doesn't get this common-random-numbers benefit and isn't
offered in the browser.) For clustered designs the grid snaps to whole clusters, so
the same smooth-curve guarantee carries over to multilevel models.

## Simulations versus precision

This is the one lever worth understanding. More simulations give you a **more precise**
power estimate — a narrower confidence interval around the number — at the cost of
time. The interval width shrinks like $1/\sqrt{n_\text{sims}}$: **doubling the
simulation count roughly halves the interval.** The engine always reports that
interval, so you can see exactly how precise your estimate is.

The defaults are chosen so the precision is good enough for almost every study:

| Design | Default simulations | Why |
|---|---|---|
| OLS / logistic / factor | **1,600** | each fit is very cheap |
| Mixed-effects (clustered) | **800** | each fit costs more |

> [!tip]
> Raise the simulation count when two designs report **nearly the same power** and
> you need to tell them apart — that's exactly when the confidence intervals overlap
> and a sharper estimate pays off.

## Where the engine is strongest

MCPower is tuned for the designs that dominate social science, psychology, medicine,
and ecology: a handful of predictors, anywhere from a few dozen up to a few thousand
rows per simulated dataset, with mixed-effects models as the demanding case. Each
plain regression fit is microsecond-fast, so the engine can run thousands of them in
the blink of an eye; mixed-effects fitting is the main cost, which is why those
designs default to fewer simulations.

---

See also [[concepts/simulation-settings|simulation settings]],
[[internals/engine-architecture|one engine, four ports]], and
[[validation/index|how we know it's right]].
