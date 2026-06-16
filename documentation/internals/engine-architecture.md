# One engine, four ports

MCPower ships in four forms — a **Python** package, an **R** package, a **desktop
app**, and a **browser app** — but there is only one piece of software doing the
statistics. All four are thin wrappers around the same compiled **native engine**
(written in Rust). The engine never changes between them, so a power number computed
in Python is produced by the exact same calculation as the one you'd get in R, in
the desktop app, or in your browser. There is one implementation to trust, not four
that could quietly drift apart.

## Two operations

Underneath every analysis, the engine does just two things:

- **Power estimation** — you fix a sample size, and it estimates how much power your
  design has at that size.
- **Sample-size search** — you fix a target power, and it finds the smallest sample
  size that reaches it (the headline N is read off a fitted power curve — see
  [[concepts/required-sample-size|how required N is estimated]]).

Everything you do in the [[tutorial-python/01_first-analysis|first analysis]] — and
in every later tutorial — is one of these two calls with a different model in front
of it. The model changes; the engine entry points don't.

## Parallelism depends on the product

Simulating thousands of datasets is embarrassingly parallel, and the engine takes
advantage of that — but *how* depends on where it runs:

- **Python, R, and the desktop app** run the engine with full access to your
  machine, so it spreads the simulations across **all your CPU cores automatically**,
  in a single call. You don't configure anything.
- **The browser** can't use multiple cores inside one page thread. So the browser
  app instead launches several independent single-core workers, each simulating a
  share of the datasets, and then **merges** their partial results into one final
  answer. The path is different; the result has the same shape.

## Why two runs aren't byte-identical — and why that's fine

Because the browser splits the work differently, it walks a *different*
random-number path than a single native run does. Splitting 1,600 simulations
across, say, 8 browser workers is not the same sequence of draws as 1,600
simulations on one native thread. The individual simulated numbers differ.

The **power estimate**, though, agrees — within ordinary Monte-Carlo sampling
error. A power estimate from $n_\text{sims}$ simulations carries a standard error of
about $\sqrt{p(1-p)/n_\text{sims}}$; at 50% power over the default 1,600 simulations
that is roughly **±1.25 percentage points**, so two runs typically land within a few
points of each other. Reproducibility in MCPower is a **per-machine, per-seed**
guarantee — same machine, same seed, same inputs, same numbers every time — not a
promise that different products produce byte-identical draws. That last part is by
design. See [[internals/optimizations|the reproducible random-number stream]] for
the per-machine guarantee.

## The same result everywhere

Both the single-call native path and the merge-the-workers browser path return the
**same result structure**. A result computed in one environment can move to another
unchanged — the answer doesn't depend on which door you came in through.

---

See also [[internals/optimizations|Fast, parallel, reproducible]],
[[validation/index|how we know it's right]], and the
[[about/index|app vs. the packages]] for choosing a product.
