# Example page anatomy (authoring reference)

Every example page, in this order:

1. `# <Scenario title>` — a recognisable analysis in researcher terms (the
   recognition hook), then the model as an MCPower formula in prose.
2. `## Variations` — near-miss swaps in *prose only* (e.g. "swap the continuous
   `dose` for a 3-level factor `dose_group`"). Descriptive, never runnable code.
   Includes a **Same design, other fields** sub-list (cross-domain twins, below).
3. `## Not this setup?` — wikilinks to the nearest *other* example pages.
4. `## If you'd rather have…` — cross-domain / alternative-framing wikilinks.
5. `## Copy-paste setup` — Python then R, each in a paired chunk marker (below).
   No displayed output. These are setups, not transcripts. **This is the last
   section before the screenshot — copy-paste sits at the bottom of the page**,
   after the prose a reader uses to confirm they're on the right page. Python,
   R, and the screenshot each sit in their own `<details><summary>...</summary>`
   accordion, **closed by default** (see "Accordion tail" below) — the page
   defaults to recognition prose, not a wall of code.
6. Screenshot embeds — the theme-swapping pair
   `![[assets/<id>-setup.png|600|theme-light]]` /
   `![[assets/<id>-setup-dark.png|600|theme-dark]]`, inside the screenshot
   accordion.

## Accordion tail (canonical — closed by default)

Goldmark (Leyline's renderer, `html.WithUnsafe()` on) treats `<details>` as a
raw-HTML block, so inner fenced code and `![[embed]]` render **only** with
blank lines around the inner markdown. Canonical tail (verified against
goldmark's raw-HTML-block handling — the blank lines are load-bearing, not
cosmetic):

```
## Copy-paste setup

<details><summary>Python setup</summary>

<!-- chunk:py:<id> -->
```python
```
<!-- /chunk:py:<id> -->

</details>

<details><summary>R setup</summary>

<!-- chunk:r:<id> -->
```r
```
<!-- /chunk:r:<id> -->

</details>

<details><summary>App setup screenshot</summary>

![[assets/<id>-setup.png|600|theme-light]]
![[assets/<id>-setup-dark.png|600|theme-dark]]

</details>
```

`inject_chunks.py` still targets the `<!-- chunk:… -->` fences by regex; the
surrounding `<details>` block is inert to it — injection fills the fence
exactly the same whether or not it's wrapped.

## Realistic names + domain rotation

Variable names are realistic and domain-specific — never `x1` / `y` / `group`.
The book rotates three domains: **clinical/health, ecology/biology, social
science** (no education/psych framing like `test_score` / `study_hours`;
educational *attainment* — `graduate`, `years_education` — belongs under social
science). Vary names within a domain: not every continuous predictor is `dose`.
Because variable names are opaque to the engine's formula parser, renaming never
changes the numbers — types, effect sizes, N, seed, and factor level-counts stay
fixed; only names and prose move. Factor dummies stay integer-indexed
(`treatment[2]`, `habitat[3]`) because every chunk uses simulated data, not
uploads.

## Cross-domain twins (in `## Variations`)

Each page carries a **Same design, other fields** sub-list giving the identical
model structure in the **other two** domains as prose-only formulas, e.g. for a
clustered binary trial whose primary domain is clinical:

- Social science: `graduate ~ treatment + (1|school)` — pupils nested in schools.
- Ecology: `survived ~ treatment + (1|tank)` — fish nested in rearing tanks.

This sits alongside the effect-dial bullets and the `find_sample_size` swap
bullet in `## Variations`.

## Chunk markers (single-sourcing — never write the code twice)

Leave the fence EMPTY. `inject_chunks.py` fills it from `chunks/<id>.py` /
`chunks/<id>.R` after validation, so the shown code is exactly what was run.

    <!-- chunk:py:<id> -->
    ```python
    ```
    <!-- /chunk:py:<id> -->

    <!-- chunk:r:<id> -->
    ```r
    ```
    <!-- /chunk:r:<id> -->

The chunk file (`chunks/<id>.py`) is a self-contained runnable script: imports,
build chain, and exactly one `find_power(...)` or `find_sample_size(...)` call.
No `print`, no plotting — execution is validation-only.
