# Documentation example scripts

In-repo tooling for the documentation set. **Not published** — leyline must
exclude this `scripts/` directory from the rendered vault.

## What's here

| File | Purpose |
|------|---------|
| `examples-python.json` | Authored tutorial chunks for the Python pages + their cached real output. |
| `examples-r.json` | Same, for the R pages. |
| `run_examples.py` | Runs each Python chunk, captures stdout into `output`, renders plots, and converts any R plot specs to PNG. |
| `run_examples.R` | Runs each R chunk, captures output, and emits R plot specs (`.vl.json`) for `run_examples.py` to render. |
| `inject_examples.py` | Pastes each cached chunk (code + output + plot) into the tutorial pages at their markers. |

These chunks are the tutorial's **own** example set — separate from the
`ports/{py,r}/examples/` L4 scripts, so tutorial snippets can stay small and
pedagogically shaped. The chunk schema is `{id, label, code, output, plot,
captured_at}`; you write `id`/`label`/`code` (and `plot` for a chart), the runner
fills the rest.

## Workflow

1. **Author** a chunk in the relevant `examples-*.json` (`id`, `label`, `code`;
   leave `output` and `captured_at` empty). Each chunk runs in a fresh namespace,
   so it must be self-contained. To attach a chart, set `"plot"` to a filename and
   leave the result object to be plotted in a variable named `result`.
2. **Run** the matching runner to capture **real** output (and emit/render plots):
   - R first: `Rscript run_examples.R [id ...]` — captures output and writes any
     plot spec to `../assets/examples/<plot>.vl.json`.
   - Python second: activate the workspace `.venv` (so the editable `mcpower` build
     is used, not a stale install), then `python run_examples.py [id ...]` — captures
     output, renders the Python plots, and converts the R `.vl.json` specs to PNG
     (one renderer for both ports). Run Python after R so the R charts materialise.
3. **Inject** the cache into the pages — the markers, not hand-copying:
   - `python inject_examples.py examples-python.json ../tutorial-python`
   - `python inject_examples.py examples-r.json   ../tutorial-r`

A tutorial page marks each example region with a marker pair:

```
<!-- example:01-power -->
<!-- /example -->
```

`inject_examples.py` rewrites everything between the markers with the chunk's
fenced code, captured output, and plot embed. Re-running it on an up-to-date page
is a no-op, so pages never drift from the cache. Nothing here runs at leyline
build time — the cache is refreshed manually and published pages stay static.

## Plots

Set a chunk's `"plot"` to the output filename (e.g. `01-curve-py.png`) and leave
the plottable result in `result`. The Python runner renders it via the result's
`save_plot` (print theme — the default); the R runner instead applies the print
theme and writes the Vega-Lite spec (the R port ships no built-in PNG renderer),
which `run_examples.py` then converts to PNG with `vl_convert` — so both ports'
charts come out of the same renderer in matching print style.
`inject_examples.py` embeds the image with `![[examples/<plot>|600]]`.

> [!note] R object system is provisional
> The R chains use `MCPower$new(...)` (R6-style). The R port's object system
> (R6 / S4 / S7) is not finalised; if it changes, update these chains to match.
