# Formula Fixture Corpus

This corpus exercises `_engine.parse_formula` and `_engine.parse_assignments` — the Rust parser
entry points exposed by `engine-spec-builder` via the `engine-py` PyO3 adapter.  It is the
single source of truth for parser behaviour across all MCPower ports; every port that wraps
`engine-spec-builder` should drive its parser tests from these same files.

## Layout

```
formula-fixtures/
├── cases.json            # parse_formula cases — one JSON array (positive + negative)
├── canonical-suite.json  # cross-port semantic-projection suite (separate harness)
└── assignments/          # parse_assignments cases (input + kind + known + expected in one file)
    └── <stem>.json
```

### Formula cases (`cases.json`)

`cases.json` is `{"version": 1, "cases": [...]}`.  Each case is an object with an `id` and a
`formula`, plus exactly one of:

- `expected` — a JSON object that `parse_formula` must return exactly (positive case), or
- `error` — a regex substring that the raised `ValueError` message must match (negative case).

A case is negative **iff** it carries an `error` key; the `id` is only a label.  Most negatives
are named `err_*`, but a positive-looking id such as `016_intercept_and_slope_same_group` can also
be a negative case.

### Assignment cases (`assignments/`)

Each `.json` file is a self-contained fixture:

```json
{
  "input":    "<assignment string>",
  "kind":     "effect" | "variable_type" | "correlation",
  "known":    {"predictors": [...], "interaction_terms": [...]},
  "expected": <dict from parse_assignments>  |  {"error": "<regex>"}
}
```

A top-level `{"error": "..."}` key in `expected` means the call must raise `ValueError` with a
matching message.  A soft-error (unknown name, etc.) is not a raised exception — it appears in
`expected.errors[]` just as `parse_assignments` returns it.

## Port consumption

**Python:** `mcpower/ports/py/tests/test_formula_fixtures.py` — parametrised pytest, run as part
of the standard `python -m pytest mcpower/ports/py/tests/` invocation.

Other ports (R, WASM, Tauri) should each add a parallel test driver that reads the same
`cases.json` / `assignments/*.json` files and calls their local binding of `parse_formula` /
`parse_assignments`.

## Regenerating goldens

Run the generator from the workspace root after activating the workspace venv:

```bash
source .venv/bin/activate
python mcpower/scripts/gen_formula_goldens.py
```

The generator rewrites the `expected` block of every positive formula case in `cases.json` (those
without an `error` key) and every non-error assignment case by calling the live engine.  Re-run
whenever the parser output shape changes intentionally, then commit the updated goldens together
with the engine change.

## Drift policy

Golden files are committed.  Any unintentional parser change will cause fixture tests to fail in
CI.  Intentional changes require: (1) updating the parser, (2) re-running the generator, (3)
committing both together in the same PR with a clear rationale.  Do not regenerate goldens without
also landing the corresponding engine change.
