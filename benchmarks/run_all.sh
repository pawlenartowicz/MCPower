#!/usr/bin/env bash
# Run the full benchmark pipeline: py + R harnesses (multi-core, plus a
# single-core mcpower pass each for the mcpower-1t series), then combine.
# Simple cache: a stage is skipped when its results file already exists —
# delete the file (or results/*.json) to re-run that stage.
set -euo pipefail
cd "$(dirname "$0")"

PY="$(realpath -s ../../.venv/bin/python)"   # absolute (avoids the relative ../.. py3.14 sys.prefix warning) but -s keeps the venv symlink UNRESOLVED: resolving it lands on the bare pyenv interpreter, which misses pyvenv.cfg and imports a stale mcpower from pyenv's base site-packages instead of the venv

run() {  # run <output> <cmd...>: skip when <output> already exists
  local out=$1; shift
  if [ -f "$out" ]; then
    echo "== skip: $out exists"
  else
    echo "== run: $* (-> $out)"
    "$@"
  fi
}

run results/py.json    "$PY" harness.py --case all --out results/py.json
run results/py_1t.json "$PY" harness.py --case all --methods mcpower_find_power --threads 1 --out results/py_1t.json
run results/r.json     Rscript harness.R --case all --out results/r.json
run results/r_1t.json  Rscript harness.R --case all --methods mcpower_find_power --threads 1 --out results/r_1t.json

"$PY" combine.py results/py.json results/r.json \
  --py-1t results/py_1t.json --r-1t results/r_1t.json \
  --plot results/summary_fp.png --plot-fss results/summary_fss.png
