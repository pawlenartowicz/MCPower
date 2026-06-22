#!/usr/bin/env bash
# Run the full benchmark pipeline: py + R harnesses (multi-core, plus a
# single-core mcpower pass each for the mcpower-1t series), then combine, then
# the power-agreement cross-check.
#
# Optional:  --scale S   scales every tier's sim count (S=0.1 -> a ~10% quick
#            preview; S=1.0 default = the real run). The result cache below is
#            scale-aware: a stage re-runs when its results file was built at a
#            different scale, so a 10% run never silently reuses full-scale
#            results. Delete results/*.json to force a re-run at the same scale.
set -euo pipefail
cd "$(dirname "$0")"

PY="$(realpath -s ../../.venv/bin/python)"   # absolute (avoids the relative ../.. py3.14 sys.prefix warning) but -s keeps the venv symlink UNRESOLVED: resolving it lands on the bare pyenv interpreter, which misses pyvenv.cfg and imports a stale mcpower from pyenv's base site-packages instead of the venv

SCALE=1.0
while [ $# -gt 0 ]; do
  case "$1" in
    --scale)   SCALE="$2"; shift 2 ;;
    --scale=*) SCALE="${1#*=}"; shift ;;
    -h|--help) echo "usage: $0 [--scale S]   (S=0.1 -> ~10% sims quick preview)"; exit 0 ;;
    *) echo "unknown arg: $1" >&2; exit 1 ;;
  esac
done
# Normalize + validate (so "1" and "1.0" compare equal to the recorded scale).
SCALE="$("$PY" -c "import sys; print(float(sys.argv[1]))" "$SCALE" 2>/dev/null)" \
  || { echo "invalid --scale value" >&2; exit 1; }

scale_matches() {  # success when results file $1 records the current SCALE
  "$PY" - "$1" "$SCALE" <<'EOF'
import json, sys
try:
    s = json.load(open(sys.argv[1]))["meta"]["n_sims_scale"]
except Exception:
    sys.exit(1)
sys.exit(0 if abs(float(s) - float(sys.argv[2])) < 1e-9 else 1)
EOF
}

run() {  # run <output> <cmd...>: skip only when <output> exists AT THE CURRENT SCALE
  local out=$1; shift
  if [ -f "$out" ] && scale_matches "$out"; then
    echo "== skip: $out exists (scale $SCALE)"
  else
    echo "== run: $* (-> $out)"
    "$@"
  fi
}

run results/py.json    "$PY" harness.py --case all --scale "$SCALE" --out results/py.json
run results/py_1t.json "$PY" harness.py --case all --methods mcpower_find_power --threads 1 --scale "$SCALE" --out results/py_1t.json
run results/r.json     Rscript harness.R --case all --scale "$SCALE" --out results/r.json
run results/r_1t.json  Rscript harness.R --case all --methods mcpower_find_power --threads 1 --scale "$SCALE" --out results/r_1t.json

"$PY" combine.py results/py.json results/r.json \
  --py-1t results/py_1t.json --r-1t results/r_1t.json \
  --plot results/summary_fp.png --plot-fss results/summary_fss.png

# Power-agreement cross-check (MCPower vs dedicated tools vs DIY loops) — printed
# right after the timing results so a benchmark run reports both speed and the
# statistical sanity check. Reads results/r.json (the only run carrying tools).
"$PY" power_agreement.py
