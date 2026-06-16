#!/usr/bin/env bash
# bench.sh — native + wasm throughput in one go, same flags for both.
#
# Runs the native bin (cargo release build) pinned to PIN_CORE, then the wasm
# Node runner (ports/wasm/bench/throughput.mjs) on the same core. Native runs
# first so a --save run refreshes engine.json before the wasm runner computes
# its wasm_tax column against it; without --save, wasm_tax reads vs the last
# *saved* native baseline, not this run.
#
# Usage (from anywhere, machine clock-locked):
#   scripts/bench.sh                     # both grids (native 21 rows, wasm default subset)
#   scripts/bench.sh --case glm_multi    # one case, both sides
#   scripts/bench.sh --mode off
#   scripts/bench.sh --save              # overwrite engine.json AND wasm.json
#   scripts/bench.sh --force             # skip the clock-lock check (numbers = noise)
#
# Env: PIN_CORE (default 2 — a P-core; never 6-13/14-15 on this grid's dev box).
# Both underlying tools refuse --save with --case/--mode (partial baseline).

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PIN_CORE="${PIN_CORE:-2}"

FORCE=0
ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode)
      MODE="${2:?--mode requires off|on}"
      [[ "$MODE" == "off" || "$MODE" == "on" ]] || { echo "bench.sh: --mode requires off|on" >&2; exit 1; }
      ARGS+=("--mode" "$MODE"); shift 2 ;;
    --case)
      ARGS+=("--case" "${2:?--case requires a case id}"); shift 2 ;;
    --save) ARGS+=("--save"); shift ;;
    --force) FORCE=1; shift ;;
    *) echo "bench.sh: unknown flag $1; usage: bench.sh [--case <id>] [--mode off|on] [--save] [--force]" >&2; exit 1 ;;
  esac
done

# ── clock-lock check (mirrors profile.sh) ────────────────────────────────────
NO_TURBO="$(cat /sys/devices/system/cpu/intel_pstate/no_turbo 2>/dev/null || echo '?')"
GOVERNOR="$(cat "/sys/devices/system/cpu/cpu${PIN_CORE}/cpufreq/scaling_governor" 2>/dev/null || echo '?')"
if [[ "$NO_TURBO" != "1" || "$GOVERNOR" != "performance" ]]; then
  if [[ "$FORCE" == "1" ]]; then
    echo "bench.sh: UNSTABILIZED (no_turbo=$NO_TURBO, governor=$GOVERNOR) — numbers are noise" >&2
  else
    echo "bench.sh: machine not stabilized (no_turbo=$NO_TURBO, cpu${PIN_CORE} governor=$GOVERNOR)" >&2
    echo "  lock the clock first (bench-l), or --force" >&2
    exit 1
  fi
fi

echo "── native (core $PIN_CORE) ──"
(cd "$REPO_ROOT" && cargo build --release --bin throughput -p engine-core)
taskset -c "$PIN_CORE" "$REPO_ROOT/target/release/throughput" "${ARGS[@]+"${ARGS[@]}"}"

echo
echo "── wasm (core $PIN_CORE) ──"
WASM_ARGS=("${ARGS[@]+"${ARGS[@]}"}")
[[ "$FORCE" == "1" ]] && WASM_ARGS+=("--force")
taskset -c "$PIN_CORE" node "$REPO_ROOT/ports/wasm/bench/throughput.mjs" "${WASM_ARGS[@]+"${WASM_ARGS[@]}"}"
