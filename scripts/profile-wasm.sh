#!/usr/bin/env bash
# profile-wasm.sh — perf-profile wasm throughput cases via node --perf-basic-prof.
#
# The wasm analog of profile.sh: V8 emits a /tmp/perf-<pid>.map for JIT'd wasm,
# so linux perf attributes samples to wasm FUNCTION NAMES — but only if the
# wasm name section survives. The standard release artifact strips it (wasm-opt
# without -g → useless wasm-function[123] symbols), so this script builds its
# own names-kept artifact (wasm-pack --profiling: optimized + debug names) for
# the perf run, while the clean timing row keeps the standard build. No
# srclines report — there are no line tables through the wasm JIT; symbol
# shares are the deliverable. Raw perf.data + the profiling artifact stay in
# /tmp; the text reports under benchmarks/profiles/ are the durable record.
#
# Usage (from anywhere):
#   scripts/profile-wasm.sh glm_multi                 # off-mode profile
#   scripts/profile-wasm.sh glm_multi ols_multi --mode on
#   scripts/profile-wasm.sh glmm_nested --force       # skip the clock-lock check
#
# Env: PIN_CORE (default 2 — a P-core), PERF_FREQ (default 4000).
#
# Stabilize first (same gate as profile.sh):
#   echo 1 | sudo tee /sys/devices/system/cpu/intel_pstate/no_turbo
#   echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PIN_CORE="${PIN_CORE:-2}"
PERF_FREQ="${PERF_FREQ:-4000}"
SCRATCH="/tmp/mcpower-profile-wasm"

MODE="off"
FORCE=0
CASES=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode)
      MODE="${2:?--mode requires off|on}"
      [[ "$MODE" == "off" || "$MODE" == "on" ]] || { echo "profile-wasm.sh: --mode requires off|on" >&2; exit 1; }
      shift 2 ;;
    --force) FORCE=1; shift ;;
    -*) echo "profile-wasm.sh: unknown flag $1" >&2; exit 1 ;;
    *) CASES+=("$1"); shift ;;
  esac
done
# Wasm rows are slow; unlike profile.sh there is no all-grid default.
[[ ${#CASES[@]} -gt 0 ]] || { echo "profile-wasm.sh: at least one case id required" >&2; exit 1; }

# ── clock-lock check (mirrors profile.sh) ────────────────────────────────────
NO_TURBO="$(cat /sys/devices/system/cpu/intel_pstate/no_turbo 2>/dev/null || echo '?')"
GOVERNOR="$(cat "/sys/devices/system/cpu/cpu${PIN_CORE}/cpufreq/scaling_governor" 2>/dev/null || echo '?')"
if [[ "$NO_TURBO" != "1" || "$GOVERNOR" != "performance" ]]; then
  echo "profile-wasm.sh: machine not stabilized (no_turbo=$NO_TURBO, cpu${PIN_CORE} governor=$GOVERNOR)" >&2
  echo "  stabilize (see header) or pass --force to record anyway (numbers will be noise)" >&2
  [[ "$FORCE" == "1" ]] || exit 1
fi

STAMP="$(date +%F_%H%M)"
RUN_SCRATCH="$SCRATCH/$STAMP"
mkdir -p "$RUN_SCRATCH"

# Names-kept artifact for the perf runs. Built from inside the workspace so
# .cargo/config.toml's +simd128 baseline applies, exactly like the deployment build.
# Names survive only with BOTH halves: CARGO_PROFILE_RELEASE_DEBUG=2 makes rustc
# emit the name section despite the workspace's debug = false, and the crate's
# [package.metadata.wasm-pack.profile.profiling] wasm-opt = ['-O', '-g'] stops
# wasm-opt stripping it (wasm-pack #797: --profiling defaults to plain '-O').
# The grep gate below catches either half regressing.
echo "building names-kept wasm artifact (wasm-pack --profiling)..."
(cd "$REPO_ROOT" && CARGO_PROFILE_RELEASE_DEBUG=2 wasm-pack build crates/engine-wasm \
  --profiling --target web --out-dir "$RUN_SCRATCH/pkg")
if ! grep -qm1 deviance "$RUN_SCRATCH/pkg/engine_wasm_bg.wasm"; then
  echo "profile-wasm.sh: name section missing from profiling artifact — symbols would be anonymous" >&2
  exit 1
fi

GIT_REV="$(git -C "$REPO_ROOT" rev-parse --short HEAD 2>/dev/null || echo 'no-git')"
GIT_DIRTY=""
[[ -n "$(git -C "$REPO_ROOT" status --porcelain 2>/dev/null)" ]] && GIT_DIRTY="+dirty"
RUNNER="$REPO_ROOT/ports/wasm/bench/throughput.mjs"

for CASE in "${CASES[@]}"; do
  OUT="$REPO_ROOT/benchmarks/profiles/$STAMP/$CASE-wasm-$MODE"
  echo "── $CASE ($MODE) → ${OUT#"$REPO_ROOT"/}"

  # Clean timed row from the STANDARD artifact (the runner rebuilds it), no
  # profiler attached — the fits/s of record. Header-only output means the
  # case has no such mode (mixed rows are off-only): skip it.
  TIMING="$(cd "$REPO_ROOT/ports/wasm" && taskset -c "$PIN_CORE" node "$RUNNER" --case "$CASE" --mode "$MODE")"
  echo "$TIMING"
  if [[ "$(wc -l <<<"$TIMING")" -lt 2 ]]; then
    echo "  (no $MODE mode for $CASE — skipped)"
    continue
  fi
  mkdir -p "$OUT"
  printf '%s\n' "$TIMING" > "$OUT/timing.txt"

  # perf run on the names-kept artifact. --perf-basic-prof makes node write
  # /tmp/perf-<pid>.map; perf report resolves wasm frames through it.
  perf record -F "$PERF_FREQ" -o "$RUN_SCRATCH/$CASE-$MODE.data" -- \
    taskset -c "$PIN_CORE" node --perf-basic-prof "$RUNNER" \
    --case "$CASE" --mode "$MODE" --artifact "$RUN_SCRATCH/pkg" > /dev/null
  perf report --stdio --no-children -s sym --percent-limit 0.3 \
    -i "$RUN_SCRATCH/$CASE-$MODE.data" > "$OUT/symbols.txt" 2>/dev/null

  {
    echo "date: $STAMP"
    echo "git: $GIT_REV$GIT_DIRTY"
    echo "case: $CASE  mode: $MODE  runtime: node $(node --version) (--perf-basic-prof, wasm-pack --profiling artifact)"
    echo "pin_core: $PIN_CORE  perf_freq: $PERF_FREQ"
    echo "no_turbo: $NO_TURBO  governor: $GOVERNOR$( [[ "$NO_TURBO" != "1" || "$GOVERNOR" != "performance" ]] && echo '  (UNSTABILIZED — numbers are noise)')"
    echo "cpu: $(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2- | sed 's/^ //')"
    echo "perf.data + profiling artifact: $RUN_SCRATCH (tmpfs — gone on reboot)"
    echo "note: timing.txt is from the STANDARD artifact; symbols.txt from the names-kept one"
  } > "$OUT/meta.txt"
done

echo "done — reports under benchmarks/profiles/$STAMP/"
