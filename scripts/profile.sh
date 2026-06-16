#!/usr/bin/env bash
# profile.sh — perf-profile engine throughput cases and keep the reports in-repo.
#
# The committed record of "where does this case spend its time": runs the same
# case ids (and --mode filter) as the throughput bench, applies the two-build
# protocol the 2026-06 profile docs froze (function shares from the real
# debug=false binary; source-line attribution from a line-tables-only build,
# built in its own target dir so it never invalidates the main release cache),
# and writes the timing row + both perf reports under benchmarks/profiles/.
# Raw perf.data + the exact binaries stay in /tmp (rebuild-dependent, big);
# the text reports are the durable record.
#
# Usage (from anywhere):
#   scripts/profile.sh glm_rare                      # off-mode profile
#   scripts/profile.sh glm_rare ols_large_n --mode on
#   scripts/profile.sh glmm_crossed --force          # skip the clock-lock check
#   scripts/profile.sh                               # no cases = ALL grid cases
#
# Env: PIN_CORE (default 2 — a P-core; never 6-13/14-15 on this grid's dev box),
#      PERF_FREQ (default 4000).
#
# The clock-lock check refuses to record on an unstabilized machine — wall-clock
# and shares are noise otherwise. Stabilize first:
#   echo 1 | sudo tee /sys/devices/system/cpu/intel_pstate/no_turbo
#   echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PIN_CORE="${PIN_CORE:-2}"
PERF_FREQ="${PERF_FREQ:-4000}"
SCRATCH="/tmp/mcpower-profile"

MODE="off"
FORCE=0
CASES=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode)
      MODE="${2:?--mode requires off|on}"
      [[ "$MODE" == "off" || "$MODE" == "on" ]] || { echo "profile.sh: --mode requires off|on" >&2; exit 1; }
      shift 2 ;;
    --force) FORCE=1; shift ;;
    -*) echo "profile.sh: unknown flag $1" >&2; exit 1 ;;
    *) CASES+=("$1"); shift ;;
  esac
done
# No cases given = the whole grid (ids queried from the binary after the build).

# ── clock-lock check ─────────────────────────────────────────────────────────
NO_TURBO="$(cat /sys/devices/system/cpu/intel_pstate/no_turbo 2>/dev/null || echo '?')"
GOVERNOR="$(cat "/sys/devices/system/cpu/cpu${PIN_CORE}/cpufreq/scaling_governor" 2>/dev/null || echo '?')"
if [[ "$NO_TURBO" != "1" || "$GOVERNOR" != "performance" ]]; then
  echo "profile.sh: machine not stabilized (no_turbo=$NO_TURBO, cpu${PIN_CORE} governor=$GOVERNOR)" >&2
  echo "  stabilize (see header) or pass --force to record anyway (numbers will be noise)" >&2
  [[ "$FORCE" == "1" ]] || exit 1
fi

# ── two builds ───────────────────────────────────────────────────────────────
echo "building nodebug + line-tables binaries..."
(cd "$REPO_ROOT" && cargo build --release --bin throughput -p engine-core)
(cd "$REPO_ROOT" && CARGO_PROFILE_RELEASE_DEBUG=line-tables-only \
  CARGO_TARGET_DIR="$REPO_ROOT/target/lines" \
  cargo build --release --bin throughput -p engine-core)

# Freeze the exact binaries next to the perf.data so reports stay regenerable
# after the next cargo build.
STAMP="$(date +%F_%H%M)"
RUN_SCRATCH="$SCRATCH/$STAMP"
mkdir -p "$RUN_SCRATCH"
cp "$REPO_ROOT/target/release/throughput" "$RUN_SCRATCH/throughput-nodebug"
cp "$REPO_ROOT/target/lines/release/throughput" "$RUN_SCRATCH/throughput-lines"

if [[ ${#CASES[@]} -eq 0 ]]; then
  mapfile -t CASES < <("$RUN_SCRATCH/throughput-nodebug" --list)
  echo "profiling all ${#CASES[@]} grid cases (mode: $MODE)"
fi

GIT_REV="$(git -C "$REPO_ROOT" rev-parse --short HEAD 2>/dev/null || echo 'no-git')"
GIT_DIRTY=""
[[ -n "$(git -C "$REPO_ROOT" status --porcelain 2>/dev/null)" ]] && GIT_DIRTY="+dirty"

for CASE in "${CASES[@]}"; do
  OUT="$REPO_ROOT/benchmarks/profiles/$STAMP/$CASE-$MODE"
  echo "── $CASE ($MODE) → ${OUT#"$REPO_ROOT"/}"

  # Clean timed row (no profiler attached) — the fits/s of record. A header-only
  # table means the case has no such mode (mixed rows are off-only): skip it.
  TIMING="$(taskset -c "$PIN_CORE" "$RUN_SCRATCH/throughput-nodebug" --case "$CASE" --mode "$MODE")"
  echo "$TIMING"
  if [[ "$(wc -l <<<"$TIMING")" -lt 2 ]]; then
    echo "  (no $MODE mode for $CASE — skipped)"
    continue
  fi
  mkdir -p "$OUT"
  printf '%s\n' "$TIMING" > "$OUT/timing.txt"

  # Function-level shares from the real binary; line attribution from the
  # line-tables build. Both runs re-execute the full row (warm-up + 3 reps).
  for BUILD in nodebug lines; do
    perf record -F "$PERF_FREQ" -o "$RUN_SCRATCH/$CASE-$MODE-$BUILD.data" -- \
      taskset -c "$PIN_CORE" "$RUN_SCRATCH/throughput-$BUILD" --case "$CASE" --mode "$MODE" \
      > /dev/null
  done
  perf report --stdio --no-children -s sym --percent-limit 0.3 \
    -i "$RUN_SCRATCH/$CASE-$MODE-nodebug.data" > "$OUT/symbols.txt" 2>/dev/null
  perf report --stdio --no-children -s sym,srcline --percent-limit 0.3 \
    -i "$RUN_SCRATCH/$CASE-$MODE-lines.data" > "$OUT/srclines.txt" 2>/dev/null

  {
    echo "date: $STAMP"
    echo "git: $GIT_REV$GIT_DIRTY"
    echo "case: $CASE  mode: $MODE"
    echo "pin_core: $PIN_CORE  perf_freq: $PERF_FREQ"
    echo "no_turbo: $NO_TURBO  governor: $GOVERNOR$( [[ "$NO_TURBO" != "1" || "$GOVERNOR" != "performance" ]] && echo '  (UNSTABILIZED — numbers are noise)')"
    echo "cpu: $(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2- | sed 's/^ //')"
    echo "perf.data + binaries: $RUN_SCRATCH (tmpfs — gone on reboot)"
  } > "$OUT/meta.txt"
done

echo "done — reports under benchmarks/profiles/$STAMP/"
