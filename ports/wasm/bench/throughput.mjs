#!/usr/bin/env node
// WASM browser-throughput bench — Node-on-V8 runner over the native grid's
// dumped cases. Node's V8 is the normative wasm engine for per-case rows
// (same Liftoff→TurboFan tiering as Chrome, pinnable, perf-profilable);
// the worker-pool scaling case lives in bench/scaling/ (real Chrome) instead.
//
// Per row this mirrors the native bin's protocol exactly: the same dumped
// pass specs (preset triple for `on` at half n_sims), the same call-level
// seed, single-threaded `run_batch_st` via the dev-only `bench_run_pass`
// entry. Wasm tiers, so the discipline is stricter than native: TWO discarded
// warm-up runs, then the MIN of 5 timed reps.
//
// Usage (from ports/wasm/, machine clock-locked, pinned to one P-core):
//   taskset -c 2 node bench/throughput.mjs                  # default subset, compare
//   taskset -c 2 node bench/throughput.mjs --case glm_wide  # any grid case
//   taskset -c 2 node bench/throughput.mjs --mode off
//   taskset -c 2 node bench/throughput.mjs --save           # write benchmarks/results/wasm.json
//   ... --force            skip the clock-lock/affinity check (numbers = noise)
//   ... --artifact <dir>   prebuilt wasm-pack output dir (skips the standard
//                          rebuild; used by scripts/profile-wasm.sh)
//
// Columns: the native table shape + `wasm_tax` (native fits/sec from
// benchmarks/results/engine.json ÷ wasm fits/sec, same machine) and a sanity
// gate: `hash=` (byte-identical to native, build proven), `Δk=a/b ok`
// (Rust-libm vs glibc last-bit drift flipping a few borderline sims), or
// `Δk=a/b BUILD SUSPECT` (wrong feature flags / seed path). vs_baseline diffs
// against the saved wasm.json, so wasm before/afters read directly per group.

import { execFileSync } from 'node:child_process';
import { readFileSync, writeFileSync } from 'node:fs';
import { readFile } from 'node:fs/promises';
import os from 'node:os';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const PORT_ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const REPO_ROOT = resolve(PORT_ROOT, '../..');
const RESULTS_DIR = join(REPO_ROOT, 'benchmarks/results');
const FNV_OFFSET_HEX = 'cbf29ce484222325';
const WARMUPS = 2;
const REPS = 5;

// Default subset: covers the four bench hypotheses (OLS overhead, GLM tax,
// LME/GLMM scaling, wasm_tax vs native) while keeping the slow dense-GLMM
// rows down to one — glmm_intercept is already the slowest per-fit case and
// adding more GLMM rows adds noise without new information. Full grid via --case.
const DEFAULT_SUBSET = [
  ['ols_large_n', ['off', 'on']],
  ['ols_multi', ['off', 'on']],
  ['glm_multi', ['off', 'on']],
  ['glm_wide', ['off']],
  ['lme_multi', ['off']],
  ['lmm_slope', ['off']],
  ['glmm_intercept', ['off']],
  ['glmm_nested', ['off']],
];

function die(msg) {
  console.error(`throughput.mjs: ${msg}`);
  process.exit(1);
}

function parseArgs(argv) {
  const args = { save: false, force: false, case: null, mode: null, artifact: null };
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    if (a === '--save') args.save = true;
    else if (a === '--force') args.force = true;
    else if (a === '--case') args.case = argv[++i] ?? die('--case requires a case id');
    else if (a === '--mode') {
      args.mode = argv[++i];
      if (args.mode !== 'off' && args.mode !== 'on') die('--mode requires off or on');
    } else if (a === '--artifact') args.artifact = argv[++i] ?? die('--artifact requires a dir');
    else die(`unknown flag ${a}; usage: throughput.mjs [--save] [--case <id>] [--mode off|on] [--force] [--artifact <dir>]`);
  }
  if (args.save && (args.case || args.mode)) {
    die('--save with --case/--mode would overwrite the baseline with a partial run; drop one');
  }
  return args;
}

// Same stance as scripts/profile.sh: refuse to time on an unstabilized or
// unpinned process — wasm numbers are noise otherwise.
function checkLock(force) {
  const read = (p) => {
    try { return readFileSync(p, 'utf8').trim(); } catch { return '?'; }
  };
  const noTurbo = read('/sys/devices/system/cpu/intel_pstate/no_turbo');
  const allowed = (read('/proc/self/status').match(/^Cpus_allowed_list:\s*(\S+)$/m) ?? [])[1] ?? '?';
  const pinned = /^\d+$/.test(allowed);
  const governor = pinned
    ? read(`/sys/devices/system/cpu/cpu${allowed}/cpufreq/scaling_governor`)
    : '?';
  if (noTurbo !== '1' || !pinned || governor !== 'performance') {
    console.error(`throughput.mjs: machine not stabilized/pinned (no_turbo=${noTurbo}, cpus_allowed=${allowed}, governor=${governor})`);
    console.error('  lock the clock and pin: taskset -c 2 node bench/throughput.mjs …  (or --force; numbers will be noise)');
    if (!force) process.exit(1);
  }
  return { noTurbo, allowed, governor };
}

function loadJson(path) {
  try { return JSON.parse(readFileSync(path, 'utf8')); } catch { return null; }
}

async function loadWasm(artifactDir) {
  const mod = await import(pathToFileURL(join(artifactDir, 'engine_wasm.js')).href);
  const bytes = await readFile(join(artifactDir, 'engine_wasm_bg.wasm'));
  await mod.default({ module_or_path: bytes });
  return mod;
}

// One full row run: every pass back-to-back (the native run_row_once shape).
// Per-pass spec JSON.parse happens inside the wasm call — counted in the
// timing like the native bin counts its per-pass spec clone; once per pass,
// amortized over thousands of fits.
function runRowOnce(mod, dump, nSims, passJsons) {
  let hash = FNV_OFFSET_HEX;
  let kUnc = 0;
  let kConv = 0;
  let kUncFirst = 0;
  let convMin = 1;
  passJsons.forEach((pj, i) => {
    const out = JSON.parse(mod.bench_run_pass(pj, dump.n, nSims, BigInt(dump.seed), hash));
    hash = out.hash_state;
    kUnc += out.k_unc;
    kConv += out.k_conv;
    if (i === 0) kUncFirst = out.k_unc;
    convMin = Math.min(convMin, out.k_conv / nSims);
  });
  return { hash, kUnc, kConv, kUncFirst, convMin };
}

// Tier 1: hash equality → byte-identical build. Tier 2: |Δk| within a few
// borderline-sim flips (at least 3, scaled by 0.2% of total fits for the row).
function gateVerdict(native, row) {
  if (!native || !native.sig_hash) return 'no-native-ctrl';
  if (native.sig_hash === row.sig_hash) return 'hash=';
  const tol = Math.max(3, Math.ceil(0.002 * row.passes * row.n_sims));
  const dUnc = Math.abs(Number(native.k_unc) - row.k_unc);
  const dConv = Math.abs(Number(native.k_conv) - row.k_conv);
  const ok = dUnc <= tol && dConv <= tol;
  return `Δk=${dUnc}/${dConv}${ok ? ' ok' : ' BUILD SUSPECT'}`;
}

const args = parseArgs(process.argv.slice(2));
const lock = checkLock(args.force);

// Standard artifact rebuild keeps the timing honest against the working tree;
// --artifact (profile-wasm.sh's names-kept build) skips it. All informational
// output goes to stderr — stdout is the table (profile-wasm.sh captures it).
if (!args.artifact) {
  console.error('building standard wasm artifact (pnpm build:wasm)…');
  execFileSync('pnpm', ['build:wasm'], { cwd: PORT_ROOT, stdio: ['ignore', 2, 2] });
}
console.error('dumping native case specs (--dump-cases)…');
const dumps = JSON.parse(
  execFileSync(
    'cargo',
    ['run', '--release', '--bin', 'throughput', '-p', 'engine-core', '--', '--dump-cases'],
    { cwd: REPO_ROOT, encoding: 'utf8', maxBuffer: 64 * 1024 * 1024, stdio: ['ignore', 'pipe', 'inherit'] },
  ),
);
const byId = new Map(dumps.map((d) => [d.id, d]));

let rows; // [dump, mode] pairs
if (args.case) {
  const dump = byId.get(args.case);
  if (!dump) die(`unknown case ${args.case}; valid: ${dumps.map((d) => d.id).join(', ')}`);
  const natural = dump.has_on_mode ? ['off', 'on'] : ['off'];
  rows = natural.filter((m) => !args.mode || m === args.mode).map((m) => [dump, m]);
} else {
  rows = DEFAULT_SUBSET.flatMap(([id, modes]) => {
    const dump = byId.get(id) ?? die(`default-subset case ${id} missing from dump`);
    return modes.filter((m) => !args.mode || m === args.mode).map((m) => [dump, m]);
  });
}

const native = loadJson(join(RESULTS_DIR, 'engine.json'));
const wasmBaseline = loadJson(join(RESULTS_DIR, 'wasm.json'));
const findRec = (b, id, mode) => b?.records?.find((r) => r.case_id === id && r.mode === mode);

const mod = await loadWasm(args.artifact ?? join(PORT_ROOT, 'vendor/engine-wasm'));

console.log(
  `${'case'.padEnd(18)} ${'mode'.padEnd(4)} ${'n'.padStart(5)} ${'n_sims'.padStart(7)} ${'elapsed_s'.padStart(10)} ${'fits_per_sec'.padStart(12)} ${'conv'.padStart(6)} ${'power'.padStart(7)} ${'wasm_tax'.padStart(8)}  vs_baseline  gate`,
);

const records = [];
for (const [dump, mode] of rows) {
  const nSims = mode === 'on' ? Math.floor(dump.n_sims / 2) : dump.n_sims;
  const passJsons = dump.modes[mode].map((s) => JSON.stringify(s));

  let ctrl;
  for (let i = 0; i < WARMUPS; i++) ctrl = runRowOnce(mod, dump, nSims, passJsons);
  let best = Infinity;
  for (let i = 0; i < REPS; i++) {
    const t0 = performance.now();
    runRowOnce(mod, dump, nSims, passJsons);
    best = Math.min(best, (performance.now() - t0) / 1000);
  }

  const fits = passJsons.length * nSims;
  const fps = fits / best;
  const power = ctrl.kUncFirst / nSims;
  const row = {
    case_id: dump.id,
    mode,
    n: dump.n,
    n_sims: nSims,
    elapsed_s: best,
    fits_per_sec: fps,
    convergence_rate: ctrl.convMin,
    power_first_target: power,
    k_unc: ctrl.kUnc,
    k_conv: ctrl.kConv,
    sig_hash: ctrl.hash,
    passes: passJsons.length,
  };

  const nat = findRec(native, dump.id, mode);
  const tax = nat && nat.fits_per_sec > 0 ? nat.fits_per_sec / fps : null;
  const base = findRec(wasmBaseline, dump.id, mode);
  const vs = base && base.fits_per_sec > 0
    ? `${(fps / base.fits_per_sec).toFixed(2)}x${fps / base.fits_per_sec < 0.9 ? '  <<< REGRESSION' : ''}`
    : '-';

  console.log(
    `${dump.id.padEnd(18)} ${mode.padEnd(4)} ${String(dump.n).padStart(5)} ${String(nSims).padStart(7)} ${best.toFixed(3).padStart(10)} ${fps.toFixed(0).padStart(12)} ${ctrl.convMin.toFixed(3).padStart(6)} ${power.toFixed(3).padStart(7)} ${(tax ? `${tax.toFixed(2)}x` : '-').padStart(8)}  ${vs.padEnd(11)}  ${gateVerdict(nat, row)}`,
  );

  const { passes, ...rec } = row;
  records.push({ ...rec, wasm_tax: tax });
}

if (rows.length === 0) console.error('(no rows — the selected case has no such mode)');

if (args.save) {
  const out = {
    meta: {
      timestamp_utc: new Date().toISOString().replace(/\.\d{3}Z$/, '+00:00'),
      os: `${process.platform}-${process.arch}`,
      cpu_model: os.cpus()[0]?.model ?? 'unknown',
      cores_logical: os.cpus().length,
      threads_mode: '1',
      node_version: process.version,
      pinned_cpu: lock.allowed,
    },
    records,
  };
  const path = join(RESULTS_DIR, 'wasm.json');
  writeFileSync(path, JSON.stringify(out, null, 2));
  console.error(`\nbaseline saved to ${path}`);
}
