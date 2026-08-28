// Pool sizing + per-worker seed/sim assignment for the WASM worker pool.
// The engine does the real RNG decorrelation internally via pcg_mix64(base_seed,
// sim_id) (crates/engine-core/src/rng.rs); JS only offsets the master seed by
// the worker index so each worker starts from a distinct stream.
//
// INVARIANT: results are NOT bit-equal across worker counts. Each worker
// receives a different base_seed → different sims → statistically equivalent
// outputs, but not byte-identical. This is intentional and mirrors the note in
// engine-core/src/single_core.rs: never assert byte-equality across pool sizes.
import type { AppSpec } from './types';

const U64 = 1n << 64n;

/** Case class for the routing heuristic. Each class carries a relative per-fit
 *  cost weight (see `CLASS_WEIGHT`); the classes exist only to separate costs
 *  that differ by more than a factor of two. */
export type CaseClass =
  | 'ols'
  | 'glm'
  | 'lmm'
  | 'glmm_intercept'
  | 'glmm_slopes'
  | 'glmm_crossed_nested';

/**
 * Classify an AppSpec into a routing case class. Mirror of the `app-spec.ts`
 * `family` union (change together): `linear`/`anova` ⇒ `ols`, `logit`/`poisson`
 * ⇒ `glm`; `mixed` is `lmm` unless its outcome is a GLMM one (`binary` —
 * logit/probit — or `poisson`), which then splits three ways by grouping
 * structure. Cracks the otherwise-opaque AppSpec for the structural fields
 * `family`, `outcome.kind`, `slopes`, `extra_groupings` only — pinned by the
 * schema-drift test in `tests/seeds.test.ts`.
 *
 * `extra_groupings` is tested before `slopes`: a crossed/nested GLMM that also
 * carries random slopes was never measured, and of the two candidate weights
 * the crossed one is higher, so taking it routes such a spec to the larger
 * pool. Over-provisioning cost nothing up to the w12 cap in the sweep;
 * under-provisioning cost up to 4.13x.
 */
export function classifyCase(spec: AppSpec): CaseClass {
  const s = spec as Record<string, unknown>;
  if (s['family'] !== 'mixed') {
    return s['family'] === 'logit' || s['family'] === 'poisson' ? 'glm' : 'ols';
  }
  const outcome = s['outcome'] as { kind?: string } | undefined;
  // A clustered binary (logit/probit) OR Poisson outcome is a GLMM; gaussian /
  // absent is a plain LMM.
  if (outcome?.kind !== 'binary' && outcome?.kind !== 'poisson') return 'lmm';
  const extra = s['extra_groupings'] as unknown[] | undefined;
  if (extra && extra.length > 0) return 'glmm_crossed_nested';
  const slopes = s['slopes'] as unknown[] | undefined;
  return slopes && slopes.length > 0 ? 'glmm_slopes' : 'glmm_intercept';
}

/**
 * Per-fit cost of each class relative to OLS, measured as serial (single-worker)
 * wall time per fit per observation and snapped to a power of two. OLS cost is
 * linear in n, so one weight per class holds across the whole n range.
 *
 * Measured 2026-08-27/28 on Chrome-for-testing 152, clock locked, min of 3 reps:
 * ols 1.0, glm 2.3, lmm 1.5, glmm_intercept 30-33, glmm_slopes 123-128,
 * glmm_crossed_nested 211-301 (converging on ~209 as run size grows).
 */
const CLASS_WEIGHT: Record<CaseClass, number> = {
  ols: 1,
  glm: 2,
  lmm: 2,
  glmm_intercept: 32,
  glmm_slopes: 128,
  glmm_crossed_nested: 256,
};

/** Inputs to the worker-count routing rule. `n` is the run's sample size
 *  (a sample-size grid passes its largest N). `scenarioCount` is the number of
 *  scenarios fanned out (0 or 1 both mean one baseline contract). `forced` (the
 *  bench scaling page's override) bypasses routing entirely. */
export interface RouteInputs {
  caseClass: CaseClass;
  scenarioCount: number;
  n: number;
  nSims: number;
  hardwareConcurrency: number;
  forced?: number;
}

/**
 * Workers to spawn for a run. `forced` bypasses the heuristic, clamped to
 * [1, nSims] and uncapped (lets the scaling page sweep w12/w16).
 *
 * Otherwise: estimate the run's total serial work as
 * `n x nSims x scenarios x classWeight`, then map that score onto a four-rung
 * ladder. Scoring the work directly is what the previous rule got wrong — it
 * multiplied an n-tier and an nSims-tier independently, which mis-ranks a run
 * that is large on one axis and small on the other (n=10000 x 1000 sims routed
 * to w2 where w8 was optimal).
 *
 * Ladder and weights come from a 504-run locked-clock sweep over 32 distinct
 * (model, n, nSims) points on Chrome-for-testing 152 (6 P-core + 8 E-core +
 * 2 LP-E, no HT). Against the best worker count measured per point, this rule
 * is 1.03x off optimal on average and 1.34x at worst; the previous rule was
 * 1.96x average and 4.13x worst. The 12-cap is the measured knee: w16 bought
 * 4% over w12 on the one class still scaling there and nothing on the others,
 * which does not pay for oversubscribing a 16-thread box.
 *
 * Known gap: grid points are NOT a routing input, so a sample-size search over
 * a 10-point grid scores the same as a single power run at the same nSims. The
 * heavy classes now clear the top rung on weight alone, which covers the common
 * case, but the term itself is unmeasured.
 *
 * This is port-local routing tuning, not a cross-port perf contract — it does
 * not go in `mcpower/configs/`.
 */
export function poolSize(o: RouteInputs): number {
  if (o.forced !== undefined) return Math.max(1, Math.min(o.nSims, Math.floor(o.forced)));
  const score = o.n * o.nSims * Math.max(1, o.scenarioCount) * CLASS_WEIGHT[o.caseClass];
  const workers = score < 2e5 ? 1 : score < 3e6 ? 4 : score < 1.2e7 ? 8 : 12;
  const cores = Math.max(1, o.hardwareConcurrency || 1);
  // `nSims` clamp preserves splitSims' no-zero-share-worker invariant; it never
  // binds for realistic runs (anchors have nSims >= 800, workers <= 12).
  return Math.max(1, Math.min(workers, 12, cores, o.nSims));
}

/** Split n_sims into per-worker shares; remainder spread across the first workers. */
export function splitSims(nSims: number, nWorkers: number): number[] {
  const base = Math.floor(nSims / nWorkers);
  const rem = nSims % nWorkers;
  return Array.from({ length: nWorkers }, (_, i) => base + (i < rem ? 1 : 0));
}

/**
 * Per-worker base seeds: masterSeed + i, wrapped to u64 (seeds are u64).
 *
 * Worker i gets `(masterSeed + i) % 2^64`. The engine's `pcg_mix64(base_seed,
 * sim_id)` handles the real per-sim decorrelation from there; this offset just
 * ensures each worker drives a distinct stream.
 *
 * Caller's responsibility: `masterSeed` must already be a valid u64 bigint
 * (0 ≤ masterSeed < 2^64). No clamping is done here.
 *
 * NOTE: results are NOT bit-equal across different worker counts — see file
 * header for the invariant.
 */
export function workerSeeds(masterSeed: bigint, nWorkers: number): bigint[] {
  return Array.from({ length: nWorkers }, (_, i) => (masterSeed + BigInt(i)) % U64);
}
