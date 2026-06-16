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

/** Case class for the routing heuristic. Only `glmm_*` carry a case multiplier;
 *  `ols`/`glm`/`anova`/`lmm` all map to `simple` and rely on N / n_sims /
 *  scenarios to scale up (which is why their defaults stay at 1 worker). */
export type CaseClass = 'simple' | 'glmm_intercept' | 'glmm_slopes';

/**
 * Classify an AppSpec into a routing case class. Mirror of the `app-spec.ts`
 * `family` union (change together): `linear` ⇒ ols, `logit` ⇒ glm, `anova` ⇒
 * ols-like — all `simple`; `mixed` is `lmm` (simple) unless its outcome is
 * binary, in which case it is a GLMM, split intercept-only (empty/absent
 * `slopes`) vs random-slopes. Cracks the otherwise-opaque AppSpec for the four
 * structural fields `family`, `outcome.kind`, `slopes` only — pinned by the
 * schema-drift test in `tests/seeds.test.ts`.
 */
export function classifyCase(spec: AppSpec): CaseClass {
  const s = spec as Record<string, unknown>;
  if (s['family'] !== 'mixed') return 'simple';
  const outcome = s['outcome'] as { kind?: string } | undefined;
  if (outcome?.kind !== 'binary') return 'simple'; // gaussian / absent ⇒ lmm
  const slopes = s['slopes'] as unknown[] | undefined;
  return slopes && slopes.length > 0 ? 'glmm_slopes' : 'glmm_intercept';
}

/** Inputs to the worker-count routing rule. `n` is the run's sample size
 *  (a sample-size grid passes its largest N). `forced` (the bench scaling
 *  page's override) bypasses routing entirely. */
export interface RouteInputs {
  caseClass: CaseClass;
  hasScenarios: boolean;
  n: number;
  nSims: number;
  hardwareConcurrency: number;
  forced?: number;
}

/**
 * Workers to spawn for a run. `forced` bypasses the heuristic, clamped to
 * [1, nSims] and uncapped (lets the scaling page sweep w12/w16). Otherwise a
 * multiplicative routing rule: start at 1 and multiply by per-factor tiers,
 * then clamp to min(8, hardwareConcurrency). Every factor is ×2/×4 so the
 * pre-clamp count is in {1,2,4,8,16,…} — the measured sweep points; the
 * machine clamp can land between (e.g. 6 on a 6-core box).
 *
 * Factors and the 8-cap come from the 2026-06-13 real-Chrome scaling sweep on
 * this box (6 P-core + 8 E-core + 2 LP-E, no HT): glmm_intercept scaled
 * 1.87×(w2)→3.10×(w4)→4.07×(w8) then flat/negative; ols_multi peaked at w4
 * and fell off by w8. Past 8 the extra workers spill onto E/LP-E cores and
 * oversubscribe, so 8 is the heavy-arm knee and a safe ceiling. This is
 * port-local routing tuning, not a cross-port perf contract — it does not go
 * in `mcpower/configs/`.
 */
export function poolSize(o: RouteInputs): number {
  if (o.forced !== undefined) return Math.max(1, Math.min(o.nSims, Math.floor(o.forced)));
  let workers = 1;
  workers *= o.caseClass === 'glmm_slopes' ? 4 : o.caseClass === 'glmm_intercept' ? 2 : 1;
  workers *= o.hasScenarios ? 2 : 1;
  workers *= o.n > 24000 ? 4 : o.n > 8000 ? 2 : 1;
  workers *= o.nSims > 8000 ? 4 : o.nSims > 2000 ? 2 : 1;
  const cores = Math.max(1, o.hardwareConcurrency || 1);
  // `nSims` clamp preserves splitSims' no-zero-share-worker invariant; it never
  // binds for realistic runs (anchors have nSims ≥ 800, workers ≤ 8).
  return Math.max(1, Math.min(workers, 8, cores, o.nSims));
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
