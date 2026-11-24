// Standardized β, Cohen's d and model R² of a model's generating effects: the
// live values beside the effect inputs (refreshed while the user edits) and the
// per-run copy the Summary tab shows. The engine computes the moments the
// values depend on (`effect_moments_json`), none of which depend on the effect
// sizes; this module evaluates the values for the current effect sizes, names
// and stores them.
import { type EffectMoments, effectMoments } from '$lib/api/engine';
import type { AppSpec } from '$lib/domain/app-spec';
import { buildVarLevels, renderDescriptor } from '$lib/domain/effect-names';

export interface StandardizedRow {
  /** Effect name as the cards and result tables spell it (`x1`, `group[B]`, `x1:x2`). */
  name: string;
  /** The effect input β (residual-SD scale). */
  input: number;
  /** null where undefined (count outcomes, random slopes). */
  std_beta: number | null;
  /** Binary predictors and factor levels only; null elsewhere or where undefined. */
  cohen_d: number | null;
}

export interface StandardizedValues {
  rows: StandardizedRow[];
  r2: number | null;
  /** A clustered (mixed) model, whose R² is the fixed-effects share. */
  clustered: boolean;
}

/** β′·cov·β for a `k × k` row-major `cov`. */
function quad(cov: number[], beta: number[]): number {
  const k = beta.length;
  let total = 0;
  for (let i = 0; i < k; i++) {
    let row = 0;
    for (let j = 0; j < k; j++) row += cov[i * k + j]! * beta[j]!;
    total += beta[i]! * row;
  }
  return total;
}

/** The values of `spec`'s current effect sizes from its moments `out`:
 *  `V_η = β′Σβ`, std β_j = β_j·sd_j / √(V_η + noise), R² = V_η / (V_η + noise),
 *  d = β′(μ₁ − μ₀) / √(w·β′Σ₁β + (1 − w)·β′Σ₀β + noise). Mirrors `evaluate` in
 *  engine-core `standardized.rs`; change together. */
export function evaluateStandardized(out: EffectMoments, spec: AppSpec): StandardizedValues {
  const varLevels = buildVarLevels();
  const inputs = new Map(spec.effects.map((e) => [e.name, e.value]));
  const names = out.descriptors.map((d) => renderDescriptor(d, varLevels));
  const beta = names.map((name) => inputs.get(name) ?? 0);
  const clustered = spec.family === 'mixed';
  const m = out.moments;
  if (!m) {
    return {
      rows: names.map((name, i) => ({ name, input: beta[i]!, std_beta: null, cohen_d: null })),
      r2: null,
      clustered,
    };
  }
  const vEta = quad(m.cov, beta);
  const sdY = Math.sqrt(vEta + m.noise);
  return {
    rows: names.map((name, i) => {
      const g = m.groups[i];
      const cohen_d = g
        ? beta.reduce((s, b, j) => s + b * g.mean_diff[j]!, 0) /
          Math.sqrt(g.w * quad(g.cov_1, beta) + (1 - g.w) * quad(g.cov_0, beta) + m.noise)
        : null;
      return { name, input: beta[i]!, std_beta: (beta[i]! * m.sd[i]!) / sdY, cohen_d };
    }),
    r2: vEta / (vEta + m.noise),
    clustered,
  };
}

/** Fetch `spec`'s moments and evaluate its values. */
export async function resolveStandardized(spec: AppSpec): Promise<StandardizedValues> {
  return evaluateStandardized(await effectMoments(spec), spec);
}

// The engine call takes about 1 ms without an upload; the wait only coalesces a
// burst of edits (a held stepper) into one call.
const DEBOUNCE_MS = 50;

function createStandardizedStore() {
  let current = $state<StandardizedValues | null>(null);
  let timer: ReturnType<typeof setTimeout> | null = null;
  let seq = 0;
  // The last fetched moments and the model they describe: the spec without its
  // effect sizes, plus the upload by identity (the upload store replaces the
  // object whenever the data or its mode changes, and serializing it costs
  // more than the engine call for a large upload).
  let cached: { key: string; csv: unknown; moments: EffectMoments } | null = null;

  /** Recompute for the model being edited. An edit to effect sizes alone is
   *  evaluated at once from the cached moments; any other edit fetches new
   *  moments after the debounce, and the previous values stay shown until they
   *  land, so the readouts do not flicker while typing. A spec that cannot be
   *  built, or that the engine rejects, shows a dash. */
  function refresh(spec: AppSpec | null) {
    if (timer) clearTimeout(timer);
    const id = ++seq;
    if (!spec) {
      current = null;
      return;
    }
    const key = JSON.stringify({ ...spec, effects: undefined, csv: undefined });
    const csv = 'csv' in spec ? spec.csv : null;
    if (cached && cached.key === key && cached.csv === csv) {
      current = evaluateStandardized(cached.moments, spec);
      return;
    }
    timer = setTimeout(() => {
      effectMoments(spec).then(
        (moments) => {
          cached = { key, csv, moments };
          if (id === seq) current = evaluateStandardized(moments, spec);
        },
        // A spec the engine rejects mid-edit shows a dash; the run path
        // reports the same error when the user runs it.
        () => {
          if (id === seq) current = null;
        },
      );
    }, DEBOUNCE_MS);
  }

  return {
    get current() {
      return current;
    },
    refresh,
  };
}

export const standardizedStore = createStandardizedStore();
