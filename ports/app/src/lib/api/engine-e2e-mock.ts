// Deterministic in-process mock for the Tauri engine API; used by E2E tests to avoid spawning a real Tauri backend.
// Must export exactly the same surface as engine.ts. This mock must match engine.ts's full export surface.
import type { AppSpec, EffectDescriptor, EffectSize, EffectsFromData } from '$lib/domain/app-spec';

export type { EffectDescriptor, EffectSize, EffectsFromData };
import type {
  PowerResult,
  SampleSizeResult,
  SampleSizeMethod,
  ScenarioResult,
  ProgressEvent,
  PlotSpecs,
  FormulaParse,
} from '$lib/domain/result';
import { stubParseFormula } from '../../tests/parse-formula-stub';

export interface FindPowerResponse {
  run_id: string;
  result: ScenarioResult<PowerResult>;
  plots: PlotSpecs;
}
export interface FindSampleSizeResponse {
  run_id: string;
  result: ScenarioResult<SampleSizeResult>;
  plots: PlotSpecs;
}

/** Deterministic histogram: all sims land in the top bucket (fully-powered mock point). */
function histogramFor(targetCount: number, nSims: number): number[] {
  const buckets = new Array(targetCount + 1).fill(0);
  buckets[targetCount] = nSims;
  return buckets;
}

/** Minimal valid theme-naked Vega-Lite bar spec for E2E (chart mounts). */
function mockBarSpec(k: number): string {
  return JSON.stringify({
    $schema: 'https://vega.github.io/schema/vega-lite/v5.json',
    // target_{idx} tokens use the engine's β̂-column space (intercept at 0,
    // targets 1..k) so buildTargetLabelMap relabels them like a real run.
    data: { values: Array.from({ length: k }, (_, i) => ({ target: `target_${i + 1}`, power: 0.6 + i * 0.05 })) },
    mark: 'bar',
    encoding: {
      x: { field: 'target', type: 'nominal' },
      y: { field: 'power', type: 'quantitative', scale: { domain: [0, 1] } },
    },
  });
}

/** Minimal valid theme-naked Vega-Lite line spec for E2E. */
function mockLineSpec(ns: number[]): string {
  return JSON.stringify({
    $schema: 'https://vega.github.io/schema/vega-lite/v5.json',
    data: { values: ns.map((n) => ({ n, power: 0.7 })) },
    mark: { type: 'line', point: true },
    encoding: {
      x: { field: 'n', type: 'quantitative' },
      y: { field: 'power', type: 'quantitative', scale: { domain: [0, 1] } },
    },
  });
}

/** Number of fake inference targets for the mock to emit. */
function mockTargetCount(spec: AppSpec): number {
  // ANOVA has no parsed_formula; its bars correspond to the effect rows.
  if (spec.family === 'anova') return spec.effects.length;
  return spec.parsed_formula.predictors.length + spec.parsed_formula.interaction_terms.length;
}

function fakePowerResult(spec: AppSpec, n: number): PowerResult {
  const k = mockTargetCount(spec);
  return {
    n,
    n_sims: spec.n_sims,
    // β̂-column indices: intercept at 0, targets 1..k (buildRows maps idx-1).
    target_indices: Array.from({ length: k }, (_, i) => i + 1),
    power_uncorrected: Array.from({ length: k }, (_, i) => 0.6 + i * 0.05),
    power_corrected: Array.from({ length: k }, (_, i) => 0.55 + i * 0.05),
    ci_uncorrected: Array.from({ length: k }, () => ({ lo: 0.5, hi: 0.7 })),
    ci_corrected: Array.from({ length: k }, () => ({ lo: 0.45, hi: 0.65 })),
    convergence_rate: 1.0,
    boundary_hit: [],
    estimator_extras:
      spec.family === 'logit'
        ? { estimator: 'glm', baseline_prob_realized: spec.baseline_probability }
        : { estimator: 'ols' },
    success_count_histogram_uncorrected: histogramFor(k, spec.n_sims),
    success_count_histogram_corrected: histogramFor(k, spec.n_sims),
  };
}

// E2E hook: a run whose outcome is the sentinel "boom" rejects with a representative
// engine error, so the run-failure error card can be exercised end-to-end (regression
// guard against the old "see console" dead-end). Valid formula → run button enables →
// the failure surfaces at run time, not as a field error.
function failIfSentinel(spec: AppSpec): void {
  if (spec.family !== 'anova' && spec.parsed_formula.outcome === 'boom') {
    throw new Error("factor 'group' has only one observed level");
  }
}

export async function findPower(spec: AppSpec, sampleSize: number): Promise<FindPowerResponse> {
  await new Promise((r) => setTimeout(r, 50));
  failIfSentinel(spec);
  const bar = mockBarSpec(mockTargetCount(spec));
  return {
    run_id: `e2e-${Date.now()}`,
    result: { scenarios: [['default', fakePowerResult(spec, sampleSize)]] },
    plots: { blocks: [{ key: 'power', spec: bar }] },
  };
}

export async function findSampleSize(
  spec: AppSpec,
  bounds: [number, number],
  method: SampleSizeMethod,
): Promise<FindSampleSizeResponse> {
  await new Promise((r) => setTimeout(r, 50));
  const step =
    'Grid' in method
      ? ('Fixed' in method.Grid.by ? method.Grid.by.Fixed : method.Grid.by.Auto.count)
      : 0;
  const ns = 'Grid' in method
    ? Array.from(
        { length: Math.floor((bounds[1] - bounds[0]) / step) + 1 },
        (_, i) => bounds[0] + i * step,
      )
    : [bounds[0], bounds[1]];
  const curve = mockLineSpec(ns);
  const k = mockTargetCount(spec);
  const blocks = [{ key: 'curve', spec: curve }];
  if (k >= 2) {
    blocks.push({ key: 'at_least_k', spec: mockLineSpec(ns) });
    blocks.push({ key: 'exactly_k', spec: mockLineSpec(ns) });
  }
  // fitted / fitted_joint: one stub "fitted" entry per target (n_achievable == bounds[0],
  // consistent with first_achieved == bounds[0]). ci_lo/ci_hi null keeps the mock simple.
  const stubbedFit = { status: 'fitted' as const, n_star: bounds[0], n_achievable: bounds[0], ci_lo: null, ci_hi: null };
  return {
    run_id: `e2e-${Date.now()}`,
    result: {
      scenarios: [['default', {
        grid_or_trace: ns.map((n) => fakePowerResult(spec, n)),
        first_achieved: Array.from({ length: k }, () => bounds[0]),
        first_joint_achieved: Array.from({ length: k }, () => bounds[0]),
        target_power: spec.target_power,
        method,
        fitted: Array.from({ length: k }, () => stubbedFit),
        fitted_joint: Array.from({ length: k }, () => stubbedFit),
        cluster_atom: 1,
      }]],
    },
    plots: { blocks },
  };
}

export async function cancelRun(_runId: string): Promise<boolean> { return true; }

// No-op stub, mirroring engine-wasm.ts: keeps this mock's export surface in sync
// with engine.ts (App.svelte imports setNThreads from $lib/api/engine).
export async function setNThreads(_n: number): Promise<void> {}

export async function onProgress(_listener: (e: ProgressEvent) => void): Promise<() => void> {
  return () => {};
}

export async function parseFormula(formula: string): Promise<FormulaParse> {
  const state = stubParseFormula(formula);
  if (state.error !== null) throw new Error(state.error);
  return state.result!;
}

export async function getEffectsFromData(spec: AppSpec): Promise<EffectsFromData> {
  // Deterministic stand-in: echo the spec's effect names with a fixed magnitude;
  // no ICC / baseline preview (the mock fit recovers neither).
  const effects = 'effects' in spec ? spec.effects.map((e) => ({ name: e.name, value: 0.3 })) : [];
  return { effects, cluster_icc: null, baseline_probability: null };
}

export async function effectSkeleton(spec: AppSpec): Promise<EffectDescriptor[]> {
  // Stub: build a continuous descriptor per predictor + one intercept at index 0.
  // This keeps the E2E mock coherent — real rendering uses the Tauri command.
  const predictors = 'parsed_formula' in spec ? spec.parsed_formula.predictors : [];
  return [
    { kind: 'intercept' },
    ...predictors.map((p): EffectDescriptor => ({ kind: 'continuous', predictor: p })),
  ];
}
