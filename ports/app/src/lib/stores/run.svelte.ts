// Run store: orchestrates engine invocations (find-power / find-sample-size), tracks run state and progress, and manages result tabs.
import type { AppSpec, EffectDescriptor } from '$lib/domain/app-spec';
import type {
  PowerResult,
  SampleSizeResult,
  SampleSizeMethod,
  ProgressEvent,
  PlotSpecs,
} from '$lib/domain/result';
import { findPower, findSampleSize, cancelRun, onProgress, effectSkeleton } from '$lib/api/engine';
import { historyStore } from './history.svelte';
import { familyStore } from './family.svelte';
import { toRunError, type AppError } from '$lib/errors/report';

export type ResultTab = 'summary' | 'joint' | 'script' | 'export';

export type RunState = 'idle' | 'running' | 'done' | 'error' | 'cancelled';

export interface RunTab {
  id: string;
  label: string;
  kind: 'find-power' | 'find-sample-size';
  subView: ResultTab;
  spec: AppSpec;
  sample_size?: number;
  bounds?: [number, number];
  method?: SampleSizeMethod;
  effect_names: string[];
  result: PowerResult | SampleSizeResult;
  scenarios: [string, PowerResult | SampleSizeResult][];
  /** Vega-Lite plot blocks for the Summary chart and Joint tab.
   *  Absent on history-reopened tabs predating this field. */
  plots?: PlotSpecs;
}

export interface LastRunDescriptor {
  kind: RunTab['kind'];
  spec: AppSpec;
  sample_size?: number;
  bounds?: [number, number];
  method?: SampleSizeMethod;
}

function createRunStore() {
  let runState = $state<RunState>('idle');
  let currentRunId = $state<string | null>(null);
  const progress = $state<{ completed: number; total: number }>({ completed: 0, total: 0 });
  const runTabs = $state<RunTab[]>([]);
  let activeTabId = $state<string | null>(null);
  let lastRun = $state<LastRunDescriptor | null>(null);
  // The last run failure, as the 'run' card reads it. Set in startRun's catch,
  // cleared when the next run begins or when the user dismisses the card.
  let lastError = $state<AppError | null>(null);

  let _unlisten: (() => void) | null = null;
  let _tabCounter = 0;

  async function ensureSubscribed() {
    if (_unlisten) return;
    _unlisten = await onProgress(handleEvent);
  }

  function handleEvent(e: ProgressEvent) {
    switch (e.kind) {
      case 'run_started':
        currentRunId = e.run_id;
        break;
      case 'started':
        progress.completed = 0;
        progress.total = Number(e.total_sims);
        break;
      case 'sims_completed':
        // Monotone guard: rayon workers can race tick emission, so a slightly older count may land after a newer one.
        progress.completed = Math.max(progress.completed, Number(e.completed));
        break;
      case 'cancelled':
        runState = 'cancelled';
        break;
      default:
        break;
    }
  }

  function nextLabel() {
    // Monotonic, matching the id `pushTab` is about to mint (`tab-${_tabCounter+1}`).
    // `runTabs.length` would shrink when a tab is closed and reissue a stale "Run N".
    return `Run ${_tabCounter + 1}`;
  }

  // Render one EffectDescriptor to a display name using the port's label store.
  // Mirrors Python _render_descriptor / _factor_label (engine-py report).
  function renderDescriptor(desc: EffectDescriptor, varLevels: Map<string, string[]>): string {
    if (desc.kind === 'intercept') return '(Intercept)';
    if (desc.kind === 'continuous') return desc.predictor;
    if (desc.kind === 'factor_level') {
      const levels = varLevels.get(desc.factor) ?? [];
      const label = (desc.level >= 0 && desc.level < levels.length)
        ? levels[desc.level]!
        : String(desc.level + 1);
      return `${desc.factor}[${label}]`;
    }
    // interaction
    return desc.components.map((c) => renderDescriptor(c, varLevels)).join(':');
  }

  // Build a name → levels map from the live familyStore config so skeleton
  // rendering uses whatever labels the user has typed on the card.
  function buildVarLevels(): Map<string, string[]> {
    const map = new Map<string, string[]>();
    // We look up the active config variables; for history replays with no live
    // config this returns an empty map and skeleton renders index+1 names.
    try {
      const cfg = familyStore.byFamily[familyStore.active];
      for (const v of cfg.variables) {
        if (v.kind === 'factor' && Array.isArray(v.levels) && v.levels.length > 0) {
          map.set(v.name, v.levels);
        }
      }
    } catch {
      // No live config — fall back to empty map (skeleton renders index+1 names).
    }
    return map;
  }

  async function resolveEffectNames(spec: AppSpec): Promise<string[]> {
    // ANOVA has its own expanded effect rows — keep using them directly.
    if (spec.family === 'anova') {
      return spec.effects.map((e) => e.name);
    }
    // Regression / mixed: fetch the engine's index-only skeleton and render
    // names using the port's label store. Falls back to bare predictor names
    // if the skeleton call fails (network, spec validation error, etc.).
    try {
      const skeleton = await effectSkeleton(spec);
      const varLevels = buildVarLevels();
      // skeleton[0] is the intercept; result names are intercept-excluded
      // (effect_names[idx-1] for target_indices[i]). Slice off element 0.
      return skeleton.slice(1).map((desc) => renderDescriptor(desc, varLevels));
    } catch (err) {
      console.warn('effectSkeleton failed, falling back to bare predictor names:', err);
      const names = [...spec.parsed_formula.predictors];
      for (const interaction of spec.parsed_formula.interaction_terms) {
        names.push(interaction.join(':'));
      }
      return names;
    }
  }

  function pushTab(tab: Omit<RunTab, 'id'> & { id?: string }) {
    _tabCounter += 1;
    const id = tab.id ?? `tab-${_tabCounter}`;
    const t: RunTab = { ...tab, id } as RunTab;
    runTabs.push(t);
    activeTabId = t.id;
    return t;
  }

  function removeTab(id: string) {
    const idx = runTabs.findIndex((t) => t.id === id);
    if (idx === -1) return;
    runTabs.splice(idx, 1);
    if (activeTabId === id) activeTabId = runTabs[runTabs.length - 1]?.id ?? null;
  }

  function clearTabs() {
    runTabs.length = 0;
    activeTabId = null;
  }

  async function startRun(
    descriptor: LastRunDescriptor,
    invokeFn: () => Promise<{
      run_id: string;
      result: { scenarios: Array<[string, PowerResult | SampleSizeResult]> };
      plots?: PlotSpecs;
    }>,
    extras: Partial<Pick<RunTab, 'sample_size' | 'bounds' | 'method'>>,
  ) {
    runState = 'running';
    lastError = null;
    lastRun = descriptor;
    // Clear the previous run's final progress before the strip renders. The
    // `started` event also zeroes these, but only once the engine call spins
    // up — without this the bar inherits the prior run's 100% and the
    // indicator's CSS transition animates *down* from it, sitting far ahead of
    // the (untransitioned) percent text until it catches up.
    progress.completed = 0;
    progress.total = 0;
    await ensureSubscribed();
    try {
      const [{ run_id, result, plots }, effect_names] = await Promise.all([
        invokeFn(),
        resolveEffectNames(descriptor.spec),
      ]);
      runState = 'done';
      const scenarios = result.scenarios;
      const inner = scenarios[0]?.[1];
      if (!inner) throw new Error('engine returned no scenarios');
      pushTab({
        ...extras,
        id: run_id,
        label: nextLabel(),
        kind: descriptor.kind,
        subView: 'summary',
        spec: descriptor.spec,
        effect_names,
        result: inner,
        scenarios,
        plots,
      });
      void historyStore.push({
        ...extras,
        id: run_id,
        ts: Date.now(),
        family: descriptor.spec.family,
        kind: descriptor.kind,
        spec: descriptor.spec,
        effect_names,
        result: inner,
      });
    } catch (err) {
      if ((runState as RunState) !== 'cancelled') {
        runState = 'error';
        lastError = toRunError(err);
      }
      console.error(err);
    } finally {
      currentRunId = null;
    }
  }

  async function startFindPower(spec: AppSpec, sampleSize: number) {
    await startRun(
      { kind: 'find-power', spec, sample_size: sampleSize },
      () => findPower(spec, sampleSize),
      { sample_size: sampleSize },
    );
  }

  async function startFindSampleSize(
    spec: AppSpec,
    bounds: [number, number],
    method: SampleSizeMethod,
  ) {
    // Cluster-size mode is fine for sample-size search: the engine's grid atom
    // becomes the cluster size (N sweeps in multiples of it, n_clusters varies).
    await startRun(
      { kind: 'find-sample-size', spec, bounds, method },
      () => findSampleSize(spec, bounds, method),
      { bounds, method },
    );
  }

  async function cancel() {
    if (currentRunId) await cancelRun(currentRunId);
  }

  async function replayLast() {
    if (!lastRun) return;
    if (lastRun.kind === 'find-power' && lastRun.sample_size !== undefined) {
      await startFindPower(lastRun.spec, lastRun.sample_size);
    } else if (lastRun.kind === 'find-sample-size' && lastRun.bounds && lastRun.method) {
      await startFindSampleSize(lastRun.spec, lastRun.bounds, lastRun.method);
    }
  }

  return {
    get runState() { return runState; },
    set runState(v: RunState) { runState = v; },
    get currentRunId() { return currentRunId; },
    get progress() { return progress; },
    get runTabs() { return runTabs; },
    get activeTabId() { return activeTabId; },
    set activeTabId(v: string | null) { activeTabId = v; },
    get lastRun() { return lastRun; },
    get lastError() { return lastError; },
    set lastError(v: AppError | null) { lastError = v; },
    startFindPower,
    startFindSampleSize,
    cancel,
    pushTab,
    removeTab,
    clearTabs,
    replayLast,
  };
}

export const runStore = createRunStore();
