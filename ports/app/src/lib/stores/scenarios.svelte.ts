// Scenarios store: reactive list of ScenarioConfig entries, hydrated from persistence and debounce-saved on every mutation.
// clone()'s shallow spread must mirror ScenarioConfig's nested fields; adding a nested object without updating clone() silently loses data.
import { BUNDLED_SCENARIOS } from '$lib/configs/scenarios-bundled';
import type { ScenarioConfig } from '$lib/configs/scenarios';
import { loadScenarios, saveScenarios } from '$lib/persistence/scenarios';

function clone(scenarios: readonly ScenarioConfig[]): ScenarioConfig[] {
  return scenarios.map((s) => ({
    ...s,
    new_distributions: [...s.new_distributions],
    residual_dists: [...s.residual_dists],
    lme: { ...s.lme },
  }));
}

function createScenariosStore() {
  let scenarios = $state<ScenarioConfig[]>(clone(BUNDLED_SCENARIOS));
  let ready = $state<boolean>(false);
  let saveTimer: ReturnType<typeof setTimeout> | null = null;

  function scheduleSave() {
    if (!ready) return;
    if (saveTimer) clearTimeout(saveTimer);
    saveTimer = setTimeout(() => {
      // $state.snapshot before persisting — see family.svelte.ts for why (DataCloneError).
      void saveScenarios($state.snapshot(scenarios) as ScenarioConfig[]);
    }, 200);
  }

  void loadScenarios()
    .then((stored) => {
      if (stored && stored.length > 0) scenarios = clone(stored);
    })
    .finally(() => {
      ready = true;
    });

  return {
    get scenarios() {
      return scenarios;
    },
    get ready() {
      return ready;
    },
    update(name: string, patch: Partial<ScenarioConfig>) {
      const idx = scenarios.findIndex((s) => s.name === name);
      if (idx < 0) return;
      scenarios[idx] = { ...scenarios[idx]!, ...patch };
      scheduleSave();
    },
    updateLme(name: string, patch: Partial<ScenarioConfig['lme']>) {
      const idx = scenarios.findIndex((s) => s.name === name);
      if (idx < 0) return;
      const current = scenarios[idx]!;
      scenarios[idx] = { ...current, lme: { ...current.lme, ...patch } };
      scheduleSave();
    },
    resetToBundled() {
      scenarios = clone(BUNDLED_SCENARIOS);
      scheduleSave();
    },
    /** Restore only the two More-options distribution pools to bundled values.
     *  Family reset calls this — the pools are the dialog's only state living
     *  outside FamilyConfig — while Settings-tab knobs (heterogeneity, λ, …)
     *  keep their edits. */
    resetDistributionPools() {
      for (const b of BUNDLED_SCENARIOS) {
        const idx = scenarios.findIndex((s) => s.name === b.name);
        if (idx < 0) continue;
        scenarios[idx] = {
          ...scenarios[idx]!,
          new_distributions: [...b.new_distributions],
          residual_dists: [...b.residual_dists],
        };
      }
      scheduleSave();
    },
  };
}

export const scenariosStore = createScenariosStore();
