// Family store: active model family selection and per-family config state.
import {
  defaultFamilyConfig,
  FAMILIES,
  mixedOutcomeKind,
  type Entrypoint,
  type FamilyConfig,
  type OutcomeKind,
} from '$lib/domain/family';
import { loadFamily, saveFamily } from '$lib/persistence/per-family';

function buildByFamily(): Record<Entrypoint, FamilyConfig> {
  const out = {} as Record<Entrypoint, FamilyConfig>;
  for (const f of FAMILIES) out[f] = defaultFamilyConfig(f);
  return out;
}

function createFamilyStore() {
  let active = $state<Entrypoint>('regression');
  /** Outcome toggle for the Regression entrypoint. linear → Ols; logit/probit →
   *  binary Glm; poisson → count Glm. */
  let regressionOutcome = $state<OutcomeKind>('linear');
  const byFamily = $state<Record<Entrypoint, FamilyConfig>>(buildByFamily());
  const hydrated = new Set<Entrypoint>();

  // 'linear'/'logit' persist keys are abandoned; no migration (same regression config object).
  void loadFamily('regression').then((cfg) => {
    if (cfg) byFamily.regression = cfg;
    hydrated.add('regression');
  });

  // Persist a family's config. $state.snapshot strips the reactive proxy: the
  // wasm IndexedDB backend stores via structured clone, which throws
  // DataCloneError on a Svelte state proxy (the Tauri backend tolerated it via
  // JSON, so this only bit on the web). Mirrors the scenarios store.
  const persist = (f: Entrypoint) =>
    void saveFamily(f, $state.snapshot(byFamily[f]) as FamilyConfig);

  return {
    get active() { return active; },
    set active(v: Entrypoint) {
      // Persist outgoing family before switching.
      persist(active);
      active = v;
      // Hydrate incoming family on demand.
      if (!hydrated.has(v)) {
        void loadFamily(v).then((cfg) => {
          if (cfg) byFamily[v] = cfg;
          hydrated.add(v);
        });
      }
    },
    get byFamily() { return byFamily; },
    get regressionOutcome() { return regressionOutcome; },
    set regressionOutcome(v: OutcomeKind) { regressionOutcome = v; },
    /** Resolved outcome for the active family: regression reads its store-level toggle,
     *  mixed reads its persisted cluster outcome. Read-only; writes still target
     *  each family's home (ModelSection.setOutcome). */
    get activeOutcome(): OutcomeKind {
      if (active === 'regression') return regressionOutcome;
      if (active === 'mixed') return mixedOutcomeKind(byFamily.mixed.cluster);
      return 'linear';
    },
    resetActive() {
      byFamily[active] = defaultFamilyConfig(active);
      persist(active);
    },
    resetAll() {
      active = 'regression';
      regressionOutcome = 'linear';
      for (const f of FAMILIES) {
        byFamily[f] = defaultFamilyConfig(f);
        persist(f);
      }
    },
  };
}

export const familyStore = createFamilyStore();
