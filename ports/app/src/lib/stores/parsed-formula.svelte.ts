// Parsed-formula store: per-formula parse-state cache backed by the Tauri `parse_formula_cmd` command; SvelteMap ensures reactive per-key granularity.
import { SvelteMap } from 'svelte/reactivity';
import type { ParsedFormula } from '$lib/domain/app-spec';
import type { FormulaParse } from '$lib/domain/result';
import { parseFormula } from '$lib/api/engine';

// Vite statically replaces this, so the E2E branch compiles out of the real build
// (and the test-only stub it imports is tree-shaken away). Mirrors engine.ts.
const E2E = import.meta.env.VITE_E2E === 'true';

export type { FormulaParse };

export interface ParseState {
  result: FormulaParse | null;
  error: string | null;
  pending: boolean;
}

const EMPTY: ParseState = { result: null, error: null, pending: false };

function createParsedFormulaStore() {
  // SvelteMap (NOT $state(new Map())): plain $state does not track Map .get/.set
  // mutations in Svelte 5, so reactive reads would never re-run after the async
  // parse resolves. SvelteMap instruments per-key signals — do not wrap in $state.
  const cache = new SvelteMap<string, ParseState>();

  function ensure(formula: string): ParseState {
    const key = formula;
    const hit = cache.get(key); // reactive read — tracks this key even while absent
    if (hit) return hit;
    // Trigger the parse OUTSIDE the current reactive computation. Mutating the
    // SvelteMap synchronously here would be a state write inside a `$derived`
    // (consumers read via `$derived(effectNames(cfg))` etc.), which Svelte 5
    // rejects as `state_unsafe_mutation`. queueMicrotask defers the write so the
    // read stays pure; the `cache.get(key)` dependency above re-runs the reader
    // once the pending state — and later the result — lands.
    queueMicrotask(() => {
      if (cache.has(key)) return;
      cache.set(key, { ...EMPTY, pending: true });
      if (E2E) {
        // No Tauri runtime under Playwright — parse with the same JS stub the
        // unit tests use, so formula-driven UI (effect rows, clusters) renders.
        void import('../../tests/parse-formula-stub').then(({ stubParseFormula }) =>
          cache.set(key, stubParseFormula(formula)),
        );
        return;
      }
      void parseFormula(formula)
        .then((result) => cache.set(key, { result, error: null, pending: false }))
        .catch((e: unknown) => cache.set(key, { result: null, error: String(e), pending: false }));
    });
    return { ...EMPTY, pending: true };
  }

  // Last state with a successful result handed out by getStable. Plain
  // variable, not $state — reactivity comes from the SvelteMap read inside
  // ensure(); this is only a fallback snapshot for resultless windows.
  let lastGood: ParseState | null = null;

  return {
    /** Reactive read; triggers a parse on first sight of `formula`. */
    get(formula: string): ParseState {
      return ensure(formula);
    },
    /**
     * Like get(), but while the current formula has no result (parse still
     * pending, or a mid-typing syntax error) this returns the last successful
     * parse instead of `{result: null}`. UI that derives rows from the parse
     * (predictor cards, effect rows, cluster cards) must use this — acting on
     * the transient null state wipes user values (effect sizes, factor
     * configs) on every formula edit. Error reporting stays on get().
     */
    getStable(formula: string): ParseState {
      // An empty/cleared formula resolves to the empty state, NOT the last good
      // parse. Without this, Reset and manual clear leave the stale parse alive
      // and the derived cards/effects/tests revive on the next reconcile. The
      // lastGood fallback below is kept only for NON-empty formulas (genuine
      // mid-typing / transient syntax error), where collapsing to null would
      // wipe user values.
      if (formula.trim() === '') {
        lastGood = null;
        return EMPTY;
      }
      const st = ensure(formula);
      if (st.result) {
        lastGood = st;
        return st;
      }
      return lastGood ?? st;
    },
  };
}

export const parsedFormulaStore = createParsedFormulaStore();

export function toWireParsed(fp: FormulaParse): ParsedFormula {
  return {
    outcome: fp.dependent,
    predictors: fp.predictors,
    interaction_terms: fp.terms
      .filter((t) => t.kind === 'interaction')
      .map((t) => (t as { vars: string[] }).vars),
  };
}

/** One cluster term derived from the formula's random effects — the UI's view
 *  of `(1|g)` / `(1|a/b)` / `(1+x|g)`. Order matches the parser's extraction
 *  order; the first term is the primary cluster. */
export interface ClusterTerm {
  cluster: string;
  /** Parent grouping name for the child of a nested `(1|A/B)` term; null = crossed/primary. */
  parent: string | null;
  /** Random-slope predictor names from `(1+x|g)` syntax; empty = intercept-only. */
  slopeVars: string[];
}

export function toClusterTerms(fp: FormulaParse): ClusterTerm[] {
  // A `slope` RE carries an implicit random intercept (lme4 convention — the
  // parser rejects intercept suppression), so every RE maps to one cluster term.
  return fp.random_effects.map((r) =>
    r.kind === 'intercept'
      ? { cluster: r.group, parent: r.parent, slopeVars: [] }
      : { cluster: r.group, parent: null, slopeVars: r.vars },
  );
}
