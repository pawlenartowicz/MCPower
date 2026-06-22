import type { FormulaParse } from '$lib/domain/result';
import type { ParseState } from '$lib/stores/parsed-formula.svelte';

/** TEST-ONLY formula parser stub. Production reaches the real Rust parser via the
 *  `parse_formula_cmd` Tauri command, unavailable in the node/vitest environment.
 *  This stand-in covers the formula shapes exercised by the consumer-logic unit
 *  tests (outcome `~`/`=`, `+`-separated mains, `*` star-expansion, `:` interaction,
 *  random effects `(1|g)` / `(1+x|g)`). It is NOT a source of truth — the canonical
 *  suite + the Rust integration test guard the real grammar. */
export function stubParseFormula(input: string): ParseState {
  const trimmed = input.trim();

  // Empty formula → error
  if (!trimmed) {
    return { result: null, error: 'formula is empty', pending: false };
  }

  let dependent = 'explained_variable';
  let rhs = trimmed;
  const m = trimmed.match(/^(.*?)\s*(?:~|=)\s*(.*)$/);
  if (m) {
    dependent = m[1]!.trim() || 'explained_variable';
    rhs = m[2]!;
  }

  const rhsTrimmed = rhs.trim();

  // RHS is empty or whitespace → no predictor error
  if (!rhsTrimmed) {
    return { result: null, error: 'no predictor found', pending: false };
  }

  const predictors: string[] = [];
  const terms: FormulaParse['terms'] = [];
  // Mirror the Rust extraction ORDER (nested pairs, then slopes, then plain
  // intercepts) — it decides which cluster term is primary, so the stub must
  // agree with `extract_random_effects` rather than use token order.
  const nestedREs: FormulaParse['random_effects'] = [];
  const slopeREs: FormulaParse['random_effects'] = [];
  const interceptREs: FormulaParse['random_effects'] = [];

  const addMain = (n: string) => {
    if (!predictors.includes(n)) predictors.push(n);
    if (!terms.some((t) => t.kind === 'main' && t.name === n))
      terms.push({ kind: 'main', name: n });
  };

  // Split on top-level '+' only — a '+' inside parentheses belongs to a
  // random-effect term like (1+x|g).
  const tokens: string[] = [];
  let depth = 0;
  let current = '';
  for (const ch of rhsTrimmed) {
    if (ch === '(') depth++;
    if (ch === ')') depth--;
    if (ch === '+' && depth === 0) {
      tokens.push(current);
      current = '';
    } else {
      current += ch;
    }
  }
  tokens.push(current);

  for (const tok of tokens.map((s) => s.trim()).filter(Boolean)) {
    // Nested random intercept: (1|A/B) → parent A + child "A:B" (parent set).
    const nested = tok.match(/^\(\s*1\s*\|\s*([A-Za-z_][\w.]*)\s*\/\s*([A-Za-z_][\w.]*)\s*\)$/);
    if (nested) {
      const parent = nested[1]!;
      nestedREs.push({ kind: 'intercept', group: parent, parent: null });
      nestedREs.push({ kind: 'intercept', group: `${parent}:${nested[2]!}`, parent });
      continue;
    }
    // Random effect: (1|g) or (1+x|g) or (1+x+z|g)
    const re = tok.match(/^\(\s*1\s*(\+[^|]*)?\|\s*([A-Za-z_][\w.:]*)\s*\)$/);
    if (re) {
      const group = re[2]!;
      // Redundant explicit intercept tokens collapse (mirrors Rust): (1+1|g) ≡ (1|g).
      const vars = (re[1] ?? '')
        .slice(1)
        .split('+')
        .map((s) => s.trim())
        .filter((s) => s !== '' && s !== '1');
      if (vars.length > 0) {
        slopeREs.push({ kind: 'slope', group, vars });
      } else {
        interceptREs.push({ kind: 'intercept', group, parent: null });
      }
      continue;
    }
    // Star expansion: a*b expands to a + b + a:b (and triples get all pairs + triple)
    if (tok.includes('*')) {
      const vars = tok.split('*').map((s) => s.trim());
      vars.forEach(addMain);
      // All pairwise interactions
      for (let i = 0; i < vars.length; i++)
        for (let j = i + 1; j < vars.length; j++)
          terms.push({ kind: 'interaction', vars: [vars[i]!, vars[j]!] });
      // Higher-order interaction for 3+ vars
      if (vars.length > 2) terms.push({ kind: 'interaction', vars });
      continue;
    }
    // Colon interaction: x1:x2 (or x1:x2:x3)
    if (tok.includes(':')) {
      terms.push({ kind: 'interaction', vars: tok.split(':').map((s) => s.trim()) });
      continue;
    }
    // Plain main effect
    addMain(tok);
  }

  const random_effects = [...nestedREs, ...slopeREs, ...interceptREs];

  return {
    result: { dependent, predictors, terms, random_effects },
    error: null,
    pending: false,
  };
}
