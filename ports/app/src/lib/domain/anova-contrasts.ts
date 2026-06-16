// Rebuild pairwise ANOVA contrasts from current factor/level state; preserves manual contrasts and enable/disable toggles.
// NOTE: the `_signature` param is unused for output; callers pass it only so Svelte's `$effect` re-runs on factor-state change.
import type { FamilyConfig, VariableRow } from './family';
import { factorLevels } from './effect-names';

/** Rebuild auto-contrasts for ANOVA primary factors (role === 'factor').
 *  Idempotent given the same factor set + level labels; `signature` lets the
 *  caller's $effect re-trigger without depending on object identity.
 *  - Registers every pairwise level contrast of each primary factor.
 *  - Preserves enable/disable toggles for pairs that still exist.
 *  - Drops contrasts whose factor/level no longer exists.
 *  - Never clobbers a still-valid manually-added contrast (dedup by pair). */
export function regenAutoContrasts(cfg: FamilyConfig, _signature: string): void {
  const factors = cfg.variables.filter((v) => v.role === 'factor');
  const existing = new Map(
    (cfg.contrasts ?? []).map((c) => [`${c.positiveName} ${c.negativeName}`, c]),
  );

  // Valid effect names per primary factor, for liveness checks.
  const liveNames = new Set<string>();
  for (const f of factors) for (const lv of factorLevels(f)) liveNames.add(`${f.name}[${lv}]`);

  const next: FamilyConfig['contrasts'] = [];
  const seen = new Set<string>();

  for (const f of factors) {
    const levels = factorLevels(f);
    for (let i = 0; i < levels.length; i++) {
      for (let j = i + 1; j < levels.length; j++) {
        const pos = `${f.name}[${levels[i]}]`;
        const neg = `${f.name}[${levels[j]}]`;
        const key = `${pos} ${neg}`;
        seen.add(key);
        const prev = existing.get(key);
        next.push({ positiveName: pos, negativeName: neg, enabled: prev?.enabled ?? true });
      }
    }
  }
  for (const [key, c] of existing) {
    if (seen.has(key)) continue;
    if (liveNames.has(c.positiveName) && liveNames.has(c.negativeName)) {
      next.push(c);
      seen.add(key);
    }
  }
  cfg.contrasts = next;
}

export type { VariableRow };
