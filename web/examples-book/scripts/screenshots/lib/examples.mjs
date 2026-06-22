import { readdirSync } from 'node:fs';

// Book family (id prefix) → how to drive the app to that family's form.
// familyKey is the per-family IndexedDB record key; it equals the engine entrypoint.
const FAMILY_DRIVE = {
  ols:  { entrypoint: 'regression', outcomeKind: 'continuous', familyKey: 'regression' },
  glm:  { entrypoint: 'regression', outcomeKind: 'binary',     familyKey: 'regression' },
  anova:{ entrypoint: 'anova',      outcomeKind: 'continuous', familyKey: 'anova' },
  lmm:  { entrypoint: 'mixed',      outcomeKind: 'continuous', familyKey: 'mixed' },
  glmm: { entrypoint: 'mixed',      outcomeKind: 'binary',     familyKey: 'mixed' },
};

export function discoverIds(chunksDir) {
  return readdirSync(chunksDir)
    .filter((f) => f.endsWith('.py'))
    .map((f) => f.slice(0, -3))
    .sort();
}

// Per-id drive overrides. Factorial ANOVA examples carry interaction effects
// (`a:b`, `a:b:c`) that the app's ANOVA entrypoint cannot represent — projectAnova
// builds interaction_terms:[] and reconcileEffects strips interaction effects for
// ANOVA. Driving them through the formula-based regression entrypoint (which expands
// `*` to main + interaction terms) is the only way the captured form shows the full
// effect set faithfully. Their configs are authored as family:'regression'.
const ID_DRIVE = {
  'anova-04': { entrypoint: 'regression', outcomeKind: 'continuous', familyKey: 'regression' },
  'anova-05': { entrypoint: 'regression', outcomeKind: 'continuous', familyKey: 'regression' },
  'anova-06': { entrypoint: 'regression', outcomeKind: 'continuous', familyKey: 'regression' },
};

export function driveFor(id) {
  if (ID_DRIVE[id]) return ID_DRIVE[id];
  const family = id.split('-')[0];
  const plan = FAMILY_DRIVE[family];
  if (!plan) throw new Error(`unknown book family for id "${id}"`);
  return plan;
}
