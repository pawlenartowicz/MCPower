// Maps the engine's generic plot target tokens (`target_{idx}`) to real effect
// names, and rewrites them into a theme-naked Vega-Lite spec before embedding.
//
// The engine emits `target_{idx}` for every plot's `target` field (see
// `engine-orchestrator/src/plot.rs::target_label`) because effect labels are
// host-owned — the engine never sees the user's variable names. The tables
// already resolve names via `buildRows(effect_names)`; this brings the same
// names onto the plot axes/legends so a chart reads `sleep` instead of
// `target_2`. Only the data `target` field is rewritten — Vega-Lite keys the
// y-axis and the strokeDash legend off the field's value, so relabelling the
// data relabels the axis without touching the engine's encoding.

import { contrastLabel } from '$lib/report/rows';
import { REPORT_CONFIG } from '$lib/configs/report-config';

/**
 * Build a `target_{idx}` → display-name map from the run's target indices and
 * the (intercept-excluded) effect names. `targetIndices` are β̂-column indices
 * (intercept at 0, first effect at 1), so the name is `effectNames[idx - 1]`,
 * matching the table's `buildRows` convention. Out-of-range indices are skipped.
 * Contrast entries carry `target_{p}_vs_{n}` tokens (plot.rs `entry_label`);
 * they're labelled via `contrastLabel` — the table rows' own label source.
 *
 * `estimator` selects the overall/omnibus label (F-test for OLS, LRT for GLM):
 * the sample-size curve carries an extra `"overall"` series, relabelled here
 * from `overall_label_by_estimator` so the F-test-vs-LRT choice isn't hardcoded.
 */
export function buildTargetLabelMap(
  targetIndices: number[],
  effectNames: string[],
  contrastPairs: [number, number][] = [],
  estimator = 'ols',
  baselineContrasts: Map<number, string> = new Map(),
): Record<string, string> {
  const map: Record<string, string> = {};
  for (const idx of targetIndices) {
    // A marginal collapsed from a baseline contrast is labelled "ref vs level",
    // matching the table; same shared map source so the two never diverge.
    const relabel = baselineContrasts.get(idx);
    if (relabel !== undefined) {
      map[`target_${idx}`] = relabel;
      continue;
    }
    const name = effectNames[idx - 1];
    if (name) map[`target_${idx}`] = name;
  }
  for (const [p, n] of contrastPairs) {
    if (effectNames[p - 1] && effectNames[n - 1]) {
      map[`target_${p}_vs_${n}`] = contrastLabel(effectNames, p, n);
    }
  }
  map.overall = REPORT_CONFIG.overall_label_by_estimator[estimator] ?? 'Overall';
  return map;
}

/**
 * In-place: rewrite every `target` field value in the spec's data rows using
 * `map`. Walks the whole spec tree (top-level `data`, layered marks' own
 * `data`, faceted `data`, vconcat children) so per-scenario, overlay, marker,
 * and faceted curve data are all covered. A no-op when `map` is empty.
 */
export function relabelTargets(spec: unknown, map: Record<string, string>): void {
  if (Object.keys(map).length === 0) return;
  walk(spec, map);
}

function walk(node: unknown, map: Record<string, string>): void {
  if (Array.isArray(node)) {
    for (const child of node) walk(child, map);
    return;
  }
  if (!node || typeof node !== 'object') return;
  const obj = node as Record<string, unknown>;
  const data = obj.data as { values?: unknown[] } | undefined;
  if (data && Array.isArray(data.values)) {
    for (const row of data.values) {
      if (row && typeof row === 'object') {
        const r = row as Record<string, unknown>;
        if (typeof r.target === 'string' && r.target in map) {
          r.target = map[r.target];
        }
      }
    }
  }
  for (const value of Object.values(obj)) walk(value, map);
}
