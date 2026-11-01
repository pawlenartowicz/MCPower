<script lang="ts">
  // Power-result table for find-power runs; supports single-scenario and multi-scenario layouts with a Δ column.
  import type { RunTab } from '$lib/stores/run.svelte';
  import type { PowerResult, SampleSizeResult, EstimatorExtras } from '$lib/domain/result';
  import { REPORT_CONFIG } from '$lib/configs/report-config';
  import { buildRows, factorReferenceLabels, baselineContrastLabels } from '$lib/report/rows';
  import { requiredNHeadline, requiredNCI } from '$lib/report/crossing';
  import { unitColumnHeader } from '$lib/domain/unit-label';

  interface Props { tab: RunTab; }
  const { tab }: Props = $props();

  // find-power → PowerResult (per-test power table); find-sample-size →
  // SampleSizeResult (per-target marginal required-N table). The joint required-N
  // table lives in JointDistTab; this stays per-target marginal only.
  const r = $derived(tab.kind === 'find-power' ? (tab.result as PowerResult) : null);
  const ssr = $derived(tab.kind === 'find-sample-size' ? (tab.result as SampleSizeResult) : null);
  // Requested baseline event probability of a binary outcome (plain logit/probit
  // or a binary mixed model); null for any other outcome.
  const requestedBaseline = $derived.by((): number | null => {
    const spec = tab.spec;
    if (spec.family === 'logit') return spec.baseline_probability;
    if (spec.family === 'mixed' && spec.outcome?.kind === 'binary') return spec.outcome.baseline_probability;
    return null;
  });
  const useCorrected = $derived(tab.spec.correction !== 'none');

  // Determine estimator string for overall label lookup. For find-sample-size
  // the estimator lives on the grid points (identical across the grid).
  const estimator = $derived(
    r?.estimator_extras
      ? (r.estimator_extras as { estimator: string }).estimator
      : ((ssr?.grid_or_trace[0]?.estimator_extras as { estimator: string } | undefined)?.estimator ??
        'ols'),
  );

  const overallLabel = $derived(
    REPORT_CONFIG.overall_label_by_estimator[estimator] ?? 'Overall',
  );

  // Build display rows with idx-1 mapping (β̂-col → intercept-excluded effect name).
  // factors: the app spec has factor var_types; pass them as {name: {baseline: ref_level}}.
  const factors = $derived((): Record<string, { baseline?: string }> => {
    if (!tab.spec || tab.spec.family === 'anova') return {};
    const result: Record<string, { baseline?: string }> = {};
    for (const vt of tab.spec.var_types) {
      if (vt.kind === 'factor') {
        result[vt.name] = {};
      }
    }
    return result;
  });

  // Target indices: from the find-power result, or any grid point of a
  // find-sample-size result (identical across the grid). Drives effect-name rows.
  const targetIndices = $derived(
    r ? r.target_indices : (ssr?.grid_or_trace[0]?.target_indices ?? []),
  );

  // Pairwise contrast identities — the engine appends one power/ci entry per
  // contrast after the marginals; buildRows turns them into `contrast` rows.
  const contrastPairs = $derived(
    r ? (r.contrast_pairs ?? []) : (ssr?.grid_or_trace[0]?.contrast_pairs ?? []),
  );

  // Baseline contrasts (e.g. ANOVA F[1] vs F[2]) collapse to plain marginals in
  // the engine result (the reference side has no β-column); recover them from
  // spec.contrasts so they render as `vs` rows. ANOVA ships family 'linear', so
  // the early-return matches the `factors()` guard above (anova has no var_types).
  const factorRefs = $derived.by(() =>
    !tab.spec || tab.spec.family === 'anova' ? {} : factorReferenceLabels(tab.spec.var_types),
  );
  const baselineContrasts = $derived.by(() =>
    baselineContrastLabels(targetIndices, tab.effect_names, factorRefs, tab.spec?.contrasts ?? []),
  );

  const displayRows = $derived(() => {
    if (!r && !ssr) return [];
    try {
      return buildRows(targetIndices, tab.effect_names, factors(), contrastPairs, baselineContrasts);
    } catch {
      // Fallback: return raw continuous rows if buildRows throws (e.g. stale data)
      return targetIndices.map((idx, pos) => ({
        kind: 'continuous' as const,
        label: tab.effect_names[idx - 1] ?? `eff[${idx}]`,
        pos,
      }));
    }
  });

  const fmt = REPORT_CONFIG.format;
  const dec = fmt.power_decimals_short;
  const target = $derived(tab.spec?.target_power ?? 0.8);

  // The one percent formatter of this view. A missing or non-finite rate
  // (null on the JSON wire: the engine's NaN for a corrected power the
  // correction does not define, or an event rate with nothing to average)
  // renders "n/a".
  function fmtPct(v: number | null, decimals: number = dec) {
    return v != null && Number.isFinite(v) ? (v * 100).toFixed(decimals) + '%' : 'n/a';
  }
  function fmtCi(ci: { lo: number | null; hi: number | null }) {
    if (ci.lo == null || ci.hi == null || !Number.isFinite(ci.lo) || !Number.isFinite(ci.hi)) {
      return 'n/a';
    }
    return `[${(ci.lo * 100).toFixed(dec)}, ${(ci.hi * 100).toFixed(dec)}]`;
  }
  // Target column: target power (integer %) + a met/not-met marker, matching the
  // Python/R short-form `Target` column for cross-port parity. No marker when
  // the power is not applicable.
  function targetCell(power: number | null) {
    const pct = `${(target * 100).toFixed(fmt.target_decimals)}%`;
    if (power == null || !Number.isFinite(power)) return pct;
    return `${pct} ${power >= target ? '✓' : '✗'}`;
  }

  // Under Tukey HSD a target outside every factor has no corrected power: the
  // engine flags it in `corrected_not_applicable` (and its corrected rate and
  // CI arrive as null). The flag only matters when the table shows corrected
  // values; a find-sample-size run reads it off any grid point (the spec's
  // flags, identical across the grid).
  function notApplicable(res: PowerResult | undefined, pos: number): boolean {
    return useCorrected && (res?.corrected_not_applicable?.[pos] ?? false);
  }
  function ssrNotApplicable(pos: number): boolean {
    return ssr?.grid_or_trace[0]?.corrected_not_applicable?.[pos] ?? false;
  }
  // The displayed rate at `pos`: null when not applicable, never a 0 stand-in.
  function displayedPower(res: PowerResult, pos: number): number | null {
    if (notApplicable(res, pos)) return null;
    return (useCorrected ? res.power_corrected : res.power_uncorrected)[pos] ?? null;
  }

  // ── find-sample-size: per-target marginal required-N (charter-faithful) ────
  // The main table is `Effect | Required N` (+ CI when single-scenario).
  // `first_achieved[pos]` is the grid-empirical fallback; `fitted[pos]` is the
  // model-based crossing fit (see requiredNHeadline for the fallback chain).
  // Largest N the grid evaluated — the search ceiling. An unreached target shows
  // "≥ ceiling" (charter), never a bare "—".
  const ceilingN = $derived(ssr ? Math.max(...ssr.grid_or_trace.map((p) => p.n)) : 0);
  // Smallest N on the grid — lower bound for CI display when ci_lo is null.
  const floorN = $derived(ssr ? Math.min(...ssr.grid_or_trace.map((p) => p.n)) : 0);
  // A target with no corrected power has no required N: "n/a" in every cell.
  function requiredNDisplay(pos: number): string {
    if (ssrNotApplicable(pos)) return 'n/a';
    return requiredNHeadline(ssr?.fitted?.[pos] ?? undefined, ssr?.first_achieved[pos] ?? null, ceilingN);
  }
  // Same cell, counted in the unit the search ran in (divisor 1 = observations,
  // and then the unit column is not shown at all).
  function requiredUnitDisplay(pos: number): string {
    if (ssrNotApplicable(pos)) return 'n/a';
    return requiredNHeadline(
      ssr?.fitted?.[pos] ?? undefined,
      ssr?.first_achieved[pos] ?? null,
      ceilingN,
      unitDivisor,
    );
  }
  function requiredNCIDisplay(pos: number): string {
    if (ssrNotApplicable(pos)) return 'n/a';
    return requiredNCI(ssr?.fitted?.[pos] ?? undefined, floorN, ceilingN);
  }
  // First N where ALL targets reach target power: the slowest target's required
  // N, or "≥ ceiling" if any target never reached it within the search bounds.
  // A target with no corrected power makes it not applicable: "n/a".
  const firstAllAchieved = $derived(() => {
    if (!ssr) return '';
    if (ssr.first_achieved.some((_, pos) => ssrNotApplicable(pos))) return 'n/a';
    const fa = ssr.first_achieved;
    if (fa.length === 0 || fa.some((n) => n == null)) return `≥ ${ceilingN}`;
    return String(Math.max(...(fa as number[])));
  });

  // Sample-size unit: the engine reports the divisor it searched with, the spec
  // names the unit, and the label helper spells it. Divisor 1 (observations)
  // leaves the table exactly as it was — no extra column.
  const unitDivisor = $derived(ssr?.unit_divisor ?? 1);
  const unitHeader = $derived(
    tab.spec.family === 'mixed'
      ? unitColumnHeader(tab.spec.sample_size_unit ?? 'observations', tab.spec.cluster_name)
      : '',
  );
  const showUnitColumn = $derived(!!ssr && unitDivisor > 1 && unitHeader !== '');

  // When the tab carries >1 scenario, the table switches to a per-scenario column layout with a Δ column;
  // single-scenario layout is unchanged.
  const multi = $derived(tab.scenarios.length > 1);

  // CI column shown only in the single-scenario sample-size case — mirrors how
  // find-power's single-scenario branch shows a "95% CI" column.
  const showSsrCI = $derived(!multi && !!ssr);

  // Baseline scenario index: prefer REPORT_CONFIG label if present, else 0.
  const baselineIdx = $derived((): number => {
    const preferLabel = REPORT_CONFIG.baseline_scenario.prefer_label;
    const idx = tab.scenarios.findIndex(([label]) => label === preferLabel);
    if (idx !== -1) return idx;
    return 0;
  });

  // Per-row Δ: min over all scenarios of (scenarioPower - baselinePower).
  // For the baseline scenario itself the diff is 0; for worse scenarios it's
  // negative. null when the row's power is not applicable.
  function computeDelta(pos: number): number | null {
    const baseEntry = tab.scenarios[baselineIdx()];
    if (!baseEntry) return 0;
    const basePower = displayedPower(baseEntry[1] as PowerResult, pos);
    if (basePower == null) return null;
    let minDiff = 0;
    for (const [, sr] of tab.scenarios) {
      const sp = displayedPower(sr as PowerResult, pos);
      if (sp == null) return null;
      const diff = sp - basePower;
      if (diff < minDiff) minDiff = diff;
    }
    return minDiff;
  }

  function fmtDelta(delta: number | null): string {
    if (delta == null) return 'n/a';
    const pp = (delta * 100).toFixed(fmt.drop_decimals);
    return delta >= 0 ? `+${pp}pp` : `${pp}pp`;
  }

  // Overall row Δ: same logic using overall_significant_rate across scenarios.
  function computeOverallDelta(): number {
    const baseEntry = tab.scenarios[baselineIdx()];
    if (!baseEntry) return 0;
    const basePower = (baseEntry[1] as PowerResult).overall_significant_rate ?? 0;
    let minDiff = 0;
    for (const [, sr] of tab.scenarios) {
      const sp = (sr as PowerResult).overall_significant_rate ?? 0;
      const diff = sp - basePower;
      if (diff < minDiff) minDiff = diff;
    }
    return minDiff;
  }

  // Total column count for colspan on factor-header rows.
  // Single: 5 (Effect, n, Power, 95% CI, Target)
  // Multi:  1 (Effect) + nScenarios + 1 (Δ) + 1 (Target)
  const colCount = $derived(multi ? 1 + tab.scenarios.length + 2 : 5);
</script>

{#if r}
  <div class="space-y-2">
    <div class="text-xs text-muted-foreground">
      n = {r.n} · sims = {r.n_sims} · convergence = {(r.convergence_rate * 100).toFixed(1)}% · correction = {tab.spec.correction}
    </div>
    {#if r.estimator_extras && (r.estimator_extras as { estimator: string }).estimator === 'glm'}
      {@const fx = r.estimator_extras as Extract<EstimatorExtras, { estimator: 'glm' }>}
      <div class="text-xs text-muted-foreground">
        {`event rate = ${fmtPct(fx.baseline_prob_realized, 1)}`}{requestedBaseline != null ? ` · baseline ${fmtPct(requestedBaseline, 1)}` : ''}
      </div>
    {/if}
    {#if r.estimator_extras && (r.estimator_extras as { estimator: string }).estimator === 'mle'}
      {@const fx = r.estimator_extras as Extract<EstimatorExtras, { estimator: 'mle' }>}
      <div class="text-xs text-muted-foreground">
        boundary hits = {fx.boundary_hits} · convergence (joint corrected) = {(fx.joint_corrected_rate * 100).toFixed(1)}%
      </div>
    {/if}
    <!-- w-auto + per-cell right padding: table sizes to content (left-aligned) instead
         of stretching edge-to-edge; last column drops the trailing gap. -->
    <table
      class="w-auto text-sm [&_td]:pr-8 [&_th]:pr-8 [&_td:last-child]:pr-0 [&_th:last-child]:pr-0"
      data-testid="summary-table"
    >
      <thead>
        <tr class="border-b border-border text-left">
          <th class="py-1">Effect</th>
          {#if multi}
            {#each tab.scenarios as [label] (label)}
              <th class="py-1">{label}</th>
            {/each}
            <th class="py-1">Δ</th>
          {:else}
            <th class="py-1">n</th>
            <th class="py-1">Power</th>
            <th class="py-1">95% CI</th>
          {/if}
          <th class="py-1">Target</th>
        </tr>
      </thead>
      <tbody>
        {#if r.overall_significant_rate != null}
          {#if multi}
            <tr class="border-b border-border/50 font-medium">
              <td class="py-1">{overallLabel}</td>
              {#each tab.scenarios as [scenLabel, sr] (scenLabel)}
                {@const sp = (sr as PowerResult).overall_significant_rate ?? 0}
                <td class="py-1 font-mono">{fmtPct(sp)}</td>
              {/each}
              <td class="py-1 font-mono">{fmtDelta(computeOverallDelta())}</td>
              <td class="py-1 font-mono">{targetCell(r.overall_significant_rate)}</td>
            </tr>
          {:else}
            {@const ci = r.overall_significant_ci}
            <tr class="border-b border-border/50 font-medium">
              <td class="py-1">{overallLabel}</td>
              <td class="py-1 font-mono">{r.n}</td>
              <td class="py-1 font-mono">{fmtPct(r.overall_significant_rate)}</td>
              <td class="py-1 font-mono">{ci ? fmtCi(ci) : '—'}</td>
              <td class="py-1 font-mono">{targetCell(r.overall_significant_rate)}</td>
            </tr>
          {/if}
        {/if}
        {#each displayRows() as row (row.kind === 'factor_header' ? `hdr-${row.label}` : `${row.kind}-${row.pos ?? row.label}`)}
          {#if row.kind === 'factor_header'}
            <tr class="border-b border-border/30 bg-muted/30">
              <td class="py-1 pl-0 text-xs font-semibold text-muted-foreground" colspan={colCount}>
                {row.label}{row.baseline ? ` (baseline: ${row.baseline})` : ''}
              </td>
            </tr>
          {:else if multi}
            {@const pos = row.pos!}
            {@const baseEntry = tab.scenarios[baselineIdx()]}
            {@const basePower = baseEntry ? displayedPower(baseEntry[1] as PowerResult, pos) : 0}
            <tr class="border-b border-border/50 {row.kind === 'factor_level' ? '' : ''}">
              <td class="py-1 {row.kind === 'factor_level' ? 'pl-4' : ''}">{row.label}</td>
              {#each tab.scenarios as [scenLabel, sr] (scenLabel)}
                {@const sp = displayedPower(sr as PowerResult, pos)}
                <td class="py-1 font-mono">{fmtPct(sp)}</td>
              {/each}
              <td class="py-1 font-mono">{fmtDelta(computeDelta(pos))}</td>
              <td class="py-1 font-mono">{targetCell(basePower)}</td>
            </tr>
          {:else if row.kind === 'factor_level'}
            {@const power = displayedPower(r, row.pos!)}
            {@const ci = (useCorrected ? r.ci_corrected : r.ci_uncorrected)[row.pos!] ?? { lo: 0, hi: 0 }}
            <tr class="border-b border-border/50">
              <td class="py-1 pl-4">{row.label}</td>
              <td class="py-1 font-mono">{r.n}</td>
              <td class="py-1 font-mono">{fmtPct(power)}</td>
              <td class="py-1 font-mono">{fmtCi(ci)}</td>
              <td class="py-1 font-mono">{targetCell(power)}</td>
            </tr>
          {:else}
            {@const power = displayedPower(r, row.pos!)}
            {@const ci = (useCorrected ? r.ci_corrected : r.ci_uncorrected)[row.pos!] ?? { lo: 0, hi: 0 }}
            <tr class="border-b border-border/50">
              <td class="py-1">{row.label}</td>
              <td class="py-1 font-mono">{r.n}</td>
              <td class="py-1 font-mono">{fmtPct(power)}</td>
              <td class="py-1 font-mono">{fmtCi(ci)}</td>
              <td class="py-1 font-mono">{targetCell(power)}</td>
            </tr>
          {/if}
        {/each}
      </tbody>
    </table>
  </div>
{:else if ssr}
  <div class="space-y-2">
    <div class="text-xs text-muted-foreground">
      target power = {fmtPct(target)} · grid points = {ssr.grid_or_trace.length} · correction = {tab.spec.correction}
    </div>
    <table
      class="w-auto text-sm [&_td]:pr-8 [&_th]:pr-8 [&_td:last-child]:pr-0 [&_th:last-child]:pr-0"
      data-testid="sample-size-table"
    >
      <caption class="mb-1 text-left text-sm font-medium text-foreground">
        {REPORT_CONFIG.text.sample_size_caption}
      </caption>
      <thead>
        <tr class="border-b border-border text-left">
          <th class="py-1">Effect</th>
          {#if showUnitColumn}
            <th class="py-1 text-right">{unitHeader}</th>
          {/if}
          <th class="py-1 text-right">Required N</th>
          {#if showSsrCI}
            <th class="py-1 text-right">95% CI</th>
          {/if}
        </tr>
      </thead>
      <tbody>
        <!-- Overall (omnibus) required-N row first, mirroring the find-power
             table's overall-row position. Shown only when the engine carried an
             overall test (OLS F / unclustered GLM LRT); the mixed family emits
             neither field. -->
        {#if ssr.fitted_overall != null || ssr.first_overall_achieved != null}
          <tr class="border-b border-border/50 font-medium">
            <td class="py-1">{overallLabel}</td>
            {#if showUnitColumn}
              <td class="py-1 text-right font-mono tabular-nums"
                >{requiredNHeadline(
                  ssr.fitted_overall ?? undefined,
                  ssr.first_overall_achieved ?? null,
                  ceilingN,
                  unitDivisor,
                )}</td
              >
            {/if}
            <td class="py-1 text-right font-mono tabular-nums"
              >{requiredNHeadline(ssr.fitted_overall ?? undefined, ssr.first_overall_achieved ?? null, ceilingN)}</td
            >
            {#if showSsrCI}
              <td class="py-1 text-right font-mono tabular-nums"
                >{requiredNCI(ssr.fitted_overall ?? undefined, floorN, ceilingN)}</td
              >
            {/if}
          </tr>
        {/if}
        {#each displayRows() as row (row.kind === 'factor_header' ? `hdr-${row.label}` : `${row.kind}-${row.pos ?? row.label}`)}
          {#if row.kind === 'factor_header'}
            <tr class="border-b border-border/30 bg-muted/30">
              <td
                class="py-1 pl-0 text-xs font-semibold text-muted-foreground"
                colspan={(showSsrCI ? 3 : 2) + (showUnitColumn ? 1 : 0)}
              >
                {row.label}{row.baseline ? ` (baseline: ${row.baseline})` : ''}
              </td>
            </tr>
          {:else}
            <tr class="border-b border-border/50">
              <td class="py-1 {row.kind === 'factor_level' ? 'pl-4' : ''}">{row.label}</td>
              {#if showUnitColumn}
                <td class="py-1 text-right font-mono tabular-nums"
                  >{requiredUnitDisplay(row.pos!)}</td
                >
              {/if}
              <td class="py-1 text-right font-mono tabular-nums">{requiredNDisplay(row.pos!)}</td>
              {#if showSsrCI}
                <td class="py-1 text-right font-mono tabular-nums">{requiredNCIDisplay(row.pos!)}</td>
              {/if}
            </tr>
          {/if}
        {/each}
      </tbody>
    </table>
    <div class="text-sm text-foreground">
      First N achieving all targets: <span class="font-mono tabular-nums">{firstAllAchieved()}</span>
    </div>
  </div>
{:else}
  <p class="text-xs text-muted-foreground">Table view only supports Find-power runs.</p>
{/if}
