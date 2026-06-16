<script lang="ts">
  // Joint-significance distribution tab: exactly-k / at-least-k tables for find-power; detection curve + required-N table for find-sample-size.
  import type { RunTab } from '$lib/stores/run.svelte';
  import type { PowerResult, SampleSizeResult } from '$lib/domain/result';
  import { jointDistribution } from '$lib/report/joint';
  import { REPORT_CONFIG } from '$lib/configs/report-config';
  import { requiredNHeadline } from '$lib/report/crossing';
  import VegaChart from './VegaChart.svelte';

  interface Props { tab: RunTab; }
  const { tab }: Props = $props();

  const dec = REPORT_CONFIG.format.joint_table_decimals;
  const pct = (x: number) => `${(x * 100).toFixed(dec)}%`;

  const hasMultipleScenarios = $derived(tab.scenarios.length > 1);

  // find-sample-size: targetPct and per-scenario row data
  const ssr0 = $derived(tab.result as SampleSizeResult);
  const targetPct = $derived(
    tab.kind === 'find-sample-size' ? `${(ssr0.target_power * 100).toFixed(0)}%` : ''
  );

  // Per-scenario data for find-sample-size tables.
  // Each entry: { label, targetCount, rows: { label: string, n: string }[] }
  const ssrScenarios = $derived(
    tab.kind === 'find-sample-size'
      ? tab.scenarios.map(([label, res]) => {
          const r = res as SampleSizeResult;
          // Joint-test count from first_joint_achieved itself (one slot per k):
          // it spans marginals + contrasts, which target_indices alone misses.
          const targetCount = r.first_joint_achieved.length;
          const ceiling = r.grid_or_trace.length > 0
            ? Math.max(...r.grid_or_trace.map((p) => p.n))
            : 0;
          const rows = Array.from({ length: targetCount }, (_, j) => {
            const k = targetCount - j; // all-k down to 1
            // fitted_joint[k-1] is the model-based crossing fit for "≥ k" joint target;
            // falls back to first_joint_achieved when the fit is absent (older payload).
            const fit = r.fitted_joint?.[k - 1];
            const gridN = r.first_joint_achieved[k - 1] ?? null;
            return {
              rowLabel: `≥ ${k} of ${targetCount} tests`,
              n: requiredNHeadline(fit, gridN, ceiling),
            };
          });
          return { label, targetCount, rows };
        })
      : []
  );

  // Per-scenario data for find-power tables.
  // Each entry: { label, jd: JointDistribution | null }
  const powerScenarios = $derived(
    tab.kind === 'find-power'
      ? tab.scenarios.map(([label, res]) => {
          const r = res as PowerResult;
          return {
            label,
            jd: jointDistribution(r.success_count_histogram_uncorrected, r.n_sims),
          };
        })
      : []
  );
</script>

<div data-testid="joint-dist-tab" class="space-y-4">
  {#if tab.kind === 'find-sample-size'}
    {#if tab.plots?.blocks.some((b) => b.key === 'at_least_k')}
      <figure class="space-y-1.5">
        <figcaption class="text-sm font-medium text-foreground">Joint detection — P(≥ k effects significant)</figcaption>
        <VegaChart spec={tab.plots.blocks.find((b) => b.key === 'at_least_k')!.spec} testid="at-least-k-view" />
        <p class="text-xs text-muted-foreground">Each curve is the probability that at least k effects are jointly significant (corrected) at a given sample size; the dashed line marks the {targetPct} target.</p>
      </figure>
    {/if}
    {#if tab.plots?.blocks.some((b) => b.key === 'exactly_k')}
      <figure class="space-y-1.5">
        <figcaption class="text-sm font-medium text-foreground">Exactly-k detection — P(exactly k effects significant)</figcaption>
        <VegaChart spec={tab.plots.blocks.find((b) => b.key === 'exactly_k')!.spec} testid="exactly-k-view" />
        <p class="text-xs text-muted-foreground">Each curve is the probability that exactly k effects are jointly significant (corrected) at a given sample size.</p>
      </figure>
    {/if}
    {#each ssrScenarios as scenario (scenario.label)}
      <table class="w-full text-sm">
        <caption class="mb-1 text-left font-medium">
          {#if hasMultipleScenarios}{scenario.label} — {/if}Joint detection → required N (target {targetPct})
        </caption>
        <thead>
          <tr>
            <th class="pr-4 text-left font-medium text-muted-foreground">Joint target</th>
            <th class="text-right font-medium text-muted-foreground">Required N</th>
          </tr>
        </thead>
        <tbody>
          {#each scenario.rows as row}
            <tr>
              <td class="pr-4">{row.rowLabel}</td>
              <td class="text-right tabular-nums">{row.n}</td>
            </tr>
          {/each}
        </tbody>
      </table>
    {/each}
  {:else}
    {#each powerScenarios as scenario (scenario.label)}
      {#if hasMultipleScenarios}
        <p class="text-sm font-medium text-foreground">{scenario.label}</p>
      {/if}
      {#if !scenario.jd}
        <p class="text-sm text-muted-foreground">
          Joint significance distribution is unavailable for this result.
        </p>
      {:else}
        <table class="w-full text-sm">
          <thead>
            <tr>
              <th class="pr-4 text-left font-medium text-muted-foreground">k</th>
              <th class="pr-4 text-left font-medium text-muted-foreground">Exactly</th>
              <th class="text-left font-medium text-muted-foreground">At least</th>
            </tr>
          </thead>
          <tbody>
            {#each scenario.jd.exactly as p, k}
              <tr>
                <td class="pr-4">{k}</td>
                <td class="pr-4">{pct(p)}</td>
                <td>{pct(scenario.jd.atLeast[k] ?? 0)}</td>
              </tr>
            {/each}
          </tbody>
        </table>
      {/if}
    {/each}
  {/if}
</div>
