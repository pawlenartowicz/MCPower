<script lang="ts">
  // The run's effects block: every simulated effect with its input β,
  // standardized β and Cohen's d (dash where undefined), OR = exp(β) on the
  // logit link or RR = exp(β) for a count outcome, and the model R² line.
  // Mirrors the Python/R effects block (tables.py effects_table).
  import type { AppSpec } from '$lib/domain/app-spec';
  import type { StandardizedValues } from '$lib/stores/standardized.svelte';
  import { REPORT_CONFIG } from '$lib/configs/report-config';

  interface Props {
    values: StandardizedValues;
    spec: AppSpec;
  }
  const { values, spec }: Props = $props();
  const cols = REPORT_CONFIG.text.columns;

  const ratio = $derived.by((): string | null => {
    if (spec.family === 'logit') return spec.link === 'probit' ? null : cols.or;
    if (spec.family === 'poisson') return cols.rr;
    if (spec.family === 'mixed') {
      if (spec.outcome?.kind === 'binary') return spec.outcome.link === 'probit' ? null : cols.or;
      if (spec.outcome?.kind === 'poisson') return cols.rr;
    }
    return null;
  });
  const fmt = (v: number | null) => (v === null ? '—' : v.toFixed(3));
  const r2Label = $derived(values.clustered ? REPORT_CONFIG.text.r2_fixed_label : REPORT_CONFIG.text.r2_label);
</script>

<div class="space-y-1">
  <table
    class="w-auto text-sm [&_td]:pr-8 [&_th]:pr-8 [&_td:last-child]:pr-0 [&_th:last-child]:pr-0"
    data-testid="effects-block"
  >
    <thead>
      <tr class="border-b border-border text-left">
        <th class="py-1">{cols.effect}</th>
        <th class="py-1">{cols.input}</th>
        <th class="py-1">{cols.std_beta}</th>
        <th class="py-1">{cols.cohen_d}</th>
        {#if ratio}<th class="py-1">{ratio}</th>{/if}
      </tr>
    </thead>
    <tbody>
      {#each values.rows as row (row.name)}
        <tr class="border-b border-border/50">
          <td class="py-1">{row.name}</td>
          <td class="py-1 font-mono">{row.input.toFixed(2)}</td>
          <td class="py-1 font-mono">{fmt(row.std_beta)}</td>
          <td class="py-1 font-mono">{fmt(row.cohen_d)}</td>
          {#if ratio}<td class="py-1 font-mono">{Math.exp(row.input).toFixed(2)}</td>{/if}
        </tr>
      {/each}
    </tbody>
  </table>
  <p class="font-mono text-xs text-muted-foreground" data-testid="effects-block-r2">
    {r2Label} = {fmt(values.r2)}
  </p>
</div>
