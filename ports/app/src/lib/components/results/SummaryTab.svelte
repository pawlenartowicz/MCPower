<script lang="ts">
  // Summary sub-view: power table, optional scenario selector, plot (bars or curve), and convergence notice.
  import { untrack } from 'svelte';
  import type { RunTab } from '$lib/stores/run.svelte';
  import type { SampleSizeResult } from '$lib/domain/result';
  import { REPORT_CONFIG } from '$lib/configs/report-config';
  import { UPLOAD } from '$lib/configs/app-config';
  import { reuseFraction, strictReuseWarning } from '$lib/domain/reuse-diagnostic';
  import { runTabLabelMap } from '$lib/charts/labels';
  import TableView from './TableView.svelte';
  import EffectsBlock from './EffectsBlock.svelte';
  import VegaChart from './VegaChart.svelte';
  import ConvergenceNotice from './ConvergenceNotice.svelte';

  interface Props {
    tab: RunTab;
  }
  const { tab }: Props = $props();

  // Derive the selectable scenario chips from the block keys.
  // `scenario:<label>` blocks → each chip is the label after the colon.
  // `overlay` block → "⧉ Overlay" chip.
  // Single-block runs (find-power `power`, single-scenario `curve`) → no chips.
  const scenarioChips = $derived((): { key: string; label: string }[] => {
    const blocks = tab.plots?.blocks ?? [];
    const chips: { key: string; label: string }[] = [];
    for (const b of blocks) {
      if (b.key.startsWith('scenario:')) {
        chips.push({ key: b.key, label: b.key.slice('scenario:'.length) });
      }
    }
    return chips;
  });

  const hasOverlayBlock = $derived(
    (tab.plots?.blocks ?? []).some((b) => b.key === 'overlay')
  );

  // The active block key driving the plot. Default: the baseline-config label's
  // scenario block if present, else first scenario chip, else the main block
  // (power / curve). `fallback_to_first` is intentionally not consulted: in the
  // old findBaselineIndex both flag branches resolved to index 0, so the flag
  // never changed the outcome — this replicates that single behaviour.
  function defaultBlockKey(): string {
    const blocks = tab.plots?.blocks ?? [];
    const preferLabel = REPORT_CONFIG.baseline_scenario.prefer_label;
    const preferred = blocks.find((b) => b.key === `scenario:${preferLabel}`);
    if (preferred) return preferred.key;
    const firstScenario = blocks.find((b) => b.key.startsWith('scenario:'));
    if (firstScenario) return firstScenario.key;
    const first = blocks.find((b) => b.key === 'power' || b.key === 'curve');
    return first?.key ?? (blocks[0]?.key ?? '');
  }

  let selectedBlockKey = $state(untrack(defaultBlockKey));

  // Re-validate the selection whenever the tab's plots change (e.g. user switches
  // the active run-tab). If the stale key doesn't exist in the new block list,
  // reset to the default so the plot always renders. Mirrors ExportTab's pattern.
  $effect(() => {
    const blocks = tab.plots?.blocks ?? [];
    if (blocks.length && !blocks.some((b) => b.key === selectedBlockKey)) {
      selectedBlockKey = defaultBlockKey();
    }
  });

  const currentSpec = $derived((): string | undefined => {
    const blocks = tab.plots?.blocks ?? [];
    const block = blocks.find((b) => b.key === selectedBlockKey);
    return block?.spec;
  });

  // `target_{idx}` / `target_{p}_vs_{n}` / `overall` → effect-name map so the
  // chart axes/legend read real names instead of the engine's generic tokens.
  const labelMap = $derived(runTabLabelMap(tab));

  // Figure heading + one-line description for the plot (the engine emits no
  // title; the host captions it so a chart isn't a bare, unlabelled box).
  const targetPct = $derived(`${((tab.spec?.target_power ?? 0.8) * 100).toFixed(0)}%`);
  const plotTitle = $derived(
    tab.kind === 'find-power' ? 'Power by effect' : 'Power vs sample size',
  );
  const plotDescription = $derived(
    tab.kind === 'find-power'
      ? `Bars show estimated power; whiskers are 95% Monte-Carlo CIs; the dashed line marks the ${targetPct} target.`
      : `Curves show power at each sample size; shaded bands are 95% Monte-Carlo CIs; the dashed line marks the ${targetPct} target.`,
  );

  // Strict-mode reuse diagnostic — only shown when the run used mode="strict" and U>0.
  const reuseInfo = $derived((): {
    show: boolean;
    U: number;
    // For find-power: one (fraction, warning) pair. For find-sample-size: one per target.
    entries: Array<{ label: string; fraction: number; warning: string | null }>;
  } => {
    const csv = tab.spec.csv;
    if (!csv || csv.mode !== 'strict' || csv.n_rows <= 0) return { show: false, U: 0, entries: [] };
    const U = csv.n_rows;
    const ratio = UPLOAD.strict_warning_ratio;
    if (tab.kind === 'find-power') {
      const N = tab.sample_size ?? 0;
      return {
        show: true,
        U,
        entries: [{ label: '', fraction: reuseFraction(U, N), warning: strictReuseWarning(U, N, ratio) }],
      };
    } else {
      // find-sample-size: evaluate per first_achieved N per target.
      // `first_achieved` has one slot per target (marginals, then contrast
      // pairs), not one per effect name, so labels come from the targets.
      const r = tab.result as SampleSizeResult;
      const g = r.grid_or_trace[0];
      const labels = [
        ...(g?.target_indices ?? []).map((idx) => labelMap[`target_${idx}`] ?? `target_${idx}`),
        ...(g?.contrast_pairs ?? []).map(([p, n]) => labelMap[`target_${p}_vs_${n}`] ?? `target_${p}_vs_${n}`),
      ];
      const entries = r.first_achieved.map((N0, i) => {
        const name = labels[i] ?? `target ${i + 1}`;
        const N = N0 ?? null;
        if (N === null) return { label: name, fraction: 0, warning: null };
        return { label: name, fraction: reuseFraction(U, N), warning: strictReuseWarning(U, N, ratio) };
      });
      return { show: true, U, entries };
    }
  });
</script>

<div class="space-y-4">
  <!-- What the simulated effects mean on reported scales (absent on
       history-reopened tabs, which predate the values). -->
  {#if tab.standardized}
    <EffectsBlock values={tab.standardized} spec={tab.spec} />
  {/if}

  <!-- Full-width table (always shown) -->
  <TableView {tab} />

  <!-- Scenario selector — only shown when there are scenario:<label> blocks -->
  {#if scenarioChips().length > 0}
    <div data-testid="scenario-selector" class="flex flex-wrap items-center gap-2">
      <span class="text-xs text-muted-foreground">Plot scenario:</span>
      {#each scenarioChips() as chip (chip.key)}
        <button
          type="button"
          class="rounded px-2 py-0.5 text-xs {selectedBlockKey === chip.key
            ? 'bg-primary text-primary-foreground'
            : 'bg-muted text-muted-foreground hover:bg-muted/80'}"
          onclick={() => { selectedBlockKey = chip.key; }}
        >
          {chip.label}
        </button>
      {/each}
      <!-- Overlay chip: shows all scenarios overlaid in one spec -->
      {#if hasOverlayBlock}
        <button
          type="button"
          class="rounded border border-border px-2 py-0.5 text-xs {selectedBlockKey === 'overlay'
            ? 'bg-primary text-primary-foreground'
            : 'text-muted-foreground hover:bg-muted/50'}"
          onclick={() => (selectedBlockKey = 'overlay')}
          title="Overlay all scenarios"
        >
          ⧉ Overlay
        </button>
      {/if}
    </div>
  {/if}

  <!-- Plot: engine-emitted Vega-Lite spec (bar for find-power, curve for find-sample-size),
       wrapped in a captioned figure with real effect names on the axes. -->
  {#if currentSpec()}
    <figure class="space-y-1.5">
      <figcaption class="text-sm font-medium text-foreground">{plotTitle}</figcaption>
      <VegaChart
        spec={currentSpec()!}
        {labelMap}
        testid={tab.kind === 'find-power' ? 'bars-view' : 'curve-view'}
      />
      <p class="text-xs text-muted-foreground">{plotDescription}</p>
    </figure>
  {/if}
  {#if tab.kind === 'find-power'}
    {#each tab.effect_names as name (name)}
      <span data-testid={`bar-${name}`} class="sr-only">{name}</span>
    {/each}
  {/if}

  <!-- Strict-mode reuse diagnostic (shown only when mode="strict" and U>0) -->
  {#if reuseInfo().show}
    <div data-testid="reuse-diagnostic" class="mt-3 rounded-md border border-border bg-muted/40 px-3 py-2 text-xs text-muted-foreground space-y-1">
      <p class="font-medium text-foreground">Bootstrap reuse (strict mode, U={reuseInfo().U} rows)</p>
      {#each reuseInfo().entries as entry (entry.label)}
        <p>
          {entry.label ? `${entry.label}: ` : ''}{entry.fraction.toFixed(1)}% of uploaded rows are expected to be drawn more than once in each simulated dataset.
        </p>
        {#if entry.warning}
          <p class="text-amber-700">{entry.warning}</p>
        {/if}
      {/each}
    </div>
  {/if}

  <!-- Diagnostics footer badge -->
  <ConvergenceNotice {tab} />
</div>
