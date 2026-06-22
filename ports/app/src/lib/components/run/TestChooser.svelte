<script lang="ts">
  // Test/contrast chooser; selects which coefficients or pairwise contrasts are reported for the active family.
  import { Button } from '$lib/components/ui/button';
  import InfoIcon from '$lib/components/guidance/InfoIcon.svelte';
  import { familyStore } from '$lib/stores/family.svelte';
  import { effectNames, expandMainEffect, factorLevels } from '$lib/domain/effect-names';
  import { REPORT_CONFIG } from '$lib/configs/report-config';
  import type { TestSelection } from '$lib/domain/family';

  const cfg = $derived(familyStore.byFamily[familyStore.active]);

  const selectionKind: TestSelection['kind'] = 'effects';

  // ANOVA factor effects are reported only as pairwise contrasts (the chips
  // above); the per-coefficient list is COVARIATE tests only — built straight
  // from the covariate variables (EffectRowMeta carries no role to filter on).
  const candidates = $derived(
    familyStore.active === 'anova'
      ? cfg.variables
          .filter((v) => v.role === 'covariate')
          .flatMap((v) => expandMainEffect(v, v.name, false))
      : effectNames(cfg),
  );

  const showOmnibus = $derived(
    familyStore.active === 'regression',
  );

  const selectedNames = $derived.by(() => {
    if (cfg.tests.kind === 'all') return new Set(candidates);
    return new Set(cfg.tests.names);
  });

  const perCoeffAll = $derived(
    cfg.tests.kind === 'all' || selectedNames.size === candidates.length,
  );
  const perCoeffNone = $derived(
    cfg.tests.kind !== 'all' && selectedNames.size === 0,
  );
  const allSelected = $derived((!showOmnibus || cfg.reportOverall) && perCoeffAll);
  const noneSelected = $derived(
    (!showOmnibus || !cfg.reportOverall) && perCoeffNone,
  );

  function setSelection(next: TestSelection) {
    cfg.tests = next;
  }

  function toggle(name: string) {
    const current = cfg.tests.kind === 'all' ? new Set(candidates) : new Set(cfg.tests.names);
    if (current.has(name)) current.delete(name);
    else current.add(name);
    if (current.size === candidates.length && candidates.length > 0) {
      setSelection({ kind: 'all' });
    } else {
      setSelection({ kind: selectionKind, names: candidates.filter((n) => current.has(n)) });
    }
  }

  function selectAll() {
    setSelection({ kind: 'all' });
    if (showOmnibus) cfg.reportOverall = true;
  }
  function selectNone() {
    setSelection({ kind: selectionKind, names: [] });
    if (showOmnibus) cfg.reportOverall = false;
  }

  const emptyHint = $derived(
    familyStore.active === 'anova'
      ? 'Add an ANOVA factor with ≥ 2 levels to enable contrasts.'
      : 'Add predictors to the formula to choose which to test.',
  );

  const heading = $derived(familyStore.active === 'anova' ? 'Contrasts' : 'Tests');

  // --- Contrast picker (regression + ANOVA) ---
  const isAnova = $derived(familyStore.active === 'anova');
  // Results render every contrast with "vs"; ANOVA mirrors that in the chips.
  // Regression/logit keep the "−" dash (per user choice).
  const vsToken = REPORT_CONFIG.text.vs_token ?? 'vs';

  /** Factor variables with >= 2 levels (for contrast picker). */
  const factorVars = $derived(
    cfg.variables.filter((v) =>
      v.kind === 'factor' &&
      (familyStore.active !== 'anova' || v.role === 'factor') &&
      factorLevels(v).length >= 2,
    ),
  );
  // ANOVA auto-populates every pairwise contrast from the factor levels, so the
  // manual picker is redundant there (and lets users create duplicates); only
  // regression, which has no auto-contrasts, exposes "+ Add contrast".
  const showAddContrast = $derived(!isAnova && factorVars.length > 0);

  // Section gate. ANOVA's box now holds the covariate-tests expander + the
  // contrasts it drives, so it must show whenever there are factors (even with
  // no covariates) — keyed on factors, not on `candidates` (which is covariates
  // only for ANOVA). Non-ANOVA stays keyed on its candidate effect list.
  const sectionEmpty = $derived(
    isAnova ? factorVars.length === 0 : candidates.length === 0,
  );

  let pickerOpen = $state(false);
  let pickerFactor = $state('');
  let pickerPos = $state('');
  let pickerNeg = $state('');

  /** Level placeholder list for a given factor variable name. */
  function factorLevelNames(factorName: string): string[] {
    const v = cfg.variables.find((x) => x.name === factorName);
    if (!v || v.kind !== 'factor') return [];
    if (Array.isArray(v.levels) && v.levels.length > 0) return v.levels;
    const k = typeof v.nLevels === 'number' && v.nLevels >= 2 ? Math.floor(v.nLevels) : 0;
    return Array.from({ length: k }, (_, i) => `${i + 1}`);
  }

  const pickerLevels = $derived(pickerFactor ? factorLevelNames(pickerFactor) : []);

  function openPicker() {
    pickerFactor = factorVars[0]?.name ?? '';
    pickerPos = '';
    pickerNeg = '';
    pickerOpen = true;
  }

  function closePicker() {
    pickerOpen = false;
  }

  function confirmContrast() {
    if (!pickerFactor || !pickerPos || !pickerNeg || pickerPos === pickerNeg) return;
    const posName = `${pickerFactor}[${pickerPos}]`;
    const negName = `${pickerFactor}[${pickerNeg}]`;
    // Deduplicate: skip if exact pair or reverse already exists.
    const alreadyExists = cfg.contrasts.some(
      (c) =>
        (c.positiveName === posName && c.negativeName === negName) ||
        (c.positiveName === negName && c.negativeName === posName),
    );
    if (!alreadyExists) {
      cfg.contrasts = [...cfg.contrasts, { positiveName: posName, negativeName: negName, enabled: true }];
    }
    closePicker();
  }

  function removeContrast(idx: number) {
    cfg.contrasts = cfg.contrasts.filter((_, i) => i !== idx);
  }

  // --- ANOVA beta section expander ---
  let anovaBetaExpanded = $state(false);
</script>

<div class="space-y-1">
  <div class="flex items-center gap-2">
    <span class="text-sm font-semibold">{heading}</span>
    <InfoIcon tipKey="tests" />
  </div>

  <!-- Custom contrast chips -->
  {#if cfg.contrasts.length > 0}
    <div class="flex flex-wrap gap-1 pb-1">
      {#each cfg.contrasts as c, idx (idx)}
        <span class={`inline-flex items-center gap-1 rounded-full px-2 py-0.5 text-xs font-mono ${c.enabled !== false ? 'bg-accent' : 'bg-muted opacity-60'}`}>
          <input
            type="checkbox"
            class="h-3 w-3 shrink-0 accent-primary"
            checked={c.enabled !== false}
            aria-label={`Include contrast ${c.positiveName} ${isAnova ? vsToken : '−'} ${c.negativeName}`}
            onchange={() => {
              cfg.contrasts = cfg.contrasts.map((entry, i) =>
                i === idx ? { ...entry, enabled: entry.enabled === false } : entry,
              );
            }}
          />
          {c.positiveName} {isAnova ? vsToken : '−'} {c.negativeName}
          <!-- ANOVA auto-populates every pairwise contrast, so removing one is a
               no-op (it regenerates); only the enable/disable checkbox applies.
               Regression keeps × for user-added contrasts. -->
          {#if !isAnova}
            <button
              type="button"
              class="ml-0.5 text-muted-foreground hover:text-foreground"
              aria-label="Remove contrast"
              onclick={() => removeContrast(idx)}
            >
              ×
            </button>
          {/if}
        </span>
      {/each}
    </div>
  {/if}

  <!-- + Add contrast button + inline picker -->
  {#if showAddContrast}
    {#if pickerOpen}
      <div class="rounded-md border border-border/60 p-2 space-y-2">
        <div class="flex items-center gap-2">
          <span class="text-xs font-medium">Add contrast</span>
          <InfoIcon tipKey="contrasts" />
        </div>
        <div class="flex flex-wrap items-center gap-2">
          <select
            class="rounded border border-input bg-background text-foreground px-2 py-1 text-xs"
            value={pickerFactor}
            onchange={(e) => {
              pickerFactor = (e.currentTarget as HTMLSelectElement).value;
              pickerPos = '';
              pickerNeg = '';
            }}
          >
            {#each factorVars as fv (fv.name)}
              <option value={fv.name}>{fv.name}</option>
            {/each}
          </select>
          <select
            class="rounded border border-input bg-background text-foreground px-2 py-1 text-xs"
            value={pickerPos}
            onchange={(e) => { pickerPos = (e.currentTarget as HTMLSelectElement).value; }}
          >
            <option value="" disabled>positive level</option>
            {#each pickerLevels as lv (lv)}
              <option value={lv}>{lv}</option>
            {/each}
          </select>
          <span class="text-xs text-muted-foreground">−</span>
          <select
            class="rounded border border-input bg-background text-foreground px-2 py-1 text-xs"
            value={pickerNeg}
            onchange={(e) => { pickerNeg = (e.currentTarget as HTMLSelectElement).value; }}
          >
            <option value="" disabled>negative level</option>
            {#each pickerLevels.filter((lv) => lv !== pickerPos) as lv (lv)}
              <option value={lv}>{lv}</option>
            {/each}
          </select>
        </div>
        <div class="flex gap-2">
          <Button
            variant="default"
            size="sm"
            class="h-7 px-2 text-xs"
            disabled={!pickerPos || !pickerNeg || pickerPos === pickerNeg}
            onclick={confirmContrast}
          >
            Add
          </Button>
          <Button variant="ghost" size="sm" class="h-7 px-2 text-xs" onclick={closePicker}>
            Cancel
          </Button>
        </div>
      </div>
    {:else}
      <Button variant="outline" size="sm" class="h-7 px-2 text-xs" onclick={openPicker}>
        + Add contrast
      </Button>
    {/if}
  {/if}

  {#if sectionEmpty}
    <p class="text-xs text-muted-foreground">{emptyHint}</p>
  {:else}
    <div class="space-y-1 rounded-md border border-border/60 p-2" role="group" aria-label={heading}>
      {#if showOmnibus}
        <label class="flex cursor-pointer items-center gap-2 px-1 py-0.5 text-sm">
          <input
            type="checkbox"
            class="h-4 w-4 rounded border-input accent-primary"
            checked={cfg.reportOverall}
            onchange={() => (cfg.reportOverall = !cfg.reportOverall)}
          />
          <span class="font-mono">overall</span>
        </label>
      {/if}

      <!-- ANOVA: beta section behind expander; non-ANOVA: always visible -->
      {#if isAnova}
        <button
          type="button"
          class="flex w-full items-center gap-1 px-1 py-0.5 text-xs text-muted-foreground hover:text-foreground"
          onclick={() => (anovaBetaExpanded = !anovaBetaExpanded)}
        >
          <span class="font-mono">{anovaBetaExpanded ? '▾' : '▸'}</span>
          {anovaBetaExpanded ? '− Hide covariate tests' : '+ Show covariate tests'}
        </button>
        {#if anovaBetaExpanded}
          {#if candidates.length === 0}
            <p class="px-1 py-0.5 text-xs text-muted-foreground">Add a covariate to test it.</p>
          {:else}
            {#each candidates as name (name)}
              <label class="flex cursor-pointer items-center gap-2 px-1 py-0.5 text-sm">
                <input
                  type="checkbox"
                  class="h-4 w-4 rounded border-input accent-primary"
                  checked={selectedNames.has(name)}
                  onchange={() => toggle(name)}
                />
                <span class="font-mono">{name}</span>
              </label>
            {/each}
          {/if}
        {/if}
      {:else}
        {#each candidates as name (name)}
          <label class="flex cursor-pointer items-center gap-2 px-1 py-0.5 text-sm">
            <input
              type="checkbox"
              class="h-4 w-4 rounded border-input accent-primary"
              checked={selectedNames.has(name)}
              onchange={() => toggle(name)}
            />
            <span class="font-mono">{name}</span>
          </label>
        {/each}
      {/if}

      <div class="flex gap-2 pt-1">
        <Button
          variant="ghost"
          size="sm"
          class="h-7 px-2 text-xs"
          disabled={allSelected}
          onclick={selectAll}
        >
          select all
        </Button>
        <Button
          variant="ghost"
          size="sm"
          class="h-7 px-2 text-xs"
          disabled={noneSelected}
          onclick={selectNone}
        >
          select none
        </Button>
      </div>
    </div>
  {/if}
</div>
