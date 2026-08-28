<script lang="ts">
  // "Find sample size" card — from/to/by grid inputs with an Auto toggle for
  // sample-size search, plus (mixed only) the unit the bounds are given in:
  // total observations, or the one clustered quantity the fixed cluster knob
  // leaves free. The echo line restates the same range in the other unit.
  import { untrack } from 'svelte';
  import { NumberInput } from '$lib/components/ui/number-input';
  import { Label } from '$lib/components/ui/label';
  import InfoIcon from '$lib/components/guidance/InfoIcon.svelte';
  import { familyStore } from '$lib/stores/family.svelte';
  import {
    derivedUnit,
    unitChipParts,
    unitEchoParts,
    type SampleSizeUnit,
  } from '$lib/domain/unit-label';

  const cfg = $derived(familyStore.byFamily[familyStore.active]);
  const auto = $derived(cfg.findSampleSize.by === 'auto');
  let fixedStep = $state(10);

  function toggleAuto(on: boolean) {
    cfg.findSampleSize.by = on ? 'auto' : fixedStep;
  }
  function setStep(v: number) {
    fixedStep = v;
    if (!auto) cfg.findSampleSize.by = v;
  }

  // Chips exist only for a mixed family with a cluster to name: every other
  // family searches in observations and the card is unchanged.
  const cluster = $derived(cfg.family === 'mixed' ? (cfg.cluster ?? null) : null);
  const unit = $derived<SampleSizeUnit>(cfg.findSampleSize.unit ?? 'observations');
  const otherUnit = $derived(cluster ? derivedUnit(cluster.dimKind) : 'observations');
  // The card value the echo converts through: observations per cluster under a
  // fixed size, number of clusters under a fixed count.
  const fixedValue = $derived(
    cluster ? (cluster.dimKind === 'cluster_size' ? cluster.clusterSize : cluster.nClusters) : 0,
  );
  const echoParts = $derived(
    cluster
      ? unitEchoParts(
          unit,
          cluster.dimKind,
          cluster.clusterName,
          cfg.findSampleSize.from,
          cfg.findSampleSize.to,
          fixedValue,
        )
      : [],
  );

  function setUnit(u: SampleSizeUnit) {
    cfg.findSampleSize.unit = u;
  }

  // A derived unit only means something under the dimension it came from, so
  // switching the Clusters card's dimension drops back to observations. The
  // first run only records the current dimension — mount must not clobber a
  // persisted unit. A family without clusters has no dimension at all: the card
  // is a singleton, so recording that null would make a round trip through
  // regression look like a dimension change and silently reset the chip.
  let seenDim = $state<string | null>(null);
  $effect(() => {
    const dim = cluster?.dimKind ?? null;
    untrack(() => {
      if (dim === null) return;
      if (seenDim !== null && dim !== seenDim) cfg.findSampleSize.unit = 'observations';
      seenDim = dim;
    });
  });
</script>

<div class="space-y-1" data-testid="find-sample-size">
  <div class="flex items-center gap-2">
    <h3 class="text-xs font-medium text-muted-foreground">Find sample</h3>
    <InfoIcon tipKey="sampleSizeSearch" />
  </div>
  <div class="flex flex-wrap items-end gap-2">
    <div>
      <Label for="find-n-from">from</Label>
      <NumberInput
        id="find-n-from"
        class="w-28 shrink-0"
        step={10}
        min={1}
        bind:value={cfg.findSampleSize.from}
      />
    </div>
    <div>
      <Label for="find-n-to">to</Label>
      <NumberInput
        id="find-n-to"
        class="w-28 shrink-0"
        step={10}
        min={1}
        bind:value={cfg.findSampleSize.to}
      />
    </div>
    <!-- h-9 matches NumberInput height so the checkbox centers on the input row -->
    <div class="flex h-9 items-center gap-1">
      <input
        id="find-n-auto"
        type="checkbox"
        class="h-4 w-4 rounded border-input accent-primary"
        checked={auto}
        onchange={(e) => toggleAuto((e.currentTarget as HTMLInputElement).checked)}
      />
      <Label for="find-n-auto">auto step</Label>
    </div>
    {#if !auto}
      <div>
        <Label for="find-n-step">step</Label>
        <NumberInput
          id="find-n-step"
          class="w-28 shrink-0"
          step={1}
          min={1}
          value={fixedStep}
          oninput={setStep}
        />
      </div>
    {/if}
  </div>
  {#if cluster}
    <!-- Own row: the input row above already fills the config pane at its 240px floor. -->
    <div
      role="radiogroup"
      aria-label="Sample-size unit"
      class="flex flex-wrap items-center gap-2 text-sm"
    >
      <span class="text-xs text-muted-foreground">search in:</span>
      <button
        type="button"
        role="radio"
        aria-checked={unit === 'observations'}
        data-testid="unit-observations"
        class="rounded px-3 py-1 {unit === 'observations'
          ? 'bg-primary text-primary-foreground'
          : 'bg-muted text-muted-foreground hover:bg-muted/70'}"
        onclick={() => setUnit('observations')}>observations</button
      >
      <button
        type="button"
        role="radio"
        aria-checked={unit === otherUnit}
        data-testid="unit-derived"
        class="rounded px-3 py-1 {unit === otherUnit
          ? 'bg-primary text-primary-foreground'
          : 'bg-muted text-muted-foreground hover:bg-muted/70'}"
        onclick={() => setUnit(otherUnit)}
        >{#each unitChipParts(otherUnit) as part, i (i)}{part.text}{/each}</button
      >
    </div>
  {/if}
  {#if cluster && echoParts.length > 0}
    <p class="text-xs tabular-nums text-muted-foreground" data-testid="unit-echo">
      {#each echoParts as part, i (i)}{#if part.mono}<span class="font-mono">{part.text}</span
        >{:else}{part.text}{/if}{/each}
    </p>
  {/if}
</div>
