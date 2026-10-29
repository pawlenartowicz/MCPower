<script lang="ts">
  // Mixed-only unit switch shared by Find power's n and Find sample's bounds:
  // total observations, or the one clustered quantity the fixed cluster knob
  // leaves free. Renders nothing for families without clusters.
  import { untrack } from 'svelte';
  import { familyStore } from '$lib/stores/family.svelte';
  import { derivedUnit, unitChipParts, type SampleSizeUnit } from '$lib/domain/unit-label';

  const cfg = $derived(familyStore.byFamily[familyStore.active]);
  const cluster = $derived(cfg.family === 'mixed' ? (cfg.cluster ?? null) : null);
  const unit = $derived<SampleSizeUnit>(cfg.findSampleSize.unit ?? 'observations');
  const otherUnit = $derived(cluster ? derivedUnit(cluster.dimKind) : 'observations');

  function setUnit(u: SampleSizeUnit) {
    cfg.findSampleSize.unit = u;
  }

  // A derived unit only means something under the dimension it came from, so
  // switching the Clusters card's dimension drops back to observations. The
  // first run only records the current dimension; mount must not clobber a
  // persisted unit. A family without clusters has no dimension at all: the
  // switch is a singleton, so recording that null would make a round trip
  // through regression look like a dimension change and silently reset it.
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

{#if cluster}
  <div
    role="radiogroup"
    aria-label="Sample-size unit"
    class="flex flex-wrap items-center gap-2 text-sm"
  >
    <span class="text-xs text-muted-foreground">sample size in:</span>
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
