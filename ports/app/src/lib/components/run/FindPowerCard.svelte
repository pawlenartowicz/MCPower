<script lang="ts">
  // "Find power" card — sample-size input for computing power at a fixed n.
  // For a mixed family n is in the unit chosen on SampleSizeUnitSwitch; the
  // echo line restates it in the other unit, as on FindSampleSizeCard.
  import { NumberInput } from '$lib/components/ui/number-input';
  import { Label } from '$lib/components/ui/label';
  import InfoIcon from '$lib/components/guidance/InfoIcon.svelte';
  import { familyStore } from '$lib/stores/family.svelte';
  import { unitEchoParts, unitNoun, type SampleSizeUnit } from '$lib/domain/unit-label';

  const cfg = $derived(familyStore.byFamily[familyStore.active]);
  const cluster = $derived(cfg.family === 'mixed' ? (cfg.cluster ?? null) : null);
  const unit = $derived<SampleSizeUnit>(cfg.findSampleSize.unit ?? 'observations');
  const fixedValue = $derived(
    cluster ? (cluster.dimKind === 'cluster_size' ? cluster.clusterSize : cluster.nClusters) : 0,
  );
  const echoParts = $derived(
    cluster
      ? unitEchoParts(
          unit,
          cluster.dimKind,
          cluster.clusterName,
          cfg.findPower.n,
          cfg.findPower.n,
          fixedValue,
        )
      : [],
  );
</script>

<div class="space-y-1">
  <div class="flex items-center gap-2">
    <h3 class="text-xs font-medium text-muted-foreground">Find power</h3>
    <InfoIcon tipKey="findPowerN" />
  </div>
  <div class="flex items-end gap-2">
    <div>
      <Label for="find-power-n"
        >n{#if cluster}&nbsp;({unitNoun(unit, cluster.clusterName)}){/if}</Label
      >
      <NumberInput
        id="find-power-n"
        class="w-32"
        step={10}
        min={1}
        bind:value={cfg.findPower.n}
      />
    </div>
  </div>
  {#if cluster && echoParts.length > 0}
    <p class="text-xs tabular-nums text-muted-foreground" data-testid="power-unit-echo">
      {#each echoParts as part, i (i)}{#if part.mono}<span class="font-mono">{part.text}</span
        >{:else}{part.text}{/if}{/each}
    </p>
  {/if}
</div>
