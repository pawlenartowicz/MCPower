<script lang="ts">
  // "Find power" card — sample-size input for computing power at a fixed n.
  // For a mixed family n is in the unit chosen on SampleSizeUnitSwitch; the
  // echo line restates it in the other unit, as on FindSampleSizeCard. The
  // input's bounds are `findPowerLimits` on total observations, restated in
  // that unit; the adapter refuses an n outside them.
  import { NumberInput } from '$lib/components/ui/number-input';
  import { Label } from '$lib/components/ui/label';
  import InfoIcon from '$lib/components/guidance/InfoIcon.svelte';
  import { familyStore } from '$lib/stores/family.svelte';
  import {
    familyConfigToAppSpec,
    findPowerLimits,
    findPowerSampleSizeError,
  } from '$lib/domain/app-spec-adapter';
  import {
    derivedUnit,
    limitsInUnit,
    unitEchoParts,
    unitNoun,
    type SampleSizeUnit,
  } from '$lib/domain/unit-label';

  const cfg = $derived(familyStore.byFamily[familyStore.active]);
  const cluster = $derived(cfg.family === 'mixed' ? (cfg.cluster ?? null) : null);
  const unit = $derived<SampleSizeUnit>(cfg.findSampleSize.unit ?? 'observations');
  const fixedValue = $derived(
    cluster ? (cluster.dimKind === 'cluster_size' ? cluster.clusterSize : cluster.nClusters) : 0,
  );
  const spec = $derived(
    familyConfigToAppSpec(familyStore.active, cfg, familyStore.regressionOutcome).spec,
  );
  // The adapter drops a unit the cluster dimension does not allow; so do the bounds.
  const bounds = $derived(
    limitsInUnit(
      cluster && unit === derivedUnit(cluster.dimKind) ? unit : 'observations',
      findPowerLimits(spec),
      fixedValue,
    ),
  );
  const nError = $derived(spec ? findPowerSampleSizeError(spec, cfg.findPower.n) : null);
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
        min={bounds?.[0] ?? 1}
        max={bounds?.[1]}
        bind:value={cfg.findPower.n}
      />
    </div>
  </div>
  {#if nError}
    <p class="text-xs text-destructive" data-testid="find-power-n-error">{nError}</p>
  {/if}
  {#if cluster && echoParts.length > 0}
    <p class="text-xs tabular-nums text-muted-foreground" data-testid="power-unit-echo">
      {#each echoParts as part, i (i)}{#if part.mono}<span class="font-mono">{part.text}</span
        >{:else}{part.text}{/if}{/each}
    </p>
  {/if}
</div>
