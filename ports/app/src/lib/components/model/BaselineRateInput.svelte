<script lang="ts">
  // Numeric input for the Poisson GLM/GLMM baseline (intercept) rate λ; rendered
  // only when the active family's Poisson baseline target is defined. Mixed stores
  // it on the cluster config; regression on the family config — bind to whichever
  // the active family owns. Mirrors BaselineProbabilityInput.
  import { NumberInput } from '$lib/components/ui/number-input';
  import { Label } from '$lib/components/ui/label';
  import InfoIcon from '$lib/components/guidance/InfoIcon.svelte';
  import { familyStore } from '$lib/stores/family.svelte';

  const cfg = $derived(familyStore.byFamily[familyStore.active]);
  const target = $derived(familyStore.active === 'mixed' ? cfg.cluster : cfg);
</script>

{#if target?.baselineRate !== undefined}
  <div class="flex items-center justify-between gap-3">
    <div class="flex items-center gap-2">
      <Label for="baseline-rate">Baseline rate</Label>
      <InfoIcon tipKey="baselineProbability" />
    </div>
    <NumberInput
      id="baseline-rate"
      class="w-32"
      step={0.5}
      min={0}
      value={target.baselineRate}
      oninput={(v) => { if (target) target.baselineRate = v; }}
    />
  </div>
{/if}
