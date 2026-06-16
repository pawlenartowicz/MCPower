<script lang="ts">
  // Numeric input for the GLM/GLMM baseline (intercept) probability; rendered only when the
  // active family's baseline target is defined. Mixed stores it on the cluster config (binary
  // GLMM); regression on the family config — bind to whichever the active family owns.
  import { NumberInput } from '$lib/components/ui/number-input';
  import { Label } from '$lib/components/ui/label';
  import InfoIcon from '$lib/components/guidance/InfoIcon.svelte';
  import { familyStore } from '$lib/stores/family.svelte';

  const cfg = $derived(familyStore.byFamily[familyStore.active]);
  const target = $derived(familyStore.active === 'mixed' ? cfg.cluster : cfg);
</script>

{#if target?.baselineProbability !== undefined}
  <div class="space-y-1">
    <div class="flex items-center gap-2">
      <Label for="baseline-prob">Baseline probability</Label>
      <InfoIcon tipKey="baselineProbability" />
    </div>
    <NumberInput
      id="baseline-prob"
      class="w-full"
      step={0.05}
      min={0}
      max={1}
      value={target.baselineProbability}
      oninput={(v) => { if (target) target.baselineProbability = v; }}
    />
  </div>
{/if}
