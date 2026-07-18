<script lang="ts">
  // Shared run parameters (target power, alpha, correction method, test selection,
  // test-formula override) used by all analysis families.
  import { Input } from '$lib/components/ui/input';
  import { NumberInput } from '$lib/components/ui/number-input';
  import { Label } from '$lib/components/ui/label';
  import InfoIcon from '$lib/components/guidance/InfoIcon.svelte';
  import { familyStore } from '$lib/stores/family.svelte';
  import CorrectionSelect from './CorrectionSelect.svelte';
  import EstimationSelect from './EstimationSelect.svelte';
  import TestChooser from './TestChooser.svelte';

  const cfg = $derived(familyStore.byFamily[familyStore.active]);
  // Show the estimation control (Fast/Accurate/AGQ) ONLY for a GLMM — the mixed
  // family with a non-Gaussian outcome (logit, probit, or Poisson). OLS, Gaussian
  // LME, and unclustered GLMs have exact/fixed SEs and no random effect to
  // integrate, so the knob would read as a control that does nothing.
  const showEstimation = $derived(
    familyStore.active === 'mixed' && familyStore.activeOutcome !== 'linear',
  );
</script>

<div class="space-y-3">
  <div class="flex flex-wrap items-end gap-3">
    <div>
      <div class="flex items-center gap-2">
        <Label for="target-power">Target power</Label>
        <InfoIcon tipKey="targetPower" />
      </div>
      <NumberInput
        id="target-power"
        class="w-32"
        step={1}
        min={0}
        max={100}
        bind:value={cfg.targetPower}
      />
    </div>
    <div>
      <Label for="alpha">Alpha</Label>
      <NumberInput
        id="alpha"
        class="w-28"
        step={0.01}
        min={0}
        max={1}
        bind:value={cfg.alpha}
      />
    </div>
    <div class="min-w-40">
      <CorrectionSelect />
    </div>
    {#if showEstimation}
      <div class="min-w-40">
        <EstimationSelect />
      </div>
    {/if}
  </div>
  <TestChooser />
  <!-- Same format as the model Formula field: label above, full-width input. -->
  <div class="space-y-1">
    <div class="flex items-center gap-2">
      <Label for="test-formula" class="font-semibold">Test formula override</Label>
      <InfoIcon tipKey="testFormulaOverride" />
    </div>
    <Input
      id="test-formula"
      type="text"
      placeholder="(optional) restrict which effects are tested — e.g. y = x1 + x2"
      bind:value={cfg.advanced.testFormulaOverride}
    />
  </div>
</div>
