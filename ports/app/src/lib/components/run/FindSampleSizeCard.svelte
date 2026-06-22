<script lang="ts">
  // "Find sample size" card — from/to/by grid inputs with an Auto toggle for sample-size search.
  import { NumberInput } from '$lib/components/ui/number-input';
  import { Label } from '$lib/components/ui/label';
  import InfoIcon from '$lib/components/guidance/InfoIcon.svelte';
  import { familyStore } from '$lib/stores/family.svelte';

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
</script>

<div class="space-y-1" data-testid="find-sample-size">
  <div class="flex items-center gap-2">
    <h3 class="text-xs font-medium text-muted-foreground">Find sample</h3>
    <InfoIcon tipKey="sampleSizeSearch" />
  </div>
  <div class="flex items-end gap-2">
    <div>
      <Label for="find-n-from">from</Label>
      <NumberInput
        id="find-n-from"
        class="w-28"
        step={10}
        min={1}
        bind:value={cfg.findSampleSize.from}
      />
    </div>
    <div>
      <Label for="find-n-to">to</Label>
      <NumberInput
        id="find-n-to"
        class="w-28"
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
          class="w-28"
          step={1}
          min={1}
          value={fixedStep}
          oninput={setStep}
        />
      </div>
    {/if}
  </div>
</div>
