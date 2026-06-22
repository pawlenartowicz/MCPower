<script lang="ts">
  // "Find power" card — sample-size input for computing power at a fixed n.
  import { NumberInput } from '$lib/components/ui/number-input';
  import { Label } from '$lib/components/ui/label';
  import InfoIcon from '$lib/components/guidance/InfoIcon.svelte';
  import { familyStore } from '$lib/stores/family.svelte';

  const cfg = $derived(familyStore.byFamily[familyStore.active]);
</script>

<div class="space-y-1">
  <div class="flex items-center gap-2">
    <h3 class="text-xs font-medium text-muted-foreground">Find power</h3>
    <InfoIcon tipKey="findPowerN" />
  </div>
  <div class="flex items-end gap-2">
    <div>
      <Label for="find-power-n">n</Label>
      <NumberInput
        id="find-power-n"
        class="w-32"
        step={10}
        min={1}
        bind:value={cfg.findPower.n}
      />
    </div>
  </div>
</div>
