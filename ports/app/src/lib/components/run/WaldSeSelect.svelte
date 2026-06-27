<script lang="ts">
  // Wald SE method selector for GLMM families; the family default is 'hessian'
  // (see family.ts). Only a value in WALD_SE_OPTIONS is accepted on change.
  import {
    Select,
    SelectContent,
    SelectItem,
    SelectTrigger,
  } from '$lib/components/ui/select';
  import { Label } from '$lib/components/ui/label';
  import InfoIcon from '$lib/components/guidance/InfoIcon.svelte';
  import { familyStore } from '$lib/stores/family.svelte';
  import {
    WALD_SE_LABEL,
    WALD_SE_OPTIONS,
    type WaldSe,
  } from '$lib/domain/wald-se-options';

  const cfg = $derived(familyStore.byFamily[familyStore.active]);
</script>

<div>
  <div class="flex items-center gap-2">
    <Label for="wald-se">SE method</Label>
    <InfoIcon tipKey="waldSe" />
  </div>
  <Select
    type="single"
    value={cfg.advanced.wald_se}
    onValueChange={(v: string) => {
      if (WALD_SE_OPTIONS.includes(v as WaldSe)) {
        cfg.advanced.wald_se = v as WaldSe;
      }
    }}
  >
    <SelectTrigger id="wald-se">{WALD_SE_LABEL[cfg.advanced.wald_se]}</SelectTrigger>
    <SelectContent>
      {#each WALD_SE_OPTIONS as opt (opt)}
        <SelectItem value={opt}>{WALD_SE_LABEL[opt]}</SelectItem>
      {/each}
    </SelectContent>
  </Select>
</div>
