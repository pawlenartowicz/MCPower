<script lang="ts">
  // Denominator-df selector for mixed models; the default comes from the shared
  // estimation config (see family.ts). Only a value in DF_METHOD_OPTIONS is
  // accepted on change. Mirrors WaldSeSelect.svelte in shape.
  import {
    Select,
    SelectContent,
    SelectItem,
    SelectTrigger,
  } from '$lib/components/ui/select';
  import { Label } from '$lib/components/ui/label';
  import InfoIcon from '$lib/components/guidance/InfoIcon.svelte';
  import { familyStore } from '$lib/stores/family.svelte';
  import { ESTIMATION } from '$lib/configs/app-config';
  import {
    DF_METHOD_LABEL,
    DF_METHOD_OPTIONS,
    type DfMethod,
  } from '$lib/domain/df-method-options';

  const cfg = $derived(familyStore.byFamily[familyStore.active]);
  // Config state persisted before the knob existed carries no df_method.
  const value = $derived(cfg.advanced.df_method ?? ESTIMATION.df_method);
</script>

<div>
  <div class="flex items-center gap-2">
    <Label for="df-method">Denominator df</Label>
    <InfoIcon tipKey="dfMethod" />
  </div>
  <Select
    type="single"
    {value}
    onValueChange={(v: string) => {
      if (DF_METHOD_OPTIONS.includes(v as DfMethod)) {
        cfg.advanced.df_method = v as DfMethod;
      }
    }}
  >
    <SelectTrigger id="df-method">{DF_METHOD_LABEL[value]}</SelectTrigger>
    <SelectContent>
      {#each DF_METHOD_OPTIONS as opt (opt)}
        <SelectItem value={opt}>{DF_METHOD_LABEL[opt]}</SelectItem>
      {/each}
    </SelectContent>
  </Select>
</div>
