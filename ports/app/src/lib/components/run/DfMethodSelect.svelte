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
  <Label for="df-method">Denominator df</Label>
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
  <p class="mt-1 text-xs text-muted-foreground">
    Between-within tests each fixed effect against a t rather than a Wald z — between-cluster
    df if the predictor is cluster-level, within-cluster df otherwise. Mark cluster-level
    predictors (cluster settings → Cluster-level variables) for the between df to apply;
    crossed or nested extra groupings fall back to Wald z.
  </p>
</div>
