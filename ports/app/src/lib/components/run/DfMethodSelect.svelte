<script lang="ts">
  // Denominator-df selector for mixed models; the default comes from the shared
  // estimation config (see family.ts). It offers only the values defined on the
  // current design (outcome kind, crossed/nested groupings); a stored value the
  // design does not admit stays shown and is reported by the spec adapter.
  // Mirrors WaldSeSelect.svelte in shape.
  import {
    Select,
    SelectContent,
    SelectItem,
    SelectTrigger,
  } from '$lib/components/ui/select';
  import { Label } from '$lib/components/ui/label';
  import InfoIcon from '$lib/components/guidance/InfoIcon.svelte';
  import { familyStore } from '$lib/stores/family.svelte';
  import { parsedFormulaStore, toClusterTerms } from '$lib/stores/parsed-formula.svelte';
  import { ESTIMATION } from '$lib/configs/app-config';
  import {
    DF_METHOD_LABEL,
    dfMethodOptionsFor,
    type DfMethod,
  } from '$lib/domain/df-method-options';

  const cfg = $derived(familyStore.byFamily[familyStore.active]);
  // Absent = the shared default; the default itself is never stored.
  const value = $derived(cfg.advanced.df_method ?? ESTIMATION.df_method);
  // Grouping structure from the parsed formula, the source the spec adapter
  // reads (getStable: no flicker during the parse round-trip after an edit).
  const singleGrouping = $derived.by(() => {
    const result = parsedFormulaStore.getStable(cfg.formula).result;
    return (result ? toClusterTerms(result) : []).length <= 1;
  });
  const options = $derived(
    dfMethodOptionsFor(familyStore.activeOutcome === 'linear', singleGrouping),
  );
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
      if (options.includes(v as DfMethod)) {
        // Picking the default clears the stored value rather than saving it.
        if (v === ESTIMATION.df_method) delete cfg.advanced.df_method;
        else cfg.advanced.df_method = v as DfMethod;
      }
    }}
  >
    <SelectTrigger id="df-method">{DF_METHOD_LABEL[value]}</SelectTrigger>
    <SelectContent>
      {#each options as opt (opt)}
        <SelectItem value={opt}>{DF_METHOD_LABEL[opt]}</SelectItem>
      {/each}
    </SelectContent>
  </Select>
</div>
