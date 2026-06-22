<script lang="ts">
  // Card editor for ANOVA covariate rows (continuous, binary, or factor).
  // Kind-specific state defaults and the Advanced dialog come with VariableCard.
  import { Button } from '$lib/components/ui/button';
  import Plus from '@lucide/svelte/icons/plus';
  import VariableCard from './VariableCard.svelte';
  import { familyStore } from '$lib/stores/family.svelte';
  import { effectGroups } from '$lib/domain/effect-names';
  import type { VariableRow } from '$lib/domain/family';

  const cfg = $derived(familyStore.byFamily[familyStore.active]);
  const covariates = $derived(cfg.variables.filter((v) => v.role === 'covariate'));
  const groups = $derived(effectGroups(cfg));

  function addCovariate() {
    // Skip taken names — mirrors addFactor in AnovaFactorEditor (duplicate
    // names ghost a card via the name-keyed group lookup).
    let n = covariates.length + 1;
    while (cfg.variables.some((v) => v.name === `cov${n}`)) n++;
    cfg.variables.push({ name: `cov${n}`, kind: 'continuous', role: 'covariate' });
  }
  function removeCovariate(v: VariableRow) {
    const i = cfg.variables.indexOf(v);
    if (i >= 0) cfg.variables.splice(i, 1);
  }
</script>

<div class="space-y-2">
  <span class="text-sm font-semibold">Covariates</span>
  {#each covariates as v (v)}
    {@const g = groups.variables.find((x) => x.name === v.name)}
    {#if g}
      <VariableCard
        variable={v}
        group={g}
        effects={cfg.effects}
        variables={cfg.variables}
        minLevels={2}
        nameEditable
        onRemove={() => removeCovariate(v)}
      />
    {/if}
  {/each}
  <Button variant="outline" size="sm" onclick={addCovariate}>
    <Plus class="mr-1 h-3 w-3" /> Add covariate
  </Button>
</div>
