<script lang="ts">
  // Card editor for ANOVA primary-factor rows. Owns two reconciles for the
  // ANOVA family: auto-contrasts (regenAutoContrasts on factor/level changes)
  // and the cfg.effects name reconcile.
  // cfg.effects reconcile mirrors PredictorCards — keep in sync if that $effect changes.
  // Per-factor detail (labels, shares, reference, sampled shares) lives in the card's Advanced dialog.
  import { untrack } from 'svelte';
  import { Button } from '$lib/components/ui/button';
  import Plus from '@lucide/svelte/icons/plus';
  import InfoIcon from '$lib/components/guidance/InfoIcon.svelte';
  import VariableCard from './VariableCard.svelte';
  import { familyStore } from '$lib/stores/family.svelte';
  import { regenAutoContrasts } from '$lib/domain/anova-contrasts';
  import { effectGroups, effectNames, factorLevels, reconcileEffects } from '$lib/domain/effect-names';
  import type { VariableRow } from '$lib/domain/family';

  const cfg = $derived(familyStore.byFamily[familyStore.active]);
  const factors = $derived(cfg.variables.filter((v) => v.role === 'factor'));
  const groups = $derived(effectGroups(cfg));

  function addFactor() {
    // Skip names already taken (incl. covariates) — delete-then-add would
    // otherwise mint a duplicate, and the name-keyed group lookup below
    // ghosts one of the twin cards.
    let n = factors.length + 1;
    while (cfg.variables.some((v) => v.name === `F${n}`)) n++;
    cfg.variables.push({
      name: `F${n}`,
      kind: 'factor',
      role: 'factor',
      nLevels: 2,
      levelProportions: [0.5, 0.5],
    });
  }
  function removeFactor(v: VariableRow) {
    const i = cfg.variables.indexOf(v);
    if (i >= 0) cfg.variables.splice(i, 1);
  }

  // Keep auto-contrasts in sync with the primary-factor set + level labels
  // (factorLevels covers count AND renames — a renamed level changes the
  // effect name, so stale pairs would be rejected by the engine at run time).
  // Depend ONLY on the signature: regenAutoContrasts reads and rewrites
  // cfg.contrasts, so tracking its internals would re-trigger this effect on
  // its own write (effect_update_depth_exceeded). untrack scopes the regen out.
  $effect(() => {
    const sig = factors.map((f) => `${f.name}:${factorLevels(f).join(',')}`).join('|');
    untrack(() => regenAutoContrasts(cfg, sig));
  });

  // Keep cfg.effects aligned with the derived effect names (shared reconcile).
  $effect(() => {
    const names = effectNames(cfg);
    untrack(() => reconcileEffects(cfg, names));
  });
</script>

<div class="space-y-2">
  <div class="flex items-center gap-2">
    <span class="text-sm font-semibold">ANOVA factors</span>
    <InfoIcon tipKey="anovaFactors" />
  </div>
  {#each factors as v (v)}
    <!-- Group lookup is name-keyed: rows renamed to a duplicate share one
         group entry and the loser is hidden by the guard below. -->
    {@const g = groups.variables.find((x) => x.name === v.name)}
    {#if g}
      <VariableCard
        variable={v}
        group={g}
        effects={cfg.effects}
        variables={cfg.variables}
        minLevels={2}
        nameEditable
        fixedKind
        onRemove={() => removeFactor(v)}
      />
    {/if}
  {/each}
  <Button variant="outline" size="sm" onclick={addFactor}>
    <Plus class="mr-1 h-3 w-3" /> Add factor
  </Button>
</div>
