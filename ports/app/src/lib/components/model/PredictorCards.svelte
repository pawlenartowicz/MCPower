<script lang="ts">
  // Card-per-predictor editor for the formula families (regression/mixed) —
  // merges variable type, distribution,
  // and effect sizes into one VariableCard per predictor, then an InteractionCard
  // per interaction term. Both the cfg.variables reconcile (was in VarTypesEditor)
  // and the cfg.effects reconcile (shared helper, also used by AnovaFactorEditor)
  // live here so exactly one copy runs for the non-ANOVA families.
  import { untrack } from 'svelte';
  import { Button } from '$lib/components/ui/button';
  import InfoIcon from '$lib/components/guidance/InfoIcon.svelte';
  import VariableCard from './VariableCard.svelte';
  import InteractionCard from './InteractionCard.svelte';
  import EffectVisualizerDialog from './EffectVisualizerDialog.svelte';
  import { familyStore } from '$lib/stores/family.svelte';
  import { uploadStore } from '$lib/stores/upload.svelte';
  import { parsedFormulaStore } from '$lib/stores/parsed-formula.svelte';
  import { effectGroups, effectNames, reconcileEffects, reconcileTestSelection } from '$lib/domain/effect-names';
  import { uploadedColumnByName } from '$lib/domain/upload-detect';
  import type { VariableRow } from '$lib/domain/family';

  const cfg = $derived(familyStore.byFamily[familyStore.active]);
  // getStable: keeps the last good parse while a formula edit is mid-parse, so
  // the reconciles below never see a transient empty predictor list (which
  // would wipe variable configs and effect sizes on every edit).
  const parsed = $derived(parsedFormulaStore.getStable(cfg.formula).result);
  const groups = $derived(effectGroups(cfg));

  // Keep cfg.variables aligned with the formula's predictors, preserving rows by
  // name. (Relocated from VarTypesEditor — non-ANOVA only.)
  $effect(() => {
    const names = parsed?.predictors ?? [];
    untrack(() => {
      const byName = new Map(cfg.variables.map((v) => [v.name, v]));
      const current = cfg.variables.map((v) => v.name);
      const matches = current.length === names.length && current.every((n, i) => n === names[i]);
      if (matches) return;
      const next: VariableRow[] = names.map(
        (name) => byName.get(name) ?? { name, kind: 'continuous' },
      );
      cfg.variables = next;
    });
  });

  // (#2) Upload-driven type-sync: force each predictor matched to an uploaded
  // column to use the detected type/levels/proportions. Runs after formula-reconcile
  // (#1) so cfg.variables exists, and before effects-reconcile (#3) so factor
  // dummies are derived from the correct levels.
  $effect(() => {
    const colByName = uploadedColumnByName(uploadStore.csvData);
    // Read formula predictors to ensure this re-runs when the formula changes.
    void parsed?.predictors;
    untrack(() => {
      for (const v of cfg.variables) {
        const col = colByName.get(v.name);
        if (!col) continue;
        if (col.col_type === 'factor') {
          const labels = col.labels;
          // Empirical proportions: count code occurrences / total.
          const counts = new Array<number>(labels.length).fill(0);
          for (const code of col.values) {
            if (code >= 0 && code < labels.length) counts[code]!++;
          }
          const total = col.values.length || 1;
          const levelProportions = counts.map((c) => c / total);
          v.kind = 'factor';
          v.levels = labels;
          v.nLevels = labels.length;
          v.levelProportions = levelProportions;
          // referenceLevel left as-is if already set to a valid label; else default to first.
          if (!v.referenceLevel || !labels.includes(v.referenceLevel)) {
            v.referenceLevel = labels[0];
          }
        } else if (col.col_type === 'binary') {
          // Empirical proportion of the "1" category.
          const ones = col.values.filter((x) => x === 1).length;
          const total = col.values.length || 1;
          v.kind = 'binary';
          v.binaryProportion = ones / total;
        } else {
          // continuous — only force kind; leave declared distribution untouched.
          v.kind = 'continuous';
        }
      }
    });
  });

  // (#3) Keep cfg.effects — and the test/contrast selection — aligned with the
  // derived effect names. Pruning stale test names here prevents a formula edit
  // that drops a previously-selected effect (e.g. a no-longer-promoted
  // interaction var) from reaching the adapter as an `unknown effect name`.
  $effect(() => {
    const names = effectNames(cfg);
    untrack(() => {
      reconcileEffects(cfg, names);
      reconcileTestSelection(cfg, names);
    });
  });

  // Locked set: predictor names that match an uploaded column (drives the
  // locked prop on VariableCard). Clears automatically when upload is cleared.
  const lockedNames = $derived(
    new Set(uploadedColumnByName(uploadStore.csvData).keys()),
  );

  // The "Get effects from data" fit flow lives in EffectVisualizerDialog (the
  // dedicated effect-setting surface); this editor only opens that dialog.
  let effectVisualizerOpen = $state(false);
</script>

<div class="space-y-2">
  <div class="flex flex-wrap items-center gap-2">
    <span class="text-sm font-semibold">Predictors</span>
    <InfoIcon tipKey="variableTypes" />
    <InfoIcon tipKey="effects" />
    <Button
      variant="default"
      size="sm"
      class="ml-auto h-7 px-2 text-xs"
      onclick={() => (effectVisualizerOpen = true)}
    >
      Visual effect builder
    </Button>
  </div>
  <EffectVisualizerDialog open={effectVisualizerOpen} onOpenChange={(v) => (effectVisualizerOpen = v)} />

  {#if groups.variables.length === 0}
    <p class="text-xs text-muted-foreground">
      Variables appear here as you add predictors to the formula.
    </p>
  {/if}

  {#each groups.variables as g (g.name)}
    {@const v = cfg.variables.find((x) => x.name === g.name)}
    {#if v}
      <VariableCard variable={v} group={g} effects={cfg.effects} variables={cfg.variables} locked={lockedNames.has(g.name)} />
    {/if}
  {/each}

  {#each groups.interactions as g (g.term)}
    <InteractionCard group={g} effects={cfg.effects} variables={cfg.variables} />
  {/each}
</div>
