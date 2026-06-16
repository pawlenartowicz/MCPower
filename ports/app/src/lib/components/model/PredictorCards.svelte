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
  import { familyStore } from '$lib/stores/family.svelte';
  import { uploadStore } from '$lib/stores/upload.svelte';
  import { parsedFormulaStore } from '$lib/stores/parsed-formula.svelte';
  import { effectGroups, effectNames, reconcileEffects } from '$lib/domain/effect-names';
  import { familyConfigToAppSpec } from '$lib/domain/app-spec-adapter';
  import { getEffectsFromData } from '$lib/api/engine';
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

  // (#3) Keep cfg.effects aligned with the derived effect names (shared reconcile).
  $effect(() => {
    const names = effectNames(cfg);
    untrack(() => reconcileEffects(cfg, names));
  });

  // Locked set: predictor names that match an uploaded column (drives the
  // locked prop on VariableCard). Clears automatically when upload is cleared.
  const lockedNames = $derived(
    new Set(uploadedColumnByName(uploadStore.csvData).keys()),
  );

  // ---------------------------------------------------------------------------
  // Fit from data — recovers effects from an uploaded dataset for any formula
  // family. The engine's get_effects_from_data dispatches the estimator off the
  // spec family (OLS / GLM / MLE), so the only gate is "a formula family with
  // data loaded" (ANOVA uses AnovaFactorEditor cards, not this editor).
  // ---------------------------------------------------------------------------
  let fitError = $state<string | null>(null);
  let fitPending = $state(false);
  let fitSucceeded = $state(false);
  const canFitFromData = $derived(
    (familyStore.active === 'regression' || familyStore.active === 'mixed') &&
      uploadStore.csvData !== null,
  );
  // Estimator-aware caveat shown after a successful fit. regressionOutcome is
  // ignored for the mixed family (it routes by entrypoint).
  const fitNote = $derived(
    familyStore.active === 'mixed'
      ? 'mixed-model fixed-effect approximations'
      : familyStore.regressionOutcome === 'binary'
        ? 'logistic log-odds approximations'
        : 'standardized OLS approximations',
  );

  async function fitFromData() {
    fitError = null;
    fitPending = true;
    fitSucceeded = false;
    try {
      const { spec, errors } = familyConfigToAppSpec(
        familyStore.active,
        cfg,
        familyStore.regressionOutcome,
      );
      if (!spec || errors.length > 0) {
        fitError = errors[0] ?? 'Spec is not ready — fix formula errors first.';
        return;
      }
      const fitted = await getEffectsFromData(spec);
      untrack(() => {
        const byName = new Map(fitted.map((e) => [e.name, e.value]));
        cfg.effects = cfg.effects.map((e) => ({
          name: e.name,
          value: byName.has(e.name) ? byName.get(e.name)! : e.value,
        }));
      });
      fitSucceeded = true;
    } catch (err) {
      fitError = err instanceof Error ? err.message : String(err);
      fitSucceeded = false;
    } finally {
      fitPending = false;
    }
  }
</script>

<div class="space-y-2">
  <div class="flex flex-wrap items-center gap-2">
    <span class="text-sm font-semibold">Predictors</span>
    <InfoIcon tipKey="variableTypes" />
    <InfoIcon tipKey="effects" />
    {#if canFitFromData}
      <Button
        variant="outline"
        size="sm"
        class="h-7 px-2 text-xs"
        disabled={fitPending}
        onclick={fitFromData}
      >
        {fitPending ? 'Fitting…' : 'Fit from data'}
      </Button>
    {/if}
  </div>
  {#if fitError}
    <p class="text-xs text-destructive">{fitError}</p>
  {/if}
  {#if fitSucceeded && !fitError && !fitPending}
    <p class="text-xs italic text-muted-foreground">
      Note: fitted effects are {fitNote} from the uploaded data — useful as a
      starting point, not exact power targets.
    </p>
  {/if}

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
