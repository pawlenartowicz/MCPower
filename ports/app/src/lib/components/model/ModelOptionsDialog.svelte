<script lang="ts">
  // Model "More options" popup — structural outcome-level generation knobs.
  // Magnitudes (λ, heterogeneity, df) live in the per-scenario Scenarios settings.
  //
  // Three controls:
  //   1. Residual distribution: normal by default, or one of the other four shapes,
  //      the same in every scenario ("high_kurtosis" is Student t with 10 df).
  //   2. Heteroskedasticity driver: which predictor (or the linear predictor) drives
  //      residual variance; the ratio λ comes from the active scenario.
  //   3. Variable distribution pool: multi-select of distributions scenarios may
  //      swap any unpinned continuous predictor into; edits all scenarios at once.
  import { Dialog, DialogContent, DialogTitle } from '$lib/components/ui/dialog';
  import {
    Select,
    SelectContent,
    SelectItem,
    SelectTrigger,
  } from '$lib/components/ui/select';
  import { Label } from '$lib/components/ui/label';
  import { familyStore } from '$lib/stores/family.svelte';
  import { parsedFormulaStore } from '$lib/stores/parsed-formula.svelte';
  import { scenariosStore } from '$lib/stores/scenarios.svelte';
  import { defaultOutcomeOptions, type OutcomeOptionsConfig } from '$lib/domain/family';
  import type { NewDistribution } from '$lib/configs/scenarios';
  import { configPaneDialogStyle } from './dialog-position.svelte';
  import InfoIcon from '$lib/components/guidance/InfoIcon.svelte';

  let { open = $bindable(false) }: { open?: boolean } = $props();

  const cfg = $derived(familyStore.byFamily[familyStore.active]);

  // Residual + heteroskedasticity controls are continuous-outcome-only (excludes
  // every non-linear GLM family: logit, probit, and Poisson).
  const continuousOutcome = $derived(familyStore.activeOutcome === 'linear');

  // Lazily attach options so older saved configs stay untouched until the dialog opens.
  $effect(() => {
    if (open && !cfg.outcomeOptions) cfg.outcomeOptions = defaultOutcomeOptions();
  });
  const o = $derived(cfg.outcomeOptions);

  // ---- Residual distribution ----
  const RESIDUALS: Array<{ value: string; label: string; hint?: string }> = [
    { value: 'normal', label: 'normal (default)' },
    { value: 'right_skewed', label: 'right-skewed' },
    { value: 'left_skewed', label: 'left-skewed' },
    { value: 'high_kurtosis', label: 'heavy tails', hint: 'Student t with 10 df' },
    { value: 'uniform', label: 'uniform' },
  ];

  // The select value: the distribution name; null (never chosen) shows as normal.
  const residualSelectValue = $derived(o?.residualDistribution ?? 'normal');
  const residualLabel = $derived(
    RESIDUALS.find((r) => r.value === residualSelectValue)?.label ?? RESIDUALS[0]!.label,
  );

  // Stores the choice as given: null and 'normal' project the same.
  function setResidualDist(v: string) {
    if (!o) return;
    o.residualDistribution = v as OutcomeOptionsConfig['residualDistribution'];
  }

  // ---- Variable distribution pool (new_distributions on all scenarios) ----
  const VAR_DIST_OPTIONS: Array<{ value: NewDistribution; label: string }> = [
    { value: 'right_skewed', label: 'right-skewed' },
    { value: 'left_skewed', label: 'left-skewed' },
    { value: 'high_kurtosis', label: 'heavy tails' },
    { value: 'uniform', label: 'uniform' },
  ];

  // Derive the current pool from the first scenario (all scenarios share the same pool;
  // the ModelOptionsDialog is the single editing surface).
  const varDistPool = $derived<NewDistribution[]>(
    (scenariosStore.scenarios[0]?.new_distributions ?? []) as NewDistribution[],
  );

  function toggleVarDist(dist: NewDistribution) {
    const current = (scenariosStore.scenarios[0]?.new_distributions ?? []) as NewDistribution[];
    const next: NewDistribution[] = current.includes(dist)
      ? current.filter((d) => d !== dist)
      : [...current, dist];
    for (const s of scenariosStore.scenarios) {
      scenariosStore.update(s.name, { new_distributions: next });
    }
  }

  // ---- Heteroskedasticity driver ----
  const continuousPredictors = $derived(
    (parsedFormulaStore.getStable(cfg.formula).result?.predictors ?? []).filter((name) => {
      const v = cfg.variables.find((x) => x.name === name);
      return !v || v.kind === 'continuous';
    }),
  );
  const LP = '__linear_predictor__';
  const driverLabel = $derived(
    o?.heteroskedasticityDriver ? o.heteroskedasticityDriver : 'linear predictor',
  );
</script>

<Dialog bind:open>
  <DialogContent class="max-h-[85vh] overflow-y-auto sm:max-w-md" style={configPaneDialogStyle()}>
    <DialogTitle>Model — more options</DialogTitle>

    {#if o}
      {#if continuousOutcome}
        <div class="space-y-1.5">
          <div class="flex items-center gap-1.5">
            <Label>Residual distribution</Label>
            <InfoIcon tipKey="residualDistribution" />
          </div>
          <Select
            type="single"
            value={residualSelectValue}
            onValueChange={(v: string) => setResidualDist(v)}
          >
            <SelectTrigger class="h-8 w-full text-xs" data-testid="residual-dist">{residualLabel}</SelectTrigger>
            <SelectContent>
              {#each RESIDUALS as r (r.value)}
                <SelectItem value={r.value}>
                  {r.label}{r.hint ? ` — ${r.hint}` : ''}
                </SelectItem>
              {/each}
            </SelectContent>
          </Select>
        </div>

        <div class="space-y-1.5 border-t border-border pt-3">
          <div class="flex items-center gap-1.5">
            <Label>Heteroskedasticity driver</Label>
            <InfoIcon tipKey="heteroskedasticityDriver" />
          </div>
          <p class="text-xs text-muted-foreground">
            Residual variance scales with this predictor; variance ratio λ comes from the active
            scenarios.
          </p>
          <Select
            type="single"
            value={o.heteroskedasticityDriver === '' ? LP : o.heteroskedasticityDriver}
            onValueChange={(v: string) => {
              o.heteroskedasticityDriver = v === LP ? '' : v;
            }}
          >
            <SelectTrigger class="h-8 w-44 text-xs" data-testid="het-driver">{driverLabel}</SelectTrigger>
            <SelectContent>
              <SelectItem value={LP}>linear predictor</SelectItem>
              {#each continuousPredictors as name (name)}
                <SelectItem value={name}>{name}</SelectItem>
              {/each}
            </SelectContent>
          </Select>
        </div>
      {/if}

      <div class="space-y-1.5 {continuousOutcome ? 'border-t border-border pt-3' : ''}">
        <Label>Allowed variable swap distributions</Label>
        <p class="text-xs text-muted-foreground">
          Distributions scenarios may swap unpinned continuous predictors into (applies to all
          scenarios). Pinned predictors are unaffected.
        </p>
        <div class="flex flex-wrap gap-1.5">
          {#each VAR_DIST_OPTIONS as opt (opt.value)}
            {@const active = varDistPool.includes(opt.value)}
            <button
              type="button"
              onclick={() => toggleVarDist(opt.value)}
              class="rounded border px-2.5 py-1 text-xs transition-colors
                {active
                  ? 'border-primary bg-primary/10 text-foreground'
                  : 'border-border text-muted-foreground hover:bg-muted/60'}"
              aria-pressed={active}
            >{opt.label}</button>
          {/each}
        </div>
      </div>
    {/if}
  </DialogContent>
</Dialog>
