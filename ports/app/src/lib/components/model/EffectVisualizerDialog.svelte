<script lang="ts">
  // Sizes-only view. The rail lists ONE entry per predictor (then one per
  // interaction term) via effectGroups; the detail pane reuses EffectControls
  // bound to the LIVE cfg.effects proxy entry (so mutations propagate back to
  // FamilyConfig) plus an EffectCartoon illustration. A factor is a single rail
  // entry: its reference level is shown locked ("reference") and each
  // non-reference level gets its own control. Vocabulary: standardized β.
  import {
    Dialog,
    DialogContent,
    DialogHeader,
    DialogTitle,
    DialogDescription,
  } from '$lib/components/ui/dialog';
  import { configPaneDialogStyle } from './dialog-position.svelte';
  import { Button } from '$lib/components/ui/button';
  import EffectControls from '$lib/components/model/EffectControls.svelte';
  import EffectCartoon from '$lib/components/model/EffectCartoon.svelte';
  import BaselineProbabilityInput from '$lib/components/model/BaselineProbabilityInput.svelte';
  import { readChartColors } from '$lib/charts/theme';
  import { familyStore } from '$lib/stores/family.svelte';
  import { uploadStore } from '$lib/stores/upload.svelte';
  import { parsedFormulaStore } from '$lib/stores/parsed-formula.svelte';
  import { effectGroups } from '$lib/domain/effect-names';
  import { logit } from '$lib/domain/effect-cartoon';
  import { familyConfigToAppSpec } from '$lib/domain/app-spec-adapter';
  import { uploadedColumnByName } from '$lib/domain/upload-detect';
  import { getEffectsFromData, type EffectsFromData } from '$lib/api/engine';

  let { open, onOpenChange }: { open: boolean; onOpenChange: (v: boolean) => void } = $props();

  const cfg = $derived(familyStore.byFamily[familyStore.active]);
  const outcomeKind = $derived(familyStore.activeOutcome); // OutcomeKind
  const isBinary = $derived(outcomeKind === 'logit' || outcomeKind === 'probit');
  // EffectCartoon only illustrates binary vs continuous; collapse the four-way
  // OutcomeKind to that pair (Poisson falls through to the continuous cartoon).
  const cartoonOutcome = $derived<'continuous' | 'binary'>(isBinary ? 'binary' : 'continuous');
  // Baseline lives on the cluster config for mixed (binary GLMM), on the family
  // config for regression — mirror BaselineProbabilityInput's resolution.
  const baselineTarget = $derived(familyStore.active === 'mixed' ? cfg.cluster : cfg);
  // The cartoon only illustrates binary vs continuous; probit reuses the logit
  // transform as a visual approximation and Poisson falls through to a 0 intercept.
  const intercept = $derived(
    isBinary ? logit(baselineTarget?.baselineProbability ?? 0.2) : 0,
  );

  const colors = readChartColors();
  const groupColors = $derived([colors.chart1, colors.chart2, colors.chart3, colors.chart4, colors.chart5]);

  const groups = $derived(effectGroups(cfg));
  // Rail keys are stable per variable group / interaction term (v:<name> / i:<term>),
  // not per dummy — a factor collapses to a single entry.
  const rail = $derived([
    ...groups.variables.map((g) => ({ key: `v:${g.name}`, label: g.name, kindTag: g.kind, summary: variableSummary(g) })),
    ...groups.interactions.map((g) => ({ key: `i:${g.term}`, label: g.term, kindTag: 'interaction', summary: interactionSummary(g) })),
  ]);

  let selected = $state<string | null>(null);
  $effect(() => {
    if (open && (selected === null || !rail.some((r) => r.key === selected))) {
      selected = rail[0]?.key ?? null;
    }
  });

  // Live proxy lookups — never copies.
  function effEntry(name: string) {
    return cfg.effects.find((e) => e.name === name);
  }
  function effVal(name: string): number {
    return effEntry(name)?.value ?? 0;
  }
  function kindOf(name: string): 'continuous' | 'binary' | 'factor' {
    return cfg.variables.find((v) => v.name === name)?.kind ?? 'continuous';
  }
  // Level token inside a `var[level]` dummy name.
  function levelOf(name: string): string {
    return name.slice(name.indexOf('[') + 1, name.lastIndexOf(']'));
  }

  function variableSummary(g: (typeof groups.variables)[number]): string {
    if (g.kind === 'factor') {
      return g.rows.filter((r) => !r.isReference).map((r) => effVal(r.name).toFixed(2)).join('/') || '0';
    }
    return effVal(g.rows[0]?.name ?? g.name).toFixed(2);
  }
  function interactionSummary(g: (typeof groups.interactions)[number]): string {
    return g.rows.map((r) => effVal(r.name).toFixed(2)).join('/');
  }

  const selVar = $derived(
    selected?.startsWith('v:') ? groups.variables.find((g) => `v:${g.name}` === selected) : undefined,
  );
  const selInter = $derived(
    selected?.startsWith('i:') ? groups.interactions.find((g) => `i:${g.term}` === selected) : undefined,
  );

  // Grouped detail (binary/factor predictor): reference label + the editable
  // non-reference level rows. Binary has no factor levels, so synthesize a single
  // "reference" group plus the bare effect as its one level.
  const detail = $derived.by(() => {
    if (!selVar || selVar.kind === 'continuous') return null;
    if (selVar.kind === 'binary') {
      const bare = selVar.rows[0]?.name ?? selVar.name;
      return { referenceLabel: 'reference', rows: [{ name: bare, level: selVar.name }] };
    }
    const refRow = selVar.rows.find((r) => r.isReference);
    return {
      referenceLabel: refRow ? levelOf(refRow.name) : 'reference',
      rows: selVar.rows.filter((r) => !r.isReference).map((r) => ({ name: r.name, level: levelOf(r.name) })),
    };
  });
  const levelBetas = $derived(detail ? detail.rows.map((r) => effVal(r.name)) : []);

  // Cartoon β: the bare/continuous value; for a factor the first level (caption only).
  const varBeta = $derived.by(() => {
    if (!selVar) return 0;
    if (selVar.kind === 'factor') return levelBetas[0] ?? 0;
    return effVal(selVar.rows[0]?.name ?? selVar.name);
  });

  // Interaction cartoon params. Looks up the two components' kinds + main effects;
  // for a binary×continuous term partnerMainA carries the continuous main (the
  // slope), binShift the binary main (the group-1 intercept). grid uses
  // partnerMainA/B as the two binary mains.
  const inter = $derived.by(() => {
    if (!selInter) return null;
    const comps = selInter.term.split(':');
    const kindA = kindOf(comps[0] ?? '');
    const kindB = kindOf(comps[1] ?? '');
    const mainA = effVal(comps[0] ?? '');
    const mainB = effVal(comps[1] ?? '');
    const contMain = kindA === 'continuous' ? mainA : mainB;
    const binMain = kindA === 'binary' ? mainA : mainB;
    const bothBinary = kindA === 'binary' && kindB === 'binary';
    return {
      kindA,
      kindB,
      beta: effVal(selInter.rows[0]?.name ?? ''),
      partnerMainA: bothBinary ? mainA : contMain,
      partnerMainB: mainB,
      binShift: binMain,
    };
  });

  // ---------------------------------------------------------------------------
  // Get effects from data — recovers effects from an uploaded dataset for any
  // formula family. The engine's get_effects_from_data dispatches the estimator
  // off the spec family (OLS / GLM / MLE), so the only gate is "a formula family
  // with data loaded" (ANOVA uses AnovaFactorEditor cards, not this dialog).
  // ---------------------------------------------------------------------------
  const parsed = $derived(parsedFormulaStore.getStable(cfg.formula).result);
  let fitError = $state<string | null>(null);
  let fitPending = $state(false);
  // Fit produces a preview, not a write: the user reviews the values and clicks
  // Apply to commit them. Null when no fit has run (or after Apply clears it).
  let fittedPreview = $state<EffectsFromData | null>(null);
  const canFitFromData = $derived(
    (familyStore.active === 'regression' || familyStore.active === 'mixed') &&
      uploadStore.csvData !== null,
  );
  // Estimator-aware caveat shown with the preview. regressionOutcome is
  // ignored for the mixed family (it routes by entrypoint).
  const fitNote = $derived(
    familyStore.active === 'mixed'
      ? 'mixed-model fixed-effect approximations'
      : familyStore.regressionOutcome === 'logit' || familyStore.regressionOutcome === 'probit'
        ? 'binary log-odds approximations'
        : familyStore.regressionOutcome === 'poisson'
          ? 'Poisson log-rate approximations'
          : 'standardized OLS approximations',
  );
  // Non-blocking warning: the outcome the engine will fit is binary, but the
  // uploaded column the user designated isn't detected as binary. col_type is
  // already on the upload column, so this costs nothing.
  const binaryOutcomeWarning = $derived.by(() => {
    if (!canFitFromData || !isBinary) return null;
    const col = uploadedColumnByName(uploadStore.csvData).get(parsed?.dependent ?? '');
    return col && col.col_type !== 'binary'
      ? `Outcome '${parsed!.dependent}' is detected as ${col.col_type}, not binary — the binary fit treats any non-zero value as a "1".`
      : null;
  });

  async function fitFromData() {
    fitError = null;
    fittedPreview = null;
    // Guard: the outcome must be a real uploaded column, else get_effects_from_data
    // fails cryptically. Reuse the same name→column map the type-sync uses.
    const cols = uploadedColumnByName(uploadStore.csvData);
    const outcomeName = parsed?.dependent;
    if (!outcomeName || !cols.has(outcomeName)) {
      fitError = `The outcome '${outcomeName ?? 'y'}' isn't in your uploaded data — set one of your columns as the outcome (the 'y' toggle in the upload panel).`;
      return;
    }
    fitPending = true;
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
      // canFitFromData already gates to regression/mixed; this narrows the union
      // off 'anova' (which has no parsed_formula) for the reduction below.
      if (spec.family === 'anova') return;
      // Per-variable recovery: get_effects_from_data builds its design from
      // parsed_formula + var_types and requires EVERY referenced column to exist,
      // so a partial upload errors ("no matching uploaded column"). Drop the
      // predictors (and any interaction referencing one) that aren't in the upload
      // and fit the recoverable sub-model instead; applyPreview only overwrites the
      // names the fit returns, so dropped predictors keep their current effect.
      const present = (name: string) => cols.has(name);
      const predictors = spec.parsed_formula.predictors.filter(present);
      const kept = new Set(predictors);
      // An effect name references predictors as `a`, `g[level]`, or `a:b`
      // (interaction). Keep an effect only when every referenced predictor
      // survived: assemble_spec validates spec.effects' length against the reduced
      // design, so a stale effect for a dropped predictor errors ("effect count
      // mismatch").
      const effectVars = (name: string) =>
        name.split(':').map((part) => {
          const b = part.indexOf('[');
          return b === -1 ? part : part.slice(0, b);
        });
      const fitSpec = {
        ...spec,
        parsed_formula: {
          ...spec.parsed_formula,
          predictors,
          interaction_terms: spec.parsed_formula.interaction_terms.filter((comps) =>
            comps.every((c) => kept.has(c)),
          ),
        },
        var_types: spec.var_types.filter((vt) => kept.has(vt.name)),
        effects: spec.effects.filter((e) => effectVars(e.name).every((n) => kept.has(n))),
      };
      if (predictors.length === 0) {
        fitError = `None of your predictors match an uploaded column — add a predictor that exists in your data.`;
        return;
      }
      // Store the recovered preview only; nothing is written into cfg until Apply.
      fittedPreview = await getEffectsFromData(fitSpec);
    } catch (err) {
      fitError = err instanceof Error ? err.message : String(err);
      fittedPreview = null;
    } finally {
      fitPending = false;
    }
  }

  // Commit the preview into cfg: fitted effects onto matching effect rows, the
  // estimated ICC onto the cluster card (mixed), and the recovered baseline
  // probability onto whichever knob the active outcome exposes — the cluster
  // card for a mixed binary outcome, the top-level baseline for OLS-logit.
  // Clears the preview so a second click cannot double-apply.
  function applyPreview() {
    const preview = fittedPreview;
    if (!preview) return;
    const byName = new Map(preview.effects.map((e) => [e.name, e.value]));
    cfg.effects = cfg.effects.map((e) => ({
      name: e.name,
      value: byName.has(e.name) ? byName.get(e.name)! : e.value,
    }));
    if (preview.cluster_icc != null && cfg.cluster) {
      cfg.cluster.icc = preview.cluster_icc;
    }
    if (preview.baseline_probability != null) {
      if (familyStore.active === 'mixed' && cfg.cluster) {
        cfg.cluster.baselineProbability = preview.baseline_probability;
      } else {
        cfg.baselineProbability = preview.baseline_probability;
      }
    }
    fittedPreview = null;
  }
</script>

<Dialog {open} {onOpenChange}>
  <!-- sm:max-w-2xl = 42rem = 672px; +32px margin → needs ≥704px of left pane, else center. -->
  <DialogContent class="max-h-[85vh] overflow-y-auto sm:max-w-2xl" style={configPaneDialogStyle(704)}>
    <DialogHeader>
      <DialogTitle>Set effect sizes</DialogTitle>
      <DialogDescription>Standardized β for each term, with an illustration of its magnitude.</DialogDescription>
    </DialogHeader>

    <div class="grid grid-cols-[12rem_1fr] gap-3">
      <nav class="space-y-1 border-r border-[var(--border)] pr-2" aria-label="effect terms">
        {#each rail as r (r.key)}
          <button
            type="button"
            class="block w-full rounded px-2 py-1 text-left text-sm {selected === r.key ? 'bg-[var(--primary)] text-[var(--primary-foreground)]' : 'hover:bg-[var(--muted)]'}"
            onclick={() => (selected = r.key)}>
            <span class="font-medium">{r.label}</span>
            <span class="float-right text-[var(--muted-foreground)]">{r.summary}</span>
            <span class="block text-xs opacity-70">{r.kindTag}</span>
          </button>
        {/each}
        {#if rail.length === 0}
          <p class="p-2 text-sm text-[var(--muted-foreground)]">No effects yet — define a formula first.</p>
        {/if}
      </nav>

      <div class="space-y-3">
        {#if isBinary}
          <!-- Model-level baseline (intercept) — shapes every binary illustration,
               so it sits once above the per-term controls and writes the live config. -->
          <div class="rounded-md border border-[var(--border)] bg-[var(--muted)]/40 p-2">
            <BaselineProbabilityInput />
          </div>
        {/if}
        {#if selVar && selVar.kind === 'continuous'}
          {@const eff = effEntry(selVar.rows[0]?.name ?? selVar.name)}
          <EffectCartoon outcomeKind={cartoonOutcome} predictorKind="continuous" role="main" beta={varBeta} {intercept} />
          {#if eff}
            <EffectControls effect={eff} variables={cfg.variables} />
          {/if}
        {:else if selVar && detail}
          <EffectCartoon
            outcomeKind={cartoonOutcome}
            predictorKind={selVar.kind}
            role="main"
            beta={varBeta}
            {levelBetas}
            {intercept} />
          <div class="text-xs text-[var(--muted-foreground)]">
            Effect of each level vs. <strong class="text-[var(--foreground)]">{detail.referenceLabel}</strong>
            (in {isBinary ? 'log-odds' : 'SD'} units)
          </div>
          <!-- reference row: shown but not editable -->
          <div class="flex items-center gap-2 text-sm">
            <span class="inline-block h-3 w-3 rounded-sm" style="background:{groupColors[0]}"></span>
            <span class="font-medium">{detail.referenceLabel}</span>
            <span class="text-[var(--muted-foreground)]">reference</span>
          </div>
          {#each detail.rows as lr, i (lr.name)}
            {@const eff = effEntry(lr.name)}
            <div class="flex items-center gap-2">
              <span class="inline-block h-3 w-3 shrink-0 rounded-sm" style="background:{groupColors[(i + 1) % groupColors.length]}"></span>
              <span class="w-20 shrink-0 truncate text-sm font-medium" title={lr.level}>{lr.level}</span>
              {#if eff}
                <EffectControls effect={eff} variables={cfg.variables} />
              {/if}
            </div>
          {/each}
        {:else if selInter && inter}
          <EffectCartoon
            outcomeKind={cartoonOutcome}
            predictorKind={inter.kindA}
            partnerKind={inter.kindB}
            role="interaction"
            beta={inter.beta}
            partnerMainA={inter.partnerMainA}
            partnerMainB={inter.partnerMainB}
            binShift={inter.binShift}
            {intercept} />
          {#each selInter.rows as ir (ir.name)}
            {@const eff = effEntry(ir.name)}
            <div class="flex items-center gap-2">
              {#if selInter.rows.length > 1}
                <span class="w-28 shrink-0 truncate text-sm font-medium" title={ir.name}>{ir.name}</span>
              {/if}
              {#if eff}
                <EffectControls effect={eff} variables={cfg.variables} />
              {/if}
            </div>
          {/each}
        {/if}

        {#if canFitFromData}
          <!-- Get effects from data: fit the uploaded rows to seed β / baseline /
               ICC. Sits at the bottom, beside the per-term value inputs above.
               Preview-then-Apply so nothing is written until the user commits. -->
          <div class="space-y-2 border-t border-[var(--border)] pt-3">
            <Button
              variant="default"
              size="sm"
              class="h-7 px-2 text-xs"
              disabled={fitPending}
              onclick={fitFromData}
            >
              {fitPending ? 'Fitting…' : 'Get effects from data'}
            </Button>
            {#if binaryOutcomeWarning}
              <p class="text-xs text-[var(--muted-foreground)]">{binaryOutcomeWarning}</p>
            {/if}
            {#if fitError}
              <p class="text-xs text-destructive">{fitError}</p>
            {/if}
            {#if fittedPreview && !fitError && !fitPending}
              <div class="space-y-1 rounded-md border border-[var(--border)] bg-[var(--muted)]/30 p-2">
                <p class="text-xs italic text-[var(--muted-foreground)]">
                  Note: fitted effects are {fitNote} from the uploaded data — useful as a
                  starting point, not exact power targets. Nothing changes until you Apply.
                </p>
                {#if fittedPreview.effects.length > 0}
                  <ul class="text-xs font-mono text-[var(--foreground)]">
                    {#each fittedPreview.effects as e (e.name)}
                      <li>{e.name} = {e.value.toFixed(4)}</li>
                    {/each}
                  </ul>
                {/if}
                {#if fittedPreview.cluster_icc != null}
                  <p class="text-xs text-[var(--foreground)]">
                    Estimated ICC: {fittedPreview.cluster_icc.toFixed(4)} — approximation.
                  </p>
                {/if}
                {#if fittedPreview.baseline_probability != null}
                  <p class="text-xs text-[var(--foreground)]">
                    Baseline probability: {fittedPreview.baseline_probability.toFixed(4)} — approximation.
                  </p>
                {/if}
                <Button variant="default" size="sm" class="h-7 px-2 text-xs" onclick={applyPreview}>
                  Apply
                </Button>
              </div>
            {/if}
          </div>
        {/if}
      </div>
    </div>
  </DialogContent>
</Dialog>
