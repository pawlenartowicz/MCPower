<script lang="ts">
  // Authoring view over FamilyConfig — NOT a re-implementation of VariableCard.
  // It writes ONLY cfg.formula on "Use this model"; PredictorCards' reconcile
  // ($effect over cfg.formula) does the variables/effects resync. On open it
  // hydrates rows from the SYNC getStable() parse (no await), and detects any
  // ≥3-way term so it is carried through, never dropped.
  import {
    Dialog,
    DialogContent,
    DialogHeader,
    DialogTitle,
    DialogDescription,
    DialogFooter,
  } from '$lib/components/ui/dialog';
  import { Button } from '$lib/components/ui/button';
  import { configPaneDialogStyle } from './dialog-position.svelte';
  import { familyStore } from '$lib/stores/family.svelte';
  import { parsedFormulaStore } from '$lib/stores/parsed-formula.svelte';
  import { uploadStore } from '$lib/stores/upload.svelte';
  import { uploadedColumnByName } from '$lib/domain/upload-detect';
  import type { UploadColumn } from '$lib/domain/app-spec';
  import {
    hydrateBuilder,
    assembleFormula,
    applyBuilderVariables,
    validatePredictorName,
    FACTOR_DEFAULT_LEVELS,
    type BuilderState,
    type PredictorKind,
  } from '$lib/domain/model-builder';

  let { open, onOpenChange }: { open: boolean; onOpenChange: (v: boolean) => void } = $props();

  const cfg = $derived(familyStore.byFamily[familyStore.active]);

  let state = $state<BuilderState>({ dependent: 'y', predictors: [], interactions: [], carried: [] });

  // Hydrate every time the dialog opens (sync parse; may be slightly stale — the
  // existing getStable contract, fine here). Overlay dataBacked from the upload.
  $effect(() => {
    if (!open) return;
    const parse = parsedFormulaStore.getStable(cfg.formula).result;
    const next = hydrateBuilder(parse, cfg.variables);
    const cols = uploadedColumnByName(uploadStore.csvData);
    for (const p of next.predictors) p.dataBacked = cols.has(p.name);
    state = next;
  });

  const formulaPreview = $derived(assembleFormula(state));

  function nameError(p: { name: string }): string | null {
    return validatePredictorName(
      p.name,
      state.predictors.filter((q) => q !== p).map((q) => q.name),
    );
  }
  const hasErrors = $derived(state.predictors.some((p) => nameError(p) !== null));

  function addPredictor() {
    state.predictors = [...state.predictors, { name: '', kind: 'continuous', levels: [], dataBacked: false }];
  }

  // Uploaded columns not yet used (as the dependent or a predictor) — offered as
  // one-click chips so an uploaded column lands as a data-backed predictor
  // pre-typed from its detection (factor levels carried over), no retyping.
  const usedNames = $derived(new Set([state.dependent, ...state.predictors.map((p) => p.name)]));
  const uploadCols = $derived(
    (uploadStore.csvData?.columns ?? []).filter((c) => !usedNames.has(c.name)),
  );
  function addUploadColumn(col: UploadColumn) {
    state.predictors = [
      ...state.predictors,
      {
        name: col.name,
        kind: col.col_type,
        levels: col.col_type === 'factor' ? [...col.labels] : [],
        dataBacked: true,
      },
    ];
  }
  function removePredictor(p: unknown) {
    state.predictors = state.predictors.filter((q) => q !== p);
    state.interactions = state.interactions.filter(
      (i) => state.predictors.some((q) => q.name === i.a) && state.predictors.some((q) => q.name === i.b),
    );
  }
  function setKind(p: { kind: PredictorKind; levels: string[] }, kind: PredictorKind) {
    p.kind = kind;
    if (kind === 'factor' && p.levels.length === 0) p.levels = [...FACTOR_DEFAULT_LEVELS];
    if (kind !== 'factor') p.levels = [];
  }
  function addInteraction() {
    const names = state.predictors.map((p) => p.name).filter(Boolean);
    if (names.length >= 2) state.interactions = [...state.interactions, { a: names[0]!, b: names[1]! }];
  }
  function removeInteraction(i: unknown) {
    state.interactions = state.interactions.filter((x) => x !== i);
  }

  function useModel() {
    if (hasErrors) return;
    // Write variables BEFORE the formula: PredictorCards' reconcile preserves rows
    // by name, so the kinds/levels we set here survive and the effect dummies derive
    // from the right factor levels — even from a blank-formula start (where the async
    // parse would otherwise default every variable to continuous).
    cfg.variables = applyBuilderVariables(state, cfg.variables);
    cfg.formula = assembleFormula(state);
    onOpenChange(false);
  }
</script>

<Dialog {open} {onOpenChange}>
  <DialogContent class="max-h-[85vh] overflow-y-auto sm:max-w-lg" style={configPaneDialogStyle()}>
    <DialogHeader>
      <DialogTitle>Build model visually</DialogTitle>
      <DialogDescription>Assemble the formula by clicking. Edits sync with the text box.</DialogDescription>
    </DialogHeader>

    {#if state.carried.length}
      <p class="rounded bg-[var(--muted)] p-2 text-sm text-[var(--muted-foreground)]">
        This formula has terms the visual builder can't edit ({state.carried.join(', ')}). They are kept unchanged.
      </p>
    {/if}

    <label class="block text-sm">
      Dependent variable
      <input class="mt-1 w-full rounded border border-[var(--border)] bg-[var(--card)] px-2 py-1" bind:value={state.dependent} />
    </label>

    <div class="space-y-2">
      <div class="text-sm font-medium">Predictors</div>
      {#each state.predictors as p (p)}
        <div class="space-y-1 rounded border border-[var(--border)] p-2">
          <div class="flex items-center gap-2">
            <input
              class="flex-1 rounded border border-[var(--border)] bg-[var(--card)] px-2 py-1"
              bind:value={p.name}
              placeholder="name"
            />
            <!-- Data-backed predictors keep the detected type — the upload sets it,
                 mirroring the locked VariableCard / disabled factor-level inputs below. -->
            <div role="group" aria-label="predictor type" class="flex gap-1 text-xs">
              {#each ['continuous', 'binary', 'factor'] as k}
                <button
                  type="button"
                  disabled={p.dataBacked}
                  class="rounded px-2 py-1 disabled:cursor-default disabled:opacity-40 {p.kind === k ? 'bg-[var(--primary)] text-[var(--primary-foreground)]' : 'bg-[var(--muted)]'}"
                  onclick={() => setKind(p, k as PredictorKind)}>{k}</button>
              {/each}
            </div>
            <button type="button" aria-label="remove predictor" onclick={() => removePredictor(p)}>×</button>
          </div>
          {#if nameError(p)}
            <p class="text-xs text-[var(--destructive)]">{nameError(p)}</p>
          {/if}
          {#if p.kind === 'factor'}
            <div class="pl-2 text-xs">
              {#each p.levels as _, li}
                <div class="my-0.5 flex items-center gap-1">
                  <input
                    class="block w-40 rounded border border-[var(--border)] bg-[var(--card)] px-1"
                    bind:value={p.levels[li]}
                    disabled={p.dataBacked}
                    placeholder={li === 0 ? 'reference' : `level ${li + 1}`}
                  />
                  <!-- Remove only above the 2-level minimum, and never for uploaded levels. -->
                  {#if p.levels.length >= 3 && !p.dataBacked}
                    <button
                      type="button"
                      aria-label={`remove level ${li + 1}`}
                      onclick={() => (p.levels = p.levels.filter((_, j) => j !== li))}>×</button>
                  {/if}
                </div>
              {/each}
              {#if !p.dataBacked}
                <button type="button" onclick={() => (p.levels = [...p.levels, `${p.levels.length + 1}`])}>+ level</button>
              {/if}
            </div>
          {/if}
        </div>
      {/each}
      <button type="button" onclick={addPredictor}>+ Add predictor</button>
      {#if uploadCols.length > 0}
        <div class="flex flex-wrap items-center gap-1 text-xs">
          <span class="text-[var(--muted-foreground)]">Add from data:</span>
          {#each uploadCols as col (col.name)}
            <button
              type="button"
              class="rounded border border-[var(--border)] px-2 py-0.5 font-mono hover:bg-[var(--muted)]"
              onclick={() => addUploadColumn(col)}>+ {col.name}</button>
          {/each}
        </div>
      {/if}
    </div>

    <div class="space-y-2">
      <div class="text-sm font-medium">Interactions</div>
      {#each state.interactions as i (i)}
        <div class="flex items-center gap-2 text-sm">
          <select bind:value={i.a} class="rounded border border-[var(--border)] bg-[var(--card)] px-1">
            {#each state.predictors as p}<option value={p.name}>{p.name}</option>{/each}
          </select>
          <span>×</span>
          <select bind:value={i.b} class="rounded border border-[var(--border)] bg-[var(--card)] px-1">
            {#each state.predictors as p}<option value={p.name}>{p.name}</option>{/each}
          </select>
          <button type="button" aria-label="remove interaction" onclick={() => removeInteraction(i)}>×</button>
        </div>
      {/each}
      <button type="button" onclick={addInteraction}>+ Add interaction</button>
    </div>

    <div class="rounded bg-[var(--muted)] p-2 font-mono text-sm" data-testid="formula-preview">{formulaPreview}</div>

    <DialogFooter>
      <Button variant="ghost" onclick={() => onOpenChange(false)}>Cancel</Button>
      <Button onclick={useModel} disabled={hasErrors}>Use this model</Button>
    </DialogFooter>
  </DialogContent>
</Dialog>
