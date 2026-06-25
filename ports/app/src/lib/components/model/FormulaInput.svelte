<script lang="ts">
  // Free-text formula input with plain-language, clickable example formulas shown
  // below the field (word-based, not x1/x2). Clicking an example fills the formula.
  import { Input } from '$lib/components/ui/input';
  import { Label } from '$lib/components/ui/label';
  import { Button } from '$lib/components/ui/button';
  import RotateCcw from '@lucide/svelte/icons/rotate-ccw';
  import { familyStore } from '$lib/stores/family.svelte';
  import { uiStore } from '$lib/stores/ui.svelte';
  import type { Entrypoint } from '$lib/domain/family';
  import { familyConfigToAppSpec } from '$lib/domain/app-spec-adapter';
  import InfoIcon from '$lib/components/guidance/InfoIcon.svelte';
  import ModelBuilderDialog from './ModelBuilderDialog.svelte';

  const cfg = $derived(familyStore.byFamily[familyStore.active]);

  // Visual builder is regression-only: mixed formulas carry random effects the
  // pairwise builder would strip on "Use this model".
  let modelBuilderOpen = $state(false);

  // Debounced commit: the input edits a local draft and writes cfg.formula
  // only after a typing pause (or Enter/blur). Committing per keystroke would
  // trigger an engine parse + cache entry per prefix and churn the cards.
  const COMMIT_DELAY_MS = 350;
  let draft = $state('');
  let timer: ReturnType<typeof setTimeout> | undefined;
  // Follow external changes (family switch, reset, example click) into the draft.
  $effect(() => {
    draft = cfg.formula;
  });
  function commit() {
    clearTimeout(timer);
    timer = undefined;
    if (cfg.formula !== draft) cfg.formula = draft;
  }
  function onDraftInput(e: Event) {
    draft = (e.target as HTMLInputElement).value;
    clearTimeout(timer);
    timer = setTimeout(commit, COMMIT_DELAY_MS);
  }

  // Surface formula/spec validation as a 'field' error inline under the control. Gated on
  // a non-empty formula so the untouched empty state stays guided by the checklist, not
  // red text; the transient "parsing formula…" placeholder is filtered to avoid flicker.
  const formulaErrors = $derived.by(() => {
    if (cfg.formula.trim().length === 0) return [];
    const { errors } = familyConfigToAppSpec(familyStore.active, cfg, familyStore.regressionOutcome);
    return errors.filter((e) => e !== 'parsing formula…');
  });

  // Word-based examples per (entrypoint, outcome); the parser accepts `=` and `~`.
  // Binary lists reuse the continuous predictors and change only the outcome word
  // (score→passed, income→promoted) so the reader sees that going binary changes only
  // the outcome — matching the toggle's behaviour (it never rewrites the live formula).
  // ANOVA doesn't render this component, but the record stays total for the type.
  const EXAMPLES: Record<Entrypoint, { continuous: string[]; binary: string[] }> = {
    anova: { continuous: ['score = teaching_method + class_size'], binary: [] },
    regression: {
      continuous: ['score = hours_studied + sleep', 'income = education + experience'],
      binary: ['passed = hours_studied + sleep', 'promoted = education + experience'],
    },
    mixed: {
      continuous: ['score = lesson_time + (1|classroom)'],
      binary: ['passed = lesson_time + (1|classroom)'],
    },
  };
  const examples = $derived(EXAMPLES[familyStore.active][familyStore.activeOutcome]);

  function useExample(formula: string) {
    cfg.formula = formula;
  }
</script>

<div class="space-y-1">
  <div class="flex items-center gap-2">
    <Label for="formula" class="font-semibold">Formula</Label>
    <InfoIcon tipKey="formula" />
    <Button
      type="button"
      variant="ghost"
      size="sm"
      class="ml-auto h-7 px-2 text-xs text-muted-foreground"
      aria-label="Reset configuration"
      title="Reset the {familyStore.active} configuration to defaults"
      onclick={() => (uiStore.resetConfirmOpen = true)}
    >
      <RotateCcw class="mr-1 h-3.5 w-3.5" /> Reset
    </Button>
    {#if familyStore.active === 'regression'}
      <Button
        type="button"
        variant="default"
        size="sm"
        class="h-7 px-2 text-xs"
        onclick={() => (modelBuilderOpen = true)}
      >
        Visual formula builder
      </Button>
    {/if}
  </div>
  <Input
    id="formula"
    type="text"
    placeholder="Write your formula — e.g. y = x1 + x2"
    value={draft}
    oninput={onDraftInput}
    onblur={commit}
    onkeydown={(e: KeyboardEvent) => {
      if (e.key === 'Enter') commit();
    }}
  />
  {#if formulaErrors.length > 0}
    <div class="space-y-0.5">
      {#each formulaErrors as err (err)}
        <p class="text-xs text-destructive">{err}</p>
      {/each}
    </div>
  {/if}
  <p class="text-xs text-muted-foreground">
    e.g.
    {#each examples as ex, i (ex)}
      {#if i > 0}<span aria-hidden="true">, </span>{/if}
      <button
        type="button"
        class="text-primary underline decoration-dotted underline-offset-2 hover:opacity-80"
        data-testid={i === 0 ? 'formula-sample' : undefined}
        onclick={() => useExample(ex)}
      >
        {ex}
      </button>
    {/each}
  </p>
  {#if familyStore.active === 'regression'}
    <ModelBuilderDialog open={modelBuilderOpen} onOpenChange={(v) => (modelBuilderOpen = v)} />
  {/if}
</div>
