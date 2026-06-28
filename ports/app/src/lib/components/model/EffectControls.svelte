<script lang="ts">
  // Shared effect-size control unit: a stepper, a dashed ± sign-flip, then the
  // Cohen presets — left to right, so the ± sits between the number and the
  // effect-size choices. Presets show full words (small/medium/large) and collapse
  // to s/m/l when the row is too narrow (container query).
  // Used by every effect row across VariableCard and InteractionCard. Presets are
  // chosen by name+kind via the single-sourced presetsFor() helper; the ± relabel
  // disambiguates it from the stepper's own − button.
  import { NumberInput } from '$lib/components/ui/number-input';
  import { presetsFor } from '$lib/domain/effect-names';
  import type { EffectRow, VariableRow } from '$lib/domain/family';
  import { familyStore } from '$lib/stores/family.svelte';

  // `effect` is the live cfg.effects proxy entry — we mutate effect.value in
  // place (never reassign), so it propagates without a two-way binding.
  let { effect, variables }: { effect: EffectRow; variables: VariableRow[] } = $props();

  // A binary outcome (logistic regression / binary GLMM) makes β a log-odds, so
  // the presets switch to the odds (beta) set and each input gains a live
  // OR = exp(β) readout. The β value is unchanged — only the scale's reading is.
  const isLogit = $derived(familyStore.activeOutcome === 'binary');
  const presets = $derived(presetsFor(effect.name, variables, isLogit));
  const orText = $derived(`OR ${Math.exp(effect.value).toFixed(2)}`);
  // exp(β) is per-1-SD for a continuous predictor, per-category for binary/factor;
  // the number is the same, only the interpretation differs.
  const orTooltip =
    'Odds ratio = exp(β). Per 1 SD for a continuous predictor, per category for a binary/factor predictor.';

  function presetTitle(p: { long: string; value: number }): string {
    return isLogit
      ? `${p.long} effect = OR ${Math.exp(p.value).toFixed(2)} (β ${p.value}, Chen et al.)`
      : `${p.long} effect = ${p.value} (Cohen)`;
  }

  function applyPreset(magnitude: number) {
    const sign = effect.value < 0 ? -1 : 1;
    effect.value = sign * magnitude;
  }
  function flipSign() {
    effect.value = -effect.value;
  }
</script>

<div class="@container/effects flex min-w-0 flex-1 items-center gap-1.5">
  <NumberInput
    class="h-8 w-24 shrink-0"
    dense
    step={0.05}
    bind:value={effect.value}
    data-testid={`effect-${effect.name}`}
  />
  {#if isLogit}
    <span
      class="shrink-0 font-mono text-xs text-muted-foreground tabular-nums"
      title={orTooltip}
      data-testid={`effect-or-${effect.name}`}
    >
      {orText}
    </span>
  {/if}
  <button
    type="button"
    class="inline-flex h-8 min-w-8 items-center justify-center rounded-md border border-dashed border-border bg-secondary px-2 font-mono text-sm text-secondary-foreground hover:bg-secondary/80"
    title="Flip sign"
    aria-label="Flip sign"
    onclick={flipSign}
  >
    ±
  </button>
  <div class="inline-flex items-center gap-1">
    {#each presets as p (p.short)}
      <button
        type="button"
        class="rounded-md border border-border bg-secondary px-2 py-1 text-[11px] font-semibold text-secondary-foreground hover:bg-secondary/80"
        title={presetTitle(p)}
        aria-label="{p.long} effect = {p.value}"
        onclick={() => applyPreset(p.value)}
      >
        <!-- 22rem (352px): full words fit beside the stepper+± in the default
             config pane; the old @lg (512px) collapsed to letters almost always. -->
        <span class="@[22rem]/effects:hidden">{p.short.toLowerCase()}</span>
        <span class="hidden @[22rem]/effects:inline">{p.long}</span>
      </button>
    {/each}
    {#if isLogit}
      <span
        class="rounded bg-muted px-1 text-[9px] font-semibold uppercase tracking-wide text-muted-foreground"
        title="Odds-ratio benchmarks (Chen et al. 2010) — beta"
      >
        beta
      </span>
    {/if}
  </div>
</div>
