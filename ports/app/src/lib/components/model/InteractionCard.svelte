<script lang="ts">
  // One interaction term as an amber-tinted card. A non-factor interaction is a
  // single coefficient and collapses to a one-line header (term + tag + controls).
  // A factor-involved interaction expands to one row per non-reference level
  // combination and shows a plain-language hint. The rows come verbatim from
  // effectGroups — the card is a pure projection, no string-sniffing.
  import EffectControls from './EffectControls.svelte';
  import type { InteractionGroup } from '$lib/domain/effect-names';
  import type { EffectRow, VariableRow } from '$lib/domain/family';

  let {
    group,
    effects,
    variables,
  }: { group: InteractionGroup; effects: EffectRow[]; variables: VariableRow[] } = $props();

  const inline = $derived(!group.isFactorInteraction && group.rows.length === 1);
  const parts = $derived(group.term.split(':'));

  function effectFor(name: string): EffectRow | undefined {
    return effects.find((e) => e.name === name);
  }
</script>

<div class="rounded-md border border-primary/40 bg-primary/5 px-2.5 py-2">
  {#if inline}
    {@const e = effectFor(group.rows[0]!.name)}
    <div class="flex flex-wrap items-center gap-2">
      <span class="font-mono text-[13px] font-semibold">{group.term}</span>
      <span
        class="rounded border border-primary px-1.5 py-px text-[10px] font-semibold uppercase tracking-wide text-primary"
        >interaction</span
      >
      {#if e}
        <EffectControls effect={e} {variables} />
      {/if}
    </div>
  {:else}
    <div class="flex flex-wrap items-center gap-2">
      <span class="font-mono text-[13px] font-semibold">{group.term}</span>
      <span class="flex-1"></span>
      <span
        class="rounded border border-primary px-1.5 py-px text-[10px] font-semibold uppercase tracking-wide text-primary"
        >interaction</span
      >
    </div>
    {#if group.isFactorInteraction}
      <p class="mt-1.5 text-[11px] italic text-muted-foreground">
        how much {parts[0]}'s effect changes at each level of {parts.slice(1).join(' & ')} (vs. its
        reference)
      </p>
    {/if}
    {#each group.rows as row (row.name)}
      {@const e = effectFor(row.name)}
      {#if e}
        <div class="mt-2 flex items-center gap-2">
          <span class="min-w-16 font-mono text-xs">{row.name}</span>
          <EffectControls effect={e} {variables} />
        </div>
      {/if}
    {/each}
  {/if}
</div>
