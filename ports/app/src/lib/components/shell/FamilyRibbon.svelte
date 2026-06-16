<script lang="ts">
  // Horizontal ribbon of analysis-family selector buttons; disabled families show as non-interactive.
  import {
    FAMILIES,
    FAMILY_LABEL,
    FAMILY_ICON,
    SELECTABLE_FAMILIES,
    type Entrypoint,
  } from '$lib/domain/family';
  import { familyStore } from '$lib/stores/family.svelte';

  // narrow=true (phone header row 2): fill the width with equal-width buttons
  // instead of the fixed-width desktop ribbon.
  let { narrow = false }: { narrow?: boolean } = $props();
</script>

<div role="radiogroup" aria-label="Analysis family" class="flex items-stretch gap-1 {narrow ? 'w-full' : ''}">
  {#each FAMILIES as f (f)}
    {@const Icon = FAMILY_ICON[f]}
    {@const active = familyStore.active === f}
    {@const selectable = SELECTABLE_FAMILIES.includes(f)}
    <button
      type="button"
      role="radio"
      aria-checked={active}
      aria-label={FAMILY_LABEL[f]}
      aria-disabled={!selectable}
      disabled={!selectable}
      title={selectable ? undefined : `${FAMILY_LABEL[f]} is not available yet`}
      class="inline-flex h-14 {narrow ? 'flex-1 min-w-0' : 'w-26'} flex-col items-center justify-center gap-1 rounded-md border-b-2 px-2 text-xs transition-colors {selectable
        ? 'hover:bg-muted'
        : 'cursor-not-allowed opacity-40'} {active
        ? 'border-primary bg-primary/10 text-foreground'
        : 'border-transparent text-muted-foreground'}"
      onclick={() => {
        if (selectable) familyStore.active = f as Entrypoint;
      }}
    >
      <Icon class="h-7 w-7" />
      <span class="font-medium whitespace-nowrap">{FAMILY_LABEL[f]}</span>
    </button>
  {/each}
</div>
