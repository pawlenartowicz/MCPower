<script lang="ts">
  // Onboarding checklist shown in the results panel before any run has been executed.
  import Check from '@lucide/svelte/icons/check';
  import Circle from '@lucide/svelte/icons/circle';
  import { familyStore } from '$lib/stores/family.svelte';

  const cfg = $derived(familyStore.byFamily[familyStore.active]);
  // ANOVA defines its model through structured factor rows, not a formula string;
  // regression/mixed use the formula field. Mirror each family's readiness check.
  const modelStep = $derived(
    cfg.family === 'anova'
      ? { label: 'Add a factor', done: cfg.variables.some((v) => v.role === 'factor') }
      : { label: 'Enter a formula', done: cfg.formula.trim().length > 0 },
  );
  const items = $derived([
    { label: 'Choose an analysis type', done: true },
    modelStep,
    { label: 'Set effect sizes', done: cfg.effects.length > 0 },
  ]);
</script>

<div class="rounded-lg border border-border p-5">
  <h2 class="mb-4 text-lg font-semibold">Get started</h2>
  <ol class="space-y-3 text-sm">
    {#each items as item, i (i)}
      <li class="flex items-center gap-3">
        {#if item.done}
          <Check class="h-5 w-5 text-green-600" />
        {:else}
          <Circle class="h-5 w-5 text-muted-foreground" />
        {/if}
        <span class:line-through={item.done} class:text-muted-foreground={item.done}>
          {i + 1}. {item.label}
        </span>
      </li>
    {/each}
  </ol>
</div>
