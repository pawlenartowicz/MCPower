<script lang="ts">
  // Progress bar + cancel button shown at the top of ResultsPane while a simulation run is active.
  import { Progress } from '$lib/components/ui/progress';
  import { Button } from '$lib/components/ui/button';
  import X from '@lucide/svelte/icons/x';
  import { runStore } from '$lib/stores/run.svelte';

  // The bar reads the *continuous* fraction so it nudges forward on every tick;
  // the readout rounds to a whole percent. Feeding the bar the rounded value
  // instead makes it advance in 1% lurches (several percent per tick on short
  // runs), which the indicator's transition then renders as janky motion.
  const fraction = $derived(
    runStore.progress.total === 0
      ? 0
      : (runStore.progress.completed / runStore.progress.total) * 100,
  );
  const pct = $derived(Math.round(fraction));
</script>

<div class="flex items-center gap-3 border-b border-border bg-amber-50 px-3 py-2 dark:bg-amber-950/30">
  <Progress value={fraction} class="flex-1" />
  <span class="font-mono text-xs">
    {pct}% · {runStore.progress.completed.toLocaleString()} / {runStore.progress.total.toLocaleString()} sims
  </span>
  <Button variant="ghost" size="icon" aria-label="Cancel run" onclick={() => runStore.cancel()}>
    <X class="h-4 w-4" />
  </Button>
</div>
