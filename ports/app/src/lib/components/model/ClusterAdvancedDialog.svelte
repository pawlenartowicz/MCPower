<script lang="ts">
  // Advanced popup for the PRIMARY cluster (opened from the primary card's ⚙):
  // cluster-level variables (drawn once per cluster, constant within it) and
  // random-slope parameters. The slope SET is formula-driven — `(1+x|g)` adds a
  // slope on x, removing it from the formula removes it here; the popup only
  // edits each slope's variance and intercept correlation (ClusterEditor's
  // reconcile keeps cl.slopes mirroring the formula).
  import { Dialog, DialogContent, DialogTitle } from '$lib/components/ui/dialog';
  import { NumberInput } from '$lib/components/ui/number-input';
  import { familyStore } from '$lib/stores/family.svelte';
  import { parsedFormulaStore } from '$lib/stores/parsed-formula.svelte';
  import { configPaneDialogStyle } from './dialog-position.svelte';
  import InfoIcon from '$lib/components/guidance/InfoIcon.svelte';

  let { open = $bindable(false) }: { open?: boolean } = $props();

  const cfg = $derived(familyStore.byFamily[familyStore.active]);
  const predictors = $derived(parsedFormulaStore.getStable(cfg.formula).result?.predictors ?? []);
  const slopes = $derived(cfg.cluster?.slopes ?? []);

  function isClusterLevel(name: string): boolean {
    return (cfg.cluster?.clusterLevelVars ?? []).includes(name);
  }
  function toggleClusterLevel(name: string) {
    const cl = cfg.cluster;
    if (!cl) return;
    const current = cl.clusterLevelVars ?? [];
    cl.clusterLevelVars = current.includes(name)
      ? current.filter((n) => n !== name)
      : [...current, name];
  }
</script>

<Dialog bind:open>
  <DialogContent class="max-h-[85vh] overflow-y-auto sm:max-w-md" style={configPaneDialogStyle()}>
    <DialogTitle>
      <span class="font-mono">{cfg.cluster?.clusterName ?? 'cluster'}</span>
      <span class="ml-1 text-sm font-normal text-muted-foreground">— advanced</span>
    </DialogTitle>

    <div class="space-y-1.5">
      <div class="flex items-center gap-1.5">
        <p class="text-xs font-medium">Cluster-level variables</p>
        <InfoIcon tipKey="clusterLevelVars" />
      </div>
      <p class="text-xs text-muted-foreground">
        Drawn once per {cfg.cluster?.clusterName ?? 'cluster'}, constant within it (e.g. age within ID).
      </p>
      {#if predictors.length === 0}
        <p class="text-xs text-muted-foreground">Add predictors to the formula first.</p>
      {:else}
        <div class="flex flex-wrap gap-x-4 gap-y-1.5">
          {#each predictors as name (name)}
            <label class="flex items-center gap-1.5 text-xs">
              <input
                type="checkbox"
                checked={isClusterLevel(name)}
                data-testid={`cluster-level-${name}`}
                onchange={() => toggleClusterLevel(name)}
              />
              <span class="font-mono">{name}</span>
            </label>
          {/each}
        </div>
      {/if}
    </div>

    <div class="space-y-1.5 border-t border-border pt-3">
      <div class="flex items-center gap-1.5">
        <p class="text-xs font-medium">Random slopes</p>
        <InfoIcon tipKey="randomSlopes" />
      </div>
      <p class="text-xs text-muted-foreground">
        Declared in the formula — <span class="font-mono">(1 + x | {cfg.cluster?.clusterName ?? 'g'})</span>
        adds a slope on <span class="font-mono">x</span>. Set each slope's variance and
        intercept correlation here.
      </p>
      {#if slopes.length === 0}
        <p class="text-xs text-muted-foreground">
          No random slopes in the formula.
        </p>
      {:else}
        <div class="grid grid-cols-[1fr_7rem_7rem] items-center gap-x-2 gap-y-1.5 text-xs">
          <span></span>
          <span class="text-muted-foreground">slope variance</span>
          <span class="text-muted-foreground">corr w/ intercept</span>
          {#each slopes as slope (slope.predictorName)}
            <span class="font-mono" data-testid={`slope-${slope.predictorName}`}>{slope.predictorName}</span>
            <NumberInput
              class="h-7 w-full"
              step={0.05}
              min={0}
              value={slope.slopeVariance}
              oninput={(v: number) => (slope.slopeVariance = v)}
            />
            <NumberInput
              class="h-7 w-full"
              step={0.1}
              min={-1}
              max={1}
              value={slope.slopeInterceptCorr}
              oninput={(v: number) => (slope.slopeInterceptCorr = v)}
            />
          {/each}
        </div>
      {/if}
    </div>
  </DialogContent>
</Dialog>
