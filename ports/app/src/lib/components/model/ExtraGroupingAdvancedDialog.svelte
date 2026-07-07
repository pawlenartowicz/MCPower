<script lang="ts">
  // Advanced popup for ONE EXTRA grouping factor (opened from its card's ⚙) —
  // the secondary-grouping mirror of ClusterAdvancedDialog. Extra groupings carry
  // only random-slope parameters (cluster-level variables are a primary-only
  // knob), so this dialog shows just the slopes section. The slope SET is
  // formula-driven — `(1+x|g)` adds a slope on x; ClusterEditor's reconcile keeps
  // this grouping's `slopes` mirroring the formula and the popup only edits each
  // slope's variance and intercept correlation.
  import { Dialog, DialogContent, DialogTitle } from '$lib/components/ui/dialog';
  import { NumberInput } from '$lib/components/ui/number-input';
  import { familyStore } from '$lib/stores/family.svelte';
  import { configPaneDialogStyle } from './dialog-position.svelte';
  import InfoIcon from '$lib/components/guidance/InfoIcon.svelte';

  let { open = $bindable(false), groupingIndex }: { open?: boolean; groupingIndex: number } =
    $props();

  const cfg = $derived(familyStore.byFamily[familyStore.active]);
  const grouping = $derived(cfg.cluster?.extraGroupings?.[groupingIndex] ?? null);
  const slopes = $derived(grouping?.slopes ?? []);
</script>

<Dialog bind:open>
  <DialogContent class="max-h-[85vh] overflow-y-auto sm:max-w-md" style={configPaneDialogStyle()}>
    <DialogTitle>
      <span class="font-mono">{grouping?.clusterName ?? 'grouping'}</span>
      <span class="ml-1 text-sm font-normal text-muted-foreground">— advanced</span>
    </DialogTitle>

    <div class="space-y-1.5">
      <div class="flex items-center gap-1.5">
        <p class="text-xs font-medium">Random slopes</p>
        <InfoIcon tipKey="randomSlopes" />
      </div>
      <p class="text-xs text-muted-foreground">
        Declared in the formula — <span class="font-mono">(1 + x | {grouping?.clusterName ?? 'g'})</span>
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
            <span class="font-mono" data-testid={`extra-slope-${slope.predictorName}`}>{slope.predictorName}</span>
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
