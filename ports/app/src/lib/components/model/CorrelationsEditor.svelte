<script lang="ts">
  // Lower-triangular correlation matrix editor; mirrors the symmetric upper half as read-only display.
  // The section heading is provided by the enclosing Collapsible (see ModelSection).
  import { NumberInput } from '$lib/components/ui/number-input';
  import { familyStore } from '$lib/stores/family.svelte';
  import { uploadStore } from '$lib/stores/upload.svelte';
  import {
    allContinuousUploaded,
    correlatableVariables,
    correlationDisplayValue,
  } from '$lib/domain/correlations';

  const cfg = $derived(familyStore.byFamily[familyStore.active]);
  const n = $derived(cfg.variables.length);

  $effect(() => {
    if (cfg.correlations.length !== n) cfg.correlations.length = n;
    for (let i = 0; i < n; i++) {
      let row = cfg.correlations[i];
      if (!Array.isArray(row)) {
        row = Array.from({ length: n }, (_, j) => (i === j ? 1 : 0));
        cfg.correlations[i] = row;
        continue;
      }
      if (row.length !== n) row.length = n;
      for (let j = 0; j < n; j++) {
        const want = i === j ? 1 : (row[j] ?? 0);
        if (row[j] !== want) row[j] = want;
      }
    }
  });

  function setCell(i: number, j: number, v: number) {
    if (i === j) return;
    const lo = Math.max(i, j);
    const hi = Math.min(i, j);
    const loRow = cfg.correlations[lo];
    const hiRow = cfg.correlations[hi];
    if (loRow) loRow[hi] = v;
    if (hiRow) hiRow[lo] = v;
  }

  // The subset of variables that are shown in the correlation triangle.
  const entries = $derived(
    correlatableVariables(cfg.variables, uploadStore.csvData, uploadStore.mode),
  );

  /**
   * Display value for cell (entryI.idx, entryJ.idx): the stored value, or the
   * measured Pearson r for an untouched uploaded-continuous pair in partial
   * mode. The measured value is NEVER written to cfg.correlations.
   */
  function displayValue(iIdx: number, jIdx: number): number {
    return correlationDisplayValue(
      cfg.correlations[iIdx]?.[jIdx] ?? 0,
      uploadStore.mode,
      uploadStore.csvData,
      cfg.variables[iIdx]?.name,
      cfg.variables[jIdx]?.name,
    );
  }
</script>

<div class="space-y-3">
  {#if entries.length < 2}
    {#if allContinuousUploaded(cfg.variables, uploadStore.csvData)}
      <p class="text-sm text-muted-foreground">In strict mode, correlations come from the uploaded data.</p>
    {:else}
      <p class="text-sm text-muted-foreground">Add continuous predictors to edit correlations.</p>
    {/if}
  {:else}
    <div class="overflow-x-auto">
      <table class="border-collapse text-sm">
        <thead>
          <tr>
            <th></th>
            {#each entries as e (e.row.name)}
              <th class="px-2 py-1 text-sm font-medium">{e.row.name}</th>
            {/each}
          </tr>
        </thead>
        <tbody>
          {#each entries as re, dispI (re.row.name)}
            <tr>
              <th class="pr-2 text-right text-sm font-medium">{re.row.name}</th>
              {#each entries as _ce, dispJ (_ce.row.name + dispJ)}
                <td class="p-1">
                  {#if dispI === dispJ}
                    <span class="inline-flex h-9 w-28 items-center justify-center text-muted-foreground">1</span>
                  {:else if dispJ < dispI}
                    <NumberInput
                      class="w-28"
                      step={0.05}
                      min={-1}
                      max={1}
                      value={displayValue(re.idx, entries[dispJ]!.idx)}
                      oninput={(v) => setCell(re.idx, entries[dispJ]!.idx, v)}
                    />
                  {:else}
                    <span class="inline-flex h-9 w-28 items-center justify-center text-muted-foreground">
                      {displayValue(re.idx, entries[dispJ]!.idx)}
                    </span>
                  {/if}
                </td>
              {/each}
            </tr>
          {/each}
        </tbody>
      </table>
    </div>
  {/if}
</div>
