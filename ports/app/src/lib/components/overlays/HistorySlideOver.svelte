<script lang="ts">
  // Right-side slide-over panel listing past runs; supports search, replay, and delete.
  import { Sheet, SheetContent, SheetHeader, SheetTitle } from '$lib/components/ui/sheet';
  import { Input } from '$lib/components/ui/input';
  import { Button } from '$lib/components/ui/button';
  import X from '@lucide/svelte/icons/x';
  import { uiStore } from '$lib/stores/ui.svelte';
  import { historyStore } from '$lib/stores/history.svelte';
  import { runStore } from '$lib/stores/run.svelte';
  import type { HistoryEntry } from '$lib/persistence/history';

  let q = $state('');

  // Floating panel offset below the header — mirrors SettingsSlideOver, change together.
  let panelTop = $state(116);
  // Phones: take the whole viewport instead of the floating card. Mirrors
  // SettingsSlideOver — change together.
  let fullScreen = $state(false);
  $effect(() => {
    if (uiStore.historyOpen) {
      panelTop = (document.querySelector('header')?.getBoundingClientRect().bottom ?? 104) + 12;
      fullScreen = window.innerWidth < 640;
    }
  });

  const filtered = $derived(
    historyStore.entries.filter((e) =>
      `${labelOf(e)} ${formulaOf(e)}`.toLowerCase().includes(q.toLowerCase()),
    ),
  );

  function labelOf(e: HistoryEntry): string {
    const what =
      e.kind === 'find-power'
        ? `Find power · n=${e.sample_size}`
        : `Find sample · ${e.bounds?.[0]}→${e.bounds?.[1]}`;
    return `${e.family} · ${what}`;
  }

  function formulaOf(e: HistoryEntry): string {
    if (e.spec.family === 'anova') {
      const terms = [...e.spec.factors, ...e.spec.covariates].map((t) => t.name);
      return `${e.spec.outcome} ~ ${terms.join(' + ')}`;
    }
    const f = e.spec.parsed_formula;
    const terms = [...f.predictors, ...f.interaction_terms.map((t: string[]) => t.join(':'))];
    return `${f.outcome} ~ ${terms.join(' + ')}`;
  }

  function whenOf(e: HistoryEntry): string {
    const d = new Date(e.ts);
    const p = (n: number) => String(n).padStart(2, '0');
    return `${p(d.getDate())}.${p(d.getMonth() + 1)}.${d.getFullYear()}, ${p(d.getHours())}:${p(d.getMinutes())}`;
  }

  async function replay(entry: HistoryEntry) {
    uiStore.historyOpen = false;
    if (entry.kind === 'find-power' && entry.sample_size !== undefined) {
      await runStore.startFindPower(entry.spec, entry.sample_size);
    } else if (entry.kind === 'find-sample-size' && entry.bounds && entry.method) {
      await runStore.startFindSampleSize(entry.spec, entry.bounds, entry.method);
    }
  }
</script>

<Sheet open={uiStore.historyOpen} onOpenChange={(v: boolean) => (uiStore.historyOpen = v)}>
  <SheetContent
    side="right"
    class={fullScreen
      ? '!inset-0 !w-full !max-w-full !rounded-none border-0'
      : 'w-full !max-w-[calc(100vw-1.5rem)] sm:!max-w-xl !right-3 !bottom-3 rounded-xl border shadow-xl'}
    style={fullScreen ? undefined : `top: ${panelTop}px`}
  >
    <SheetHeader>
      <SheetTitle>History</SheetTitle>
    </SheetHeader>
    <div class="min-h-0 flex-1 space-y-2 overflow-y-auto px-4 pt-4 pb-4">
      <Input placeholder="Search…" bind:value={q} />
      {#if filtered.length === 0}
        <p class="text-xs text-muted-foreground">No runs yet.</p>
      {/if}
      <ul class="space-y-1.5">
        {#each filtered as e (e.id)}
          <li class="flex items-center justify-between gap-2 rounded-md border border-border px-3 py-2">
            <button type="button" class="min-w-0 flex-1 text-left" onclick={() => replay(e)}>
              <div class="flex items-baseline justify-between gap-2">
                <span class="truncate text-sm font-medium">{labelOf(e)}</span>
                <span class="shrink-0 text-xs text-muted-foreground">{whenOf(e)}</span>
              </div>
              <div class="truncate font-mono text-xs text-muted-foreground">{formulaOf(e)}</div>
            </button>
            <Button
              variant="ghost"
              size="icon"
              aria-label="Delete"
              onclick={() => historyStore.remove(e.id)}
            >
              <X class="h-4 w-4" />
            </Button>
          </li>
        {/each}
      </ul>
    </div>
  </SheetContent>
</Sheet>
