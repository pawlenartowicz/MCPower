<script lang="ts">
  // Four-item toggle row (Summary / Joint dist / Script / Export) that switches the active sub-view for a run tab.
  import { ToggleGroup, ToggleGroupItem } from '$lib/components/ui/toggle-group';
  import type { RunTab, ResultTab } from '$lib/stores/run.svelte';

  interface Props {
    tab: RunTab;
  }
  const { tab }: Props = $props();

  const TABS: { id: ResultTab; label: string }[] = [
    { id: 'summary', label: 'Summary' },
    { id: 'joint', label: 'Joint dist' },
    { id: 'script', label: 'Script' },
    { id: 'export', label: 'Export' },
  ];
</script>

<div class="border-b border-border px-2 py-1">
  <ToggleGroup
    type="single"
    value={tab.subView}
    onValueChange={(v: string) => {
      if (v === 'summary' || v === 'joint' || v === 'script' || v === 'export') tab.subView = v;
    }}
  >
    {#each TABS as t (t.id)}
      <ToggleGroupItem value={t.id} data-testid={`tab-${t.id}`}>{t.label}</ToggleGroupItem>
    {/each}
  </ToggleGroup>
</div>
