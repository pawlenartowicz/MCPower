<script lang="ts">
  // Horizontal tab strip listing all open run tabs; clicking activates, X closes.
  import X from '@lucide/svelte/icons/x';
  import { runStore } from '$lib/stores/run.svelte';
</script>

<div class="flex items-center gap-1 border-b border-border px-2 py-1">
  {#each runStore.runTabs as tab (tab.id)}
    <div
      class="flex items-center gap-1 rounded-t border border-b-0 px-2 py-1 text-sm"
      class:bg-background={runStore.activeTabId === tab.id}
      class:bg-muted={runStore.activeTabId !== tab.id}
    >
      <button type="button" class="text-left" onclick={() => (runStore.activeTabId = tab.id)}>
        {tab.label}
      </button>
      <button
        type="button"
        aria-label="Close tab"
        class="rounded p-0.5 hover:bg-destructive/20"
        onclick={(e) => {
          e.stopPropagation();
          runStore.removeTab(tab.id);
        }}
      >
        <X class="h-3 w-3" />
      </button>
    </div>
  {/each}
  <!-- Permanent, non-closable New-run tab. Active when no run tab is selected
       (activeTabId === null); shows the family-aware tutorial preview. -->
  <button
    type="button"
    class="rounded-t border border-b-0 px-2 py-1 text-sm"
    class:bg-background={runStore.activeTabId === null}
    class:bg-muted={runStore.activeTabId !== null}
    onclick={() => (runStore.activeTabId = null)}
  >
    New run
  </button>
</div>
