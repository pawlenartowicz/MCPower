<script lang="ts">
  // Root results panel: shows the get-started checklist, progress strip, run tabs, and the active sub-view.
  import { runStore } from '$lib/stores/run.svelte';
  import GetStartedChecklist from './GetStartedChecklist.svelte';
  import WhatToFind from './WhatToFind.svelte';
  import FamilyTutorial from './FamilyTutorial.svelte';
  import ProgressStrip from './ProgressStrip.svelte';
  import RunErrorCard from './RunErrorCard.svelte';
  import RunTabs from './RunTabs.svelte';
  import TabSelector from './TabSelector.svelte';
  import ScriptView from './ScriptView.svelte';
  import SummaryTab from './SummaryTab.svelte';
  import JointDistTab from './JointDistTab.svelte';
  import ExportTab from './ExportTab.svelte';
  import ResultsCitation from './ResultsCitation.svelte';

  const active = $derived(runStore.runTabs.find((t) => t.id === runStore.activeTabId) ?? null);
</script>

<div class="flex h-full flex-col">
  {#if runStore.runState === 'running'}
    <ProgressStrip />
  {/if}
  {#if runStore.lastError}
    <!-- A failed run: the card replaces the empty-state checklist (failed first run) and
         otherwise sits above the preserved prior result (failed later run). -->
    <RunErrorCard />
  {/if}
  <!-- Run-tab strip is always visible; the permanent New-run tab lives at its right edge. -->
  <RunTabs />
  {#if active}
    <TabSelector tab={active} />
    <div class="flex-1 overflow-auto p-4">
      {#if active.subView === 'summary'}
        <SummaryTab tab={active} />
      {:else if active.subView === 'joint'}
        <JointDistTab tab={active} />
      {:else if active.subView === 'script'}
        <ScriptView tab={active} />
      {:else}
        <ExportTab tab={active} />
      {/if}
      <ResultsCitation />
    </div>
  {:else if !runStore.lastError}
    <!-- New-run tab: family-aware empty state as a tutorial preview. The trailing
         spacer keeps the tutorial prose off the bottom edge of the scroll area. -->
    <div class="flex-1 overflow-auto">
      <div class="flex flex-col gap-4 p-4 min-[900px]:flex-row">
        <div class="flex shrink-0 flex-col gap-4 min-[900px]:w-[260px]">
          <GetStartedChecklist />
          <WhatToFind />
        </div>
        <div class="min-w-0 flex-1">
          <FamilyTutorial />
        </div>
      </div>
      <div class="h-12" aria-hidden="true"></div>
    </div>
  {/if}
</div>
