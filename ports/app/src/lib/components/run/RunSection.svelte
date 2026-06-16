<script lang="ts">
  // "Run" section composing SharedParams, FindPowerCard, and
  // FindSampleSizeCard inside the shared collapsible shell.
  import CollapsibleSection from '$lib/components/shell/CollapsibleSection.svelte';
  import { sharedPrefs } from '$lib/stores/shared-prefs.svelte';
  import { familyStore } from '$lib/stores/family.svelte';
  import SharedParams from './SharedParams.svelte';
  import FindPowerCard from './FindPowerCard.svelte';
  import FindSampleSizeCard from './FindSampleSizeCard.svelte';

  const cfg = $derived(familyStore.byFamily[familyStore.active]);
  const testSummary = $derived.by(() => {
    const t = cfg.tests;
    if (t.kind === 'all') return 'all tests';
    if (t.names.length === 0) return 'no tests';
    if (t.names.length === 1) return `1 test`;
    return `${t.names.length} tests`;
  });
  const summary = $derived(`target=${cfg.targetPower}% · ${testSummary}`);
</script>

<CollapsibleSection title="Run" {summary} bind:open={sharedPrefs.runExpanded}>
  <SharedParams />
  <div class="space-y-2 border-t border-border/40 pt-2">
    <FindPowerCard />
  </div>
  <div class="space-y-2 border-t border-border/40 pt-2">
    <FindSampleSizeCard />
  </div>
</CollapsibleSection>
