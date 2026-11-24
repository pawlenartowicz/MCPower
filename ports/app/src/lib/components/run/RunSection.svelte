<script lang="ts">
  // "Run" section composing SharedParams, the sample-size unit switch,
  // FindPowerCard, and FindSampleSizeCard inside the shared collapsible shell.
  import CollapsibleSection from '$lib/components/shell/CollapsibleSection.svelte';
  import { sharedPrefs } from '$lib/stores/shared-prefs.svelte';
  import { familyStore } from '$lib/stores/family.svelte';
  import SharedParams from './SharedParams.svelte';
  import SampleSizeUnitSwitch from './SampleSizeUnitSwitch.svelte';
  import FindPowerCard from './FindPowerCard.svelte';
  import FindSampleSizeCard from './FindSampleSizeCard.svelte';

  const cfg = $derived(familyStore.byFamily[familyStore.active]);
  const testSummary = $derived.by(() => {
    const t = cfg.tests;
    if (t.kind === 'all') return 'all tests';
    // Contrasts live in cfg.contrasts, not cfg.tests, and run only where
    // TestChooser's contrastsAllowed holds (OLS: ANOVA or linear regression).
    const contrastsRun =
      familyStore.active === 'anova' ||
      (familyStore.active === 'regression' && familyStore.activeOutcome === 'linear');
    const nContrasts = contrastsRun ? cfg.contrasts.filter((c) => c.enabled).length : 0;
    const n = t.names.length + nContrasts;
    if (n === 0) return 'no tests';
    if (n === 1) return `1 test`;
    return `${n} tests`;
  });
  const summary = $derived(`target=${cfg.targetPower}% · ${testSummary}`);
</script>

<CollapsibleSection title="Run" {summary} bind:open={sharedPrefs.runExpanded}>
  <SharedParams />
  <div class="space-y-2 border-t border-border/40 pt-2">
    <SampleSizeUnitSwitch />
    <FindPowerCard />
  </div>
  <div class="space-y-2 border-t border-border/40 pt-2">
    <FindSampleSizeCard />
  </div>
</CollapsibleSection>
