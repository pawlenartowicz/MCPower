<script lang="ts">
  // Left configuration pane: the four peer sections in workflow order —
  // Upload data → Model → Correlations → Run — each a collapsible card built on
  // CollapsibleSection. Family conditions gate the data-driven sections: Upload is
  // regression + mixed (matching PredictorCards' canFitFromData consumer),
  // Correlations is non-ANOVA. Model and Run own their own shells.
  import CollapsibleSection from './CollapsibleSection.svelte';
  import ModelSection from '$lib/components/model/ModelSection.svelte';
  import RunSection from '$lib/components/run/RunSection.svelte';
  import UploadSection from '$lib/components/model/UploadSection.svelte';
  import CorrelationsEditor from '$lib/components/model/CorrelationsEditor.svelte';
  import { sharedPrefs } from '$lib/stores/shared-prefs.svelte';
  import { familyStore } from '$lib/stores/family.svelte';
  import { uploadStore } from '$lib/stores/upload.svelte';

  const uploadSummary = $derived(
    uploadStore.csvData ? uploadStore.filename : 'no file selected',
  );
</script>

<div class="flex flex-col gap-3.5 p-4">
  {#if familyStore.active === 'regression' || familyStore.active === 'mixed'}
    <CollapsibleSection
      title="Upload data"
      summary={uploadSummary}
      bind:open={sharedPrefs.uploadExpanded}
    >
      <UploadSection />
    </CollapsibleSection>
  {/if}

  <ModelSection />

  {#if familyStore.active !== 'anova'}
    <CollapsibleSection
      title="Correlations"
      optional
      tipKey="correlations"
      bind:open={sharedPrefs.correlationsExpanded}
    >
      <CorrelationsEditor />
    </CollapsibleSection>
  {/if}

  <RunSection />
</div>
