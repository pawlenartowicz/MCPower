<script lang="ts">
  // Model section: the formula, variable-type, effect, cluster, and ANOVA sub-editors
  // (AnovaFactorEditor + AnovaCovariateEditor) inside the shared collapsible shell.
  // ConfigPanel composes Model, Correlations, and Upload as siblings; do not push
  // Correlations/Upload back inside here.
  import CollapsibleSection from '$lib/components/shell/CollapsibleSection.svelte';
  import { Button } from '$lib/components/ui/button';
  import Settings2 from '@lucide/svelte/icons/settings-2';
  import { sharedPrefs } from '$lib/stores/shared-prefs.svelte';
  import { familyStore } from '$lib/stores/family.svelte';
  import FormulaInput from './FormulaInput.svelte';
  import PredictorCards from './PredictorCards.svelte';
  import ClusterEditor from './ClusterEditor.svelte';
  import AnovaFactorEditor from './AnovaFactorEditor.svelte';
  import AnovaCovariateEditor from './AnovaCovariateEditor.svelte';
  import BaselineProbabilityInput from './BaselineProbabilityInput.svelte';
  import ModelOptionsDialog from './ModelOptionsDialog.svelte';
  import InfoIcon from '$lib/components/guidance/InfoIcon.svelte';

  const cfg = $derived(familyStore.byFamily[familyStore.active]);
  let modelOptionsOpen = $state(false);

  // Regression and mixed use different storage (store field vs cluster flag); kept
  // inline rather than extracted to avoid a wrapper for two callers.
  const outcomeIsBinary = $derived(familyStore.activeOutcome === 'binary');

  function setOutcome(binary: boolean) {
    if (familyStore.active === 'regression') {
      familyStore.regressionOutcome = binary ? 'binary' : 'continuous';
    } else if (familyStore.active === 'mixed' && cfg.cluster) {
      cfg.cluster.binaryOutcome = binary;
      // The adapter requires a baseline in (0,1) for binary mixed; seed the default
      // (mirrors regression's always-defined cfg.baselineProbability = 0.2).
      if (binary && cfg.cluster.baselineProbability === undefined) {
        cfg.cluster.baselineProbability = 0.2;
      }
    }
  }
  const summary = $derived(
    familyStore.active === 'anova'
      ? `${cfg.variables.filter((v) => v.kind === 'factor').length} factors · ${cfg.effects.length} effects`
      : `${cfg.formula || '(no formula)'} · ${cfg.effects.length} effects`,
  );
</script>

<CollapsibleSection title="Model" {summary} bind:open={sharedPrefs.modelExpanded}>
  {#if familyStore.active === 'regression' || familyStore.active === 'mixed'}
    <div role="group" aria-label="Outcome type" class="flex items-center gap-2 text-sm">
      <span class="text-sm font-medium">Outcome type</span>
      <InfoIcon tipKey="outcomeType" />
      <button
        type="button"
        class="rounded px-3 py-1 {!outcomeIsBinary
          ? 'bg-primary text-primary-foreground'
          : 'bg-muted text-muted-foreground hover:bg-muted/70'}"
        onclick={() => setOutcome(false)}
      >Continuous</button>
      <button
        type="button"
        class="rounded px-3 py-1 {outcomeIsBinary
          ? 'bg-primary text-primary-foreground'
          : 'bg-muted text-muted-foreground hover:bg-muted/70'}"
        onclick={() => setOutcome(true)}
      >Binary</button>
    </div>
  {/if}
  {#if familyStore.active === 'anova'}
    <AnovaFactorEditor />
    <AnovaCovariateEditor />
  {:else}
    <FormulaInput />
    <PredictorCards />
  {/if}
  {#if familyStore.active === 'mixed'}
    <ClusterEditor />
  {/if}
  {#if outcomeIsBinary}
    <BaselineProbabilityInput />
  {/if}
  {#if familyStore.active !== 'anova'}
    <div class="flex justify-end">
      <Button
        variant="ghost"
        size="sm"
        class="h-7 px-2 text-xs text-muted-foreground"
        data-testid="model-more-options"
        onclick={() => (modelOptionsOpen = true)}
      >
        <Settings2 class="mr-1 h-3.5 w-3.5" /> More options
      </Button>
    </div>
    <ModelOptionsDialog bind:open={modelOptionsOpen} />
  {/if}
</CollapsibleSection>
