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
  import BaselineRateInput from './BaselineRateInput.svelte';
  import ModelOptionsDialog from './ModelOptionsDialog.svelte';
  import InfoIcon from '$lib/components/guidance/InfoIcon.svelte';
  import { OUTCOME_KINDS, type OutcomeKind } from '$lib/domain/family';

  const cfg = $derived(familyStore.byFamily[familyStore.active]);
  let modelOptionsOpen = $state(false);

  const activeOutcome = $derived(familyStore.activeOutcome);
  const outcomeIsBinary = $derived(activeOutcome === 'logit' || activeOutcome === 'probit');
  const outcomeIsPoisson = $derived(activeOutcome === 'poisson');

  // Regression and mixed use different storage (store field vs cluster config); kept
  // inline rather than extracted to avoid a wrapper for two callers. Seed the
  // baseline the chosen outcome needs (binary → probability in (0,1); Poisson →
  // rate λ + raw τ²) so the adapter's validation has a value to read.
  function setOutcome(kind: OutcomeKind) {
    if (familyStore.active === 'regression') {
      familyStore.regressionOutcome = kind;
    } else if (familyStore.active === 'mixed' && cfg.cluster) {
      cfg.cluster.outcomeKind = kind;
      if ((kind === 'logit' || kind === 'probit') && cfg.cluster.baselineProbability === undefined) {
        cfg.cluster.baselineProbability = 0.2;
      }
      if (kind === 'poisson') {
        if (cfg.cluster.baselineRate === undefined) cfg.cluster.baselineRate = 2.0;
        if (cfg.cluster.tauSquared === undefined) cfg.cluster.tauSquared = 0.5;
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
    <div role="group" aria-label="Outcome type" class="flex flex-wrap items-center gap-2 text-sm">
      <span class="text-sm font-medium">Outcome type</span>
      <InfoIcon tipKey="outcomeType" />
      {#each OUTCOME_KINDS as entry, i (entry.kind)}
        {#if entry.group === 'advanced' && (i === 0 || OUTCOME_KINDS[i - 1]?.group !== 'advanced')}
          <span aria-hidden="true" class="mx-1 h-5 w-px bg-border"></span>
        {/if}
        <button
          type="button"
          class="rounded px-3 py-1 {activeOutcome === entry.kind
            ? 'bg-primary text-primary-foreground'
            : 'bg-muted text-muted-foreground hover:bg-muted/70'}"
          onclick={() => setOutcome(entry.kind)}
        >{entry.label}</button>
      {/each}
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
  {#if outcomeIsPoisson}
    <BaselineRateInput />
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
