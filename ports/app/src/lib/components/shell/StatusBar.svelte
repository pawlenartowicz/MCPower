<script lang="ts">
// Bottom action bar: run summary, Find Power / Find Sample buttons, Scenarios toggle, status badge, and Reset.
import Play from '@lucide/svelte/icons/play';
import Layers from '@lucide/svelte/icons/layers';
import { Badge } from '$lib/components/ui/badge';
import { Button } from '$lib/components/ui/button';
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from '$lib/components/ui/dialog';
import { Switch } from '$lib/components/ui/switch';
import { demoStore } from '$lib/stores/demo.svelte';
import { familyStore } from '$lib/stores/family.svelte';
import { scenariosStore } from '$lib/stores/scenarios.svelte';
import { uiStore } from '$lib/stores/ui.svelte';
import { sharedPrefs } from '$lib/stores/shared-prefs.svelte';
import { startRun } from '$lib/run/run';
import { runStore } from '$lib/stores/run.svelte';
import { familyConfigToAppSpec } from '$lib/domain/app-spec-adapter';
import { SELECTABLE_FAMILIES } from '$lib/domain/family';
import { SIMULATION } from '$lib/configs/app-config';

// collapsed (mobile only): hide the scenarios toggle on scroll-down, like the family
// ribbon. The Find buttons and the readiness badge stay; desktop ignores it.
let { collapsed = false }: { collapsed?: boolean } = $props();

const summary = $derived.by(() => {
  const cfg = familyStore.byFamily[familyStore.active];
  return `target=${cfg.targetPower}% · α=${cfg.alpha}`;
});

const sampleSize = $derived.by(() => {
  const s = familyStore.byFamily[familyStore.active].findSampleSize;
  const byLabel = s.by === 'auto' ? 'auto' : String(s.by);
  return `sample size | from ${s.from} to ${s.to} by ${byLabel}`;
});

const isWired = $derived(SELECTABLE_FAMILIES.includes(familyStore.active));

const adaptResult = $derived.by(() => {
  if (!isWired) return null;
  return familyConfigToAppSpec(
    familyStore.active,
    familyStore.byFamily[familyStore.active],
    familyStore.regressionOutcome,
  );
});

const wiredDisabled = $derived(
  isWired &&
  (!adaptResult?.spec || (adaptResult?.errors?.length ?? 0) > 0 || runStore.runState === 'running')
);
// Non-blocking config warnings (e.g. alpha above the usual max, extreme baseline
// probability). Shown but never gate the run — mirrors the Python/R soft warns.
const configWarnings = $derived(isWired ? (adaptResult?.warnings ?? []) : []);
const isRunning = $derived(isWired && runStore.runState === 'running');
const nonWiredDisabled = $derived(
  !isWired &&
  (demoStore.runState === 'running' || demoStore.runState === 'not-ready')
);

const status = $derived.by(() => {
  if (isWired) {
    switch (runStore.runState) {
      case 'running':   return { text: '● Running…', tone: 'amber' };
      case 'done':      return { text: '● Last run done', tone: 'green' };
      case 'error':     return { text: '● Error', tone: 'red' };
      case 'cancelled': return { text: '● Cancelled', tone: 'neutral' };
      default:          return { text: '● Ready to run', tone: 'green' };
    }
  }
  switch (demoStore.runState) {
    case 'running':
      return { text: '● Running…', tone: 'amber' };
    case 'done':
      return { text: '● Last run 1.6 s ago', tone: 'green' };
    case 'not-ready':
      return { text: '● Set an effect size', tone: 'neutral' };
    default:
      return { text: '● Ready to run', tone: 'green' };
  }
});

async function onFindPower() {
  // On phones the two panes are toggled, not side-by-side: jump to Results so the
  // run's progress and output are visible without a manual switch (no-op on desktop).
  sharedPrefs.activePane = 'results';
  if (!isWired) return startRun('find-power');
  const spec = adaptResult?.spec;
  if (!spec) return;
  const cfg = familyStore.byFamily[familyStore.active];
  await runStore.startFindPower(spec, cfg.findPower.n);
}

async function onFindSample() {
  sharedPrefs.activePane = 'results';
  if (!isWired) return startRun('find-sample-size');
  const spec = adaptResult?.spec;
  if (!spec) return;
  const cfg = familyStore.byFamily[familyStore.active];
  const ssb = cfg.findSampleSize;
  const by =
    ssb.by === 'auto'
      ? { Auto: { count: SIMULATION.cluster_auto_count } }
      : { Fixed: ssb.by };
  await runStore.startFindSampleSize(
    spec,
    [ssb.from, ssb.to],
    { Grid: { by, mode: 'Linear' } },
  );
}

async function onCancel() {
  await runStore.cancel();
}

function confirmReset() {
  familyStore.resetActive();
  // More-options dialog state outside FamilyConfig: the scenario distribution
  // pools. Reset them too so the dialog reads all-defaults after a reset.
  scenariosStore.resetDistributionPools();
  uiStore.resetConfirmOpen = false;
}
</script>

<div
  class="flex shrink-0 flex-col items-stretch border-b border-border bg-muted px-6 transition-[padding,gap] duration-200 {collapsed
    ? 'gap-2 py-2'
    : 'gap-3 py-4'} min-[900px]:flex-row min-[900px]:items-center min-[900px]:justify-between min-[900px]:gap-6 min-[900px]:py-4"
>
  <div class="flex min-w-0 flex-col text-sm leading-tight">
    <span class="hidden truncate text-muted-foreground min-[900px]:block">{summary}</span>
    <span class="hidden truncate text-muted-foreground min-[900px]:block">{sampleSize}</span>
    {#if configWarnings.length > 0}
      <span
        class="truncate text-amber-700"
        title={configWarnings.join(' · ')}
        data-testid="config-warnings"
      >⚠ {configWarnings.join(' · ')}</span>
    {/if}
  </div>
  <div class="flex flex-wrap items-center gap-3 min-[900px]:flex-nowrap min-[900px]:shrink-0 min-[900px]:gap-5">
    {#if isRunning}
      <Button variant="destructive" class="h-11 px-5 text-sm" onclick={onCancel}>Cancel</Button>
    {:else}
      <Button class="h-11 px-5 text-sm" disabled={isWired ? wiredDisabled : nonWiredDisabled} onclick={onFindPower}>
        <Play class="mr-2 h-5 w-5" /> Find power
      </Button>
      <Button class="h-11 px-5 text-sm" disabled={isWired ? wiredDisabled : nonWiredDisabled} onclick={onFindSample}>
        <Play class="mr-2 h-5 w-5" /> Find sample
      </Button>
    {/if}
    <div
      class="overflow-hidden transition-[max-height,opacity] duration-200 min-[900px]:max-h-none min-[900px]:opacity-100 {collapsed
        ? 'max-h-0 opacity-0'
        : 'max-h-14 opacity-100'}"
    >
      <label
        class="flex cursor-pointer items-center gap-2 text-sm text-muted-foreground"
        data-testid="scenarios-switcher"
        title="Run alternate scenarios — edit the three sets in Settings · Scenarios"
      >
        <Layers class="h-5 w-5" />
        <span class="leading-tight">Robustness<br />scenarios</span>
        <Switch
          bind:checked={sharedPrefs.scenariosEnabled}
          aria-label="Enable robustness scenarios"
        />
      </label>
    </div>
  </div>
  <div class="flex shrink-0 items-center gap-5">
    <Badge variant="outline" class="text-sm" data-tone={status.tone}>{status.text}</Badge>
  </div>
</div>

<Dialog open={uiStore.resetConfirmOpen} onOpenChange={(v) => (uiStore.resetConfirmOpen = v)}>
  <DialogContent>
    <DialogHeader>
      <DialogTitle>Reset {familyStore.active} configuration?</DialogTitle>
      <DialogDescription>
        This restores defaults for the active family only. Other families are untouched.
      </DialogDescription>
    </DialogHeader>
    <DialogFooter>
      <Button variant="ghost" onclick={() => (uiStore.resetConfirmOpen = false)}>Cancel</Button>
      <Button variant="destructive" onclick={confirmReset}>Reset</Button>
    </DialogFooter>
  </DialogContent>
</Dialog>
