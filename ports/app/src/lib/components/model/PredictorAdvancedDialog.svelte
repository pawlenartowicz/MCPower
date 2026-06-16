<script lang="ts">
  // Advanced popup for one predictor (opened from VariableCard's ⚙ button).
  // One sectioned dialog, content by variable kind: continuous → distribution
  // glyph cards; binary → linked 1s/0s share pair; factor → levels table (label · share ·
  // reference) + sampled-shares toggle. Upload-locked predictors open read-only.
  import { Dialog, DialogContent, DialogTitle } from '$lib/components/ui/dialog';
  import { Input } from '$lib/components/ui/input';
  import { NumberInput } from '$lib/components/ui/number-input';
  import { Button } from '$lib/components/ui/button';
  import { Label } from '$lib/components/ui/label';
  import RotateCw from '@lucide/svelte/icons/rotate-cw';
  import { familyStore } from '$lib/stores/family.svelte';
  import { factorLevels } from '$lib/domain/effect-names';
  import { configPaneDialogStyle } from './dialog-position.svelte';
  import InfoIcon from '$lib/components/guidance/InfoIcon.svelte';
  import type { ContinuousDistribution, VariableRow } from '$lib/domain/family';

  // Below this many expected observations in a level, warn that n may not
  // populate the cell (shares are weights, so expected obs = n·wᵢ/Σw).
  const SMALL_CELL_MIN_OBS = 20;

  // Factor-share allocation tri-state — mirrors VariableRow.sampledProportions
  // (Option<bool> on the wire): `default` clears it (None → inherit each
  // scenario's `sampled_factor_proportions`), the other two pin the override.
  const SHARE_MODES: Array<{ key: string; value: boolean | undefined; label: string; hint: string }> = [
    { key: 'default', value: undefined, label: 'Default', hint: "Follows each scenario's setting." },
    { key: 'exact', value: false, label: 'Exact', hint: 'Allocate level counts deterministically every run.' },
    { key: 'sampled', value: true, label: 'Sampled', hint: 'Draw level counts multinomially each run.' },
  ];

  // "default" entry unpins — scenarios decide. Choosing any explicit option (incl. normal) pins it.
  const DEFAULT_DIST = '__default__';
  const DISTRIBUTIONS: Array<{ value: ContinuousDistribution | typeof DEFAULT_DIST; glyph: string; label: string; hint?: string }> = [
    { value: DEFAULT_DIST, glyph: '·····', label: 'default', hint: 'Scenarios may swap this predictor.' },
    { value: 'normal', glyph: '▁▄█▄▁', label: 'normal', hint: 'Pinned — no scenario swaps.' },
    { value: 'right_skewed', glyph: '▁█▄▂▁', label: 'right-skewed' },
    { value: 'left_skewed', glyph: '▁▂▄█▁', label: 'left-skewed' },
    { value: 'high_kurtosis', glyph: '▂▂█▂▂', label: 'heavy tails' },
    { value: 'uniform', glyph: '▅▅▅▅▅', label: 'uniform' },
  ];

  let {
    variable = $bindable(),
    open = $bindable(false),
    locked = false,
  }: {
    variable: VariableRow;
    open?: boolean;
    locked?: boolean;
  } = $props();

  // Resolved labels (blank slots → "i+1") — same source the effect rows and
  // the engine adapter use, so what's shown here is what the run uses.
  const labels = $derived(factorLevels(variable).slice(0, variable.nLevels ?? 0));
  const refIdx = $derived.by(() => {
    const r = variable.referenceLevel;
    if (!r) return 0;
    const idx = labels.indexOf(r);
    return idx === -1 ? 0 : idx;
  });

  function setLabel(idx: number, value: string) {
    const wasReference = idx === refIdx;
    if (!variable.levels) {
      variable.levels = Array.from({ length: variable.nLevels ?? 0 }, (_, i) => String(i + 1));
    }
    variable.levels[idx] = value;
    // The reference is stored by label; keep it pointing at this slot.
    if (wasReference) variable.referenceLevel = labels[idx];
  }

  function setShare(idx: number, pctValue: number) {
    if (variable.levelProportions) variable.levelProportions[idx] = pctValue / 100;
  }

  function normalizeProportions() {
    const props = variable.levelProportions;
    if (!Array.isArray(props) || props.length === 0) return;
    const sum = props.reduce((a, b) => a + (Number.isFinite(b) ? b : 0), 0);
    variable.levelProportions =
      sum <= 0 ? props.map(() => 1 / props.length) : props.map((p) => (Number.isFinite(p) ? p / sum : 0));
  }

  // Display a stored proportion as a percentage without forcing integers.
  function pct(p: number | undefined): number {
    return Math.round((p ?? 0) * 10000) / 100;
  }

  // Small-cell check against the find-power sample size: weights are
  // normalized, so a level's expected count is n·wᵢ/Σw.
  const findPowerN = $derived(familyStore.byFamily[familyStore.active].findPower?.n ?? 0);
  const smallCells = $derived.by(() => {
    if (variable.kind !== 'factor' || findPowerN <= 0) return [];
    const weights = variable.levelProportions ?? [];
    const sum = weights.reduce((a, b) => a + (Number.isFinite(b) && b > 0 ? b : 0), 0);
    if (sum <= 0) return [];
    const out: Array<{ label: string; expected: number }> = [];
    weights.forEach((w, i) => {
      const expected = (findPowerN * (Number.isFinite(w) && w > 0 ? w : 0)) / sum;
      if (expected < SMALL_CELL_MIN_OBS) out.push({ label: labels[i] ?? String(i + 1), expected });
    });
    return out;
  });
</script>

<Dialog bind:open>
  <DialogContent class="max-h-[85vh] overflow-y-auto sm:max-w-md" style={configPaneDialogStyle()}>
    <DialogTitle>
      <span class="font-mono">{variable.name}</span>
      <span class="ml-1 text-sm font-normal text-muted-foreground">— {variable.kind}</span>
      {#if locked}
        <span class="ml-2 rounded bg-primary/10 px-1.5 py-0.5 text-xs font-medium text-primary">from data</span>
      {/if}
    </DialogTitle>

    {#if variable.kind === 'continuous'}
      <div class="space-y-2">
        <p class="text-xs font-medium text-muted-foreground">
          Distribution
          <span class="ml-1 font-normal text-muted-foreground/70">
            — choosing an explicit option pins this predictor (no scenario swaps)
          </span>
        </p>
        <div class="flex flex-wrap gap-2" role="radiogroup" aria-label="Distribution">
          {#each DISTRIBUTIONS as d (d.value)}
            {@const isDefault = d.value === DEFAULT_DIST}
            {@const selected = isDefault
              ? !variable.pinned
              : variable.pinned && (variable.distribution ?? 'normal') === d.value}
            <button
              type="button"
              role="radio"
              aria-checked={selected}
              data-testid={`dist-${d.value}`}
              disabled={locked}
              title={d.hint}
              class="flex flex-col items-center gap-1 rounded-lg border px-3 py-2 text-sm transition-colors
                {selected
                  ? 'border-primary bg-primary/10 text-foreground'
                  : 'border-border text-muted-foreground hover:bg-muted/60'}
                disabled:cursor-not-allowed disabled:opacity-60"
              onclick={() => {
                if (isDefault) {
                  variable.pinned = false;
                  variable.distribution = undefined;
                } else {
                  variable.pinned = true;
                  variable.distribution = d.value as ContinuousDistribution;
                }
              }}
            >
              <span class="font-mono text-base leading-none" aria-hidden="true">{d.glyph}</span>
              <span>{d.label}</span>
            </button>
          {/each}
        </div>
      </div>
    {:else if variable.kind === 'binary'}
      <div class="space-y-1.5">
        <!-- Linked pair: both rows edit the single binaryProportion, so
             moving one stepper moves the other to keep the sum at 100%. -->
        <div class="flex items-center justify-between gap-3">
          <Label for="adv-binary-share-zeros">Share of 0s</Label>
          <NumberInput
            id="adv-binary-share-zeros"
            class="w-32"
            step={5}
            min={0}
            max={100}
            suffix="%"
            disabled={locked}
            value={pct(1 - (variable.binaryProportion ?? 0.5))}
            oninput={(n: number) => (variable.binaryProportion = (100 - n) / 100)}
          />
        </div>
        <div class="flex items-center justify-between gap-3">
          <Label for="adv-binary-share">Share of 1s</Label>
          <NumberInput
            id="adv-binary-share"
            class="w-32"
            step={5}
            min={0}
            max={100}
            suffix="%"
            disabled={locked}
            value={pct(variable.binaryProportion ?? 0.5)}
            oninput={(n: number) => (variable.binaryProportion = n / 100)}
          />
        </div>
      </div>
    {:else if variable.kind === 'factor'}
      <div class="space-y-2">
        <div class="grid grid-cols-[1fr_8rem_5rem] items-center gap-x-2 gap-y-1.5 text-sm">
          <span class="text-muted-foreground">label</span>
          <span class="text-muted-foreground">share</span>
          <span class="text-center text-muted-foreground">reference</span>
          {#each variable.levelProportions ?? [] as _p, idx (idx)}
            <Input
              class="h-8 font-mono text-sm"
              placeholder={String(idx + 1)}
              disabled={locked}
              value={variable.levels?.[idx] ?? ''}
              data-testid={`level-label-${idx}`}
              oninput={(e) => setLabel(idx, (e.target as HTMLInputElement).value)}
            />
            <NumberInput
              class="h-8 w-full"
              step={5}
              min={0}
              disabled={locked}
              value={pct(variable.levelProportions?.[idx])}
              oninput={(n: number) => setShare(idx, n)}
            />
            <div class="flex justify-center">
              <input
                type="radio"
                name={`reference-${variable.name}`}
                aria-label={`Use ${labels[idx]} as reference`}
                checked={idx === refIdx}
                disabled={locked}
                onchange={() => (variable.referenceLevel = labels[idx])}
              />
            </div>
          {/each}
        </div>
        <div class="flex items-start gap-1.5">
          <p class="text-xs text-muted-foreground">
            Shares are weights — they don't need to sum to 100%; they're rescaled automatically.
          </p>
          {#if !locked}
            <Button
              variant="ghost"
              size="icon"
              class="h-6 w-6 shrink-0"
              aria-label="Rescale shares to sum to 100%"
              title="Rescale shares so they sum to 100% (optional)"
              onclick={normalizeProportions}
            >
              <RotateCw class="h-3.5 w-3.5" />
            </Button>
          {/if}
        </div>
        {#each smallCells as cell (cell.label)}
          <p class="text-xs text-amber-700 dark:text-amber-500" data-testid="small-cell-warning">
            ~{Math.round(cell.expected)} obs in [{cell.label}] at n={findPowerN} — make sure n is
            enough to populate this level.
          </p>
        {/each}
        <div class="space-y-2 border-t border-border pt-2">
          <div class="flex items-center gap-1.5">
            <Label>Factor shares</Label>
            <InfoIcon tipKey="sampledShares" />
          </div>
          <p class="text-xs text-muted-foreground">
            How level counts are allocated each run — <span class="font-medium">Default</span> follows
            each scenario, <span class="font-medium">Exact</span> allocates deterministically,
            <span class="font-medium">Sampled</span> draws multinomially.
          </p>
          <div class="flex flex-wrap gap-2" role="radiogroup" aria-label="Factor share allocation">
            {#each SHARE_MODES as m (m.key)}
              {@const selected = variable.sampledProportions === m.value}
              <button
                type="button"
                role="radio"
                aria-checked={selected}
                data-testid={`shares-${m.key}`}
                disabled={locked}
                title={m.hint}
                class="rounded-lg border px-3 py-2 text-sm transition-colors
                  {selected
                    ? 'border-primary bg-primary/10 text-foreground'
                    : 'border-border text-muted-foreground hover:bg-muted/60'}
                  disabled:cursor-not-allowed disabled:opacity-60"
                onclick={() => (variable.sampledProportions = m.value)}
              >
                {m.label}
              </button>
            {/each}
          </div>
        </div>
      </div>
    {/if}
  </DialogContent>
</Dialog>
