<script lang="ts">
  // One predictor as a compact card: header = name + type selector + the factor
  // levels count / binary share + a ⚙ Advanced button (labels, shares,
  // reference, distribution live in PredictorAdvancedDialog). Body = the
  // variable's effect row(s). Continuous/binary collapse to a single label-less
  // effect row; a factor lists its reference level first then a row per
  // non-reference level, labelled `[level]` (the variable name is the card).
  // ANOVA mode: pass nameEditable / fixedKind / minLevels=2 / onRemove to
  // switch from formula-derived read-only headers to user-managed ANOVA rows.
  import {
    Select,
    SelectContent,
    SelectItem,
    SelectTrigger,
  } from '$lib/components/ui/select';
  import { NumberInput } from '$lib/components/ui/number-input';
  import { Button } from '$lib/components/ui/button';
  import { Input } from '$lib/components/ui/input';
  import Settings2 from '@lucide/svelte/icons/settings-2';
  import Trash2 from '@lucide/svelte/icons/trash-2';
  import EffectControls from './EffectControls.svelte';
  import PredictorAdvancedDialog from './PredictorAdvancedDialog.svelte';
  import InfoIcon from '$lib/components/guidance/InfoIcon.svelte';
  import type { VariableGroup } from '$lib/domain/effect-names';
  import type { EffectRow, VariableKind, VariableRow } from '$lib/domain/family';

  const KINDS: VariableKind[] = ['continuous', 'binary', 'factor'];

  let {
    variable = $bindable(),
    group,
    effects,
    variables,
    locked = false,
    /** Factor floor: 3 for formula families (2-level = binary), 2 for ANOVA. */
    minLevels = 3,
    /** ANOVA: names are typed, not formula-derived. */
    nameEditable = false,
    /** ANOVA primary factors: kind is structurally 'factor' — show a badge, no select. */
    fixedKind = false,
    /** ANOVA: rows are user-managed — show a trash button. */
    onRemove,
  }: {
    variable: VariableRow;
    group: VariableGroup;
    effects: EffectRow[];
    variables: VariableRow[];
    locked?: boolean;
    minLevels?: number;
    nameEditable?: boolean;
    fixedKind?: boolean;
    onRemove?: () => void;
  } = $props();

  let advancedOpen = $state(false);

  // A bare single effect row (continuous/binary) drops its label entirely.
  const singleRow = $derived(group.rows.length === 1 && !group.rows[0]!.isReference);

  function ensureFactorState(row: VariableRow) {
    if (row.kind !== 'factor') return;
    if (typeof row.nLevels !== 'number' || row.nLevels < 2) row.nLevels = minLevels;
    if (!Array.isArray(row.levelProportions) || row.levelProportions.length !== row.nLevels) {
      const each = 1 / row.nLevels;
      row.levelProportions = Array.from({ length: row.nLevels }, () => each);
    }
  }
  function ensureBinaryState(row: VariableRow) {
    if (row.kind !== 'binary') return;
    if (typeof row.binaryProportion !== 'number') row.binaryProportion = 0.5;
  }
  function onKindChange(k: string) {
    if (!KINDS.includes(k as VariableKind)) return;
    variable.kind = k as VariableKind;
    ensureBinaryState(variable);
    ensureFactorState(variable);
  }
  function resizeLevels(n: number) {
    if (!Number.isFinite(n) || n < minLevels) return;
    const next = Math.floor(n);
    variable.nLevels = next;
    // Changing the level count resets proportions to equal — old shares no longer
    // map cleanly onto a different number of levels.
    const current = variable.levelProportions ?? [];
    if (current.length !== next) {
      const each = 1 / next;
      variable.levelProportions = Array.from({ length: next }, () => each);
    }
    // Keep levels (label strings) in sync without losing already-typed names
    const currentLabels = variable.levels ?? [];
    if (currentLabels.length !== next) {
      variable.levels = Array.from({ length: next }, (_, i) => currentLabels[i] ?? String(i + 1));
    }
    // If the referenceLevel is no longer valid after shrink, reset to first label
    if (variable.referenceLevel && !variable.levels!.includes(variable.referenceLevel)) {
      variable.referenceLevel = variable.levels![0] ?? '';
    }
  }

  function effectFor(name: string): EffectRow | undefined {
    return effects.find((e) => e.name === name);
  }
  // Display only the `[level]` part of a factor dummy row — the variable name
  // is already the card header.
  function rowLabel(name: string): string {
    return name.startsWith(variable.name) ? name.slice(variable.name.length) : name;
  }
  // Display a stored proportion as a percentage without forcing integers
  // (e.g. equal thirds → 33.33 instead of a misleading 33).
  function pct(p: number | undefined): number {
    return Math.round((p ?? 0) * 10000) / 100;
  }
</script>

<div class="rounded-md border border-border bg-card px-2.5 py-2">
  <div class="flex flex-wrap items-center gap-2">
    {#if nameEditable}
      <Input class="h-7 w-24 font-mono text-[13px]" aria-label="Variable name" bind:value={variable.name} />
    {:else}
      <span class="font-mono text-[13px] font-semibold" title={variable.name}>{variable.name}</span>
    {/if}
    {#if locked}
      <span class="rounded bg-muted px-1.5 py-0.5 text-xs text-muted-foreground">{variable.kind}</span>
      <span class="rounded bg-primary/10 px-1.5 py-0.5 text-xs font-medium text-primary">from data</span>
      <InfoIcon tipKey="uploadedType" />
    {:else if fixedKind}
      <span class="rounded bg-muted px-1.5 py-0.5 text-xs text-muted-foreground">{variable.kind}</span>
    {:else}
      <Select type="single" value={variable.kind} onValueChange={onKindChange}>
        <SelectTrigger class="h-7 w-32 text-xs">{variable.kind}</SelectTrigger>
        <SelectContent>
          {#each KINDS as k (k)}
            <SelectItem value={k}>{k}</SelectItem>
          {/each}
        </SelectContent>
      </Select>
    {/if}
    {#if variable.kind === 'factor'}
      <span class="text-xs text-muted-foreground">levels</span>
      <NumberInput
        class="h-7 w-24 shrink-0"
        step={1}
        min={minLevels}
        value={variable.nLevels ?? minLevels}
        disabled={locked}
        disableDecrement={(variable.nLevels ?? minLevels) <= minLevels}
        decrementTitle={minLevels <= 2 ? 'Minimum 2 levels for ANOVA factors' : 'Use a binary variable for 2-level factors'}
        oninput={resizeLevels}
      />
    {:else if variable.kind === 'binary'}
      <!-- Linked pair: both steppers edit binaryProportion, so moving one
           moves the other to keep the sum at 100%. -->
      <span class="text-xs text-muted-foreground">share&nbsp;0:</span>
      <NumberInput
        class="h-7 w-24 shrink-0"
        dense
        step={5}
        min={0}
        max={100}
        suffix="%"
        disabled={locked}
        aria-label="Share of 0s"
        value={pct(1 - (variable.binaryProportion ?? 0.5))}
        oninput={(n: number) => (variable.binaryProportion = (100 - n) / 100)}
      />
      <span class="text-xs text-muted-foreground">1:</span>
      <NumberInput
        class="h-7 w-24 shrink-0"
        dense
        step={5}
        min={0}
        max={100}
        suffix="%"
        disabled={locked}
        aria-label="Share of 1s"
        value={pct(variable.binaryProportion ?? 0.5)}
        oninput={(n: number) => (variable.binaryProportion = n / 100)}
      />
    {/if}
    <Button
      variant="outline"
      size="sm"
      class="ml-auto h-7 shrink-0 px-2 text-xs"
      data-testid={`advanced-${variable.name}`}
      onclick={() => (advancedOpen = true)}
    >
      <Settings2 class="mr-1 h-3.5 w-3.5" /> Advanced
    </Button>
    {#if onRemove}
      <Button variant="ghost" size="icon" class="h-7 w-7 shrink-0" aria-label="Remove" onclick={onRemove}>
        <Trash2 class="h-4 w-4" />
      </Button>
    {/if}
  </div>

  <div class="mt-2 text-[11.5px] text-muted-foreground">effects</div>
  {#if singleRow}
    {@const e = effectFor(group.rows[0]!.name)}
    {#if e}
      <div class="mt-2 flex items-center">
        <EffectControls effect={e} {variables} />
      </div>
    {/if}
  {:else}
    {#each group.rows as row (row.name)}
      {#if row.isReference}
        <div class="mt-2 flex items-center gap-2">
          <span class="min-w-16 font-mono text-xs text-muted-foreground">{rowLabel(row.name)}</span>
          <span class="text-xs italic text-muted-foreground">reference</span>
        </div>
      {:else}
        {@const e = effectFor(row.name)}
        {#if e}
          <div class="mt-2 flex items-center gap-2">
            <span class="min-w-16 font-mono text-xs">{rowLabel(row.name)}</span>
            <EffectControls effect={e} {variables} />
          </div>
        {/if}
      {/if}
    {/each}
  {/if}
</div>

<PredictorAdvancedDialog bind:variable bind:open={advancedOpen} {locked} />
