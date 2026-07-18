<script lang="ts">
  // Estimation control for GLMM families (Fast / Accurate / AGQ), mapped to the
  // (wald_se, agq) pair on AdvancedConfig. Picking AGQ reveals a node-count input.
  import {
    Select,
    SelectContent,
    SelectItem,
    SelectTrigger,
  } from '$lib/components/ui/select';
  import { NumberInput } from '$lib/components/ui/number-input';
  import { Label } from '$lib/components/ui/label';
  import InfoIcon from '$lib/components/guidance/InfoIcon.svelte';
  import { familyStore } from '$lib/stores/family.svelte';
  import {
    ESTIMATION_LABEL,
    ESTIMATION_OPTIONS,
    estimationModeOf,
    estimationPair,
    clampAgqNodes,
    AGQ_DEFAULT_NODES,
    AGQ_MIN_NODES,
    AGQ_MAX_NODES,
    type EstimationMode,
  } from '$lib/domain/estimation-options';

  const cfg = $derived(familyStore.byFamily[familyStore.active]);
  const mode = $derived(estimationModeOf(cfg.advanced.wald_se, cfg.advanced.agq));

  function setMode(v: string) {
    if (!ESTIMATION_OPTIONS.includes(v as EstimationMode)) return;
    const pair = estimationPair(v as EstimationMode, cfg.advanced.agq);
    cfg.advanced.wald_se = pair.wald_se;
    cfg.advanced.agq = pair.agq;
  }
</script>

<div>
  <div class="flex items-center gap-2">
    <Label for="estimation-mode">Estimation</Label>
    <InfoIcon tipKey="waldSe" />
  </div>
  <Select type="single" value={mode} onValueChange={setMode}>
    <SelectTrigger id="estimation-mode">{ESTIMATION_LABEL[mode]}</SelectTrigger>
    <SelectContent>
      {#each ESTIMATION_OPTIONS as opt (opt)}
        <SelectItem value={opt}>{ESTIMATION_LABEL[opt]}</SelectItem>
      {/each}
    </SelectContent>
  </Select>
  {#if mode === 'agq'}
    <div class="mt-2 flex items-center gap-2">
      <Label for="agq-nodes" class="text-xs">Quadrature points (odd)</Label>
      <NumberInput
        id="agq-nodes"
        class="w-20"
        step={2}
        min={AGQ_MIN_NODES}
        max={AGQ_MAX_NODES}
        value={cfg.advanced.agq > 1 ? cfg.advanced.agq : AGQ_DEFAULT_NODES}
        oninput={(v) => { cfg.advanced.agq = clampAgqNodes(v); }}
      />
    </div>
  {/if}
</div>
