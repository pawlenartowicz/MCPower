<script lang="ts">
  // Multiple-comparison correction selector; resets to family default when the active family changes.
  import {
    Select,
    SelectContent,
    SelectItem,
    SelectTrigger,
  } from '$lib/components/ui/select';
  import { Label } from '$lib/components/ui/label';
  import InfoIcon from '$lib/components/guidance/InfoIcon.svelte';
  import { familyStore } from '$lib/stores/family.svelte';
  import {
    CORRECTION_DEFAULT,
    CORRECTION_LABEL,
    CORRECTION_OPTIONS,
    type CorrectionMethod,
  } from '$lib/domain/correction-options';

  const cfg = $derived(familyStore.byFamily[familyStore.active]);
  const options = $derived(CORRECTION_OPTIONS[familyStore.active]);

  // Preserve current selection across family switches when it appears in the
  // new family's set; otherwise fall back to that family's default.
  $effect(() => {
    if (!options.includes(cfg.advanced.correction)) {
      cfg.advanced.correction = CORRECTION_DEFAULT[familyStore.active];
    }
  });
</script>

<div>
  <div class="flex items-center gap-2">
    <Label for="correction">Correction</Label>
    <InfoIcon tipKey="correction" />
  </div>
  <Select
    type="single"
    value={cfg.advanced.correction}
    onValueChange={(v: string) => {
      if (options.includes(v as CorrectionMethod)) {
        cfg.advanced.correction = v as CorrectionMethod;
      }
    }}
  >
    <SelectTrigger id="correction">{CORRECTION_LABEL[cfg.advanced.correction]}</SelectTrigger>
    <SelectContent>
      {#each options as c (c)}
        <SelectItem value={c}>{CORRECTION_LABEL[c]}</SelectItem>
      {/each}
    </SelectContent>
  </Select>
</div>
