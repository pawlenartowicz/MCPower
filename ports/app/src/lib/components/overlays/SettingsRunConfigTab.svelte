<script lang="ts">
  // Settings tab for per-family simulation expert options: sim count, alpha, seed, max-fail,
  // and (Tauri-only) engine thread count.
  import { NumberInput } from '$lib/components/ui/number-input';
  import { Label } from '$lib/components/ui/label';
  import InfoIcon from '$lib/components/guidance/InfoIcon.svelte';
  import { familyStore } from '$lib/stores/family.svelte';
  import { sharedPrefs } from '$lib/stores/shared-prefs.svelte';

  const cfg = $derived(familyStore.byFamily[familyStore.active]);
  // Tauri-only: show the thread-count control only in the desktop shell.
  const isTauri = import.meta.env.VITE_TARGET !== 'wasm';
</script>

<div class="space-y-3 px-1 pt-1">
  <div class="flex items-center gap-2">
    <p class="text-xs text-muted-foreground">
      Session-stable expert settings for the active family ({familyStore.active}).
    </p>
    <InfoIcon tipKey="advanced" />
  </div>
  <div class="space-y-3">
    <div class="flex items-center justify-between gap-3">
      <Label for="sims">Simulations</Label>
      <NumberInput
        id="sims"
        class="w-28"
        step={100}
        min={0}
        bind:value={cfg.advanced.simulations}
      />
    </div>
    <div class="flex items-center justify-between gap-3">
      <Label for="alpha">Alpha</Label>
      <NumberInput
        id="alpha"
        class="w-28"
        step={0.01}
        min={0}
        max={1}
        bind:value={cfg.alpha}
      />
    </div>
    <div class="flex items-center justify-between gap-3">
      <Label for="seed">Seed</Label>
      <NumberInput id="seed" class="w-28" step={1} bind:value={cfg.advanced.seed} />
    </div>
    <div class="flex items-center justify-between gap-3">
      <Label for="max-fail">Max failed sims</Label>
      <NumberInput
        id="max-fail"
        class="w-28"
        step={0.01}
        min={0}
        max={1}
        bind:value={cfg.advanced.maxFailedSimulations}
      />
    </div>
    {#if isTauri}
      <div class="flex items-center justify-between gap-3">
        <div>
          <Label for="n-threads">Threads</Label>
          <p class="text-xs text-muted-foreground">
            Takes effect on next restart. Blank = all cores.
          </p>
        </div>
        <!-- Plain input: NumberInput requires a number, but null ("all cores") is
             a valid state here. min="" + placeholder avoids a spurious 0 default. -->
        <input
          id="n-threads"
          type="number"
          class="w-28 rounded-md border border-input bg-transparent px-2 py-1 text-center text-sm text-foreground shadow-xs focus:border-ring focus:ring-3 focus:ring-ring/50 focus:outline-none [appearance:textfield] [&::-webkit-inner-spin-button]:m-0 [&::-webkit-inner-spin-button]:appearance-none [&::-webkit-outer-spin-button]:appearance-none"
          step={1}
          min={1}
          placeholder="auto"
          value={sharedPrefs.nThreads ?? ''}
          onchange={(e: Event) => {
            const raw = (e.target as HTMLInputElement).value.trim();
            const parsed = parseInt(raw, 10);
            sharedPrefs.nThreads = raw === '' || isNaN(parsed) || parsed < 1 ? null : parsed;
          }}
        />
      </div>
    {/if}
  </div>
</div>
