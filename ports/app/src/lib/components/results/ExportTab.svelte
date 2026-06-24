<script lang="ts">
  // Export tab: renders a selected plot block offscreen with the print theme,
  // then saves as PNG or SVG. No Rust command needed — all browser-side.
  import type { RunTab } from '$lib/stores/run.svelte';
  import { embedPrint } from '$lib/charts/embed';
  import { NumberInput } from '$lib/components/ui/number-input';
  import InfoIcon from '$lib/components/guidance/InfoIcon.svelte';
  import plotThemes from '$configs/plot-themes.json';

  interface Props { tab: RunTab; }
  const { tab }: Props = $props();

  type Format = 'png' | 'svg';

  let format = $state<Format>('png');
  let scale = $state(2);
  let saving = $state(false);
  let errorMsg = $state<string | null>(null);

  const hasPlots = $derived((tab.plots?.blocks?.length ?? 0) > 0);

  // Human-readable label for a block key.
  function blockLabel(key: string): string {
    if (key === 'power') return 'Power by effect';
    if (key === 'curve') return 'Power curve';
    if (key === 'overlay') return '⧉ Overlay';
    if (key === 'at_least_k') return 'At least k';
    if (key === 'exactly_k') return 'Exactly k';
    if (key.startsWith('scenario:')) return key.slice('scenario:'.length);
    return key;
  }

  let selectedKey = $state('');
  // Keep selectedKey valid when plots change (e.g. first render).
  $effect(() => {
    const blocks = tab.plots?.blocks ?? [];
    if (blocks.length > 0 && !blocks.some((b) => b.key === selectedKey)) {
      selectedKey = blocks[0]!.key;
    }
  });

  const selectedSpec = $derived(
    tab.plots?.blocks?.find((b) => b.key === selectedKey)?.spec ?? null
  );

  const printTheme = (plotThemes as Record<string, Record<string, unknown>>)['light-print']!;

  async function save() {
    const specStr = selectedSpec;
    if (!specStr) return;

    saving = true;
    errorMsg = null;

    // Browser build (no Tauri runtime): download via anchor + object URL.
    // Native build: native save dialog + plugin-fs. Mirrors menu.ts export.
    const isWasm = import.meta.env.VITE_TARGET === 'wasm';

    try {
      // Native: pick the save path via dialog first — so if the user cancels we
      // don't do the heavy offscreen embed work. The browser download has no
      // path picker; the browser's own dialog handles naming.
      let path: string | null = null;
      if (!isWasm) {
        const { save: dialogSave } = await import('@tauri-apps/plugin-dialog');
        path = await dialogSave({
          defaultPath: `mcpower-plot.${format}`,
          filters: [
            format === 'png'
              ? { name: 'PNG image', extensions: ['png'] }
              : { name: 'SVG image', extensions: ['svg'] },
          ],
        });
        if (!path) return; // user cancelled
      }

      // Offscreen embed with print theme: detached div, never appended to the document.
      const container = document.createElement('div');
      const result = await embedPrint(container, specStr, printTheme);
      try {
        if (format === 'svg') {
          const svgStr = await result.view.toSVG();
          if (isWasm) {
            const { downloadBlob } = await import('$lib/api/menu-wasm');
            downloadBlob('mcpower-plot.svg', new Blob([svgStr], { type: 'image/svg+xml' }));
          } else {
            const { writeTextFile } = await import('@tauri-apps/plugin-fs');
            await writeTextFile(path!, svgStr);
          }
        } else {
          const dataUrl = await result.view.toImageURL('png', scale);
          // data URL → Uint8Array
          const base64 = dataUrl.slice(dataUrl.indexOf(',') + 1);
          const binary = atob(base64);
          const bytes = new Uint8Array(binary.length);
          for (let i = 0; i < binary.length; i++) {
            bytes[i] = binary.charCodeAt(i);
          }
          if (isWasm) {
            const { downloadBlob } = await import('$lib/api/menu-wasm');
            downloadBlob('mcpower-plot.png', new Blob([bytes], { type: 'image/png' }));
          } else {
            const { writeFile } = await import('@tauri-apps/plugin-fs');
            await writeFile(path!, bytes);
          }
        }
      } finally {
        result.finalize();
      }
    } catch (err) {
      errorMsg = err instanceof Error ? err.message : String(err);
    } finally {
      saving = false;
    }
  }
</script>

<div data-testid="export-tab" class="space-y-4">
  {#if !hasPlots}
    <p data-testid="export-no-plots" class="text-sm text-muted-foreground">
      No plot available for this run. Re-run to generate an exportable chart.
    </p>
  {:else}
    <div class="flex flex-wrap items-end gap-4">
      <!-- Block selector: choose which plot to export -->
      <div class="flex flex-col gap-1">
        <label for="export-block" class="flex items-center gap-1 text-xs text-muted-foreground">Chart <InfoIcon tipKey="exportPlot" /></label>
        <select
          id="export-block"
          data-testid="export-block-select"
          class="rounded border border-border bg-background px-2 py-1 text-sm"
          bind:value={selectedKey}
        >
          {#each tab.plots?.blocks ?? [] as block (block.key)}
            <option value={block.key}>{blockLabel(block.key)}</option>
          {/each}
        </select>
      </div>

      <!-- Format selector -->
      <div class="flex flex-col gap-1">
        <label for="export-format" class="text-xs text-muted-foreground">Format</label>
        <select
          id="export-format"
          data-testid="export-format-select"
          class="rounded border border-border bg-background px-2 py-1 text-sm"
          bind:value={format}
        >
          <option value="png">PNG</option>
          <option value="svg">SVG</option>
        </select>
      </div>

      <!-- Scale input — PNG only -->
      {#if format === 'png'}
        <div class="flex flex-col gap-1">
          <label for="export-scale" class="text-xs text-muted-foreground">Scale</label>
          <NumberInput
            id="export-scale"
            data-testid="export-scale-input"
            class="w-28"
            min={1}
            max={4}
            step={0.5}
            bind:value={scale}
          />
        </div>
      {/if}

      <!-- Save button -->
      <button
        type="button"
        data-testid="export-save-btn"
        class="rounded bg-primary px-4 py-1.5 text-sm text-primary-foreground disabled:opacity-50"
        disabled={saving}
        onclick={() => void save()}
      >
        {saving ? 'Saving…' : 'Save'}
      </button>
    </div>

    {#if errorMsg}
      <p data-testid="export-error" class="text-sm text-destructive">{errorMsg}</p>
    {/if}

    <p class="text-xs text-muted-foreground">
      Exports the selected chart with a print-ready theme (white background, colour-blind-safe palette).
    </p>
  {/if}
</div>
