<script lang="ts">
  // Upload section: file picker → CSV parse → type detection → store CsvData.
  // Renders a type-summary panel (name / type / levels + add-to-formula button)
  // and a none/partial/strict mode selector.
  import { uploadStore } from '$lib/stores/upload.svelte';
  import { parseCsvText, csvDataFromParsed, validateUploadRowCount } from '$lib/domain/upload-detect';
  import { familyStore } from '$lib/stores/family.svelte';
  import type { UploadMode } from '$lib/domain/app-spec';
  import Plus from '@lucide/svelte/icons/plus';
  import InfoIcon from '$lib/components/guidance/InfoIcon.svelte';

  const cfg = $derived(familyStore.byFamily[familyStore.active]);

  let fileInput = $state<HTMLInputElement | undefined>(undefined);
  let parseError = $state<string | null>(null);
  let loading = $state(false);

  const MODE_OPTIONS: { value: UploadMode; label: string; title: string }[] = [
    {
      value: 'none',
      label: 'None',
      title: "Fresh synthetic rows matching each predictor's marginal only; predictors are independent unless you set correlations.",
    },
    {
      value: 'partial',
      label: 'Partial',
      title: 'Fresh synthetic rows matching the marginals plus the correlations measured among continuous predictors.',
    },
    {
      value: 'strict',
      label: 'Strict',
      title: 'Bootstrap whole rows from the uploaded data — preserves the exact joint distribution of the predictors.',
    },
  ];

  async function onFileChange(e: Event) {
    const input = e.target as HTMLInputElement;
    const file = input.files?.[0];
    if (!file) return;

    loading = true;
    parseError = null;

    try {
      const text = await file.text();
      const parsed = parseCsvText(text);
      if (parsed.header.length === 0 || parsed.nRows === 0) {
        parseError = 'File appears empty or has no data rows.';
        return;
      }
      const rowError = validateUploadRowCount(parsed.nRows);
      if (rowError) {
        parseError = rowError;
        return;
      }
      const csvData = csvDataFromParsed(parsed, uploadStore.mode);
      uploadStore.load(csvData, file.name);
    } catch (err) {
      parseError = err instanceof Error ? err.message : 'Failed to parse CSV.';
    } finally {
      loading = false;
    }
  }

  function onModeChange(v: UploadMode) {
    uploadStore.setMode(v);
  }

  function onClear() {
    uploadStore.clear();
    parseError = null;
    if (fileInput) fileInput.value = '';
  }

  // Word-boundary match (treats `.` and word chars as token chars) so e.g. `am`
  // doesn't match inside `team` or `am.x`. Cheaper + synchronous vs the async
  // Tauri-backed parsed-formula store, which is what the disabled state needs.
  function isInFormula(formula: string, name: string): boolean {
    const esc = name.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
    return new RegExp(`(^|[^\\w.])${esc}([^\\w.]|$)`).test(formula);
  }

  function addToFormula(name: string) {
    if (isInFormula(cfg.formula, name)) return;
    const f = cfg.formula.trim();
    cfg.formula = f === '' ? `y = ${name}` : `${f} + ${name}`;
  }
</script>

<div class="space-y-3">
  {#if uploadStore.csvData}
    <div class="flex items-center gap-2">
      <span class="text-xs text-muted-foreground truncate max-w-48" title={uploadStore.filename}>
        {uploadStore.filename}
      </span>
      <button
        type="button"
        class="text-xs text-muted-foreground hover:text-foreground underline"
        onclick={onClear}
      >
        Clear
      </button>
    </div>
  {/if}

  {#if !uploadStore.csvData}
    <div>
      <input
        bind:this={fileInput}
        type="file"
        accept=".csv"
        class="block text-sm file:mr-3 file:rounded file:border-0 file:bg-primary file:px-3 file:py-1 file:text-primary-foreground file:text-sm file:cursor-pointer cursor-pointer text-muted-foreground"
        onchange={onFileChange}
        disabled={loading}
      />
      {#if loading}
        <p class="mt-1 text-xs text-muted-foreground">Parsing…</p>
      {/if}
      {#if parseError}
        <p class="mt-1 text-xs text-destructive">{parseError}</p>
      {/if}
    </div>
  {:else}
    <!-- Type-summary panel: one grid so name/type/levels/add align across rows -->
    <div class="rounded border border-border bg-muted/30 p-2 text-xs space-y-1">
      <div class="grid grid-cols-[auto_auto_1fr_auto] gap-x-3 gap-y-0.5 items-baseline">
        <span class="font-medium text-muted-foreground">Column</span>
        <span class="font-medium text-muted-foreground">Type</span>
        <span class="font-medium text-muted-foreground">Levels</span>
        <span></span>
        {#each (uploadStore.summary?.columns ?? []) as col (col.name)}
          <span class="font-mono truncate max-w-40" title={col.name}>{col.name}</span>
          <span class="text-muted-foreground">{col.colType}</span>
          <span class="text-muted-foreground">
            {#if col.colType === 'factor'}
              {col.labels.length} ({col.labels.join(', ')})
            {:else if col.colType === 'binary'}
              2
            {:else}
              —
            {/if}
          </span>
          {#if col.name}
            <button
              type="button"
              title={isInFormula(cfg.formula, col.name) ? 'Already in formula' : 'Add to formula'}
              disabled={isInFormula(cfg.formula, col.name)}
              class="justify-self-end rounded p-0.5 text-muted-foreground hover:bg-muted hover:text-foreground disabled:cursor-default disabled:opacity-40 disabled:hover:bg-transparent"
              onclick={() => addToFormula(col.name)}
            >
              <Plus class="h-3.5 w-3.5" />
            </button>
          {:else}
            <span></span>
          {/if}
        {/each}
      </div>
      <p class="text-muted-foreground pt-1">
        {uploadStore.csvData.n_rows} rows · {uploadStore.csvData.columns.length} columns
      </p>
    </div>

    <!-- Mode selector -->
    <div class="flex items-center gap-2">
      <span class="text-xs text-muted-foreground">Mode:</span>
      <InfoIcon tipKey="uploadMode" />
      <div role="group" aria-label="Upload mode" class="flex gap-1 text-xs">
        {#each MODE_OPTIONS as opt (opt.value)}
          <button
            type="button"
            title={opt.title}
            class="rounded px-2 py-0.5 {uploadStore.mode === opt.value
              ? 'bg-primary text-primary-foreground'
              : 'bg-muted text-muted-foreground hover:bg-muted/70'}"
            onclick={() => onModeChange(opt.value)}
          >
            {opt.label}
          </button>
        {/each}
      </div>
    </div>
  {/if}
</div>
