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

  // Keep the levels panel compact: clip each level label and cap the list length.
  // The leading count (col.labels.length) still reports the true total.
  const MAX_LEVEL_LABEL = 24;
  const MAX_LEVELS_SHOWN = 7;
  function formatLevels(labels: string[]): string {
    const shown = labels
      .slice(0, MAX_LEVELS_SHOWN)
      .map((l) => (l.length > MAX_LEVEL_LABEL ? l.slice(0, MAX_LEVEL_LABEL) + '…' : l));
    if (labels.length > MAX_LEVELS_SHOWN) shown.push('…');
    return shown.join(', ');
  }

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
    if (f === '') cfg.formula = `y = ${name}`;
    // Trailing `=`/`~` = outcome set, no predictors yet (set via the y toggle):
    // append as the first predictor, not after a stray `+`.
    else if (/[=~]$/.test(f)) cfg.formula = `${f} ${name}`;
    else cfg.formula = `${f} + ${name}`;
  }

  // Outcome (LHS) helpers. Source of truth is cfg.formula; selected state is
  // derived by a synchronous LHS split — matching isInFormula's sync regex, not
  // the async parsed-formula store (see the comment above). Both `=` and `~` are
  // accepted LHS/RHS separators (the app writes `=`); split on whichever is first.
  function splitFormula(formula: string): { lhs: string; rhs: string } {
    const i = formula.search(/[=~]/);
    if (i === -1) return { lhs: '', rhs: formula.trim() };
    return { lhs: formula.slice(0, i).trim(), rhs: formula.slice(i + 1).trim() };
  }

  // Drop a bare main-effect token `name` from the RHS, preserving order. Splits
  // on top-level `+` only so a `+` inside a random-effect term like `(1+x|g)`
  // stays put. Interactions (`name:z`) that embed the name are intentionally left
  // — only a standalone predictor is removed (mutual exclusivity with outcome).
  function removeMainFromRhs(rhs: string, name: string): string {
    const tokens: string[] = [];
    let depth = 0;
    let current = '';
    for (const ch of rhs) {
      if (ch === '(') depth++;
      else if (ch === ')') depth--;
      if (ch === '+' && depth === 0) {
        tokens.push(current);
        current = '';
      } else {
        current += ch;
      }
    }
    tokens.push(current);
    return tokens
      .map((t) => t.trim())
      .filter((t) => t !== '' && t !== name)
      .join(' + ');
  }

  function isOutcome(formula: string, name: string): boolean {
    return splitFormula(formula).lhs === name;
  }

  // Set `name` as the outcome: rewrite the LHS and strip `name` from the RHS if it
  // was a predictor. The previous outcome (old LHS) is discarded entirely, not
  // demoted to a predictor — silent promotion is surprising.
  function setOutcomeColumn(name: string) {
    const cleaned = removeMainFromRhs(splitFormula(cfg.formula).rhs, name);
    cfg.formula = cleaned === '' ? `${name} =` : `${name} = ${cleaned}`;
  }

  // Clear the outcome: LHS falls back to the placeholder `y` (or fully empty when
  // there are no predictors either).
  function clearOutcomeColumn() {
    const { rhs } = splitFormula(cfg.formula);
    cfg.formula = rhs === '' ? '' : `y = ${rhs}`;
  }

  function toggleOutcome(name: string) {
    if (isOutcome(cfg.formula, name)) clearOutcomeColumn();
    else setOutcomeColumn(name);
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
              {col.labels.length} ({formatLevels(col.labels)})
            {:else if col.colType === 'binary'}
              2
            {:else}
              —
            {/if}
          </span>
          {#if col.name}
            <div class="justify-self-end flex items-center gap-1">
              <!-- Outcome (y) toggle: exactly one column is the LHS at a time. -->
              <button
                type="button"
                title={isOutcome(cfg.formula, col.name) ? 'Outcome (y) — click to clear' : 'Set as outcome (y)'}
                aria-pressed={isOutcome(cfg.formula, col.name)}
                class="rounded px-1.5 py-0.5 font-mono text-[11px] italic {isOutcome(cfg.formula, col.name)
                  ? 'bg-primary text-primary-foreground'
                  : 'text-muted-foreground hover:bg-muted hover:text-foreground'}"
                onclick={() => toggleOutcome(col.name)}
              >
                y
              </button>
              <button
                type="button"
                title={isOutcome(cfg.formula, col.name)
                  ? 'Is the outcome'
                  : isInFormula(cfg.formula, col.name)
                    ? 'Already in formula'
                    : 'Add to formula'}
                disabled={isInFormula(cfg.formula, col.name)}
                class="rounded p-0.5 text-muted-foreground hover:bg-muted hover:text-foreground disabled:cursor-default disabled:opacity-40 disabled:hover:bg-transparent"
                onclick={() => addToFormula(col.name)}
              >
                <Plus class="h-3.5 w-3.5" />
              </button>
            </div>
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
