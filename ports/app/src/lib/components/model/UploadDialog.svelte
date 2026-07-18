<script lang="ts">
  // Staging dialog for the data uploader: parses the picked file (CSV/TSV/XLSX/.sav)
  // in the frontend, shows a live preview, lets the user override CSV dialect or pick
  // an xlsx sheet, and commits to uploadStore only on "Use this data". Nothing touches
  // the store until confirm — cancel/close discards everything.
  import { Dialog, DialogContent, DialogFooter, DialogTitle } from '$lib/components/ui/dialog';
  import { Select, SelectContent, SelectItem, SelectTrigger } from '$lib/components/ui/select';
  import { Input } from '$lib/components/ui/input';
  import { Button } from '$lib/components/ui/button';
  import { Label } from '$lib/components/ui/label';
  import { csvDataFromParsed, validateUploadRowCount, type ParsedCsv } from '$lib/domain/upload-detect';
  import { parseFile, type Dialect } from '$lib/domain/upload-parsers';
  import { uploadStore } from '$lib/stores/upload.svelte';

  let {
    file,
    open,
    onOpenChange,
  }: { file: File | null; open: boolean; onOpenChange: (v: boolean) => void } = $props();

  const PREVIEW_ROWS = 10;
  // .sav/.xlsx materialise several full-file copies at once during parsing
  // (raw bytes + a Buffer copy + row objects + transposed columns for .sav;
  // the parsed workbook for .xlsx) — CSV/TSV only ever holds the text once.
  // There is no cheap pre-parse row count for .xlsx, so this is a coarse byte
  // cap that rejects a clearly-oversized file before the expensive full parse,
  // ahead of the precise row-count check that runs on the parsed result.
  const MAX_BINARY_UPLOAD_BYTES = 200 * 1024 * 1024;

  // ---- Dialect option maps (CSV/TSV) ----
  const SEP_CHARS: Record<string, string> = { comma: ',', semicolon: ';', tab: '\t', pipe: '|' };
  const QUOTE_CHARS: Record<string, string> = { double: '"', single: "'" };

  function sepChoiceFor(char: string): string {
    return Object.keys(SEP_CHARS).find((k) => SEP_CHARS[k] === char) ?? 'custom';
  }
  function quoteChoiceFor(char: string): string {
    return Object.keys(QUOTE_CHARS).find((k) => QUOTE_CHARS[k] === char) ?? 'custom';
  }
  function friendlySeparator(char: string): string {
    if (char === ',') return 'comma';
    if (char === ';') return 'semicolon';
    if (char === '\t') return 'tab';
    if (char === '|') return 'pipe';
    return `"${char}"`;
  }

  // ---- Staged state (reset each time a new file opens) ----
  let loading = $state(false);
  let error = $state<string | null>(null);
  let format = $state<'csv' | 'xlsx' | 'sav'>('csv');
  // `parsed` is what the preview renders — the full parse initially, a preview slice
  // after a dialect change. `totalRows` is the true file row count from the last full
  // parse (a dialect change never alters it — the file's line count is separator-agnostic).
  let parsed = $state<ParsedCsv | null>(null);
  let totalRows = $state(0);
  // `fullParsed` is non-null only while it matches the current dialect/sheet; a dialect
  // change nulls it, forcing a fresh full parse on confirm.
  let fullParsed = $state<ParsedCsv | null>(null);

  let detectedDialect = $state<Dialect>({ separator: ',', quote: '"' });
  let separatorChoice = $state('comma');
  let customSeparator = $state(',');
  let quoteChoice = $state('double');
  let customQuote = $state('"');

  let sheetNames = $state<string[]>([]);
  let sheet = $state('');
  let variableLabels = $state<Record<string, string>>({});

  const currentDialect = $derived<Dialect>({
    separator: separatorChoice === 'custom' ? customSeparator : (SEP_CHARS[separatorChoice] ?? ','),
    quote: quoteChoice === 'custom' ? customQuote : (QUOTE_CHARS[quoteChoice] ?? '"'),
  });

  // Wide files get a roomier dialog before falling back to horizontal scroll.
  const wide = $derived((parsed?.header.length ?? 0) > 8);

  const previewRows = $derived.by(() => {
    const p = parsed;
    if (!p) return [];
    const n = Math.min(PREVIEW_ROWS, p.nRows);
    const rows: string[][] = [];
    for (let r = 0; r < n; r++) rows.push(p.columns.map((col) => col[r] ?? ''));
    return rows;
  });

  const canConfirm = $derived(!loading && !error && parsed !== null && parsed.header.length > 0);

  // ---- Parse orchestration ----
  // Re-init whenever a new file is opened; clear on close.
  let lastFile: File | null = null;
  $effect(() => {
    if (open && file && file !== lastFile) {
      lastFile = file;
      void initParse(file);
    }
    if (!open) lastFile = null;
  });

  /** Pre-parse size gate for .sav/.xlsx — returns an error message, or `null` when acceptable. */
  function checkBinaryUploadSize(f: File): string | null {
    const ext = f.name.slice(f.name.lastIndexOf('.')).toLowerCase();
    if ((ext !== '.sav' && ext !== '.xlsx') || f.size <= MAX_BINARY_UPLOAD_BYTES) return null;
    const mb = (n: number) => Math.round(n / (1024 * 1024));
    return `File is ${mb(f.size)} MB; ${ext} uploads are capped at ${mb(MAX_BINARY_UPLOAD_BYTES)} MB to avoid running out of browser memory while parsing.`;
  }

  function resetState() {
    error = null;
    parsed = null;
    fullParsed = null;
    totalRows = 0;
    sheetNames = [];
    sheet = '';
    variableLabels = {};
  }

  async function initParse(f: File) {
    resetState();
    const sizeErr = checkBinaryUploadSize(f);
    if (sizeErr) {
      error = sizeErr;
      return;
    }
    loading = true;
    try {
      const outcome = await parseFile(f);
      format = outcome.format;
      parsed = outcome.parsed;
      fullParsed = outcome.parsed;
      totalRows = outcome.parsed.nRows;
      if (outcome.dialect) {
        detectedDialect = outcome.dialect;
        separatorChoice = sepChoiceFor(outcome.dialect.separator);
        customSeparator = outcome.dialect.separator;
        quoteChoice = quoteChoiceFor(outcome.dialect.quote);
        customQuote = outcome.dialect.quote;
      }
      if (outcome.sheetNames) {
        sheetNames = outcome.sheetNames;
        sheet = outcome.sheetNames[0] ?? '';
      }
      if (outcome.variableLabels) variableLabels = outcome.variableLabels;
      validateFull(outcome.parsed);
    } catch (err) {
      error = err instanceof Error ? err.message : 'Failed to parse the file.';
    } finally {
      loading = false;
    }
  }

  // Post-full-parse gates (empty file + row-count). Sets `error`, returns validity.
  function validateFull(p: ParsedCsv): boolean {
    if (p.header.length === 0 || p.nRows === 0) {
      error = 'File appears empty or has no data rows.';
      return false;
    }
    const rowErr = validateUploadRowCount(p.nRows);
    if (rowErr) {
      error = rowErr;
      return false;
    }
    error = null;
    return true;
  }

  async function reparsePreview() {
    if (!file) return;
    loading = true;
    error = null;
    fullParsed = null; // dialect changed — the cached full parse is stale.
    try {
      const outcome = await parseFile(file, { dialect: currentDialect, previewRows: PREVIEW_ROWS });
      parsed = outcome.parsed;
    } catch (err) {
      error = err instanceof Error ? err.message : 'Failed to parse the file.';
    } finally {
      loading = false;
    }
  }

  async function changeSheet(name: string) {
    if (!file) return;
    sheet = name;
    loading = true;
    error = null;
    try {
      const outcome = await parseFile(file, { sheet: name });
      parsed = outcome.parsed;
      fullParsed = outcome.parsed; // xlsx has no partial read — this is a full parse.
      totalRows = outcome.parsed.nRows;
      validateFull(outcome.parsed);
    } catch (err) {
      error = err instanceof Error ? err.message : 'Failed to read the sheet.';
    } finally {
      loading = false;
    }
  }

  function onSeparatorChange(v: string) {
    separatorChoice = v;
    void reparsePreview();
  }
  function onCustomSeparatorInput() {
    if (separatorChoice === 'custom') void reparsePreview();
  }
  function onQuoteChange(v: string) {
    quoteChoice = v;
    void reparsePreview();
  }
  function onCustomQuoteInput() {
    if (quoteChoice === 'custom') void reparsePreview();
  }

  async function confirm() {
    if (!file) return;
    // Guarantee a full parse for the current dialect/sheet before committing.
    let full = fullParsed;
    if (!full) {
      loading = true;
      try {
        const opts = format === 'csv' ? { dialect: currentDialect } : {};
        full = (await parseFile(file, opts)).parsed;
      } catch (err) {
        error = err instanceof Error ? err.message : 'Failed to parse the file.';
        loading = false;
        return;
      }
      loading = false;
    }
    if (!validateFull(full)) return;
    const data = csvDataFromParsed(full, uploadStore.mode);
    uploadStore.load(data, file.name);
    onOpenChange(false);
  }
</script>

<Dialog {open} {onOpenChange}>
  <DialogContent class="max-h-[85vh] overflow-y-auto {wide ? 'sm:max-w-4xl' : 'sm:max-w-2xl'}">
    <DialogTitle class="truncate font-mono">{file?.name ?? 'Upload'}</DialogTitle>

    {#if loading}
      <div class="py-8 text-center text-sm text-muted-foreground">
        <span class="animate-pulse">Parsing {file?.name ?? 'file'}…</span>
      </div>
    {:else}
      {#if format === 'csv'}
        <div class="flex gap-3">
          <div class="flex-1 min-w-0 space-y-1.5">
            <Label class="text-xs font-normal">
              Separator
              <span class="text-muted-foreground">(detected: {friendlySeparator(detectedDialect.separator)})</span>
            </Label>
            <Select type="single" value={separatorChoice} onValueChange={onSeparatorChange}>
              <SelectTrigger class="h-8 w-full text-xs" data-testid="upload-separator">
                {separatorChoice === 'custom' ? 'Custom…' : separatorChoice.charAt(0).toUpperCase() + separatorChoice.slice(1)}
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="comma">Comma</SelectItem>
                <SelectItem value="semicolon">Semicolon</SelectItem>
                <SelectItem value="tab">Tab</SelectItem>
                <SelectItem value="pipe">Pipe</SelectItem>
                <SelectItem value="custom">Custom…</SelectItem>
              </SelectContent>
            </Select>
            {#if separatorChoice === 'custom'}
              <Input
                bind:value={customSeparator}
                maxlength={1}
                class="h-8 w-16 text-center text-xs"
                oninput={onCustomSeparatorInput}
                aria-label="Custom separator character"
              />
            {/if}
          </div>
          <div class="flex-1 min-w-0 space-y-1.5">
            <Label class="text-xs font-normal">
              Quote
              <span class="text-muted-foreground">(detected: {detectedDialect.quote})</span>
            </Label>
            <Select type="single" value={quoteChoice} onValueChange={onQuoteChange}>
              <SelectTrigger class="h-8 w-full text-xs" data-testid="upload-quote">
                {quoteChoice === 'double' ? '"' : quoteChoice === 'single' ? "'" : 'Custom…'}
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="double">"</SelectItem>
                <SelectItem value="single">'</SelectItem>
                <SelectItem value="custom">Custom…</SelectItem>
              </SelectContent>
            </Select>
            {#if quoteChoice === 'custom'}
              <Input
                bind:value={customQuote}
                maxlength={1}
                class="h-8 w-16 text-center text-xs"
                oninput={onCustomQuoteInput}
                aria-label="Custom quote character"
              />
            {/if}
          </div>
        </div>
      {/if}

      {#if format === 'xlsx' && sheetNames.length > 1}
        <div class="max-w-64 space-y-1.5">
          <Label class="text-xs font-normal">Sheet</Label>
          <Select type="single" value={sheet} onValueChange={changeSheet}>
            <SelectTrigger class="h-8 w-full text-xs" data-testid="upload-sheet">{sheet}</SelectTrigger>
            <SelectContent>
              {#each sheetNames as name (name)}
                <SelectItem value={name}>{name}</SelectItem>
              {/each}
            </SelectContent>
          </Select>
        </div>
      {/if}

      {#if parsed}
        <!-- min-w-0 keeps this grid item from sizing to the table's (wide) min-content,
             which would blow out the dialog's grid track; horizontal scroll is confined
             to the inner wrapper so only the data scrolls — the dialect controls and
             footer stay put. -->
        <div class="min-w-0 rounded border border-border bg-muted/30 p-2 text-xs">
          <div class="w-full min-w-0 overflow-x-auto">
            <table class="w-full border-collapse">
              <thead>
                <tr>
                  {#each parsed.header as name, ci (ci)}
                    <th
                      class="max-w-40 truncate whitespace-nowrap border-b border-border pr-3 text-left font-mono font-semibold"
                      title={variableLabels[name] ?? name}
                    >
                      {name}
                    </th>
                  {/each}
                </tr>
              </thead>
              <tbody>
                {#each previewRows as row, ri (ri)}
                  <tr>
                    {#each row as cell, ci (ci)}
                      <td class="max-w-40 truncate whitespace-nowrap pr-3">{cell}</td>
                    {/each}
                  </tr>
                {/each}
              </tbody>
            </table>
          </div>
          <p class="pt-1.5 text-muted-foreground">
            showing {Math.min(PREVIEW_ROWS, parsed.nRows)} of {totalRows.toLocaleString('en-US')} rows
            · {parsed.header.length} columns
          </p>
        </div>
        {#if format === 'sav' && Object.keys(variableLabels).length > 0}
          <p class="text-xs text-muted-foreground">
            Variable labels shown as tooltips on column headers — preview only, not sent to the model.
          </p>
        {/if}
      {/if}
    {/if}

    <DialogFooter>
      {#if error}
        <span class="mr-auto self-center text-xs text-destructive">{error}</span>
      {/if}
      <Button variant="ghost" onclick={() => onOpenChange(false)}>Cancel</Button>
      <Button onclick={confirm} disabled={!canConfirm}>Use this data</Button>
    </DialogFooter>
  </DialogContent>
</Dialog>
