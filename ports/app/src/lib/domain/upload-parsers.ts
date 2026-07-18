// Multi-format upload parser layer for the app's data uploader.
// Normalizes CSV/TSV (PapaParse), XLSX (read-excel-file), and SPSS .sav
// (sav-reader) into the shared ParsedCsv shape consumed by detectColumnTypes /
// csvDataFromParsed. Every parser library is lazy-imported on first use (same
// pattern as charts/embed.ts) so it never enters the main bundle — the wasm
// binary is untouched; only the chunk for the picked format downloads.
//
// The engine still receives the existing typed CsvData: this layer does host-side
// ingestion only (RULE: the engine does no I/O and no data parsing).
import type { ParsedCsv } from './upload-detect';
import { UPLOAD } from '$lib/configs/app-config';

export type UploadFormat = 'csv' | 'xlsx' | 'sav';

/** A separator + quote pair for delimited files. `separator` is the raw
 * character ("," / "\t" / ";" / "|" / custom); `quote` is the text-quote char. */
export interface Dialect {
    separator: string;
    quote: string;
}

export interface ParseOptions {
    /** csv/tsv only: override the auto-detected separator + quote. */
    dialect?: Dialect;
    /** xlsx only: which sheet to read (name). Defaults to the first sheet. */
    sheet?: string;
    /** csv/tsv only: parse just this many data rows for a fast preview.
     * Omitted/0 = full parse. xlsx/.sav have no partial read and ignore it. */
    previewRows?: number;
}

export interface ParseOutcome {
    format: UploadFormat;
    parsed: ParsedCsv;
    /** csv/tsv: the separator + quote actually used. On an auto-parse this is the
     * detected separator (PapaParse `meta.delimiter`) and the default quote. */
    dialect?: Dialect;
    /** xlsx: every sheet name in the workbook (only when the sheet was not
     * pre-selected — the caller keeps the list across re-selections). */
    sheetNames?: string[];
    /** sav: variable name → SPSS variable label, for preview display only. */
    variableLabels?: Record<string, string>;
}

// ---------------------------------------------------------------------------
// Cell stringification — ParsedCsv.columns is string[][], so non-string cells
// from xlsx/.sav must become strings. Numbers/booleans → String(v); Date cells
// → ISO (a bare String(date) yields "Mon Jan 15 2024…", which detectColumnTypes'
// Number(cell) then misreads as a high-cardinality factor); null/undefined → ''.
// ---------------------------------------------------------------------------

function cellToString(v: unknown): string {
    if (v === null || v === undefined) return '';
    if (v instanceof Date) return v.toISOString();
    return String(v);
}

/** Transpose header + row-major data into ParsedCsv's col-major columns,
 * padding short rows and truncating long ones to the header width (mirrors the
 * old parseCsvText, which padded short rows with '' and ignored extra fields). */
function toParsedCsv(header: string[], dataRows: string[][]): ParsedCsv {
    const nCols = header.length;
    const columns: string[][] = header.map(() => []);
    for (const row of dataRows) {
        for (let c = 0; c < nCols; c++) columns[c]!.push(row[c] ?? '');
    }
    return { header, columns, nRows: dataRows.length };
}

function extensionOf(filename: string): string {
    const dot = filename.lastIndexOf('.');
    return dot === -1 ? '' : filename.slice(dot).toLowerCase();
}

// ---------------------------------------------------------------------------
// CSV / TSV — PapaParse
// ---------------------------------------------------------------------------

async function parseDelimited(file: File, options: ParseOptions): Promise<ParseOutcome> {
    const { default: Papa } = await import('papaparse');
    const text = await file.text();

    const preview = options.previewRows ?? 0;
    const result = Papa.parse<string[]>(text, {
        // Empty delimiter = auto-detect; PapaParse reports the choice in meta.delimiter.
        delimiter: options.dialect?.separator ?? '',
        quoteChar: options.dialect?.quote ?? '"',
        skipEmptyLines: true,
        // preview counts the header row too, so ask for one extra to get N data rows.
        preview: preview > 0 ? preview + 1 : 0,
    });

    // PapaParse keeps parsing past a malformed row (e.g. an unterminated quote)
    // instead of throwing, so a silent field-mismatch would otherwise reach the
    // preview as a mangled row with no indication anything went wrong.
    if (result.errors.length > 0) {
        const first = result.errors[0]!;
        const where = first.row !== undefined ? ` at row ${first.row + 1}` : '';
        throw new Error(`Couldn't parse this file${where}: ${first.message}.`);
    }

    const rows = result.data;
    const header = (rows[0] ?? []).map((c) => c ?? '');
    const dataRows = rows.slice(1);
    const separator = result.meta.delimiter || options.dialect?.separator || ',';

    return {
        format: 'csv',
        parsed: toParsedCsv(header, dataRows),
        dialect: { separator, quote: options.dialect?.quote ?? '"' },
    };
}

// ---------------------------------------------------------------------------
// XLSX — read-excel-file (browser build)
// ---------------------------------------------------------------------------

async function parseXlsx(file: File, options: ParseOptions): Promise<ParseOutcome> {
    const mod = await import('read-excel-file/browser');
    const readXlsxFile = mod.default;
    const readSheet = mod.readSheet;

    let data: (string | number | boolean | Date | null)[][];
    let sheetNames: string[] | undefined;

    if (options.sheet) {
        // readSheet's sheet arg is POSITIONAL (a { sheet } object is silently
        // ignored and returns sheet 1). The caller already holds the sheet list.
        data = (await readSheet(file, options.sheet)) as typeof data;
    } else {
        const sheets = await readXlsxFile(file);
        sheetNames = sheets.map((s) => s.sheet);
        data = (sheets[0]?.data ?? []) as typeof data;
    }

    const header = (data[0] ?? []).map(cellToString);
    const dataRows = data.slice(1).map((r) => r.map(cellToString));

    return { format: 'xlsx', parsed: toParsedCsv(header, dataRows), sheetNames };
}

// ---------------------------------------------------------------------------
// SPSS .sav — sav-reader (needs a real Buffer global; see vite.config alias recipe)
// ---------------------------------------------------------------------------

async function parseSav(file: File): Promise<ParseOutcome> {
    const { Buffer } = await import('buffer');
    // sav-reader's stream stack needs a global Buffer. Browsers have none, so install
    // the polyfill; Node/vitest already provide a native Buffer and overriding it with
    // the polyfill breaks readable-stream's chunk types — so only set it when absent.
    const g = globalThis as { Buffer?: typeof Buffer };
    if (!g.Buffer) g.Buffer = Buffer;
    const BufferImpl = g.Buffer;
    const { SavBufferReader } = await import('sav-reader');

    // A bare Uint8Array throws inside sav-reader's AsyncChunkReader — must be a real Buffer.
    const buf = BufferImpl.from(await file.arrayBuffer());
    const sav = new SavBufferReader(buf);
    try {
        await sav.open();
    } catch (err) {
        throw mapSavOpenError(err);
    }

    // open() loads the header + variable metadata without materialising any rows.
    // n_cases is the file's declared row count (-1 if the writer couldn't seek back
    // to fill it in) — checking the upper cap here rejects an oversized file before
    // readAllRows() (which holds the full row-object array plus the transposed
    // string columns in memory at once) rather than after. Only the max cap applies
    // pre-parse; a too-small file is still a valid (small) read, so min_rows stays
    // a post-parse check (validateUploadRowCount, run by the caller on the result).
    const declaredRows = sav.meta.header.n_cases;
    const maxRows = import.meta.env.VITE_TARGET === 'wasm' ? UPLOAD.max_rows_wasm : UPLOAD.max_rows;
    if (declaredRows >= 0 && declaredRows > maxRows) {
        throw new Error(`Upload has ${declaredRows} rows; the maximum is ${maxRows}.`);
    }

    // Skip sav-reader's internal continuation vars for long strings; keep user-facing columns.
    const sysvars = sav.meta.sysvars.filter((v) => !v.__is_child_string_var);
    const header = sysvars.map((v) => v.name);
    const variableLabels: Record<string, string> = {};
    for (const v of sysvars) if (v.label) variableLabels[v.name] = v.label;

    const rows = (await sav.readAllRows()) as Record<string, unknown>[];
    const columns = sysvars.map((v) => rows.map((r) => cellToString(r[v.name])));

    return {
        format: 'sav',
        parsed: { header, columns, nRows: rows.length },
        variableLabels,
    };
}

/** sav-reader throws on ZLIB-compressed ($FL3-magic) .sav files. Turn that into
 * the actionable message; anything else becomes a generic readable error. */
function mapSavOpenError(err: unknown): Error {
    const msg = err instanceof Error ? err.message : String(err);
    if (/zlib|compress|fl3|\$fl3/i.test(msg)) {
        return new Error(
            "This .sav file uses ZLIB compression, which isn't supported — re-save it uncompressed in SPSS, or export CSV.",
        );
    }
    return new Error("Couldn't read this .sav file — it may be corrupt or an unsupported variant.");
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

/** Parse an uploaded file into the shared ParsedCsv shape, switching on
 * extension. Throws a readable Error for binary-format failures and unsupported
 * types; delimited files are parsed leniently (short rows padded, not rejected). */
export async function parseFile(file: File, options: ParseOptions = {}): Promise<ParseOutcome> {
    const ext = extensionOf(file.name);
    switch (ext) {
        case '.csv':
        case '.tsv':
        case '.txt':
            return parseDelimited(file, options);
        case '.xlsx':
            return parseXlsx(file, options);
        case '.sav':
            return parseSav(file);
        default:
            throw new Error(`Unsupported file type "${ext || file.name}". Use CSV, TSV, XLSX, or .sav.`);
    }
}
