// Upload store: holds the current CsvData, chosen mode, source filename, and type summary.
// Mirrors the pattern of parsed-formula.svelte.ts and family.svelte.ts.
import type { CsvData, UploadMode } from '$lib/domain/app-spec';

export interface UploadTypeSummary {
    /** Per column: name, detected type, labels (for factors). */
    columns: Array<{
        name: string;
        colType: 'continuous' | 'binary' | 'factor';
        labels: string[];
    }>;
}

function createUploadStore() {
    let csvData = $state<CsvData | null>(null);
    let filename = $state<string>('');
    let mode = $state<UploadMode>('partial');

    const summary = $derived<UploadTypeSummary | null>(
        csvData
            ? {
                  columns: csvData.columns.map((c) => ({
                      name: c.name,
                      colType: c.col_type,
                      labels: c.labels,
                  })),
              }
            : null,
    );

    return {
        get csvData() { return csvData; },
        get filename() { return filename; },
        get mode(): UploadMode { return mode; },
        set mode(v: UploadMode) { mode = v; },
        get summary() { return summary; },

        /** Load parsed CsvData from a file; always stamps the mode from the current store mode. */
        load(data: CsvData, name: string) {
            // Propagate the current mode into the data before storing.
            csvData = { ...data, mode };
            filename = name;
        },

        /** Update mode on the stored CsvData (keeps existing data). */
        setMode(v: UploadMode) {
            mode = v;
            if (csvData) csvData = { ...csvData, mode: v };
        },

        clear() {
            csvData = null;
            filename = '';
        },
    };
}

export const uploadStore = createUploadStore();
